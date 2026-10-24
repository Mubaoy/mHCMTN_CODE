from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import torch
import torch.nn.functional as F

DEFAULT_CATEGORIES = [3, 3, 3, 2, 1, 1, 2, 1, 1, 1, 1, 3]


class PiecewiseSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return torch.where(x < .05, torch.zeros_like(x),
                           torch.where(x <= .15, 5 * (x - .05), torch.full_like(x, .5)))

    @staticmethod
    def backward(ctx, gradient):
        return gradient


def decode_continuous(logits, reference_alphas):
    # Invert the middle segment so zero residual logits reproduce the reference.
    p = ((.05 + reference_alphas / 5) / .2).clamp(1e-5, 1 - 1e-5)
    return PiecewiseSTE.apply(.2 * torch.sigmoid(logits + torch.logit(p)))


def redistribute_load(base, membership, alphas):
    """Nonnegative, energy-conserving demand response for equal-duration slots.

    membership: [..., slots, 3], categorical order valley/flat/peak.
    alphas: [..., 3], order peak->valley, peak->flat, flat->valley.
    Source withdrawals are proportional to source-slot loads. Incoming energy
    is shared uniformly across the destination slots. Empty destinations disable
    that transfer rather than silently destroying energy.
    """
    valley, flat, peak = membership.unbind(-1)
    nv, nf = valley.sum(-1, keepdim=True), flat.sum(-1, keepdim=True)
    active_v, active_f = (nv > 0).to(base.dtype), (nf > 0).to(base.dtype)
    apv, apf, afv = [a.unsqueeze(-1) for a in alphas.unbind(-1)]
    apv, apf, afv = apv * active_v, apf * active_f, afv * active_v
    ep, ef = (base * peak).sum(-1, keepdim=True), (base * flat).sum(-1, keepdim=True)
    retained = base * (1 - peak * (apv + apf) - flat * afv)
    return retained + flat / nf.clamp_min(1) * (apf * ep) + valley / nv.clamp_min(1) * (apv * ep + afv * ef)


@dataclass
class ScheduleProblem:
    pv_forecast_w: np.ndarray
    baseline_load_w: np.ndarray
    step_hours: float = 1.0
    initial_categories: list | None = None
    initial_alphas: list | None = None
    tariff_per_kwh: list | None = None
    actual_pv_w: np.ndarray | None = None
    name: str = "case"
    data_kind: str = "user_provided"

    def __post_init__(self):
        self.pv_forecast_w = np.asarray(self.pv_forecast_w, dtype=float)
        self.baseline_load_w = np.asarray(self.baseline_load_w, dtype=float)
        if (self.pv_forecast_w.shape != (12,) or self.baseline_load_w.shape != (12,)
                or not np.isfinite(self.pv_forecast_w).all() or not np.isfinite(self.baseline_load_w).all()
                or (self.pv_forecast_w < 0).any() or (self.baseline_load_w < 0).any()
                or not np.isfinite(self.step_hours) or self.step_hours <= 0):
            raise ValueError("Scheduling requires twelve finite nonnegative slots and a positive slot duration.")
        self.initial_categories = list(DEFAULT_CATEGORIES if self.initial_categories is None else self.initial_categories)
        self.initial_alphas = list([.13, .01, .12] if self.initial_alphas is None else self.initial_alphas)
        if len(self.initial_categories) != 12 or not set(self.initial_categories) <= {1, 2, 3}:
            raise ValueError("initial_categories must contain twelve labels from {1,2,3}.")
        if len(self.initial_alphas) != 3 or not np.isfinite(self.initial_alphas).all() or np.any(np.array(self.initial_alphas) < 0) or np.any(np.array(self.initial_alphas) > .5):
            raise ValueError("initial_alphas must contain three values in [0,0.5].")
        self.tariff_per_kwh = list([.80, .80, .80, .55, .30, .30, .55, .30, .30, .30, .30, .80]
                                   if self.tariff_per_kwh is None else self.tariff_per_kwh)
        if len(self.tariff_per_kwh) != 12 or not np.isfinite(self.tariff_per_kwh).all() or min(self.tariff_per_kwh) < 0:
            raise ValueError("tariff_per_kwh must contain twelve nonnegative prices.")
        if self.actual_pv_w is not None:
            self.actual_pv_w = np.asarray(self.actual_pv_w, float)
            if self.actual_pv_w.shape != (12,) or not np.isfinite(self.actual_pv_w).all() or (self.actual_pv_w < 0).any():
                raise ValueError("actual_pv_w must contain twelve finite nonnegative values.")

    @property
    def scale(self):
        return max(1.0, float(max(self.pv_forecast_w.max(), self.baseline_load_w.max())))

    def context(self, device="cpu"):
        # Observed future PV is deliberately excluded from the optimization input.
        return torch.tensor(np.r_[self.pv_forecast_w / self.scale,
                                   self.baseline_load_w / self.scale,
                                   np.array(self.tariff_per_kwh) / max(1., max(self.tariff_per_kwh))],
                            dtype=torch.float32, device=device).unsqueeze(0)

    def initial(self, device="cpu"):
        return torch.tensor(self.initial_categories + self.initial_alphas,
                            dtype=torch.float32, device=device).unsqueeze(0)

    def objective(self, scheduled):
        pv = torch.as_tensor(self.pv_forecast_w, dtype=scheduled.dtype, device=scheduled.device)
        return ((scheduled - pv) / self.scale).square().mean(-1)

    def evaluate(self, categories, alphas):
        categories = np.asarray(categories)
        alphas = np.asarray(alphas, float)
        if categories.shape != (12,) or not np.isfinite(categories).all() or not np.isin(categories, [1, 2, 3]).all():
            raise ValueError("Categories must contain twelve integer labels in {1,2,3}.")
        if alphas.shape != (3,) or not np.isfinite(alphas).all() or (alphas < 0).any() or (alphas > .5).any():
            raise ValueError("Alphas must contain three finite values in [0,0.5].")
        base = torch.as_tensor(self.baseline_load_w, dtype=torch.float64)
        membership = F.one_hot(torch.tensor(categories, dtype=torch.long) - 1, 3).to(torch.float64)
        power = redistribute_load(base, membership, torch.tensor(alphas, dtype=torch.float64))
        score = float(self.objective(power))
        return score, power.numpy().copy()

    def batch_scores(self, positions):
        positions = np.asarray(positions, float)
        categories = np.rint(np.clip(positions[:, :12], 0, 1) * 2).astype(int) + 1
        alphas = np.clip(positions[:, 12:], 0, 1) * .5
        with torch.no_grad():
            members = F.one_hot(torch.tensor(categories) - 1, 3).to(torch.float64)
            power = redistribute_load(torch.tensor(self.baseline_load_w, dtype=torch.float64), members,
                                      torch.tensor(alphas, dtype=torch.float64))
            return self.objective(power).numpy()

    def metrics(self, scheduled, actual=False):
        pv = self.actual_pv_w if actual else self.pv_forecast_w
        if pv is None:
            raise ValueError("Actual PV observations were not supplied.")
        scheduled = np.asarray(scheduled, float)
        used, imported = np.minimum(scheduled, pv), np.maximum(scheduled - pv, 0)
        return {"mse_w2": float(np.mean((scheduled - pv) ** 2)),
                "self_consumption_pct": float(100 * used.sum() / pv.sum()) if pv.sum() > 0 else None,
                "self_sufficiency_pct": float(100 * used.sum() / scheduled.sum()) if scheduled.sum() > 0 else None,
                "grid_import_kwh": float(imported.sum() * self.step_hours / 1000),
                "peak_grid_import_w": float(imported.max()),
                "grid_cost_currency": float(np.dot(imported, self.tariff_per_kwh) * self.step_hours / 1000),
                "energy_balance_error_wh": float((scheduled.sum() - self.baseline_load_w.sum()) * self.step_hours),
                "minimum_load_w": float(scheduled.min())}

    def to_dict(self):
        return {"name": self.name, "data_kind": self.data_kind,
                "pv_forecast_w": self.pv_forecast_w.tolist(), "baseline_load_w": self.baseline_load_w.tolist(),
                "step_hours": self.step_hours, "initial_categories": self.initial_categories,
                "initial_alphas": self.initial_alphas, "tariff_per_kwh": self.tariff_per_kwh,
                "actual_pv_w": None if self.actual_pv_w is None else self.actual_pv_w.tolist()}
