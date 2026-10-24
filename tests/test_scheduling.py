import numpy as np
import pytest
import torch
from torch.nn import functional as F

from mhcmtn.model import MHCMTN, ModelConfig
from mhcmtn.optimization import OptimizerConfig, optimize_neural, optimize_population, run_optimization
from mhcmtn.scheduling import ScheduleProblem, decode_continuous, redistribute_load
from mhcmtn.utils import seed_everything


def test_decoder_conserves_energy_and_nonnegativity():
    rng = np.random.default_rng(82)
    base = torch.tensor(rng.uniform(0, 12000, (128, 12)), dtype=torch.float64)
    categories = rng.integers(0, 3, (128, 12))
    categories[:3] = np.arange(3)[:, None]  # Include every empty-destination case.
    membership = F.one_hot(torch.tensor(categories), 3).to(torch.float64)
    alphas = torch.tensor(rng.uniform(0, .5, (128, 3)), dtype=torch.float64)
    alphas[:3] = .5
    result = redistribute_load(base, membership, alphas)
    torch.testing.assert_close(result.sum(-1), base.sum(-1), atol=1e-9, rtol=1e-12)
    assert float(result.min()) >= -1e-10


def test_both_neural_decision_heads_and_geometry_receive_gradients(problem):
    seed_everything(42)
    model = MHCMTN(ModelConfig())
    current = problem.initial()
    logits, incentive, regularizer = model(current, problem.context())
    memberships = F.gumbel_softmax(logits, hard=True, tau=1., dim=-1)
    alphas = decode_continuous(incentive, current[:, 12:])
    load = redistribute_load(torch.tensor(problem.baseline_load_w, dtype=torch.float32)[None], memberships, alphas)
    loss = problem.objective(load).mean() + .1 * regularizer
    loss.backward()
    for name in ["blocks.0.attention.metric_factor", "blocks.0.attention.qkv.weight", "blocks.0.hyper.weight"]:
        gradient = dict(model.named_parameters())[name].grad
        assert gradient is not None and torch.isfinite(gradient).all() and gradient.abs().sum() > 0
    output_grad = model.output.weight.grad.reshape(3, 15, -1)
    assert output_grad[:, :12].abs().sum() > 0
    assert output_grad[:, 12:].abs().sum() > 0
    assert torch.equal(memberships.detach().sum(-1), torch.ones(1, 12))


@pytest.mark.parametrize("method", ["mhcmtn", "mlpo", "ga", "pso", "iwo", "wpa"])
def test_all_optimizers_return_feasible_recorded_incumbents(problem, method):
    cfg = OptimizerConfig(epochs=4, iterations=3, population=6)
    result, history, checkpoint = (optimize_neural(problem, cfg, ModelConfig(), method) if method in {"mhcmtn", "mlpo"}
                                    else optimize_population(problem, cfg, method))
    score, power = problem.evaluate(result["categories"], result["alphas"])
    assert result["objective"] == pytest.approx(score, abs=1e-12)
    np.testing.assert_allclose(result["scheduled_load_w"], power, atol=1e-10)
    assert power.min() >= -1e-8
    assert power.sum() == pytest.approx(problem.baseline_load_w.sum(), abs=1e-7)
    assert result["objective"] <= problem.evaluate(problem.initial_categories, [0, 0, 0])[0] + 1e-12
    assert np.all(np.diff([r["best_objective"] for r in history]) <= 1e-14)
    if checkpoint:
        assert result["objective_evaluations"] == 1 + 3 * cfg.epochs


def test_observed_future_pv_cannot_change_optimization(problem):
    first = ScheduleProblem(**{**problem.to_dict(), "actual_pv_w": np.zeros(12)})
    second = ScheduleProblem(**{**problem.to_dict(), "actual_pv_w": np.full(12, 10000.)})
    torch.testing.assert_close(first.context(), second.context())
    cfg = OptimizerConfig(epochs=4)
    a = optimize_neural(first, cfg, ModelConfig())[0]
    b = optimize_neural(second, cfg, ModelConfig())[0]
    assert a["categories"] == b["categories"] and a["alphas"] == b["alphas"]
    assert a["objective"] == b["objective"]


def test_saved_checkpoint_reproduces_reported_schedule(problem, tmp_path):
    result = run_optimization(problem, tmp_path, cfg=OptimizerConfig(epochs=3), plot=False)
    checkpoint = torch.load(tmp_path / "checkpoint.pt", weights_only=True)
    recovered = ScheduleProblem(**checkpoint["problem"])
    objective, power = recovered.evaluate(**checkpoint["best_decision"])
    assert objective == result["objective"]
    np.testing.assert_allclose(power, result["scheduled_load_w"])


def test_manuscript_width_forward(problem):
    model = MHCMTN(ModelConfig(d_model=64, nhead=8, dim_feedforward=256))
    labels, incentives, penalty = model(problem.initial(), problem.context())
    assert labels.shape == (1, 12, 3) and incentives.shape == (1, 3)
    assert torch.isfinite(penalty)
