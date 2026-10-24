from __future__ import annotations

from dataclasses import asdict, dataclass
import copy
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F

from .model import MHCMTN, MLPO, ModelConfig
from .scheduling import ScheduleProblem, redistribute_load, decode_continuous
from .utils import save_json, seed_everything, environment


@dataclass
class OptimizerConfig:
    epochs: int = 1000
    learning_rate: float = .001
    tau_start: float = 1.0
    tau_end: float = .1
    grad_clip: float = 1.0
    seed: int = 42
    device: str = "cpu"
    population: int = 30
    iterations: int = 1000
    ga_mutation: float = .1
    pso_w: float = .5
    pso_c1: float = 1.6
    pso_c2: float = 1.6
    mlpo_hidden: int = 15
    mlpo_learning_rate: float = .01
    iwo_max_population: int = 50
    iwo_min_seeds: int = 0
    iwo_max_seeds: int = 5
    iwo_sigma_start: float = .5
    iwo_sigma_end: float = .001
    iwo_exponent: float = 3
    wpa_scout_ratio: float = .2
    wpa_directions: int = 8
    wpa_walks: int = 5
    wpa_step_a: float = .05
    wpa_step_b: float = .1
    wpa_step_c: float = .025
    wpa_near: float = .05
    wpa_renew_ratio: float = .2

    def __post_init__(self):
        if min(self.epochs, self.iterations, self.population, self.iwo_max_population,
               self.wpa_directions, self.wpa_walks, self.mlpo_hidden) < 1:
            raise ValueError("Epochs, iterations, populations and network sizes must be positive.")
        if self.population < 3 or self.iwo_max_population < self.population:
            raise ValueError("Require population >= 3 and IWO maximum >= initial population.")
        if min(self.learning_rate, self.mlpo_learning_rate, self.tau_start, self.tau_end, self.grad_clip) <= 0:
            raise ValueError("Learning rates, temperatures and gradient limit must be positive.")
        if not 0 <= self.ga_mutation <= 1 or not 0 < self.wpa_scout_ratio < 1 or not 0 < self.wpa_renew_ratio < 1:
            raise ValueError("Invalid mutation/scout/renewal probabilities.")
        if not 0 <= self.iwo_min_seeds <= self.iwo_max_seeds or self.iwo_max_seeds < 1:
            raise ValueError("Invalid IWO seed counts.")
        if min(self.iwo_sigma_start, self.iwo_sigma_end, self.iwo_exponent,
               self.wpa_step_a, self.wpa_step_b, self.wpa_step_c, self.wpa_near) <= 0:
            raise ValueError("IWO dispersion and WPA step/distance values must be positive.")


def optimize_neural(problem: ScheduleProblem, cfg: OptimizerConfig, model_cfg: ModelConfig, method="mhcmtn"):
    seed_everything(cfg.seed)
    if cfg.device.startswith("cuda") and not torch.cuda.is_available():
        raise ValueError("CUDA requested but unavailable; install a CUDA-enabled PyTorch build or use cpu.")
    start = time.perf_counter()
    model = (MHCMTN(model_cfg) if method == "mhcmtn" else MLPO(cfg.mlpo_hidden)).to(cfg.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate if method == "mhcmtn" else cfg.mlpo_learning_rate)
    current, context = problem.initial(cfg.device), problem.context(cfg.device)
    base = torch.tensor(problem.baseline_load_w, dtype=torch.float32, device=cfg.device).unsqueeze(0)
    # Zero response is always feasible, and guarantees a no-worse incumbent.
    best_score, best_power = problem.evaluate(problem.initial_categories, [0., 0., 0.])
    best_categories, best_alphas = list(problem.initial_categories), [0., 0., 0.]
    history, evaluations = [], 1

    def consider(membership, alphas):
        nonlocal best_score, best_power, best_categories, best_alphas, evaluations, current
        labels = (membership.detach().argmax(-1)[0] + 1).cpu().numpy()
        alpha = alphas.detach()[0].cpu().numpy().clip(0, .5)
        score, power = problem.evaluate(labels, alpha)
        evaluations += 1
        if score < best_score:
            best_score, best_power = score, power
            best_categories, best_alphas = labels.tolist(), alpha.astype(float).tolist()
            current = torch.tensor(best_categories + best_alphas, dtype=torch.float32, device=cfg.device).unsqueeze(0)

    for epoch in range(cfg.epochs):
        tau = cfg.tau_start * (cfg.tau_end / cfg.tau_start) ** (epoch / max(1, cfg.epochs - 1))
        reference = current.detach().clone()
        model.train()
        logits, alpha_logits, regularizer = model(reference, context)
        membership = F.gumbel_softmax(logits, tau=tau, hard=True, dim=-1) if method == "mhcmtn" else logits
        alphas = decode_continuous(alpha_logits, reference[:, 12:])
        scheduled = redistribute_load(base, membership, alphas)
        objective = problem.objective(scheduled).mean()
        evaluations += 1  # Include the differentiable training objective call.
        loss = objective + (model_cfg.manifold_weight * regularizer if method == "mhcmtn" else 0)
        if not torch.isfinite(loss):
            raise RuntimeError(f"Nonfinite loss at epoch {epoch + 1}.")
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip, error_if_nonfinite=True)
        consider(membership, alphas)
        optimizer.step()
        model.eval()
        with torch.no_grad():
            greedy_logits, greedy_alpha, _ = model(current, context)
            greedy = F.one_hot(greedy_logits.argmax(-1), 3).float() if method == "mhcmtn" else greedy_logits
            consider(greedy, decode_continuous(greedy_alpha, current[:, 12:]))
        history.append({"iteration": epoch + 1, "training_loss": float(loss.detach()),
                        "training_objective": float(objective.detach()), "manifold_penalty": float(regularizer.detach()),
                        "tau": tau, "best_objective": best_score,
                        "best_mse_w2": best_score * problem.scale ** 2, "objective_evaluations": evaluations})
    result = build_result(problem, method, best_categories, best_alphas, best_power, best_score,
                          time.perf_counter() - start, cfg, evaluations)
    checkpoint = {"format_version": 1, "method": method, "model_config": asdict(model_cfg),
                  "optimizer_config": asdict(cfg), "final_model_state": copy.deepcopy(model.cpu().state_dict()),
                  "best_decision": {"categories": best_categories, "alphas": best_alphas},
                  "problem": problem.to_dict(), "note": "Final weights and best incumbent are stored separately; reload the stored decision to reproduce the reported schedule."}
    return result, history, checkpoint


def build_result(problem, method, categories, alphas, power, score, elapsed, cfg, evaluations):
    return {"method": method, "case_name": problem.name, "data_kind": problem.data_kind,
            "implementation_status": "new_implementation_not_historical_result_recovery",
            "categories": [int(x) for x in categories], "alphas": [float(x) for x in alphas],
            "scheduled_load_w": np.asarray(power).tolist(), "objective": float(score),
            "elapsed_seconds": elapsed, "objective_evaluations": evaluations, "configuration": asdict(cfg),
            "forecast_metrics": problem.metrics(power),
            "observed_pv_metrics": problem.metrics(power, actual=True) if problem.actual_pv_w is not None else None,
            "baseline_forecast_metrics": problem.metrics(problem.baseline_load_w)}


def optimize_population(problem: ScheduleProblem, cfg: OptimizerConfig, method: str):
    if method not in {"ga", "pso", "iwo", "wpa"}:
        raise ValueError(f"Unknown population optimizer: {method}")
    rng = np.random.default_rng(cfg.seed)
    start = time.perf_counter()
    population = rng.random((cfg.population, 15))
    population[0] = np.r_[(np.asarray(problem.initial_categories) - 1) / 2, [0., 0., 0.]]
    evaluations = 0

    def fitness(positions):
        nonlocal evaluations
        evaluations += len(positions)
        return problem.batch_scores(positions)

    scores = fitness(population)
    index = int(scores.argmin())
    best, best_score = population[index].copy(), float(scores[index])
    velocity, personal, personal_scores = np.zeros_like(population), population.copy(), scores.copy()
    history = []
    for iteration in range(cfg.iterations):
        if method == "ga":
            order = scores.argsort()
            elites = max(2, cfg.population // 9)
            children = [p.copy() for p in population[order[:elites]]]
            pool = population[order[:max(3, cfg.population // 2)]]
            while len(children) < cfg.population:
                parents = pool[rng.choice(len(pool), 2, replace=False)]
                child = np.where(rng.random(15) < .5, parents[0], parents[1])
                mutate = rng.random(12) < cfg.ga_mutation
                child[:12][mutate] = rng.integers(0, 3, size=mutate.sum()) / 2
                child[12:] += rng.normal(0, .14, 3)  # alpha-space sigma=.07
                children.append(np.clip(child, 0, 1))
            population = np.asarray(children)
            scores = fitness(population)
        elif method == "pso":
            velocity = (cfg.pso_w * velocity + cfg.pso_c1 * rng.random(population.shape) * (personal - population)
                        + cfg.pso_c2 * rng.random(population.shape) * (best - population))
            population = np.clip(population + velocity, 0, 1)
            scores = fitness(population)
            improved = scores < personal_scores
            personal[improved], personal_scores[improved] = population[improved], scores[improved]
        elif method == "iwo":
            span = float(scores.max() - scores.min())
            quality = (scores.max() - scores) / span if span > 1e-12 else np.ones(len(scores))
            seeds = np.floor(cfg.iwo_min_seeds + quality * (cfg.iwo_max_seeds - cfg.iwo_min_seeds)).astype(int)
            fraction = (iteration + 1) / cfg.iterations
            sigma = (1 - fraction) ** cfg.iwo_exponent * (cfg.iwo_sigma_start - cfg.iwo_sigma_end) + cfg.iwo_sigma_end
            parents = np.repeat(population, seeds, axis=0)
            if len(parents):
                children = np.clip(parents + rng.normal(0, sigma, parents.shape), 0, 1)
                all_population, all_scores = np.vstack((population, children)), np.r_[scores, fitness(children)]
                keep = all_scores.argsort()[:cfg.iwo_max_population]
                population, scores = all_population[keep], all_scores[keep]
        else:
            # Reconstructed WPA variant; all steps act in normalized latent space.
            leader = int(scores.argmin())
            scouts = scores.argsort()[1:1 + max(1, round(cfg.population * cfg.wpa_scout_ratio))]
            for _ in range(cfg.wpa_walks):
                directions = rng.normal(size=(len(scouts), cfg.wpa_directions, 15))
                directions /= np.maximum(np.linalg.norm(directions, axis=-1, keepdims=True), 1e-12)
                candidates = np.clip(population[scouts, None, :] + cfg.wpa_step_a * directions, 0, 1)
                candidate_scores = fitness(candidates.reshape(-1, 15)).reshape(len(scouts), -1)
                choices = candidate_scores.argmin(-1)
                for j, wolf in enumerate(scouts):
                    if candidate_scores[j, choices[j]] < scores[wolf]:
                        population[wolf], scores[wolf] = candidates[j, choices[j]], candidate_scores[j, choices[j]]
            leader = int(scores.argmin())
            target = population[leader].copy()
            delta = target - population
            distance = np.sqrt(np.mean(delta ** 2, axis=1))
            step = cfg.wpa_step_b * delta / np.maximum(np.linalg.norm(delta, axis=1, keepdims=True), 1e-12)
            near = distance <= cfg.wpa_near
            step[near] = cfg.wpa_step_c * rng.uniform(-1, 1, size=(near.sum(), 15)) * np.abs(delta[near])
            candidates = np.clip(population + step, 0, 1)
            candidate_scores = fitness(candidates)
            improve = candidate_scores < scores
            population[improve], scores[improve] = candidates[improve], candidate_scores[improve]
            renew = max(1, round(cfg.population * cfg.wpa_renew_ratio))
            leader = int(scores.argmin())
            order = [i for i in scores.argsort() if i != leader]
            worst = np.asarray(order[-renew:], dtype=int)
            population[worst] = rng.random((len(worst), 15))
            scores[worst] = fitness(population[worst])
        index = int(scores.argmin())
        if scores[index] < best_score:
            best, best_score = population[index].copy(), float(scores[index])
        history.append({"iteration": iteration + 1, "best_objective": best_score,
                        "best_mse_w2": best_score * problem.scale ** 2, "objective_evaluations": evaluations})
    categories, alphas = np.rint(best[:12] * 2).astype(int) + 1, best[12:] * .5
    exact_score, power = problem.evaluate(categories, alphas)
    return build_result(problem, method, categories, alphas, power, exact_score,
                        time.perf_counter() - start, cfg, evaluations), history, None


def run_optimization(problem, output: Path, method="mhcmtn", cfg=None, model_cfg=None, plot=True):
    cfg, model_cfg = cfg or OptimizerConfig(), model_cfg or ModelConfig()
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if method in {"mhcmtn", "mlpo"}:
        result, history, checkpoint = optimize_neural(problem, cfg, model_cfg, method)
    else:
        result, history, checkpoint = optimize_population(problem, cfg, method)
    result["model_configuration"] = asdict(model_cfg) if method == "mhcmtn" else None
    save_json(output / "result.json", result)
    save_json(output / "case.json", problem.to_dict())
    save_json(output / "environment.json", environment())
    pd.DataFrame(history).to_csv(output / "history.csv", index=False)
    pd.DataFrame({"slot": np.arange(1, 13), "pv_forecast_w": problem.pv_forecast_w,
                  "baseline_load_w": problem.baseline_load_w, "scheduled_load_w": result["scheduled_load_w"],
                  "category": result["categories"]}).to_csv(output / "schedule.csv", index=False)
    if checkpoint is not None:
        torch.save(checkpoint, output / "checkpoint.pt")
    if plot:
        from .plotting import optimization_figures
        optimization_figures(problem, result, history, output)
    return result
