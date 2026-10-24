from __future__ import annotations

import importlib.metadata
import json
import platform
import random
from pathlib import Path

import numpy as np
import torch


def seed_everything(seed: int, threads: int = 1) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(threads)
    torch.use_deterministic_algorithms(True)


def save_json(path: str | Path, value) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")


def environment() -> dict:
    packages = {}
    for name in ["torch", "numpy", "pandas", "statsmodels", "scikit-learn", "xgboost", "matplotlib"]:
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    return {"python": platform.python_version(), "platform": platform.platform(),
            "packages": packages, "cuda_available": torch.cuda.is_available()}


def regression_metrics(actual, predicted) -> dict:
    actual, predicted = np.asarray(actual, float), np.asarray(predicted, float)
    if actual.shape != predicted.shape or not np.isfinite(predicted).all():
        raise ValueError("Actual and predicted arrays must have matching finite shapes.")
    error = predicted - actual
    variance = np.sum((actual - actual.mean()) ** 2)
    return {"mse_w2": float(np.mean(error ** 2)), "rmse_w": float(np.sqrt(np.mean(error ** 2))),
            "mae_w": float(np.mean(np.abs(error))),
            "r2": float(1 - np.sum(error ** 2) / variance) if variance > 0 else None}
