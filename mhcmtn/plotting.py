"""Editable figures from actual run outputs, never from digitized old curves."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

STYLE = {"font.family": "Times New Roman", "font.size": 18, "svg.fonttype": "none", "pdf.fonttype": 42}


def save_figure(fig, path):
    for suffix in [".png", ".svg", ".pdf"]:
        fig.savefig(Path(path).with_suffix(suffix), dpi=600, bbox_inches="tight")
    plt.close(fig)


def optimization_figures(problem, result, history, output):
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(8, 6.5), layout="constrained")
        ax.plot([r["iteration"] for r in history], [r["best_mse_w2"] for r in history], color="#1f77b4")
        ax.set(xlabel="Iteration", ylabel="Best feasible MSE (W²)")
        save_figure(fig, Path(output) / "convergence")
        fig, ax = plt.subplots(figsize=(9, 6.5), layout="constrained")
        slots = np.arange(1, 13)
        ax.plot(slots, problem.pv_forecast_w, label="PV forecast", color="#e49a24")
        ax.plot(slots, problem.baseline_load_w, label="Original load", color="#777777")
        ax.plot(slots, result["scheduled_load_w"], label="Scheduled load", color="#1f77b4")
        ax.set(xlabel="Time slot", ylabel="Average power (W)")
        ax.legend(fontsize=18)
        save_figure(fig, Path(output) / "schedule")


def forecast_figure(timestamps, actual, predictions, output):
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(11, 6.5), layout="constrained")
        n = min(100, len(actual))
        ax.plot(np.arange(n), actual[:n], color="black", label="Actual", linewidth=2)
        for name, values in predictions.items():
            ax.plot(np.arange(n), values[:n], label=name, linewidth=1)
        ax.set(xlabel="Test time step", ylabel="PV average power (W)")
        ax.legend(ncol=3, fontsize=18)
        save_figure(fig, Path(output) / "prediction_comparison")
