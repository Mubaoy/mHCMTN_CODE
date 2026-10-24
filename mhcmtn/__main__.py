from __future__ import annotations

import argparse
from dataclasses import fields, replace
import json
from pathlib import Path

import pandas as pd
import torch

from .data import prepare_data
from .forecasting import ForecastConfig, run_forecasting
from .model import ModelConfig
from .optimization import OptimizerConfig, run_optimization
from .scheduling import ScheduleProblem
from .utils import save_json

METHODS = ["mhcmtn", "ga", "pso", "mlpo", "iwo", "wpa"]


def load_case(path):
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    metadata_fields = {"forecast_origin", "forecast_protocol"}
    allowed = {f.name for f in fields(ScheduleProblem)}
    unknown = set(payload) - allowed - metadata_fields
    if unknown:
        raise ValueError(f"Unknown case fields: {sorted(unknown)}")
    return ScheduleProblem(**{key: value for key, value in payload.items() if key in allowed})


def configurations(args):
    values = json.loads(Path(args.config).read_text(encoding="utf-8")) if args.config else {}
    unknown = set(values) - {"model", "optimizer", "forecast"}
    if unknown:
        raise ValueError(f"Unknown configuration sections: {sorted(unknown)}")
    model = ModelConfig(**values.get("model", {}))
    optimizer = OptimizerConfig(**values.get("optimizer", {}))
    forecast = ForecastConfig(**values.get("forecast", {}))
    for key in ["epochs", "iterations", "population", "seed", "device"]:
        value = getattr(args, key, None)
        if value is not None:
            optimizer = replace(optimizer, **{key: value})
    for key in ["seed", "device"]:
        value = getattr(args, key, None)
        if value is not None:
            forecast = replace(forecast, **{key: value})
    if getattr(args, "forecast_epochs", None) is not None:
        forecast = replace(forecast, epochs=args.forecast_epochs)
    return model, optimizer, forecast


def benchmark(case, output, methods, optimizer, model, plot=True):
    output = Path(output)
    rows = []
    for method in methods:
        print(f"Optimizing {method}...", flush=True)
        result = run_optimization(case, output / method, method, optimizer, model, plot)
        rows.append({"method": method, "data_kind": case.data_kind, "seed": optimizer.seed,
                     "objective": result["objective"], "elapsed_seconds": result["elapsed_seconds"],
                     "objective_evaluations": result["objective_evaluations"],
                     **{f"forecast_{k}": v for k, v in result["forecast_metrics"].items()},
                     **{f"observed_{k}": v for k, v in (result["observed_pv_metrics"] or {}).items()}})
    pd.DataFrame(rows).to_csv(output / "comparison.csv", index=False)
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser(description="mHCMTN: newly implemented forecasting and feasible demand-response optimization")
    commands = parser.add_subparsers(dest="command", required=True)

    def config_options(command, optimization=False):
        command.add_argument("--config", type=Path)
        command.add_argument("--seed", type=int)
        command.add_argument("--device", choices=["cpu", "cuda"])
        command.add_argument("--no-plot", action="store_true")
        if optimization:
            command.add_argument("--epochs", type=int, help="Neural optimizer steps (not forecasting epochs)")
            command.add_argument("--iterations", type=int, help="Population optimizer generations")
            command.add_argument("--population", type=int)

    def data_options(command):
        source = command.add_mutually_exclusive_group(required=True)
        source.add_argument("--source", type=Path, help="Local six-column selected workbook, original workbook, or CSV")
        source.add_argument("--download", action="store_true", help="Download the pinned public upstream workbook")
        command.add_argument("--limit", type=int, help="Explicit small-data software check; not a paper experiment")
        command.add_argument("--input-unit", choices=["kwh", "kw", "w"], default="kwh")
        command.add_argument("--timezone", default="America/Los_Angeles")

    prepare = commands.add_parser("prepare-data", help="Select six columns, resolve DST, build causal components and chronological splits")
    data_options(prepare)
    prepare.add_argument("--output", type=Path, required=True)
    prepare.add_argument("--period", type=int, default=24)
    prepare.add_argument("--stl-window", type=int, default=168)
    prepare.add_argument("--lookback", type=int, default=24)

    forecast = commands.add_parser("forecast", help="Train forecasting models and generate held-out planning cases")
    config_options(forecast)
    forecast.add_argument("--data-dir", type=Path, required=True)
    forecast.add_argument("--output", type=Path, required=True)
    forecast.add_argument("--forecast-epochs", type=int)
    forecast.add_argument("--models", nargs="+", choices=["stl_lstm", "lstm", "rf", "xgb", "svr"])
    forecast.add_argument("--origin", type=int)

    for name in ["optimize", "benchmark"]:
        command = commands.add_parser(name)
        config_options(command, optimization=True)
        command.add_argument("--case", type=Path, required=True)
        command.add_argument("--output", type=Path, required=True)
        if name == "optimize":
            command.add_argument("--method", choices=METHODS, default="mhcmtn")
        else:
            command.add_argument("--methods", nargs="+", choices=METHODS, default=METHODS)

    pipeline = commands.add_parser("pipeline", help="Prepare data, forecast, then optimize its held-out STL-LSTM case")
    data_options(pipeline)
    config_options(pipeline, optimization=True)
    pipeline.add_argument("--forecast-epochs", type=int)
    pipeline.add_argument("--work-dir", type=Path, required=True)
    pipeline.add_argument("--methods", nargs="+", choices=METHODS, default=["mhcmtn"])

    smoke = commands.add_parser("smoke", help="Small offline synthetic software check of all six optimizers")
    smoke.add_argument("--output", type=Path, required=True)
    smoke.add_argument("--no-plot", action="store_true")

    verify = commands.add_parser("check", help="Re-evaluate the stored best decision from an optimizer checkpoint")
    verify.add_argument("--checkpoint", type=Path, required=True)

    args = parser.parse_args(argv)
    if args.command == "prepare-data":
        result = prepare_data(args.output, source=args.source, download=args.download, limit=args.limit,
                              input_unit=args.input_unit, timezone=args.timezone, period=args.period,
                              stl_window=args.stl_window, lookback=args.lookback)
    elif args.command == "check":
        checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
        problem = ScheduleProblem(**checkpoint["problem"])
        score, power = problem.evaluate(**checkpoint["best_decision"])
        result = {"objective": score, "scheduled_load_w": power.tolist(), "metrics": problem.metrics(power)}
    elif args.command == "smoke":
        case = ScheduleProblem(
            pv_forecast_w=[0, 350, 900, 1500, 2100, 2600, 2800, 2600, 2100, 1500, 700, 0],
            baseline_load_w=[1800, 1700, 1600, 1200, 800, 600, 700, 900, 1200, 1600, 1800, 1900],
            name="synthetic_day_for_software_validation", data_kind="synthetic_smoke_test_only_not_paper_evidence")
        result = benchmark(case, args.output, METHODS,
                           OptimizerConfig(epochs=8, iterations=4, population=6), ModelConfig(), not args.no_plot)
    else:
        model, optimizer, forecast = configurations(args)
        if args.command == "forecast":
            result = run_forecasting(args.data_dir, args.output, forecast, args.models, not args.no_plot, args.origin)
        elif args.command == "optimize":
            result = run_optimization(load_case(args.case), args.output, args.method, optimizer, model, not args.no_plot)
        elif args.command == "benchmark":
            result = benchmark(load_case(args.case), args.output, args.methods, optimizer, model, not args.no_plot)
        else:
            work = args.work_dir
            prepare_data(work / "prepared", source=args.source, download=args.download, limit=args.limit,
                         lookback=forecast.lookback, input_unit=args.input_unit, timezone=args.timezone)
            run_forecasting(work / "prepared", work / "forecast", forecast, plot=not args.no_plot)
            result = benchmark(load_case(work / "forecast/planning_case_stl_lstm.json"), work / "optimization",
                               args.methods, optimizer, model, not args.no_plot)
    print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
