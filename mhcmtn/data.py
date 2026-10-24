from __future__ import annotations

import hashlib
import json
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.tsa.seasonal import STL

from .utils import save_json

SOURCE_COMMIT = "b1c62966d61ab326fe71b9327f017f8264cf4ef8"
SOURCE_FILE = "Solar_Weather_FInal_Dataset.xlsx"
SOURCE_URL = ("https://raw.githubusercontent.com/riddhi-vyas/"
              f"Smart-Solar-Energy-Scheduling-with-Machine-Learning/{SOURCE_COMMIT}/{SOURCE_FILE}")
SOURCE_SHA256 = "7fa57b98aa22cca6fdec84eb18a32a3d141e41d756c52ad5d84b021dd67ce2bb"
COLUMNS = ["datetime", "temp", "humidity", "solarradiation", "Home Usage (kWh)", "Solar Production (kWh)"]


def causal_stl(values, period: int = 24, window: int = 168, robust: bool = True) -> np.ndarray:
    """Endpoint STL: row t depends ONLY on observations through t, never t+1.

    The first two seasonal periods use an expanding-mean warmup. Forecast
    training excludes these warmup rows. Rolling endpoint components need not
    equal the noncausal components of one full-series STL fit.
    """
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or not np.isfinite(values).all():
        raise ValueError("STL input must be a finite vector.")
    if period < 2 or window < 2 * period:
        raise ValueError("Require period >= 2 and window >= 2*period.")
    result = np.zeros((len(values), 3), dtype=float)
    for t in range(len(values)):
        history = values[max(0, t - window + 1):t + 1]
        if len(history) < 2 * period:
            trend, seasonal = float(history.mean()), 0.0
        else:
            fit = STL(history, period=period, seasonal=7, robust=robust).fit()
            trend, seasonal = float(fit.trend[-1]), float(fit.seasonal[-1])
        result[t] = trend, seasonal, values[t] - trend - seasonal
    return result


def chronological_split(n: int, first_origin: int, train_fraction=.8, validation_fraction=.1) -> dict:
    if not 0 < train_fraction < 1 or not 0 < validation_fraction < 1 - train_fraction:
        raise ValueError("Invalid chronological split fractions.")
    train_end, validation_end = int(n * train_fraction), int(n * (train_fraction + validation_fraction))
    if train_end <= first_origin or validation_end <= train_end or validation_end >= n:
        raise ValueError("Insufficient rows for the requested lookback, warmup and splits.")
    return {"row_count": n, "first_origin": first_origin,
            "train_end_exclusive": train_end, "validation_end_exclusive": validation_end,
            "train_origins": list(range(first_origin, train_end)),
            "validation_origins": list(range(train_end, validation_end)),
            "test_origins": list(range(validation_end, n)),
            "semantics": "Zero-based raw row of each one-step target. Input rows strictly precede the target."}


def prepare_data(output: Path, source: Path | None = None, period=24, stl_window=168,
                 lookback=24, limit=None, download=False, input_unit="kwh",
                 timezone="America/Los_Angeles") -> dict:
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if source is None:
        if not download:
            raise ValueError("Pass --source or --download; no external data are bundled.")
        request = urllib.request.Request(SOURCE_URL, headers={"User-Agent": "mHCMTN-research-code"})
        with urllib.request.urlopen(request, timeout=45) as response:
            body = response.read()
        if hashlib.sha256(body).hexdigest() != SOURCE_SHA256:
            raise ValueError("Downloaded workbook differs from the previously verified upstream file.")
        source = output / SOURCE_FILE
        source.write_bytes(body)
    source = Path(source)
    input_sha256 = hashlib.sha256(source.read_bytes()).hexdigest()
    frame = pd.read_excel(source) if source.suffix.lower() == ".xlsx" else pd.read_csv(source)
    missing = set(COLUMNS) - set(frame.columns)
    if missing:
        raise ValueError(f"Missing data columns: {sorted(missing)}")
    frame = frame.loc[:, COLUMNS].copy()
    if limit is not None:
        frame = frame.iloc[:limit].copy()
    local_dates = pd.DatetimeIndex(pd.to_datetime(frame["datetime"]))
    # Manteca, California local time includes both autumn 01:00 observations.
    # Infer their DST offsets in sequence; spring's missing 02:00 is expected.
    # No values/rows are interpolated, dropped or sorted.
    aware = (local_dates.tz_localize(timezone, ambiguous="infer", nonexistent="raise")
             if local_dates.tz is None else local_dates)
    dates = aware.tz_convert("UTC").tz_localize(None)
    # pandas 3 may store microseconds; asi8 is not always nanoseconds.
    # Divide timedeltas by an explicit hour instead of assuming integer units.
    intervals = np.asarray((dates[1:] - dates[:-1]) / pd.Timedelta(hours=1), dtype=float)
    if len(intervals) == 0 or not np.allclose(intervals, intervals[0]) or intervals[0] <= 0:
        raise ValueError("Dates must be unique, sorted and uniformly spaced; no implicit interpolation is performed.")
    numeric = frame[COLUMNS[1:]].to_numpy(dtype=float)
    if not np.isfinite(numeric).all() or (numeric[:, -2:] < 0).any():
        raise ValueError("Nonfinite data or negative energy/power values need explicit preprocessing.")
    step_hours = float(intervals[0])
    factor = {"kwh": 1000 / step_hours, "kw": 1000, "w": 1}.get(input_unit)
    if factor is None:
        raise ValueError("input_unit must be kwh, kw or w.")
    pv, load = numeric[:, -1] * factor, numeric[:, -2] * factor
    print(f"Computing causal STL for {len(frame)} rows (period={period}, window={stl_window})...", flush=True)
    components = causal_stl(pv, period, stl_window)
    split = chronological_split(len(frame), 2 * period + lookback)
    np.savez_compressed(output / "prepared.npz", timestamps=dates.to_numpy(dtype="datetime64[ns]"),
                        local_timestamps=aware.tz_convert(timezone).tz_localize(None).to_numpy(dtype="datetime64[ns]"),
                        weather=numeric[:, :3], pv_w=pv, load_w=load, components=components)
    save_json(output / "split.json", split)
    metadata = {"repository": "riddhi-vyas/Smart-Solar-Energy-Scheduling-with-Machine-Learning",
                "source_commit": SOURCE_COMMIT, "source_url": SOURCE_URL,
                "input_sha256": input_sha256,
                "input_is_verified_full_upstream_workbook": input_sha256 == SOURCE_SHA256,
                "source_reference_note": "Local selected subsets require independent provenance verification; the upstream link is a reference, not an automatic match claim.",
                "rows": len(frame), "input_unit_assumption": input_unit,
                "conversion_to_average_w": factor, "step_hours": step_hours,
                "period": period, "stl_window": stl_window, "stl_robust": True,
                "lookback": lookback, "stl_mode": "causal_rolling_endpoint",
                "start": str(dates[0]), "end": str(dates[-1]),
                "timezone": timezone, "timestamp_storage": "UTC (timezone-naive datetime64); original local times retained separately",
                "data_redistributed": False}
    save_json(output / "metadata.json", metadata)
    return metadata


def load_prepared(directory: Path):
    directory = Path(directory)
    with np.load(directory / "prepared.npz", allow_pickle=False) as archive:
        arrays = {k: archive[k].copy() for k in archive.files}
    split = json.loads((directory / "split.json").read_text())
    metadata = json.loads((directory / "metadata.json").read_text())
    return arrays, split, metadata
