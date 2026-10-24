import json
import numpy as np
import pandas as pd
import pytest

from mhcmtn.__main__ import load_case
from mhcmtn.data import causal_stl, chronological_split, load_prepared, prepare_data
from mhcmtn.forecasting import MinMax, load_forecasters, make_windows, recursive_prediction
from mhcmtn.utils import regression_metrics


def test_causal_stl_has_no_future_dependence():
    t = np.arange(100)
    values = 100 + t + 40 * np.sin(2 * np.pi * t / 12)
    altered = values.copy()
    altered[75:] += 5000
    first = causal_stl(values, period=12, window=48)
    second = causal_stl(altered, period=12, window=48)
    np.testing.assert_allclose(first[:75], second[:75], atol=1e-10)
    np.testing.assert_allclose(first.sum(-1), values, atol=1e-10)


def test_window_alignment_and_split_boundaries():
    x, y = make_windows(np.arange(40)[:, None], np.arange(40), [10, 11], 4)
    np.testing.assert_array_equal(x[0, :, 0], [6, 7, 8, 9])
    np.testing.assert_array_equal(y[:, 0], [10, 11])
    split = chronological_split(1000, 72)
    assert split["train_origins"][-1] == 799
    assert split["validation_origins"] == list(range(800, 900))
    assert split["test_origins"] == list(range(900, 1000))


def test_scaler_does_not_refit_to_test_extremes():
    scaler = MinMax().fit(np.array([[1.], [3.]]))
    assert scaler.transform([[1000.]])[0, 0] == 499.5
    np.testing.assert_allclose(MinMax.restore(scaler.serialize()).inverse([[.5]]), [[2.]])


@pytest.mark.parametrize("resolution", ["ns", "us", "ms"])
def test_hourly_energy_conversion_does_not_depend_on_datetime_storage_unit(tmp_path, monkeypatch, resolution):
    rows = 96
    energy = 1 + .1 * np.sin(np.arange(rows))
    frame = pd.DataFrame({"datetime": pd.date_range("2023-08-01", periods=rows, freq="h"),
                          "temp": np.full(rows, 70), "humidity": np.full(rows, 40),
                          "solarradiation": np.full(rows, 100),
                          "Home Usage (kWh)": energy, "Solar Production (kWh)": energy})
    source = tmp_path / "source.csv"
    frame.to_csv(source, index=False)
    convert = pd.to_datetime
    monkeypatch.setattr(pd, "to_datetime", lambda values: convert(values).dt.as_unit(resolution))
    output = tmp_path / "prepared"
    meta = prepare_data(output, source=source, timezone="UTC", stl_window=48)
    arrays, _, _ = load_prepared(output)
    assert meta["step_hours"] == 1.0
    assert meta["conversion_to_average_w"] == 1000.0
    np.testing.assert_allclose(arrays["pv_w"], energy * 1000, atol=1e-9)


def test_dst_is_resolved_without_dropping_observations(forecast_run):
    arrays, split, meta = load_prepared(forecast_run[0])
    assert len(arrays["timestamps"]) == 144
    assert len(np.unique(arrays["local_timestamps"])) == 143
    np.testing.assert_array_equal(np.diff(arrays["timestamps"]).astype("timedelta64[h]").astype(int), 1)
    assert meta["step_hours"] == 1 and not meta["input_is_verified_full_upstream_workbook"]
    scalers = json.loads((forecast_run[1] / "scalers.json").read_text())
    assert scalers["target"]["minimum"][0][0] == float(arrays["pv_w"][:split["train_end_exclusive"]].min())


@pytest.mark.parametrize("name", ["stl_lstm", "lstm", "rf", "xgb", "svr"])
def test_saved_forecast_and_rollout_ignore_all_future_observations(forecast_run, name):
    prepared, output = forecast_run
    arrays, _, _ = load_prepared(prepared)
    models, scalers, cfg = load_forecasters(output)
    case = load_case(output / f"planning_case_{name}.json")
    value = json.loads((output / f"planning_case_{name}.json").read_text())
    origin = int(np.flatnonzero(arrays["timestamps"] == np.datetime64(value["forecast_origin"]))[0])
    before = recursive_prediction(name, models, scalers, arrays, origin, 12, cfg)
    np.testing.assert_allclose(before, case.pv_forecast_w, rtol=1e-5, atol=.01)
    planning_metrics = json.loads((output / "planning_metrics.json").read_text())["models"][name]
    expected = regression_metrics(case.actual_pv_w, case.pv_forecast_w)
    assert planning_metrics == expected
    changed = {k: a.copy() for k, a in arrays.items()}
    for key in ["pv_w", "load_w", "weather", "components"]:
        changed[key][origin:] += 100000
    after = recursive_prediction(name, models, scalers, changed, origin, 12, cfg)
    np.testing.assert_array_equal(before, after)
