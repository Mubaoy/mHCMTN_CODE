import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from mhcmtn.data import prepare_data
from mhcmtn.forecasting import ForecastConfig, run_forecasting
from mhcmtn.scheduling import ScheduleProblem


@pytest.fixture
def problem():
    path = Path(__file__).parents[1] / "examples/synthetic_day.json"
    return ScheduleProblem(**json.loads(path.read_text(encoding="utf-8")))


@pytest.fixture(scope="module")
def forecast_run(tmp_path_factory):
    directory = tmp_path_factory.mktemp("forecast")
    # Deliberately includes the repeated 01:00 at California's DST transition.
    dates = pd.date_range("2023-11-03", periods=144, freq="h", tz="America/Los_Angeles")
    t = np.arange(len(dates))
    pv = np.maximum(0., np.sin((t % 24 - 6) * np.pi / 12)) * 3
    frame = pd.DataFrame({"datetime": dates.tz_localize(None), "temp": 70 + np.sin(t),
                          "humidity": 40 + np.cos(t), "solarradiation": pv * 100,
                          "Home Usage (kWh)": 1 + .1 * np.cos(t), "Solar Production (kWh)": pv})
    source = directory / "source.csv"
    frame.to_csv(source, index=False)
    prepared, output = directory / "prepared", directory / "trained"
    prepare_data(prepared, source, lookback=12, stl_window=48)
    cfg = ForecastConfig(epochs=1, batch_size=16, hidden_size=8, baseline_hidden_size=8, lookback=12)
    run_forecasting(prepared, output, cfg, plot=False)
    return prepared, output
