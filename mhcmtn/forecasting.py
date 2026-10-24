from __future__ import annotations

import copy
from dataclasses import dataclass
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR

from .data import load_prepared
from .scheduling import ScheduleProblem
from .utils import environment, regression_metrics, save_json, seed_everything


@dataclass
class ForecastConfig:
    epochs: int = 50
    batch_size: int = 32
    learning_rate: float = .001
    patience: int = 10
    hidden_size: int = 50
    baseline_hidden_size: int = 64
    dropout: float = .2
    seed: int = 42
    lookback: int = 24
    device: str = "cpu"

    def __post_init__(self):
        if min(self.epochs, self.batch_size, self.patience, self.hidden_size, self.baseline_hidden_size, self.lookback) < 1:
            raise ValueError("Forecast counts must be positive.")
        if self.learning_rate <= 0 or not 0 <= self.dropout < 1:
            raise ValueError("Invalid forecasting learning rate or dropout.")


class MinMax:
    def fit(self, x):
        self.minimum = np.min(x, axis=0, keepdims=True)
        self.span = np.max(x, axis=0, keepdims=True) - self.minimum
        self.span = np.where(self.span > 0, self.span, 1.)
        return self

    def transform(self, x):
        return (np.asarray(x) - self.minimum) / self.span

    def inverse(self, x):
        return np.asarray(x) * self.span + self.minimum

    def serialize(self):
        return {"minimum": self.minimum.tolist(), "span": self.span.tolist()}

    @classmethod
    def restore(cls, value):
        scaler = cls()
        scaler.minimum, scaler.span = np.array(value["minimum"]), np.array(value["span"])
        return scaler


def make_windows(features, target, origins, lookback):
    origins = np.asarray(origins, int)
    if len(origins) == 0 or origins.min() < lookback or origins.max() >= len(features):
        raise ValueError("Invalid target origins for lookback.")
    return (np.stack([features[t-lookback:t] for t in origins]).astype(np.float32),
            np.asarray(target)[origins].reshape(-1, 1).astype(np.float32))


class StackedLSTM(nn.Module):
    def __init__(self, hidden=50, dropout=.2):
        super().__init__()
        self.first, self.second = nn.LSTM(4, hidden, batch_first=True), nn.LSTM(hidden, hidden, batch_first=True)
        self.dropout, self.output = nn.Dropout(dropout), nn.Linear(hidden, 1)

    def forward(self, x):
        first, _ = self.first(x)
        second, _ = self.second(self.dropout(first))
        return self.output(self.dropout(second[:, -1]))


class ReluLSTM(nn.Module):
    """Keras-style activation='relu', sigmoid recurrent gates, one recurrent layer.

    A standard torch.nn.LSTM would silently use tanh, so its candidate and cell
    output activations are written explicitly here. Weights are newly trained.
    """
    def __init__(self, hidden=64):
        super().__init__()
        self.hidden = hidden
        self.input = nn.Linear(4, 4 * hidden)
        self.recurrent = nn.Linear(hidden, 4 * hidden, bias=False)
        self.output = nn.Linear(hidden, 1)
        nn.init.xavier_uniform_(self.input.weight)
        nn.init.orthogonal_(self.recurrent.weight)
        nn.init.zeros_(self.input.bias)
        with torch.no_grad():
            self.input.bias[hidden:2*hidden] = 1

    def forward(self, x):
        h = x.new_zeros(len(x), self.hidden)
        c = torch.zeros_like(h)
        for step in x.unbind(1):
            i, f, g, o = (self.input(step) + self.recurrent(h)).chunk(4, dim=-1)
            c = torch.sigmoid(f) * c + torch.sigmoid(i) * torch.relu(g)
            h = torch.sigmoid(o) * torch.relu(c)
        return self.output(h)


def fit_network(model, train_x, train_y, val_x, val_y, cfg: ForecastConfig):
    model.to(cfg.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)
    generator = torch.Generator().manual_seed(cfg.seed)
    loader = DataLoader(TensorDataset(torch.tensor(train_x), torch.tensor(train_y)),
                        batch_size=cfg.batch_size, shuffle=True, generator=generator)
    val_loader = DataLoader(TensorDataset(torch.tensor(val_x), torch.tensor(val_y)), batch_size=256)
    best, best_loss, waited, history = None, float("inf"), 0, []
    for epoch in range(cfg.epochs):
        model.train()
        total = 0.
        for x, y in loader:
            x, y = x.to(cfg.device), y.to(cfg.device)
            loss = (model(x) - y).square().mean()
            if not torch.isfinite(loss):
                raise RuntimeError("Nonfinite LSTM training loss.")
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5., error_if_nonfinite=True)
            optimizer.step()
            total += float(loss.detach()) * len(x)
        model.eval()
        validation = 0.
        with torch.no_grad():
            for x, y in val_loader:
                validation += float((model(x.to(cfg.device)) - y.to(cfg.device)).square().sum())
        validation /= len(val_y)
        history.append({"epoch": epoch + 1, "training_mse_scaled": total / len(train_y),
                        "validation_mse_scaled": validation})
        if validation < best_loss:
            best_loss, best, waited = validation, copy.deepcopy(model.cpu().state_dict()), 0
            model.to(cfg.device)
        else:
            waited += 1
            if waited >= cfg.patience:
                break
    model.load_state_dict(best)
    model.to(cfg.device).eval()
    return model, history


def neural_predict(model, x, device):
    model.eval()
    batches = []
    with torch.no_grad():
        for start in range(0, len(x), 256):
            batches.append(model(torch.tensor(x[start:start+256], dtype=torch.float32, device=device)).cpu().numpy())
    return np.vstack(batches)


def recursive_prediction(model_name, models, scalers, arrays, origin, horizon, cfg):
    """Directly produce a multi-step planning forecast without future observations.

    Future weather is held at the last observed value. Predicted targets/components
    are recursively fed back, rather than borrowing held-out future PV or weather.
    """
    if origin < cfg.lookback or origin + horizon > len(arrays["pv_w"]):
        raise ValueError("Insufficient history or observation rows for this forecast origin.")
    weather = scalers["weather"].transform(arrays["weather"][origin-cfg.lookback:origin])
    future_weather = weather[-1:].copy()
    if model_name == "stl_lstm":
        states = [np.column_stack((weather, scalers["components"][i].transform(
            arrays["components"][origin-cfg.lookback:origin, i:i+1]))) for i in range(3)]
    else:
        state = np.column_stack((weather, scalers["target"].transform(arrays["pv_w"][origin-cfg.lookback:origin, None])))
    prediction = []
    for _ in range(horizon):
        if model_name == "stl_lstm":
            components = []
            for i in range(3):
                value = float(neural_predict(models[model_name][i], states[i][None].astype(np.float32), cfg.device)[0, 0])
                components.append(float(scalers["components"][i].inverse([[value]])[0, 0]))
                states[i] = np.vstack((states[i][1:], np.r_[future_weather[0], value]))
            predicted = max(0., sum(components))
        else:
            if model_name == "lstm":
                value = float(neural_predict(models[model_name], state[None].astype(np.float32), cfg.device)[0, 0])
            else:
                value = float(models[model_name].predict(state.reshape(1, -1))[0])
            predicted = max(0., float(scalers["target"].inverse([[value]])[0, 0]))
            scaled_value = scalers["target"].transform([[predicted]])[0, 0]
            state = np.vstack((state[1:], np.r_[future_weather[0], scaled_value]))
        prediction.append(predicted)
    return np.asarray(prediction)


def run_forecasting(prepared: Path, output: Path, cfg=None, model_names=None, plot=True, origin=None):
    cfg = cfg or ForecastConfig()
    model_names = list(model_names or ["stl_lstm", "lstm", "rf", "xgb", "svr"])
    if set(model_names) - {"stl_lstm", "lstm", "rf", "xgb", "svr"}:
        raise ValueError("Unknown forecast model.")
    seed_everything(cfg.seed)
    arrays, split, metadata = load_prepared(prepared)
    if cfg.lookback != metadata["lookback"]:
        raise ValueError("Forecast lookback must match the prepared split.")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    train_end = split["train_end_exclusive"]
    train, val, test = [split[f"{name}_origins"] for name in ["train", "validation", "test"]]
    scalers = {"weather": MinMax().fit(arrays["weather"][:train_end]),
               "target": MinMax().fit(arrays["pv_w"][:train_end, None]),
               "components": [MinMax().fit(arrays["components"][:train_end, i:i+1]) for i in range(3)]}
    weather = scalers["weather"].transform(arrays["weather"])
    target = scalers["target"].transform(arrays["pv_w"][:, None])
    features = np.column_stack((weather, target))
    tx, ty = make_windows(features, target, train, cfg.lookback)
    vx, vy = make_windows(features, target, val, cfg.lookback)
    ex, _ = make_windows(features, target, test, cfg.lookback)
    predictions, models, descriptors = {}, {}, {}
    for name in model_names:
        print(f"Training forecast model {name}...", flush=True)
        seed_everything(cfg.seed)
        if name == "stl_lstm":
            branches, outputs = [], []
            for branch in range(3):
                component = scalers["components"][branch].transform(arrays["components"][:, branch:branch+1])
                inputs = np.column_stack((weather, component))
                bx, by = make_windows(inputs, component, train, cfg.lookback)
                cvx, cvy = make_windows(inputs, component, val, cfg.lookback)
                cex, _ = make_windows(inputs, component, test, cfg.lookback)
                model, history = fit_network(StackedLSTM(cfg.hidden_size, cfg.dropout), bx, by, cvx, cvy, cfg)
                branches.append(model)
                outputs.append(scalers["components"][branch].inverse(neural_predict(model, cex, cfg.device)).ravel())
                torch.save(model.cpu().state_dict(), output / f"stl_branch_{branch}.pt")
                model.to(cfg.device)
                pd.DataFrame(history).to_csv(output / f"stl_branch_{branch}_history.csv", index=False)
            models[name], predictions[name] = branches, np.maximum(np.sum(outputs, axis=0), 0)
            descriptors[name] = {"implementation": "torch_stacked_lstm", "branches": 3,
                                 "hidden_size": cfg.hidden_size, "dropout": cfg.dropout}
        elif name == "lstm":
            model, history = fit_network(ReluLSTM(cfg.baseline_hidden_size), tx, ty, vx, vy, cfg)
            models[name] = model
            predictions[name] = np.maximum(scalers["target"].inverse(neural_predict(model, ex, cfg.device)).ravel(), 0)
            torch.save(model.cpu().state_dict(), output / "lstm.pt")
            model.to(cfg.device)
            pd.DataFrame(history).to_csv(output / "lstm_history.csv", index=False)
            descriptors[name] = {"implementation": "explicit_relu_lstm", "hidden_size": cfg.baseline_hidden_size}
        else:
            if name == "rf":
                model = RandomForestRegressor(n_estimators=100, random_state=cfg.seed, n_jobs=1)
            elif name == "xgb":
                try:
                    from xgboost import XGBRegressor
                except ImportError as exc:
                    raise RuntimeError("Install xgboost or omit xgb from --models.") from exc
                model = XGBRegressor(n_estimators=100, learning_rate=.05, random_state=cfg.seed,
                                     max_depth=6, objective="reg:squarederror", n_jobs=1)
            else:
                model = SVR(kernel="rbf", C=100, gamma=.1, epsilon=.1)
            model.fit(tx.reshape(len(tx), -1), ty.ravel())
            models[name] = model
            predictions[name] = np.maximum(scalers["target"].inverse(model.predict(ex.reshape(len(ex), -1))[:, None]).ravel(), 0)
            joblib.dump(model, output / f"{name}.joblib")
            # XGBoost's `missing=nan` is an API sentinel, not a failed metric.
            # Record it as text while retaining strict finite JSON for results.
            parameters = {k: "NaN (missing-value sentinel)" if isinstance(v, float) and np.isnan(v) else v
                          for k, v in model.get_params().items()}
            descriptors[name] = {"implementation": type(model).__name__, "parameters": parameters}
    actual = arrays["pv_w"][test]
    frame = pd.DataFrame({"timestamp": arrays["timestamps"][test], "actual_pv_w": actual})
    for name, values in predictions.items():
        frame[f"{name}_pv_w"] = values
    frame.to_csv(output / "test_predictions.csv", index=False)
    metrics = {name: regression_metrics(actual, values) for name, values in predictions.items()}
    save_json(output / "metrics.json", metrics)
    save_json(output / "environment.json", environment())
    save_json(output / "configuration.json", vars(cfg))
    save_json(output / "split.json", split)
    save_json(output / "scalers.json", {"weather": scalers["weather"].serialize(),
                                      "target": scalers["target"].serialize(),
                                      "components": [s.serialize() for s in scalers["components"]]})
    save_json(output / "models.json", descriptors)
    if origin is None:
        local_times = arrays.get("local_timestamps", arrays["timestamps"])
        candidates = [i for i in test if pd.Timestamp(local_times[i]).hour == 8 and i + 12 <= len(arrays["pv_w"])]
        origin = candidates[0] if candidates else test[0]
    if origin not in test or origin < 24 or origin + 12 > len(arrays["pv_w"]):
        raise ValueError("Planning origin must be in the held-out test set with 12 subsequent observations.")
    planning_metrics = {}
    for name in model_names:
        pv_forecast = recursive_prediction(name, models, scalers, arrays, origin, 12, cfg)
        # Seasonal-naive load forecast is entirely before the planning origin.
        baseline = arrays["load_w"][origin-24:origin-12]
        problem = ScheduleProblem(pv_forecast, baseline, step_hours=metadata["step_hours"],
                                  actual_pv_w=arrays["pv_w"][origin:origin+12],
                                  name=f"{name}_{str(arrays['timestamps'][origin])}", data_kind="public_dataset_held_out_case")
        case = problem.to_dict()
        case["forecast_origin"] = str(arrays["timestamps"][origin])
        case["forecast_protocol"] = "12-step recursive; persisted last observed weather; previous-day load; future observations evaluation only"
        save_json(output / f"planning_case_{name}.json", case)
        planning_metrics[name] = regression_metrics(arrays["pv_w"][origin:origin+12], pv_forecast)
    save_json(output / "planning_metrics.json", {
        "forecast_origin_utc": str(arrays["timestamps"][origin]), "horizon": 12,
        "evaluation_scope": "one held-out recursive case; not an aggregate multi-origin result",
        "protocol": "no future observations; persisted weather; recursive target feedback",
        "models": planning_metrics})
    if plot:
        from .plotting import forecast_figure
        forecast_figure(arrays["timestamps"][test], actual, predictions, output)
    return metrics


def load_forecasters(directory, device="cpu"):
    """Reload our own saved models; joblib files must come from a trusted source."""
    directory = Path(directory)
    cfg = ForecastConfig(**json.loads((directory / "configuration.json").read_text()))
    cfg.device = device
    descriptor = json.loads((directory / "models.json").read_text())
    values = json.loads((directory / "scalers.json").read_text())
    scalers = {"weather": MinMax.restore(values["weather"]), "target": MinMax.restore(values["target"]),
               "components": [MinMax.restore(v) for v in values["components"]]}
    models = {}
    for name in descriptor:
        if name == "stl_lstm":
            models[name] = []
            for i in range(3):
                model = StackedLSTM(cfg.hidden_size, cfg.dropout).to(device)
                model.load_state_dict(torch.load(directory / f"stl_branch_{i}.pt", map_location=device, weights_only=True))
                models[name].append(model.eval())
        elif name == "lstm":
            model = ReluLSTM(cfg.baseline_hidden_size).to(device)
            model.load_state_dict(torch.load(directory / "lstm.pt", map_location=device, weights_only=True))
            models[name] = model.eval()
        else:
            models[name] = joblib.load(directory / f"{name}.joblib")
    return models, scalers, cfg
