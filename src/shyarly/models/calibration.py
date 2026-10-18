"""Temperature scaling: p_cal = sigmoid(logit(p) / T).

A single temperature with no intercept keeps the model symmetric (p = 0.5 stays 0.5).
In the backtest, each event's T is fitted only on predictions for earlier events,
so calibration never sees the outcomes it is scored on.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.special import expit, logit

from shyarly.eval.metrics import log_loss

T_BOUNDS = (0.25, 4.0)


def apply_temperature(p, T: float):
    return expit(logit(np.clip(np.asarray(p, float), 1e-12, 1 - 1e-12)) / T)


def fit_temperature(y, p) -> float:
    y, z = np.asarray(y, float), logit(np.clip(np.asarray(p, float), 1e-12, 1 - 1e-12))
    res = minimize_scalar(lambda t: log_loss(y, expit(z / t)), bounds=T_BOUNDS, method="bounded",
                          options={"xatol": 1e-6})
    return float(res.x)


def rolling_temperatures(preds: pd.DataFrame, event_order: list[str]) -> dict[str, float]:
    """T for each event, fitted on all predictions of the events before it (T = 1 for the first)."""
    temps = {}
    for i, event in enumerate(event_order):
        past = preds[preds["event"].isin(event_order[:i])]
        temps[event] = fit_temperature(past["y"], past["p"]) if len(past) else 1.0
    return temps
