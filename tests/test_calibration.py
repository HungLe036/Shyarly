"""Temperature scaling: one temperature on the logit, no intercept, fitted only on past events."""

import numpy as np
import pandas as pd
import pytest
from scipy.special import expit

from shyarly.models.calibration import apply_temperature, fit_temperature, rolling_temperatures


def test_recovers_overconfidence_factor():
    rng = np.random.default_rng(0)
    z = rng.normal(0, 1, 50_000)
    y = (rng.random(z.size) < expit(z)).astype(int)
    overconfident = expit(2.0 * z)
    assert fit_temperature(y, overconfident) == pytest.approx(2.0, rel=0.05)


def test_apply_keeps_symmetry_and_identity():
    p = np.array([0.1, 0.3, 0.5, 0.8])
    np.testing.assert_allclose(apply_temperature(p, 1.0), p)
    np.testing.assert_allclose(apply_temperature(1 - p, 1.7), 1 - apply_temperature(p, 1.7))
    assert apply_temperature(np.array([0.5]), 3.0)[0] == pytest.approx(0.5)


def _preds(seed=1):
    rng = np.random.default_rng(seed)
    rows = []
    for e in ["E1", "E2", "E3"]:
        z = rng.normal(0, 1, 400)
        y = (rng.random(z.size) < expit(z)).astype(int)
        rows.append(pd.DataFrame({"event": e, "y": y, "p": expit(1.5 * z)}))
    return pd.concat(rows, ignore_index=True)


def test_rolling_uses_only_earlier_events():
    order = ["E1", "E2", "E3"]
    d = _preds()
    temps = rolling_temperatures(d, order)
    assert temps["E1"] == 1.0                                  # no history yet
    assert temps["E2"] == pytest.approx(fit_temperature(d[d.event == "E1"].y, d[d.event == "E1"].p))
    changed = d.copy()
    later = changed["event"] == "E3"
    changed.loc[later, "y"] = 1 - changed.loc[later, "y"]      # rewrite the latest event's outcomes
    changed_temps = rolling_temperatures(changed, order)
    assert changed_temps["E1"] == temps["E1"] and changed_temps["E2"] == temps["E2"]
    assert changed_temps["E3"] == temps["E3"]                  # E3's own outcomes never set E3's T
    assert rolling_temperatures(changed, order + ["E4"])["E4"] != rolling_temperatures(d, order + ["E4"])["E4"]
