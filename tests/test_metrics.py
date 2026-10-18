import numpy as np
import pytest

from shyarly.eval.metrics import brier, calibration_slope, ece, log_loss, paired_bootstrap


def test_coinflip_reference_values():
    y = np.array([0, 1, 1, 0])
    p = np.full(4, 0.5)
    assert log_loss(y, p) == pytest.approx(np.log(2))
    assert brier(y, p) == pytest.approx(0.25)


def test_calibrated_sample_has_small_ece_and_unit_slope():
    rng = np.random.default_rng(0)
    p = rng.uniform(0.05, 0.95, 200_000)
    y = (rng.random(p.size) < p).astype(int)
    assert ece(y, p, bins=5) < 0.01
    assert calibration_slope(y, p) == pytest.approx(1.0, abs=0.03)


def test_overconfident_sample_has_slope_below_one():
    rng = np.random.default_rng(1)
    z = rng.normal(0, 1, 100_000)
    y = (rng.random(z.size) < 1 / (1 + np.exp(-z))).astype(int)
    p_over = 1 / (1 + np.exp(-2 * z))
    assert calibration_slope(y, p_over) == pytest.approx(0.5, abs=0.03)


def test_paired_bootstrap_is_seeded_and_centred():
    rng = np.random.default_rng(2)
    n = 2000
    clusters = np.repeat(np.arange(n // 2), 2)
    a = rng.normal(0.60, 0.1, n)
    b = a + 0.02 + rng.normal(0, 0.01, n)
    r1 = paired_bootstrap(a, b, clusters, resamples=500, seed=7)
    r2 = paired_bootstrap(a, b, clusters, resamples=500, seed=7)
    assert r1 == r2
    assert r1["mean"] == pytest.approx(-0.02, abs=0.002)
    assert r1["ci95"][0] < -0.02 < r1["ci95"][1]
    assert r1["ci95"][1] < 0
