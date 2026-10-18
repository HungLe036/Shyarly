"""Probability scoring: log loss (primary), Brier, ECE, calibration slope, paired bootstrap."""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit, logit

EPS = 1e-12


def _clip(p):
    return np.clip(np.asarray(p, dtype=float), EPS, 1 - EPS)


def game_log_loss(y, p) -> np.ndarray:
    p, y = _clip(p), np.asarray(y, dtype=float)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def log_loss(y, p) -> float:
    return float(game_log_loss(y, p).mean())


def brier(y, p) -> float:
    return float(np.mean((np.asarray(p, float) - np.asarray(y, float)) ** 2))


def ece(y, p, bins: int = 5) -> float:
    """Expected calibration error with equal-width bins on [0, 1]."""
    y, p = np.asarray(y, float), np.asarray(p, float)
    which = np.minimum((p * bins).astype(int), bins - 1)
    total = 0.0
    for b in range(bins):
        m = which == b
        if m.any():
            total += m.mean() * abs(p[m].mean() - y[m].mean())
    return float(total)


def calibration_slope(y, p) -> float:
    """b in y ~ Bernoulli(sigmoid(a + b * logit(p))). 1 = calibrated, < 1 = overconfident."""
    y, z = np.asarray(y, float), logit(_clip(p))

    def nll(ab):
        s = ab[0] + ab[1] * z
        return np.sum(np.logaddexp(0, s) - y * s), np.array([
            np.sum(expit(s) - y), np.sum((expit(s) - y) * z)])

    return float(minimize(nll, np.array([0.0, 1.0]), jac=True, method="L-BFGS-B").x[1])


def paired_bootstrap(loss_a, loss_b, clusters, resamples: int, seed: int) -> dict:
    """Mean of (loss_a - loss_b) per game, with CIs from resampling whole clusters."""
    diff = np.asarray(loss_a, float) - np.asarray(loss_b, float)
    _, inv = np.unique(np.asarray(clusters), return_inverse=True)
    sums = np.bincount(inv, weights=diff)
    counts = np.bincount(inv).astype(float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(sums), size=(resamples, len(sums)))
    stats = sums[idx].sum(axis=1) / counts[idx].sum(axis=1)

    def ci(lo: float, hi: float) -> tuple[float, float]:
        return float(np.quantile(stats, lo)), float(np.quantile(stats, hi))

    return {"mean": float(diff.mean()), "ci80": ci(0.10, 0.90), "ci90": ci(0.05, 0.95), "ci95": ci(0.025, 0.975)}
