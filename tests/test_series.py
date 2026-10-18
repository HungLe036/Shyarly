"""Closed-form series win probabilities (games treated as independent)."""

import numpy as np
import pytest

from shyarly.models.series import series_win_prob, side_averaged


@pytest.mark.parametrize("p, bo, expected", [
    (0.6, 1, 0.6),
    (0.6, 3, 0.648),        # p^2 (3 - 2p)
    (0.6, 5, 0.68256),      # p^3 (10 - 15p + 6p^2)
    (0.5, 3, 0.5),
    (0.5, 5, 0.5),
])
def test_known_values(p, bo, expected):
    assert series_win_prob(p, bo) == pytest.approx(expected)


@pytest.mark.parametrize("bo", [1, 3, 5])
def test_closed_form_matches_enumeration(bo):
    """Brute force: play all bo games and count who reaches a majority."""
    from itertools import product

    need = bo // 2 + 1
    for p in np.linspace(0, 1, 21):
        total = 0.0
        for outcome in product([1, 0], repeat=bo):
            if sum(outcome) >= need:
                total += np.prod([p if o else 1 - p for o in outcome])
        assert series_win_prob(p, bo) == pytest.approx(total, abs=1e-12)


@pytest.mark.parametrize("bo", [1, 3, 5])
def test_symmetry_and_bounds(bo):
    p = np.linspace(0, 1, 101)
    s = series_win_prob(p, bo)
    np.testing.assert_allclose(s + series_win_prob(1 - p, bo), 1.0, atol=1e-12)
    assert s[0] == 0.0 and s[-1] == 1.0
    assert np.all(np.diff(s) >= 0)


def test_longer_series_favour_the_favourite():
    p = np.linspace(0.51, 0.99, 25)
    assert np.all(series_win_prob(p, 5) > series_win_prob(p, 3))
    assert np.all(series_win_prob(p, 3) > series_win_prob(p, 1))


def test_rejects_even_or_unsupported_lengths():
    with pytest.raises(ValueError):
        series_win_prob(0.6, 2)
    with pytest.raises(ValueError):
        series_win_prob(0.6, 7)


def test_side_averaged():
    # A wins 0.65 on blue; B wins 0.45 on blue, so A wins 0.55 on red. Average 0.60.
    assert side_averaged(0.65, 0.45) == pytest.approx(0.60)
    np.testing.assert_allclose(side_averaged(np.array([0.5, 0.7]), np.array([0.5, 0.3])), [0.5, 0.7])
