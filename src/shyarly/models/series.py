"""Series win probabilities from a per-game probability, treating games as independent.

Spec: use the per-game p averaged over blue and red side, then the closed-form
best-of-n formula.
"""

from __future__ import annotations

import numpy as np


def side_averaged(p_a_on_blue, p_b_on_blue):
    """P(A wins a game) averaged over sides, from P(A wins as blue) and P(B wins as blue)."""
    return (np.asarray(p_a_on_blue, float) + 1.0 - np.asarray(p_b_on_blue, float)) / 2.0


def series_win_prob(p, best_of: int):
    """P(win a best-of-n series) given per-game win probability p.

    Bo1: p.  Bo3: p^2 (3 - 2p).  Bo5: p^3 (10 - 15p + 6p^2).
    """
    p = np.asarray(p, float)
    if best_of == 1:
        out = p
    elif best_of == 3:
        out = p**2 * (3 - 2 * p)
    elif best_of == 5:
        out = p**3 * (10 - 15 * p + 6 * p**2)
    else:
        raise ValueError(f"best_of must be 1, 3 or 5, got {best_of}")
    return out.item() if out.ndim == 0 else out
