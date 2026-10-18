"""Trivial baselines. (Region-prior-only is BradleyTerry(..., use_teams=False).)"""

from __future__ import annotations

import numpy as np
import pandas as pd


class CoinFlip:
    name = "coinflip"

    def predict(self, games: pd.DataFrame, queries: pd.DataFrame) -> np.ndarray:
        return np.full(len(queries), 0.5)
