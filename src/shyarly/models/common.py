"""Shared point-in-time helpers for all models.

Every model predicts a query from `training_view(games, query.cutoff)`, i.e. only
games whose `date_utc` is strictly before the cutoff. Cutoffs are UTC day starts.
"""

from __future__ import annotations

from typing import Protocol

import numpy as np
import pandas as pd

QUERY_COLUMNS = [
    "cutoff", "blue_team", "red_team", "blue_team_season_id", "red_team_season_id",
    "blue_region", "red_region",
]


def day_start(ts: pd.Series | pd.Timestamp):
    """UTC midnight of each timestamp: the cutoff for predicting games on that day."""
    return ts.dt.floor("D") if isinstance(ts, pd.Series) else ts.floor("D")


def training_view(games: pd.DataFrame, cutoff: pd.Timestamp, lookback_days: float | None) -> pd.DataFrame:
    """Games usable for a prediction at `cutoff`: strictly earlier, inside the lookback
    window, and with both teams' regions known (the model universe)."""
    mask = (games["date_utc"] < cutoff) & games["blue_region"].notna() & games["red_region"].notna()
    if lookback_days is not None:
        mask &= games["date_utc"] >= cutoff - pd.Timedelta(days=lookback_days)
    return games[mask]


class Model(Protocol):
    name: str

    def predict(self, games: pd.DataFrame, queries: pd.DataFrame) -> np.ndarray:
        """P(blue side wins) for each query row, using only games before its `cutoff`."""
        ...
