"""Region-offset Elo — a baseline, not the product.

Each team (keyed by name, carried across seasons) has a rating within its region and
each region has an offset; cross-region games also update the offsets. Ratings are
multiplied by `season_carry` at every calendar-year boundary. The blue-side bonus is
the logit of the blue win rate seen so far, in Elo points.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

from shyarly.models.common import training_view

SCALE = 400.0


@dataclass(frozen=True)
class EloParams:
    k_team: float
    k_region: float
    season_carry: float


class Elo:
    name = "elo"

    def __init__(self, params: EloParams):
        self.params = params

    def predict(self, games: pd.DataFrame, queries: pd.DataFrame) -> np.ndarray:
        """One chronological pass: before answering the queries of each cutoff, apply
        exactly the games dated before that cutoff."""
        p = self.params
        last_cutoff = queries["cutoff"].max()
        train = training_view(games, last_cutoff, lookback_days=None).sort_values(["date_utc", "game_id"], kind="stable")
        dates = train["date_utc"].tolist()   # tz-aware Timestamps, compared with tz-aware cutoffs
        cols = zip(train["date_utc"].dt.year, train["blue_team"], train["red_team"],
                   train["blue_region"], train["red_region"], train["blue_win"])
        game_rows = list(cols)

        rating: dict[str, float] = {}
        last_year: dict[str, int] = {}
        offset: dict[str, float] = {}
        blue_wins = n_games = 0

        def current(team: str, year: int) -> float:
            r = rating.get(team, 0.0)
            ly = last_year.get(team)
            return r * p.season_carry ** (year - ly) if ly is not None and year > ly else r

        def side_bonus() -> float:
            return SCALE * math.log10((blue_wins + 1) / (n_games - blue_wins + 1))

        out = np.empty(len(queries))
        i = 0
        for cutoff, q in queries.groupby("cutoff", sort=True):
            while i < len(game_rows) and dates[i] < cutoff:
                year, bt, rt, br, rr, y = game_rows[i]
                for t in (bt, rt):
                    rating[t] = current(t, year)
                    last_year[t] = year
                diff = rating[bt] + offset.get(br, 0.0) - rating[rt] - offset.get(rr, 0.0) + side_bonus()
                e = 1.0 / (1.0 + 10 ** (-diff / SCALE))
                delta = float(y) - e
                rating[bt] += p.k_team * delta
                rating[rt] -= p.k_team * delta
                if br != rr:
                    offset[br] = offset.get(br, 0.0) + p.k_region * delta
                    offset[rr] = offset.get(rr, 0.0) - p.k_region * delta
                blue_wins += int(y)
                n_games += 1
                i += 1
            year = cutoff.year
            diffs = np.array([
                current(bt, year) + offset.get(br, 0.0) - current(rt, year) - offset.get(rr, 0.0) + side_bonus()
                for bt, rt, br, rr in zip(q["blue_team"], q["red_team"], q["blue_region"], q["red_region"])
            ])
            out[queries.index.get_indexer(q.index)] = 1.0 / (1.0 + 10 ** (-diffs / SCALE))
        return out
