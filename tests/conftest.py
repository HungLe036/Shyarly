"""Shared synthetic data: games drawn from a known Bradley-Terry world."""

import numpy as np
import pandas as pd
import pytest

TRUE_REGION = {"A": 1.0, "B": 0.0, "C": -1.0}
TRUE_TEAM = {f"{r}{i}": d for r in TRUE_REGION for i, d in enumerate([0.4, 0.1, -0.1, -0.4])}
TRUE_SIDE = 0.1


def team_strength(team: str) -> float:
    return TRUE_REGION[team[0]] + TRUE_TEAM[team]


def make_games(seed: int = 0, n_days: int = 400, games_per_day: int = 6, cross_share: float = 0.3,
               gold_per_logit: float | None = None, gold_noise: float = 2500.0) -> pd.DataFrame:
    """With `gold_per_logit`, each game also gets gd15 = gold_per_logit * true logit + noise."""
    rng = np.random.default_rng(seed)
    teams = list(TRUE_TEAM)
    rows = []
    start = pd.Timestamp("2025-01-01", tz="UTC")
    for d in range(n_days):
        for k in range(games_per_day):
            blue = teams[rng.integers(len(teams))]
            if rng.random() < cross_share:
                pool = [t for t in teams if t[0] != blue[0]]
            else:
                pool = [t for t in teams if t[0] == blue[0] and t != blue]
            red = pool[rng.integers(len(pool))]
            z = team_strength(blue) - team_strength(red) + TRUE_SIDE
            win = int(rng.random() < 1 / (1 + np.exp(-z)))
            t = start + pd.Timedelta(days=d, hours=10 + k)
            extra = {} if gold_per_logit is None else {"gd15": gold_per_logit * z + rng.normal(0, gold_noise)}
            rows.append({
                **extra,
                "game_id": f"g{d}_{k}", "date_utc": t,
                "blue_team": blue, "red_team": red,
                "blue_team_season_id": f"{blue.lower()}_{t.year}",
                "red_team_season_id": f"{red.lower()}_{t.year}",
                "blue_region": blue[0], "red_region": red[0],
                "blue_win": win, "cross_region": blue[0] != red[0],
                "series_id": f"s{d}_{k}",
            })
    df = pd.DataFrame(rows)
    df["date_utc"] = df["date_utc"].astype("datetime64[us, UTC]")
    df["blue_win"] = df["blue_win"].astype("int8")
    return df


def make_queries(games: pd.DataFrame, cutoff: pd.Timestamp, n: int = 20) -> pd.DataFrame:
    q = games[games["date_utc"] >= cutoff].head(n).copy()
    q["cutoff"] = cutoff
    return q.reset_index(drop=True)


@pytest.fixture(scope="session")
def synth_games():
    return make_games()
