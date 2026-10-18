"""Time-decayed, ridge-regularized Bradley-Terry logistic regression.

    logit P(blue wins) = (r[region_blue] + u[team_blue]) - (r[region_red] + u[team_red]) + beta

r are region offsets, u are team-season deviations within region, beta is the blue-side
advantage. Each game is weighted by 0.5 ** (age / half_life), times `intl_weight` for
cross-region games. Ridge penalties shrink u (lambda_team) harder than r; minor regions
get a smaller penalty (wider prior) so they can sit far from the average region.
With `use_teams=False` this is the region-prior-only baseline.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.sparse import csr_matrix
from scipy.special import expit

from shyarly.models.common import training_view


@dataclass(frozen=True)
class BTParams:
    half_life_days: float
    lambda_team: float
    lambda_region_major: float
    lambda_region_minor: float
    intl_weight: float
    lookback_days: float
    minor_regions: tuple[str, ...] = ()


@dataclass
class BTFit:
    beta: float = 0.0
    region: dict[str, float] = field(default_factory=dict)
    team: dict[str, float] = field(default_factory=dict)

    def strength(self, team_season_id: str, region: str | None) -> float:
        return self.region.get(region, 0.0) + self.team.get(team_season_id, 0.0)


class BradleyTerry:
    def __init__(self, params: BTParams, use_teams: bool = True):
        self.params = params
        self.use_teams = use_teams
        self.name = "bt" if use_teams else "region_prior"

    def fit(self, games: pd.DataFrame, cutoff: pd.Timestamp) -> BTFit:
        p = self.params
        train = training_view(games, cutoff, p.lookback_days)
        if train.empty:
            return BTFit()

        age = (cutoff - train["date_utc"]).dt.total_seconds().to_numpy() / 86400.0
        w = 0.5 ** (age / p.half_life_days) * np.where(train["cross_region"].to_numpy(), p.intl_weight, 1.0)
        y = train["blue_win"].to_numpy(dtype=float)

        regions = sorted(set(train["blue_region"]) | set(train["red_region"]))
        teams = sorted(set(train["blue_team_season_id"]) | set(train["red_team_season_id"])) if self.use_teams else []
        r_idx = {r: 1 + i for i, r in enumerate(regions)}
        t_idx = {t: 1 + len(regions) + i for i, t in enumerate(teams)}
        n, k = len(train), 1 + len(regions) + len(teams)

        rows = [np.arange(n)] * 3
        cols = [np.zeros(n, int), train["blue_region"].map(r_idx).to_numpy(), train["red_region"].map(r_idx).to_numpy()]
        vals = [np.ones(n), np.ones(n), -np.ones(n)]
        if self.use_teams:
            rows += [np.arange(n)] * 2
            cols += [train["blue_team_season_id"].map(t_idx).to_numpy(), train["red_team_season_id"].map(t_idx).to_numpy()]
            vals += [np.ones(n), -np.ones(n)]
        X = csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(n, k))

        pen = np.zeros(k)
        minor = set(p.minor_regions)
        for r, i in r_idx.items():
            pen[i] = p.lambda_region_minor if r in minor else p.lambda_region_major
        pen[1 + len(regions):] = p.lambda_team

        def objective(theta):
            z = X @ theta
            loss = np.sum(w * (np.logaddexp(0.0, z) - y * z)) + np.sum(pen * theta**2)
            grad = X.T @ (w * (expit(z) - y)) + 2 * pen * theta
            return loss, grad

        res = minimize(objective, np.zeros(k), jac=True, method="L-BFGS-B",
                       options={"maxiter": 2000, "gtol": 1e-8})
        theta = res.x
        return BTFit(
            beta=float(theta[0]),
            region={r: float(theta[i]) for r, i in r_idx.items()},
            team={t: float(theta[i]) for t, i in t_idx.items()},
        )

    def predict(self, games: pd.DataFrame, queries: pd.DataFrame) -> np.ndarray:
        out = np.empty(len(queries))
        for cutoff, q in queries.groupby("cutoff", sort=True):
            fit = self.fit(games, cutoff)
            z = np.array([
                fit.beta + fit.strength(bt, br) - fit.strength(rt, rr)
                for bt, br, rt, rr in zip(q["blue_team_season_id"], q["blue_region"],
                                          q["red_team_season_id"], q["red_region"])
            ])
            out[queries.index.get_indexer(q.index)] = expit(z)
        return out
