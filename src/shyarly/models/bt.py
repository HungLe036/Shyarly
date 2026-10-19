"""Time-decayed, ridge-regularized Bradley-Terry logistic regression.

    logit P(blue wins) = (r[region_blue] + u[team_blue]) - (r[region_red] + u[team_red]) + beta

r are region offsets, u are team-season deviations within region, beta is the blue-side
advantage. Each game is weighted by 0.5 ** (age / half_life), times `intl_weight` for
cross-region games. Cross-region games, the only ones that move r, can use their own
longer half-life (`half_life_intl_days`) so region offsets keep a longer memory than
team form. Ridge penalties shrink u (lambda_team) harder than r; minor regions
get a smaller penalty (wider prior) so they can sit far from the average region.
With `use_teams=False` this is the region-prior-only baseline.

The fit is the mode of a Bayesian model (the ridge penalties are normal priors, the
decay weights scale each game's likelihood). `fit(..., with_cov=True)` adds the Laplace
approximation of the posterior: a normal centred on the mode with covariance equal to the
inverse Hessian of the objective, H = X' diag(w p (1-p)) X + 2 diag(penalty).
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
    half_life_intl_days: float | None = None   # None: cross-region games use half_life_days
    # Shadow model margin_bt (config/model.yaml shadow_models): games with a gold difference at
    # 15:00 add 0.5 * margin_weight * w * (gd15 / margin_scale - z)^2 / margin_sigma^2 to the loss.
    margin_weight: float = 0.0
    margin_scale: float | None = None          # gold per logit unit
    margin_sigma: float = 1.0                  # residual sd of gd15 / margin_scale, in logit units


@dataclass
class BTFit:
    beta: float = 0.0
    region: dict[str, float] = field(default_factory=dict)
    team: dict[str, float] = field(default_factory=dict)
    # Parameter vector, its layout (("beta", ""), ("region", r), ("team", t)) -> position,
    # and the Laplace posterior covariance (only with `fit(..., with_cov=True)`).
    theta: np.ndarray = field(default_factory=lambda: np.zeros(1))
    index: dict[tuple[str, str], int] = field(default_factory=lambda: {("beta", ""): 0})
    cov: np.ndarray | None = None

    def strength(self, team_season_id: str, region: str | None) -> float:
        return self.region.get(region, 0.0) + self.team.get(team_season_id, 0.0)


@dataclass
class _Problem:
    X: csr_matrix
    y: np.ndarray
    w: np.ndarray
    pen: np.ndarray
    index: dict[tuple[str, str], int]
    m: np.ndarray | None = None      # gold margin target in logit units (0 where missing)
    mw: np.ndarray | None = None     # per-game margin precision weight (0 where missing)


class BradleyTerry:
    def __init__(self, params: BTParams, use_teams: bool = True):
        self.params = params
        self.use_teams = use_teams
        self.name = "bt" if use_teams else "region_prior"

    def objective(self, games: pd.DataFrame, cutoff: pd.Timestamp):
        """(loss(theta), grad(theta)) of the penalized weighted negative log-likelihood."""
        prob = self._problem(games, cutoff)
        return (lambda th: self._loss_grad(prob, th)[0]), (lambda th: self._loss_grad(prob, th)[1])

    @staticmethod
    def _loss_grad(prob: _Problem, theta: np.ndarray):
        z = prob.X @ theta
        loss = np.sum(prob.w * (np.logaddexp(0.0, z) - prob.y * z)) + np.sum(prob.pen * theta**2)
        resid = prob.w * (expit(z) - prob.y)
        if prob.mw is not None:
            loss += 0.5 * np.sum(prob.mw * (z - prob.m) ** 2)
            resid = resid + prob.mw * (z - prob.m)
        grad = prob.X.T @ resid + 2 * prob.pen * theta
        return loss, grad

    def _problem(self, games: pd.DataFrame, cutoff: pd.Timestamp) -> _Problem | None:
        p = self.params
        train = training_view(games, cutoff, p.lookback_days)
        if train.empty:
            return None

        age = (cutoff - train["date_utc"]).dt.total_seconds().to_numpy() / 86400.0
        cross = train["cross_region"].to_numpy()
        half_life = np.where(cross, p.half_life_intl_days or p.half_life_days, p.half_life_days)
        w = 0.5 ** (age / half_life) * np.where(cross, p.intl_weight, 1.0)
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

        index = {("beta", ""): 0} | {("region", r): i for r, i in r_idx.items()} | {("team", t): i for t, i in t_idx.items()}
        m = mw = None
        if p.margin_weight > 0 and "gd15" in train:
            if not p.margin_scale:
                raise ValueError("margin_weight > 0 needs margin_scale (gold per logit unit)")
            gd = train["gd15"].to_numpy(dtype=float)
            has = np.isfinite(gd)
            if has.any():
                m = np.where(has, gd / p.margin_scale, 0.0)
                mw = np.where(has, p.margin_weight * w / p.margin_sigma**2, 0.0)
        return _Problem(X, y, w, pen, index, m, mw)

    def fit(self, games: pd.DataFrame, cutoff: pd.Timestamp, with_cov: bool = False) -> BTFit:
        prob = self._problem(games, cutoff)
        if prob is None:
            return BTFit(cov=np.eye(1) if with_cov else None)

        res = minimize(lambda th: self._loss_grad(prob, th), np.zeros(len(prob.pen)), jac=True,
                       method="L-BFGS-B", options={"maxiter": 2000, "gtol": 1e-8})
        theta = res.x
        cov = None
        if with_cov:
            z = prob.X @ theta
            pz = expit(z)
            curv = prob.w * pz * (1.0 - pz) + (prob.mw if prob.mw is not None else 0.0)
            hess = (prob.X.T @ prob.X.multiply(curv[:, None])).toarray() + np.diag(2 * prob.pen)
            cov = np.linalg.inv(hess)
            cov = (cov + cov.T) / 2
        return BTFit(
            beta=float(theta[0]),
            region={key: float(theta[i]) for (kind, key), i in prob.index.items() if kind == "region"},
            team={key: float(theta[i]) for (kind, key), i in prob.index.items() if kind == "team"},
            theta=theta, index=prob.index, cov=cov,
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
