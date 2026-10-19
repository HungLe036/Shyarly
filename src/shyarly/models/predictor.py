"""Calibrated match predictions from the production model at a point in time.

`Predictor(games, as_of)` fits BT on games strictly before `as_of` (a UTC day start),
then answers per-game and per-series probabilities with the production temperature.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.special import expit

from shyarly.config import LeagueMap, WorldsConfig
from shyarly.data.games import slugify
from shyarly.models.bt import BradleyTerry, BTParams
from shyarly.models.common import day_start
from shyarly.models.series import series_win_prob, side_averaged


@dataclass(frozen=True)
class TeamRef:
    """A team as the model sees it: OE name (-> team-season id) and region."""
    label: str          # display name or code
    oe_name: str
    region: str | None


def bt_params_from_config(model_cfg: dict) -> BTParams:
    p, section = model_cfg["production"]["bt_params"], model_cfg["bt"]
    return BTParams(
        half_life_days=p["half_life_days"], half_life_intl_days=p.get("half_life_intl_days"),
        lambda_team=p["lambda_team"], lambda_region_major=p["lambda_region_major"],
        lambda_region_minor=p["lambda_region_minor"], intl_weight=p["intl_weight"],
        lookback_days=section["lookback_days"], minor_regions=tuple(section["minor_regions"]),
    )


def shadow_params_from_config(model_cfg: dict, name: str = "margin_bt") -> tuple[BTParams, float] | None:
    """(params, temperature) of a fitted shadow model, or None if it has not been fitted yet."""
    fitted = model_cfg.get("shadow_models", {}).get(name, {}).get("fitted")
    if not fitted:
        return None
    from dataclasses import replace

    params = replace(bt_params_from_config(model_cfg), margin_weight=fitted["margin_weight"],
                     margin_scale=fitted["margin_scale"], margin_sigma=fitted["margin_sigma"])
    return params, fitted["temperature"]


class Predictor:
    def __init__(self, games: pd.DataFrame, as_of: pd.Timestamp, params: BTParams, temperature: float,
                 posterior: bool = False):
        self.as_of = day_start(as_of)
        self.temperature = temperature
        self.fit = BradleyTerry(params).fit(games, self.as_of, with_cov=posterior)

    @classmethod
    def from_config(cls, games: pd.DataFrame, as_of: pd.Timestamp, model_cfg: dict,
                    posterior: bool = False) -> Predictor:
        return cls(games, as_of, bt_params_from_config(model_cfg), model_cfg["production"]["temperature"],
                   posterior=posterior)

    def team_season_id(self, team: TeamRef) -> str:
        return f"{slugify(team.oe_name)}_{self.as_of.year}"

    def has_data(self, team: TeamRef) -> bool:
        return self.team_season_id(team) in self.fit.team

    def strength(self, team: TeamRef) -> float:
        return self.fit.strength(self.team_season_id(team), team.region)

    def p_blue(self, blue: TeamRef, red: TeamRef) -> float:
        """P(blue side team wins one game), calibrated."""
        z = self.fit.beta + self.strength(blue) - self.strength(red)
        return float(expit(z / self.temperature))

    def p_game(self, a: TeamRef, b: TeamRef) -> float:
        """P(a wins one game), averaged over sides."""
        return float(side_averaged(self.p_blue(a, b), self.p_blue(b, a)))

    def p_series(self, a: TeamRef, b: TeamRef, best_of: int) -> float:
        return float(series_win_prob(self.p_game(a, b), best_of))

    def game_matrix(self, teams: list[TeamRef]) -> np.ndarray:
        """M[i, j] = P(team i beats team j in one game), side-averaged."""
        n = len(teams)
        m = np.full((n, n), 0.5)
        for i in range(n):
            for j in range(i + 1, n):
                m[i, j] = self.p_game(teams[i], teams[j])
                m[j, i] = 1.0 - m[i, j]
        return m

    def strength_posterior(self, teams: list[TeamRef]) -> tuple[np.ndarray, np.ndarray]:
        """Laplace posterior of (beta, strength_1, ..., strength_n): mean and covariance.
        A strength is region offset + team deviation; a team with no games this season is its region."""
        if self.fit.cov is None:
            raise ValueError("Predictor was built without posterior=True")
        idx = self.fit.index
        a = np.zeros((len(teams) + 1, len(idx)))
        a[0, idx[("beta", "")]] = 1.0
        for row, t in enumerate(teams, start=1):
            for key in (("region", t.region), ("team", self.team_season_id(t))):
                if key in idx:
                    a[row, idx[key]] = 1.0
        return a @ self.fit.theta, a @ self.fit.cov @ a.T

    def game_matrix_draws(self, teams: list[TeamRef], n: int, rng: np.random.Generator) -> np.ndarray:
        """n posterior draws of game_matrix: [k, i, j] = P(i beats j in one game) under draw k."""
        mean, cov = self.strength_posterior(teams)
        d = rng.multivariate_normal(mean, cov, size=n, method="cholesky")
        beta, s = d[:, :1, None], d[:, 1:]
        diff = s[:, :, None] - s[:, None, :]
        p_i_blue = expit((beta + diff) / self.temperature)
        p_j_blue = expit((beta - diff) / self.temperature)
        return side_averaged(p_i_blue, p_j_blue)


def worlds_team_ref(code: str, worlds: WorldsConfig, leagues: LeagueMap, year: int) -> TeamRef:
    """A Worlds or Play-In team by code; region from its league via config/leagues.yaml."""
    known = {t.code: t for t in worlds.teams if t.status != "pending"} | {t.code: t for t in worlds.play_in_teams}
    t = known[code]
    mapped = leagues.lookup(t.league, year)
    return TeamRef(label=t.code, oe_name=t.oe_name, region=mapped[0] if mapped else None)


def resolve_team(query: str, worlds: WorldsConfig, leagues: LeagueMap, games: pd.DataFrame,
                 as_of: pd.Timestamp) -> TeamRef:
    """Find a team by Worlds code, Worlds name or OE name (case-insensitive). Teams outside the
    Worlds config take the region of their most recent game before `as_of`."""
    q = query.strip().casefold()
    year = day_start(as_of).year
    for t in list(worlds.teams) + list(worlds.play_in_teams):
        if t.status == "pending":
            continue
        if q in {t.code.casefold(), t.name.casefold(), (t.oe_name or "").casefold()}:
            return worlds_team_ref(t.code, worlds, leagues, year)
    past = games[games["date_utc"] < day_start(as_of)]
    side = [("blue_team", "blue_region"), ("red_team", "red_region")]
    rows = pd.concat([past[[n, r, "date_utc"]].set_axis(["team", "region", "date_utc"], axis=1) for n, r in side])
    rows = rows[rows["team"].str.casefold() == q].dropna(subset=["region"])
    if rows.empty:
        raise KeyError(f"No team called {query!r} (try a Worlds code like GEN, or an Oracle's Elixir team name)")
    last = rows.sort_values("date_utc").iloc[-1]
    return TeamRef(label=last["team"], oe_name=last["team"], region=last["region"])
