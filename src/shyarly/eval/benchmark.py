"""Model vs betting market: production BT against closing Pinnacle odds (data/manual/market_odds.csv).

Each market fixture is joined to an Oracle's Elixir series with the same two teams starting
within 12 hours, which gives the outcome (series winner, and game 1 winner for the first-map
market). The model's probability is the production BT, calibrated, trained on games before
the match's UTC day (live mode, the same rule as the prediction log). The market's
probability is the closing price with the bookmaker margin removed, so the market also
knows same-day information the model does not (earlier results that day, line-ups).

No odds history exists before ~June 2026, so this is a benchmark only: nothing is tuned on it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from shyarly.eval import metrics
from shyarly.models.bt import BTParams
from shyarly.models.common import day_start
from shyarly.models.predictor import Predictor, TeamRef
from shyarly.models.series import series_win_prob

REPORT_NAME = "market_benchmark.md"
MATCH_WINDOW = pd.Timedelta(hours=12)


def _norm(name: str) -> str:
    return "".join(ch for ch in str(name).casefold() if ch.isalnum())


def name_map(names, oe_names, aliases: dict[str, str]) -> tuple[dict[str, str], list[str]]:
    """OddsPapi team name -> OE team name, by alias first, then by case/punctuation-insensitive match."""
    by_norm = {_norm(n): n for n in oe_names}
    out, missing = {}, []
    for n in names:
        target = aliases.get(n) or by_norm.get(_norm(n))
        if target in oe_names:
            out[n] = target
        else:
            missing.append(n)
    return out, sorted(missing)


def join_market(market: pd.DataFrame, series: pd.DataFrame, games: pd.DataFrame,
                aliases: dict[str, str]) -> tuple[pd.DataFrame, list[str]]:
    """One row per matched fixture with outcomes oriented to the market's team_1."""
    wide = market.pivot_table(index=["fixture_id", "tournament", "start_time_utc", "team_1", "team_2"],
                              columns="market", values="p1_fair", aggfunc="first").reset_index()
    wide.columns = [str(c) for c in wide.columns]
    oe_names = set(series["team_a"]) | set(series["team_b"])
    names, _ = name_map(sorted(set(wide["team_1"]) | set(wide["team_2"])), oe_names, aliases)
    g1 = games[games["game_in_series"] == 1].set_index("series_id")
    rows, unmatched = [], []
    for r in wide.to_dict("records"):
        t1, t2 = names.get(r["team_1"]), names.get(r["team_2"])
        start = pd.Timestamp(r["start_time_utc"])
        cand = series[(((series["team_a"] == t1) & (series["team_b"] == t2)) |
                       ((series["team_a"] == t2) & (series["team_b"] == t1))) &
                      ((series["date_utc"] - start).abs() <= MATCH_WINDOW)] if t1 and t2 else series.iloc[0:0]
        if len(cand) != 1:
            unmatched.append(r["fixture_id"])
            continue
        s = cand.iloc[0]
        y_map1 = np.nan
        if s["series_id"] in g1.index:
            g = g1.loc[s["series_id"]]
            first_winner = g["blue_team"] if g["blue_win"] == 1 else g["red_team"]
            y_map1 = float(first_winner == t1)
        rows.append({
            "fixture_id": r["fixture_id"], "tournament": r["tournament"], "start": start, "team_1": t1, "team_2": t2,
            "series_id": s["series_id"], "event": s["event"], "best_of": int(s["inferred_best_of"]),
            "y_series": float(s["winner_team"] == t1), "y_map1": y_map1,
            "p_mkt_series": float(r.get("181", np.nan)), "p_mkt_map1": float(r.get("1847", np.nan)),
        })
    return pd.DataFrame(rows), sorted(unmatched)


def _ref(team: str, games: pd.DataFrame, cutoff: pd.Timestamp) -> TeamRef:
    past = games[games["date_utc"] < cutoff]
    regions = pd.concat([past.loc[past["blue_team"] == team, ["date_utc", "blue_region"]].set_axis(["d", "r"], axis=1),
                         past.loc[past["red_team"] == team, ["date_utc", "red_region"]].set_axis(["d", "r"], axis=1)])
    regions = regions.dropna().sort_values("d")
    return TeamRef(team, team, regions["r"].iloc[-1] if len(regions) else None)


def add_model(joined: pd.DataFrame, games: pd.DataFrame, params: BTParams, temperature: float) -> pd.DataFrame:
    out = joined.copy()
    cache: dict[pd.Timestamp, Predictor] = {}
    pg, ps = [], []
    for r in out.to_dict("records"):
        cutoff = day_start(r["start"])
        if cutoff not in cache:
            cache[cutoff] = Predictor(games, cutoff, params, temperature)
        p = cache[cutoff].p_game(_ref(r["team_1"], games, cutoff), _ref(r["team_2"], games, cutoff))
        pg.append(p)
        ps.append(float(series_win_prob(p, r["best_of"])))
    return out.assign(p_model_game=pg, p_model_series=ps)


def score_benchmark(j: pd.DataFrame, resamples: int, seed: int) -> pd.DataFrame:
    rows = []
    for target, y, pm, pk in (("series", "y_series", "p_model_series", "p_mkt_series"),
                              ("first map", "y_map1", "p_model_game", "p_mkt_map1")):
        d = j.dropna(subset=[y, pm, pk])
        if d.empty:
            continue
        lm, lk = metrics.game_log_loss(d[y], d[pm]), metrics.game_log_loss(d[y], d[pk])
        res = metrics.paired_bootstrap(lm, lk, d["fixture_id"], resamples=resamples, seed=seed)
        rows.append({"target": target, "n": len(d), "model_log_loss": float(np.mean(lm)),
                     "market_log_loss": float(np.mean(lk)), "model_brier": metrics.brier(d[y], d[pm]),
                     "market_brier": metrics.brier(d[y], d[pk]), "delta": res["mean"],
                     "ci95": f"[{res['ci95'][0]:+.4f}, {res['ci95'][1]:+.4f}]"})
    return pd.DataFrame(rows)


def benchmark_markdown(j: pd.DataFrame, scores: pd.DataFrame, unmatched: list[str], missing_names: list[str]) -> str:
    lines = [
        "# Model vs betting market", "",
        "Production BT (calibrated, trained on games before each match's UTC day) against Pinnacle's closing "
        "price with the margin removed (`data/manual/market_odds.csv`, from OddsPapi). Outcomes come from "
        "Oracle's Elixir. OddsPapi has no odds history before about June 2026, so this is a benchmark only; "
        "nothing is tuned on it. Δ = model − market log loss (negative = model better), 95% CI by bootstrap "
        "over matches. With this few matches the CI is wide.", "",
    ]
    if scores.empty:
        lines += ["No matched matches yet.", ""]
    else:
        lines += ["| target | matches | model log loss | market log loss | model Brier | market Brier | Δ | 95% CI |",
                  "|---|---|---|---|---|---|---|---|"]
        for s in scores.to_dict("records"):
            lines.append(f"| {s['target']} | {s['n']} | {s['model_log_loss']:.4f} | {s['market_log_loss']:.4f} | "
                         f"{s['model_brier']:.4f} | {s['market_brier']:.4f} | {s['delta']:+.4f} | {s['ci95']} |")
        lines.append("")
        by_event = j.groupby("event").agg(n=("fixture_id", "size"))
        lines += ["Matches per event: " + ", ".join(f"{e} {n}" for e, n in by_event["n"].items()) + ".", ""]
        lines += ["## Matches", "", "| date | team 1 | team 2 | Bo | model P(team 1) | market P(team 1) | result |",
                  "|---|---|---|---|---|---|---|"]
        pct = lambda x: "—" if pd.isna(x) else f"{100 * x:.1f}%"  # noqa: E731
        for r in j.sort_values("start").to_dict("records"):
            lines.append(f"| {r['start']:%Y-%m-%d} | {r['team_1']} | {r['team_2']} | {r['best_of']} | "
                         f"{pct(r['p_model_series'])} | {pct(r['p_mkt_series'])} | "
                         f"{'team 1' if r['y_series'] == 1 else 'team 2'} |")
        lines.append("")
    if unmatched or missing_names:
        lines += ["## Not matched", "",
                  f"- Fixtures without a matching Oracle's Elixir series: {len(unmatched)} "
                  "(not played yet, not in OE yet, or a team name below).",
                  f"- OddsPapi team names with no OE match (add to `oddspapi.team_aliases` in config/sources.yaml): "
                  f"{', '.join(missing_names) if missing_names else 'none'}.", ""]
    return "\n".join(lines)
