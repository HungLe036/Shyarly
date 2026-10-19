"""Run the Worlds 2026 Swiss simulation from config + the production model, and format the odds.

While the Play-In winner is pending, runs are split between the candidates in proportion
to the model's Bo5 win probability for the Play-In final, so every row is unconditional.

With `posterior=True`, each run draws the 16 teams' strengths (and the side term) from the
BT fit's Laplace posterior, so the uncertainty in strengths is carried into the odds:
a team that is probably strong but poorly measured is sometimes weak for a whole run.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from shyarly.config import LeagueMap, WorldsConfig
from shyarly.data.manual import KO_BEST_OF, KnockoutFixed, MatchRow, SwissFixed, knockout_fixed, swiss_fixed
from shyarly.models.predictor import Predictor, worlds_team_ref
from shyarly.sim.knockout import QF_LABELS, draw_bracket, knockout_once
from shyarly.sim.swiss import RECORDS, SwissRules, SwissSimulator


KO_COLUMNS = ["semis", "final", "champion"]


@dataclass
class SwissOdds:
    table: pd.DataFrame
    runs: int
    seed: int
    play_in: dict[str, float]           # candidate -> share of runs (empty when decided)
    round1_fixed: bool
    waived_rematches: int
    posterior: bool = False
    knockout: bool = False              # table also has semis / final / champion
    results_fixed: int = 0              # played matches taken from the results file
    rounds_complete: int = 0            # Swiss rounds fully played
    bracket_fixed: bool = False


def simulate_worlds_swiss(worlds: WorldsConfig, leagues: LeagueMap, predictor: Predictor,
                          runs: int, seed: int, posterior: bool = False,
                          results: list[MatchRow] | None = None) -> SwissOdds:
    return simulate_worlds(worlds, leagues, predictor, runs, seed, posterior=posterior, results=results)


def simulate_worlds(worlds: WorldsConfig, leagues: LeagueMap, predictor: Predictor, runs: int, seed: int,
                    posterior: bool = False, results: list[MatchRow] | None = None,
                    knockout: bool = False) -> SwissOdds:
    """Swiss Stage (and with `knockout=True` the Knockout Stage) conditional on known results.
    `results` are rows from data/manual/worlds2026_results.csv; played matches are fixed and
    drawn-but-unplayed pairings are used as the draw."""
    results = list(results or [])
    year = predictor.as_of.year
    rules = SwissRules.from_config(worlds.swiss)
    fixed = [t for t in worlds.teams if t.status != "pending"]
    pending = [t for t in worlds.teams if t.status == "pending"]

    if pending:
        if worlds.swiss["round1"].get("pairings") or results:
            raise ValueError("Round 1 pairings or results are set but the Play-In winner is still pending in the config")
        a, b = pending[0].candidates
        p_a = predictor.p_series(worlds_team_ref(a, worlds, leagues, year),
                                 worlds_team_ref(b, worlds, leagues, year), worlds.raw["play_in"]["best_of"])
        scenarios = [(worlds.candidate(a), p_a), (worlds.candidate(b), 1 - p_a)]
    else:
        scenarios = [(None, 1.0)]

    counts = np.random.default_rng(seed).multinomial(runs, [p for _, p in scenarios])
    acc: dict[str, dict] = {}
    waived = 0
    fx, kf = SwissFixed(), KnockoutFixed()
    for k, ((extra, _), n) in enumerate(zip(scenarios, counts)):
        teams = fixed + ([extra] if extra else [])
        codes = [t.code for t in teams]
        refs = [worlds_team_ref(c, worlds, leagues, year) for c in codes]
        try:
            fx = swiss_fixed(results, worlds, codes)
            kf = knockout_fixed(results, fx, codes)
        except KeyError as e:
            raise ValueError(f"Unknown team {e}; use team codes from config/worlds2026.yaml") from e
        sim = SwissSimulator(codes, [r.region for r in refs], [t.tier for t in teams],
                             predictor.game_matrix(refs), rules, fixed_pairings=fx.pairings,
                             fixed_results=fx.results, extra_best_of=(KO_BEST_OF,) if knockout else ())
        if n == 0:
            continue
        sampler = None
        if posterior:
            draw_rng = np.random.default_rng([seed, k, 1])
            sampler = lambda m, refs=refs, draw_rng=draw_rng: predictor.game_matrix_draws(refs, m, draw_rng)  # noqa: E731
        ko_counts = np.zeros((len(teams), len(KO_COLUMNS)), dtype=np.int64)
        on_run = None
        if knockout:
            def on_run(log, series_p, rng, kf=kf, ko_counts=ko_counts):
                bracket = kf.bracket or draw_bracket(log.final, rng)
                w = knockout_once(bracket, series_p[KO_BEST_OF], rng, kf.results)
                for label in QF_LABELS:
                    ko_counts[w[label], 0] += 1
                for label in ("SF1", "SF2"):
                    ko_counts[w[label], 1] += 1
                ko_counts[w["F"], 2] += 1
        res = sim.simulate(int(n), seed=seed + k + 1, sampler=sampler, on_run=on_run)
        waived += res.waived_rematches
        for i, t in enumerate(teams):
            row = acc.setdefault(t.code, {
                "team": t.code, "name": t.name, "region": refs[i].region,
                "league_seed": f"{t.league}{t.seed}", "tier": t.tier,
                "in_swiss": 0, "qualify": 0, **{r: 0 for r in RECORDS},
                **({c: 0 for c in KO_COLUMNS} if knockout else {}),
            })
            row["in_swiss"] += int(n)
            row["qualify"] += int(res.qualify[i])
            for j, r in enumerate(RECORDS):
                row[r] += int(res.records[i, j])
            if knockout:
                for j, c in enumerate(KO_COLUMNS):
                    row[c] += int(ko_counts[i, j])

    table = pd.DataFrame(acc.values())
    for col in ["in_swiss", "qualify", *RECORDS, *(KO_COLUMNS if knockout else [])]:
        table[col] = table[col] / runs
    sort = ["champion", "qualify"] if knockout else ["qualify", "3-0"]
    table = table.sort_values(sort, ascending=False).reset_index(drop=True)
    play_in = {s.code: float(p) for s, p in scenarios if s is not None}
    played = sum(r.winner is not None for r in results)
    return SwissOdds(table, runs, seed, play_in, 1 in fx.pairings, waived, posterior, knockout,
                     played, fx.complete_rounds, kf.bracket is not None)


def odds_markdown(odds: SwissOdds, as_of: pd.Timestamp, temperature: float, last_game: pd.Timestamp) -> str:
    t = odds.table
    pct = lambda x: f"{100 * x:.1f}%"  # noqa: E731
    lines = [
        "# Worlds 2026 Swiss Stage odds",
        "",
        f"- **Model:** BT (production params, temperature {temperature:.4f}), trained on games before "
        f"{as_of:%Y-%m-%d} UTC (latest game {last_game:%Y-%m-%d %H:%M} UTC).",
        f"- **Simulation:** {odds.runs:,} runs, seed {odds.seed}. 95% Monte Carlo margin on each "
        f"qualify % ≤ ±{196 * np.sqrt(0.25 / odds.runs):.2f}pp. Rematch rule waived in {odds.waived_rematches} runs.",
        "- **Strength uncertainty:** " + (
            "each run draws team strengths from the BT fit's Laplace posterior (one draw per run)."
            if odds.posterior else "off; every run uses the point estimate of each team's strength."),
    ]
    if odds.play_in:
        split = ", ".join(f"{c} {pct(p)}" for c, p in odds.play_in.items())
        lines.append(f"- **Play-In winner pending:** runs split by the model's Bo5 odds for the final ({split}). "
                     "`in_swiss` is each candidate's chance of being in the field; their other columns include it.")
    lines.append("- **Round 1 draw:** " + ("official pairings from config." if odds.round1_fixed
                                          else "not published yet; simulated (pools 1v4 and 2v3, no same-region)."))
    lines.append(_results_line(odds))
    lines += ["", "| team | region | seed | in Swiss | qualify | 3-0 | 3-1 | 3-2 | 2-3 | 1-3 | 0-3 |",
              "|---|---|---|---|---|---|---|---|---|---|---|"]
    for _, r in t.iterrows():
        lines.append(f"| {r['team']} | {r['region']} | {r['league_seed']} | {pct(r['in_swiss'])} | **{pct(r['qualify'])}** | "
                     + " | ".join(pct(r[c]) for c in RECORDS) + " |")
    lines += ["", f"Qualify column sums to {100 * t['qualify'].sum():.1f}% (8 of 16 teams advance).", ""]
    return "\n".join(lines)


def _results_line(odds: SwissOdds) -> str:
    if not odds.results_fixed:
        return "- **Results:** none played yet (data/manual/worlds2026_results.csv)."
    return (f"- **Results:** {odds.results_fixed} played matches fixed from data/manual/worlds2026_results.csv; "
            f"Swiss rounds complete: {odds.rounds_complete} of 5.")


def tournament_markdown(odds: SwissOdds, as_of: pd.Timestamp, temperature: float, last_game: pd.Timestamp) -> str:
    """Champion odds: every run plays the Swiss Stage, then the knockout bracket (RULESET 4.3.4)."""
    if not odds.knockout:
        raise ValueError("tournament_markdown needs a simulation with knockout=True")
    t = odds.table
    pct = lambda x: f"{100 * x:.1f}%"  # noqa: E731
    lines = [
        "# Worlds 2026 title odds",
        "",
        f"- **Model:** BT (production params, temperature {temperature:.4f}), trained on games before "
        f"{as_of:%Y-%m-%d} UTC (latest game {last_game:%Y-%m-%d %H:%M} UTC).",
        f"- **Simulation:** {odds.runs:,} runs, seed {odds.seed}; each run plays the Swiss Stage and then the "
        "Knockout Stage with the same strength draw. 95% Monte Carlo margin ≤ "
        f"±{196 * np.sqrt(0.25 / odds.runs):.2f}pp.",
        "- **Strength uncertainty:** " + ("posterior draw per run." if odds.posterior else "off (point estimates)."),
        "- **Bracket:** " + ("official quarterfinal draw from the results file." if odds.bracket_fixed else
                             "simulated: each 3-0 team meets a random 3-2 team on opposite halves, the other four "
                             "are paired at random (RULESET 4.3.4.1–2); all matches Bo5."),
        _results_line(odds),
        "",
        "| team | region | seed | quarterfinals | semifinals | final | champion |",
        "|---|---|---|---|---|---|---|",
    ]
    for _, r in t.iterrows():
        lines.append(f"| {r['team']} | {r['region']} | {r['league_seed']} | {pct(r['qualify'])} | {pct(r['semis'])} | "
                     f"{pct(r['final'])} | **{pct(r['champion'])}** |")
    lines += ["", f"Columns sum to {100 * t['qualify'].sum():.0f}% / {100 * t['semis'].sum():.0f}% / "
              f"{100 * t['final'].sum():.0f}% / {100 * t['champion'].sum():.0f}% (8 / 4 / 2 / 1 teams).", ""]
    return "\n".join(lines)
