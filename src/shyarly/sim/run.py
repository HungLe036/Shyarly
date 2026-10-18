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
from shyarly.models.predictor import Predictor, worlds_team_ref
from shyarly.sim.swiss import RECORDS, SwissRules, SwissSimulator


@dataclass
class SwissOdds:
    table: pd.DataFrame
    runs: int
    seed: int
    play_in: dict[str, float]           # candidate -> share of runs (empty when decided)
    round1_fixed: bool
    waived_rematches: int
    posterior: bool = False


def _round1_pairings(worlds: WorldsConfig, codes: list[str]) -> dict[int, list[tuple[int, int]]] | None:
    pairs = worlds.swiss["round1"].get("pairings") or []
    if not pairs:
        return None
    idx = {c: i for i, c in enumerate(codes)}
    try:
        return {1: [(idx[a], idx[b]) for a, b in pairs]}
    except KeyError as e:
        raise ValueError(f"Round 1 pairing names unknown team {e}; use team codes from config/worlds2026.yaml") from e


def simulate_worlds_swiss(worlds: WorldsConfig, leagues: LeagueMap, predictor: Predictor,
                          runs: int, seed: int, posterior: bool = False) -> SwissOdds:
    year = predictor.as_of.year
    rules = SwissRules.from_config(worlds.swiss)
    fixed = [t for t in worlds.teams if t.status != "pending"]
    pending = [t for t in worlds.teams if t.status == "pending"]

    if pending:
        if worlds.swiss["round1"].get("pairings"):
            raise ValueError("Round 1 pairings are set but the Play-In winner is still pending in the config")
        a, b = pending[0].candidates
        p_a = predictor.p_series(worlds_team_ref(a, worlds, leagues, year),
                                 worlds_team_ref(b, worlds, leagues, year), worlds.raw["play_in"]["best_of"])
        scenarios = [(worlds.candidate(a), p_a), (worlds.candidate(b), 1 - p_a)]
    else:
        scenarios = [(None, 1.0)]

    counts = np.random.default_rng(seed).multinomial(runs, [p for _, p in scenarios])
    acc: dict[str, dict] = {}
    waived = 0
    round1_fixed = False
    for k, ((extra, _), n) in enumerate(zip(scenarios, counts)):
        teams = fixed + ([extra] if extra else [])
        codes = [t.code for t in teams]
        refs = [worlds_team_ref(c, worlds, leagues, year) for c in codes]
        pairings = _round1_pairings(worlds, codes)
        round1_fixed = pairings is not None
        sim = SwissSimulator(codes, [r.region for r in refs], [t.tier for t in teams],
                             predictor.game_matrix(refs), rules, fixed_pairings=pairings)
        if n == 0:
            continue
        sampler = None
        if posterior:
            draw_rng = np.random.default_rng([seed, k, 1])
            sampler = lambda m, refs=refs, draw_rng=draw_rng: predictor.game_matrix_draws(refs, m, draw_rng)  # noqa: E731
        res = sim.simulate(int(n), seed=seed + k + 1, sampler=sampler)
        waived += res.waived_rematches
        for i, t in enumerate(teams):
            row = acc.setdefault(t.code, {
                "team": t.code, "name": t.name, "region": refs[i].region,
                "league_seed": f"{t.league}{t.seed}", "tier": t.tier,
                "in_swiss": 0, "qualify": 0, **{r: 0 for r in RECORDS},
            })
            row["in_swiss"] += int(n)
            row["qualify"] += int(res.qualify[i])
            for j, r in enumerate(RECORDS):
                row[r] += int(res.records[i, j])

    table = pd.DataFrame(acc.values())
    for col in ["in_swiss", "qualify", *RECORDS]:
        table[col] = table[col] / runs
    table = table.sort_values(["qualify", "3-0"], ascending=False).reset_index(drop=True)
    play_in = {s.code: float(p) for s, p in scenarios if s is not None}
    return SwissOdds(table, runs, seed, play_in, round1_fixed, waived, posterior)


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
    lines += ["", "| team | region | seed | in Swiss | qualify | 3-0 | 3-1 | 3-2 | 2-3 | 1-3 | 0-3 |",
              "|---|---|---|---|---|---|---|---|---|---|---|"]
    for _, r in t.iterrows():
        lines.append(f"| {r['team']} | {r['region']} | {r['league_seed']} | {pct(r['in_swiss'])} | **{pct(r['qualify'])}** | "
                     + " | ".join(pct(r[c]) for c in RECORDS) + " |")
    lines += ["", f"Qualify column sums to {100 * t['qualify'].sum():.1f}% (8 of 16 teams advance).", ""]
    return "\n".join(lines)
