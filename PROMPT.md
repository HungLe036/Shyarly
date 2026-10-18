# Kickoff prompt for Claude Code

Paste everything below the line into a new Claude Code session opened in this folder.

---

You are building **lolpred**, a Python project that predicts **League of Legends Worlds 2026**. Today is 2026-10-18.
- The Play-In (Los Angeles) ends today.
- The **Swiss Stage (16 teams, Allen, TX) starts Oct 23**.
- The knockouts run Nov 3–14, with the final in Brooklyn on Nov 14.
- The MVP must be usable **before Oct 23**.

Read `docs/02-refined-plan.md` first; it is the spec. `docs/01-brainstorm.md` is background only.

## Goal
1. Calibrated per-game and per-series (Bo1/Bo3/Bo5) win probabilities for any two teams.
2. A Monte Carlo simulation of the Worlds 2026 Swiss Stage, giving qualify % and the 3-0…0-3 record distribution per team.
3. (v2) Daily refit and re-sim from real results, plus a knockout sim → champion %.

## Hard rules
- **No leakage.** A prediction for day D may use only games with `date_utc < D`. Write a test proving that changing future rows doesn't change past predictions.
- **Never invent facts.** That covers rosters, seeds, team names and format rules. Every tournament rule goes in `config/worlds2026.yaml` with a source URL comment.
  - Verify the rules against the official Riot / lolesports 2026 rulebook: Swiss Bo1 vs Bo3 per round, pairing and draw method, same-region avoidance, knockout seeding and draw.
  - Verify rosters and seeds against Leaguepedia or lolesports.
  - If something can't be verified, stop and ask me rather than guess.
- **Simplest model that passes the gates.**
  - The MVP model is a time-decayed, ridge-regularized Bradley-Terry logistic regression with region offsets and team deviations: `logit p = (r_regA + u_A) − (r_regB + u_B) + β_side`.
  - Region-offset Elo is a baseline only.
  - Calibration is temperature scaling, with no isotonic regression.
  - No GBM, draft model or dashboard until the MVP is done.
- **Reproducibility.** Use a pinned environment (`uv` + `pyproject.toml`), fixed RNG seeds, `pathlib` paths (this is Windows), and config values in YAML or dataclasses, not literals scattered through the code.
- **Test-driven** for the data schema, leakage, series math (closed-form Bo3/Bo5) and sim invariants:
  - 8 teams qualify and 8 are eliminated.
  - Every team ends at 3 wins or 3 losses.
  - No illegal rematches.
  - Qualify % sums to 800%.

## Data
- **Primary:** Oracle's Elixir yearly match CSVs (2022–2026). Check that 2026 is published.
- **Secondary:** the Leaguepedia Cargo API. Respect rate limits and cache all responses to `data/raw/`.
- **Fallback:** a hand-entered `data/manual/worlds2026_results.csv` for live Worlds results.
- The MVP needs only game results (date, teams, side, winner, league, event), not detailed stats.
- Keep `data/` gitignored except `data/manual/`.

## Evaluation and gates (see the spec for exact numbers)
- **Rolling-origin backtest:**
  - Tune on Worlds 2022, MSI 2023 and Worlds 2023.
  - Hold out MSI 2024, Worlds 2024, MSI 2025, Worlds 2025 and MSI 2026.
  - Score international games only.
- **Metrics:** log loss (primary), Brier, 5-bin ECE, calibration slope. Use a paired bootstrap for the CIs.
- **BT must:**
  - beat 50/50,
  - beat region-prior-only by ≥ 0.010 log loss,
  - not be credibly worse than Elo,
  - and reach ECE ≤ 0.05.
- Write the results to `reports/backtest.md`.

## Repo layout
```
pyproject.toml  README.md  .gitignore
config/worlds2026.yaml
data/{raw,interim,processed,manual}/
src/lolpred/{data/,models/,sim/,eval/,cli.py}
tests/
reports/
```

## CLI targets
```
lolpred ingest                         # fetch and cache data, build data/processed/games.parquet
lolpred backtest                       # writes reports/backtest.md
lolpred predict match "<Team A>" "<Team B>" --bo 3
lolpred simulate --stage swiss --runs 50000 [--as-of 2026-10-25]
lolpred update                         # v2: ingest new results → refit → re-sim
```

## Milestones (stop after each and report what's done, the test results and any open questions)
- **M0 (today):**
  - `git init`, scaffold the project, cache raw 2022–26 data.
  - Draft `config/worlds2026.yaml` with the 16 Swiss teams and the format rules, each with a cited source.
  - **Show me the YAML before continuing.**
- **M1:** ingest to `games.parquet` with schema and duplicate tests; report the international-game count.
- **M2:** baselines (50/50, region prior, Elo) and the BT model, with the backtest table.
- **M3:** calibration and series probabilities; gates evaluated in `reports/backtest.md`.
- **M4:** Swiss Monte Carlo and the CLI; invariant tests green; output a markdown table of team odds.
- **M5–M6 (v2):** live daily update, knockout sim, champion %, minimal Streamlit page.

Start by entering plan mode. Confirm which data sources are reachable and which rules you could verify, list any blockers, then do M0.
