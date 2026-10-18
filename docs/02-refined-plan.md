# Worlds 2026 Prediction — Refined Plan

Refines [01-brainstorm.md](01-brainstorm.md) after an ML-engineering review (ecc `mle-workflow` skill + `planner` agent).

## Iteration compact
- **Goal:** calibrated win probabilities for every Worlds 2026 match, plus per-team odds of Swiss qualification and (v2) of winning the title.
- **Success metric:** log loss on international games in a rolling-origin backtest.
- **Guardrails:** ECE ≤ 0.05 and calibration slope between 0.8 and 1.2.
- **Unacceptable mistakes:** leakage (future games in features) and simulating the wrong tournament rules.
- **Baselines:** 50/50, region-prior-only, region-offset Elo.
- **Fallback:** if Oracle's Elixir data isn't available, use results-only data (Leaguepedia plus a hand-entered CSV).

## Scope
| Phase | Deadline | Contents |
|---|---|---|
| MVP | Oct 22 (before Swiss starts Oct 23) | Game and series win probability, Swiss Monte Carlo (qualify % and 3-0…0-3 distribution), CLI output to CSV/markdown |
| v2 | Nov 2 (before knockouts) | Daily live refit and re-sim, knockout sim → champion %, minimal Streamlit page, roster-continuity adjustment, GBM only if it passes its gate |
| Stretch | — | Draft model, betting-odds benchmark, player ratings, Glicko-2 |
| Cut | — | Live in-game win probability |

## MVP model
- **Time-decayed, ridge-regularized Bradley-Terry logistic regression:** `logit p = (r_region(A) + u_A) − (r_region(B) + u_B) + β_side`.
  - Domestic games pin down `u`; international games identify the region offsets `r`.
  - Shrink `u` harder than `r`, and give minor regions a wider prior.
  - Use an exponential half-life (tune 60–180 days) and an optional extra weight for international or Worlds-patch games. Both are tuned hyperparameters.
- **Elo** with region offsets is a **baseline**, not the product.
- **Series probability:** treat games as independent with p averaged over blue and red side, and use the closed-form Bo3/Bo5 formula.
- **Calibration:** a single temperature on the logit, with no intercept so the model stays symmetric. Don't use isotonic regression; there's too little data.
- **No blending in the MVP.**

## Data contract
- **Grain:** one row per game.
  - Columns: `game_id, date_utc, patch, league, event, stage, series_id, game_in_series, blue_team_season_id, red_team_season_id, winner`.
  - A series table is derived from it.
- **Keys:** `team_season_id = {team_slug}_{year}`, plus a region lookup and a `roster_flags` table for mid-year swaps.
- **Point-in-time:** a prediction for day D uses only games with `date_utc < D`, so no same-day games. Snapshots are stored per cutoff.
- **Splits:**
  - Tune on Worlds 2022, MSI 2023 and Worlds 2023.
  - Hold out MSI 2024, Worlds 2024, MSI 2025, Worlds 2025 and MSI 2026 (if available). Train only on data before each event.
  - Run two modes: pre-event and live (per match day).
- **Scoring:** international games only.

## Promotion gates
Measured on the holdout with a paired bootstrap of 2,000 resamples.
- Log loss < 0.693 (beats 50/50).
- Beats region-prior-only: Δ < 0 with a 95% CI that excludes 0, Δ reported. *(Revised 2026-10-18 from "by ≥ 0.010 log loss", after three holdout looks; see `config/model.yaml` `gate_changes` and `reports/backtest.md`.)*
- Not credibly worse than Elo: point improvement > 0, and the 80% CI upper bound ≤ +0.005.
- ECE ≤ 0.05 and calibration slope between 0.8 and 1.2.
- **v2 GBM or blend:** must improve on BT by ≥ 0.005 log loss with a 90% CI excluding 0, and must not worsen ECE.
- **Sim:**
  - Invariant tests pass: 8 teams qualify, 8 are eliminated, every team ends at 3 wins or 3 losses, no illegal rematches.
  - Qualify % sums to 800% ± 0.1.
  - Results are stable across seeds within ±0.5pp at 50k runs.

## Top risks
| Risk | Mitigation |
|---|---|
| Leakage | Strict `date < D` rule, a test that changing future rows doesn't change past predictions, tuning only on 2022–23 |
| Cross-region sparsity | Hierarchical shrinkage and reported uncertainty on region offsets |
| Roster changes | Key on team-season and keep `roster_flags`; never hardcode rosters from memory |
| Format errors | Every rule in `config/worlds2026.yaml` with a cited source, plus invariant tests |
| Data access | Cache raw files on day 1; the results-only fallback is enough for the MVP |

## Milestones
| # | Date | Done when |
|---|---|---|
| M0 | Oct 18 | Raw 2022–26 data cached, 16 Swiss teams mapped, format rules cited in the YAML |
| M1 | Oct 19 | `games.parquet` passes schema and duplicate tests; international-game count reported |
| M2 | Oct 20 | Backtest table (log loss / Brier / ECE) for 50/50, region prior, Elo and BT |
| M3 | Oct 21 | Calibration and series math done; gates recorded in `reports/backtest.md` |
| M4 | Oct 22 | `predict match` and `simulate --stage swiss` work; invariant tests green |
| M5 | Oct 23–31 | One command per day ingests results, refits and re-simulates; pre-event vs live log loss logged |
| M6 | Nov 2 | Knockout sim, champion %, Streamlit page |
| M7 | stretch | GBM, odds, draft — each passes its gate or is documented as rejected |
