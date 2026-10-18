# Worlds 2026 Prediction — Brainstorm

Date: 2026-10-18 (Play-In ends today; Swiss Stage starts Oct 23 in Allen, TX; Final Nov 14, Brooklyn).

## What could we predict?
| Target | Value | Difficulty |
|---|---|---|
| A. Single-game win probability (Bo1) | Core building block | Medium |
| B. Series win probability (Bo3/Bo5) | Derived from A | Low once A exists |
| C. Swiss record distribution (3-0 … 0-3), qualify % | Pick'Ems, fun | Needs bracket sim |
| D. Champion / finalist / semifinalist % per team | Headline output | Needs full sim |
| E. In-draft win prob (given picks/bans) | Cool, live use | High, little data |
| F. Live in-game win prob (gold@15 etc.) | Cool | Needs live feed; out of scope |

## Core challenge
International play is rare: inter-region games mostly come from MSI, EWC, Worlds history.
Regional leagues are largely closed graphs, so a domestic rating isn't comparable across regions
without a **region-strength** adjustment learned from cross-region results.

## Candidate approaches
1. **Elo / Glicko-2 per team** with region offsets, recency decay, patch/split resets toward a regional mean.
2. **Bradley-Terry / logistic** with team strength + region strength, fit with time-weighting (Bayesian or ridge-regularized logistic).
3. **Feature model (GBM / logistic)** on rolling team stats from Oracle's Elixir:
   GD@15, XPD@15, first-dragon/herald/tower rates, dragon control, baron rate,
   side (blue/red), rating diff, roster continuity, days since last game, patch.
4. **Player-level ratings** aggregated to team (handles roster swaps mid-year).
5. **Market prior**: betting odds as benchmark — strongest public baseline.
6. **Draft model**: champion meta strength on the Worlds patch — likely v2.

Ensemble = calibrated blend of (1/2) rating model + (3) feature model.

## Tournament simulation
Monte Carlo (e.g. 50k runs) of the real format:
- Swiss: 16 teams, Bo1 early rounds, Bo3 for qualification/elimination matches (verify official rules), pairing rules incl. same-region avoidance.
- Knockout: 8-team single elimination Bo5 with official seeding/draw rules.
- Re-run after every match day using actual results ("live" mode).

## Evaluation
- Rolling-origin backtest on Worlds 2022–2025 and MSI 2023–2026 (train only on data before each event).
- Metrics: log loss, Brier, calibration curve, accuracy (secondary). Baselines: 50/50, region-prior, Elo-only, betting odds where available.

## Deliverables
- CLI: `predict match T1 GEN --bo 5`, `simulate --stage swiss`.
- Streamlit dashboard: team odds table, bracket heatmap, calibration plot.
- Daily update: ingest results → refit → resimulate.

## Open questions
- Data source availability/licensing (Oracle's Elixir CSV, Leaguepedia Cargo API, lolesports unofficial API, gol.gg).
- Exact 2026 Swiss pairing and seeding rules.
- How much to weight MSI/EWC 2026 vs domestic summer.
- Worlds is on a fixed patch — weight games on/near it?
