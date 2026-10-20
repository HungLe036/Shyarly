# Shyarly

Calibrated win probabilities and a Monte Carlo simulation of the **League of Legends World
Championship 2026**: the Swiss Stage (Allen, TX, Oct 23–31; 16 teams, 8 advance) and the
Knockout Stage through the final (Brooklyn, Nov 14).

- Spec: [docs/02-refined-plan.md](docs/02-refined-plan.md). Background: [docs/01-brainstorm.md](docs/01-brainstorm.md).
- Tournament facts (teams, seeds, format rules), each with a cited source: [config/worlds2026.yaml](config/worlds2026.yaml).
- Backtest and promotion gates: [reports/backtest.md](reports/backtest.md).
- Current Swiss odds: [reports/swiss_odds.md](reports/swiss_odds.md) ([CSV](reports/swiss_odds.csv)).
- Current title odds: [reports/tournament_odds.md](reports/tournament_odds.md) ([CSV](reports/tournament_odds.csv)).
- Live scoring of logged predictions: [reports/live_scores.md](reports/live_scores.md).

## Status (2026-10-19)

| Milestone | State | Commit |
|---|---|---|
| M0 — scaffold, raw data cache, cited Worlds config | done | `4ea689d` |
| M1 — `games.parquet` with schema, duplicate and point-in-time tests | done | `6420fa5` |
| M2 — baselines (50/50, region prior, Elo) and Bradley-Terry, backtest | done | `38a6cba`, `e2fbbcb` |
| M3 — temperature calibration, series probabilities, gates | done | `2c434f7`, `4678d8d` |
| M4 — Swiss Monte Carlo, `predict` and `simulate` CLI | done | `d33deb3` |
| M5 — daily update from live results (`shyarly update`), prediction log | done | `01f19f4` |
| M6 — knockout sim, champion %, Streamlit page | done | `01f19f4`, `b4bfc30` |
| Shadow model `margin_bt` (gold difference at 15:00), pre-registered | logging; decision Nov 1 | `217b75a`, this commit |

160 tests pass (`uv run pytest`), plus a slow 50k-run seed-stability test (`uv run pytest -m slow`).

**Field and Round 1 draw are final (2026-10-19):** Karmine Corp won the Play-In (3-0 over MVK) and
the official Round 1 pairings are in `config/worlds2026.yaml`. The model data runs through the
Play-In final on Oct 18.

## Quick start (Windows)

```powershell
winget install --id astral-sh.uv -e
uv sync
uv run shyarly ingest                 # ~400 MB of Oracle's Elixir CSVs, then builds games.parquet
uv run shyarly backtest               # ~2 min; writes reports/backtest.md
uv run shyarly predict match GEN T1 --bo 5
uv run shyarly simulate --stage swiss --runs 50000
```

## Commands

```
shyarly ingest [--refresh]                                        # fetch/cache data, build data/processed/*.parquet
shyarly backtest                                                  # tune, score holdout, write reports/backtest.md
shyarly predict match "<Team A>" "<Team B>" --bo 3 [--as-of DATE] # one game and best-of-N series
shyarly simulate --stage swiss|tournament [--runs N] [--as-of DATE] [--seed N] [--posterior|--point] [--no-results]
shyarly update [--no-fetch]                                       # daily during Worlds (see below)
shyarly shadow                                                    # pre-registered shadow model run -> reports/shadow_margin.md
shyarly odds --tournament 2549 [--refresh]                        # closing betting odds (OddsPapi) -> data/manual/market_odds.csv
shyarly benchmark                                                 # model vs market -> reports/market_benchmark.md
shyarly forecast                                                  # current odds for drawn, unplayed matches -> reports/match_forecasts.csv (also run by update)
```

You can name a team by its Worlds code (`GEN`), its Worlds name, or its Oracle's Elixir name.
`--as-of` uses only games dated before that UTC day. Example output:

```
GEN (LCK) vs T1 (LCK), games before 2026-10-18 UTC, T=1.0502
  one game:    52.5%   (on blue 56.9%, on red 48.2%)
  Bo5 series:  54.7%
```

## What was built

### Data (M0–M1)

- **Source:** the [Oracle's Elixir](https://oracleselixir.com/tools/downloads) yearly match CSVs for
  2022–2026. They now live on Google Drive; the old S3 links return 404. Downloads are cached in
  `data/raw/oe/` with a sha256 manifest.
- **Leaguepedia:** the Cargo API client throttles requests and caches every response, but it isn't
  needed for the MVP (anonymous access was rate-limited).
- **`data/processed/games.parquet`:** one row per game, 53,585 games from 2022-01 to 2026-10-17.
  - UTC dates; OE timestamps were checked against the Play-In start time.
  - Team-season ids (`{slug}_{year}`) and series ids.
  - Each side's region as of that game. For an international game, a team takes its home region
    from its earlier domestic games, so appending future rows never changes past rows.
- **[config/leagues.yaml](config/leagues.yaml):** maps the home leagues that matter to regions. These
  were derived from which teams actually play cross-region events, and checked against Leaguepedia
  (LTA 2025, LCP 2025).
- **International games:** OE files regional qualifiers under the `WLDs` code. An event's main phase
  is therefore taken to start at its first cross-region game. That gives 834 scored Worlds/MSI
  main-event games.
- **Dropped:** 2 games with no winner, and 4 games that OE lists twice under different ids.

### Models (M2–M3)

- **MVP model:** a time-decayed, ridge-regularized Bradley-Terry logistic regression
  ([src/shyarly/models/bt.py](src/shyarly/models/bt.py)):
  `logit p = (r[region_A] + u[team_A]) − (r[region_B] + u[team_B]) + β_side`.
  - Domestic games decay with a 60-day half-life. Cross-region games (the only ones that move the
    region offsets) decay with a 730-day half-life.
  - Team deviations are shrunk hard (λ = 16). Minor regions get a wider prior.
- **Baselines:** 50/50, region-prior-only (the same model without team terms), and region-offset Elo.
- **Calibration:** one temperature on the logit, with no intercept. In the backtest, each event's
  temperature is fitted only on earlier events. Production uses T = 1.0502, fitted on all eight
  backtest events.
- **Series:** the side-averaged game probability goes through the closed-form Bo3
  `p²(3−2p)` and Bo5 `p³(10−15p+6p²)` formulas.

### Backtest and gates

- **Tune events:** Worlds 2022, MSI 2023, Worlds 2023.
- **Holdout events:** MSI 2024, Worlds 2024, MSI 2025, Worlds 2025, MSI 2026.
- **Modes:** "live" refits before every UTC match day. "Pre-event" freezes the model at the start of
  each event.
- **Holdout results, live mode, calibrated, 419 games:**

| model | log loss | Brier | ECE | slope |
|---|---|---|---|---|
| 50/50 | 0.6931 | 0.2500 | 0.066 | — |
| region prior | 0.6246 | 0.2173 | 0.027 | 0.85 |
| Elo | 0.6440 | 0.2257 | 0.050 | 0.66 |
| **BT** | **0.6199** | **0.2152** | 0.044 | 0.89 |

BT passes all five promotion gates:
- beats 50/50;
- beats the region prior: Δ −0.0046, 95% CI [−0.0091, −0.0001];
- is not worse than Elo: Δ −0.0240;
- ECE 0.044;
- calibration slope 0.886.

For series, the Bo-formula beats using the raw game probability: holdout Bo5 log loss 0.519 vs 0.544.

### Swiss simulation (M4)

[src/shyarly/sim/swiss.py](src/shyarly/sim/swiss.py) follows the official 2026 ruleset:
- **Round 1:** pools 1v4 and 2v3, with no same-region pairings.
- **Rounds 2–5:** random pairing within the same record, with no rematches.
- **Match length:** Bo1, except Bo3 when a match decides qualification or elimination.
- **Finish:** a team stops at 3 wins or 3 losses.
- **Round 1:** uses the official pairings when they are in the config, otherwise simulates the draw.
- **Strength uncertainty:** each run draws the 16 teams' strengths from the BT fit's Laplace
  posterior (a normal around the fitted strengths, with covariance from the inverse Hessian of the
  ridge objective). A poorly measured team is sometimes weak, sometimes strong for a whole run, so
  favourites lose about 1pp of qualify odds and longshots gain 1–1.6pp. Averaging over the posterior
  moves single-game probabilities by only 0.3pp on average, so it barely overlaps with the
  temperature calibration. `--point` turns it off.
- **Play-In slot:** while the winner is pending, runs are split between the two finalists by the
  model's Bo5 odds for the final.
- **Tests:** invariants are checked on every simulated run.
  - Exactly 8 teams advance and 8 are eliminated.
  - Every team ends on 3 wins or 3 losses.
  - Final records always split 2/3/3/3/3/2 across 3-0…0-3.
  - Paired teams share a record, and the best-of matches the record.
  - There are no rematches.
  - Qualify % sums to 800%.
- **Speed:** 50,000 runs take about 10 seconds.

Current qualify odds (50k runs with posterior draws, official Round 1 draw, games through Oct 18):
Gen.G 86%, T1 85%, HLE 84%, DK 83%, BLG 72%, AL 61%, IG 57%, TES 54%, G2 45%, KC 44%, MKOI 37%,
TLAW 32%, LYON 29%, TSW 12%, LØS 10%, CFO 10%. The full table is in [reports/swiss_odds.md](reports/swiss_odds.md).

Title odds (same runs, plus the knockout bracket): Gen.G 23%, HLE 20%, T1 18%, DK 18%, BLG 8%,
AL 4%, IG 3%, TES 2.5%, G2 1.2%, KC 1.0%, everyone else under 1%. The four LCK teams together
are about 79%. Full table: [reports/tournament_odds.md](reports/tournament_odds.md).

## Decisions and deviations from the spec

All of these are recorded in the configs and in [reports/backtest.md](reports/backtest.md).

1. **Region-prior gate revised.** The spec required BT to beat the region prior by ≥ 0.010 log loss.
   An ML review found that team-level information adds about 0.005 at best at international events,
   including on 308 cross-region games outside the holdout. The gate is now "Δ < 0 with the 95% CI
   excluding 0, Δ reported" (user decision, 2026-10-18).
2. **Holdout looked at three times.** Each look is listed in `config/model.yaml` (`holdout_looks`),
   so the holdout scores are somewhat optimistic. A grid widening that made the holdout worse was
   discarded.
3. **Separate half-life for cross-region games.** The spec had one half-life (60–180 days). One
   shared value forced a trade-off between team form and region strength.
4. **Seed-stability gate.** "Within ±0.5pp at 50k runs" is read as "each qualify % has a 95% Monte
   Carlo margin within ±0.5pp" (it is ±0.44pp). The largest difference between two seeds across 16
   teams is itself noisy: about 0.5–0.85pp at 50k runs.
5. **Draw procedure.** The ruleset says a conflicting team "moves to the next available slot" but
   does not define the slot order. Round 1 picks uniformly among valid pairings, and later rounds
   use randomized backtracking.
6. **What the model can't see:** rosters, drafts, patch 26.20 and Play-In form beyond the games in
   the data. Same-region teams come out close together because the model leans on region strength.
7. **Roster changes (checked, no adjustment).** From Oracle's Elixir player rows, five teams changed
   lineups in 2026 (AL, LØS, CFO, TES, IG; `roster_changes` in `config/worlds2026.yaml`). The
   60-day half-life already puts at least 91% of each team's weight on its current lineup, except
   IG (76%; current five since Aug 16), whose odds deserve extra caution.
8. **Shadow model `margin_bt`, pre-registered before Swiss.** BT plus the gold difference at
   15:00 as extra evidence. Oracle's Elixir has no gold data for the LPL in 2022–2025 (about 80%
   in 2026) or for MSI 2024, so the gold term only adds information where it exists; it never
   replaces a win/loss. The rules (`shadow_models.margin_bt` in `config/model.yaml`, committed in
   `217b75a` before running) say it replaces production only if it beats it on the tune events
   *and* on Worlds 2026 Swiss games. Result so far ([reports/shadow_margin.md](reports/shadow_margin.md)):
   tune Δ −0.0024 (95% CI [−0.019, +0.015]), so condition 1 passes, barely; holdout (report only,
   look 4) Δ −0.0015. Its predictions are logged next to production's from Oct 23; decision Nov 1.
9. **Betting market as a benchmark, not an input.** Closing Pinnacle odds come from OddsPapi
   (free plan, 250 billable calls a month; the key lives in a gitignored `.env`). Its odds
   history only starts around June 2026: Worlds 2025 and MSI 2025 return none, so the market
   cannot be backtested or blended in. It is compared with the model on what exists (MSI 2026,
   the 2026 Play-In, and Worlds 2026 as it is played) by `shyarly benchmark`, which writes
   `reports/market_benchmark.md` locally.
   First read, 18 series: model 0.745 vs market 0.712 series log loss (Δ +0.033, 95% CI
   [−0.084, +0.148]), so no detectable difference yet. Raw responses stay local in
   `data/raw/odds/`. The compact `data/manual/market_odds.csv` and the report are gitignored,
   because OddsPapi's terms may restrict redistribution; only the summary numbers here are published.
10. **Longer market history from a user-supplied OddsPortal scrape** (local only, in
    `data/raw/oddsportal/`; OddsPortal's terms restrict scraping and redistribution, so only the summary
    numbers below are published). Average closing odds over ~6 mostly soft bookmakers, no Pinnacle. Only
    clean rows are used: rows without per-bookmaker odds (which included every row of an earlier
    scraper bug that copied one match's odds to rematches) and duplicates are dropped. Team names
    are matched with aliases and season-specific past names (`oddsportal` in `config/sources.yaml`),
    each derived from the data with its vote count. On 442 international series, 2022–2026:
    model 0.508 vs market 0.520 series log loss (Δ −0.012, 95% CI [−0.039, +0.018]). On the
    tune events the model looks better (Δ −0.043), but its settings were chosen there; on the
    holdout the two are level (Δ +0.005, CI [−0.043, +0.054]). Recorded as holdout look 6.

## During Worlds: the daily routine

1. **Enter results and new draws** in [data/manual/worlds2026_results.csv](data/manual/worlds2026_results.csv),
   one row per match, with a source:
   - after a match: fill `winner` and `score` (games won by `team_a`-`team_b`, e.g. `2-1`);
   - after each round's draw: add the next round's pairings with `date_utc` and an empty winner;
   - knockouts: rounds `QF1`–`QF4`, `SF1`, `SF2`, `F`. QF1/QF2 feed SF1 and QF3/QF4 feed SF2, so
     enter the quarterfinals in bracket order.

   The file is checked against the rules when it is read: Round 1 must match the official draw,
   later pairs must share a record, a round needs the previous one finished, scores must fit
   the best-of, and the two 3-0 vs 3-2 quarterfinals must be on opposite halves.
2. **Run `uv run shyarly update`.** It:
   - re-downloads this year's Oracle's Elixir data (`--no-fetch` to skip) and refits the model;
   - re-simulates the whole tournament, conditional on every played match and published draw,
     and writes `reports/swiss_odds.*`, `reports/tournament_odds.*` and a dated snapshot in
     `reports/daily/`;
   - logs predictions for drawn matches in `reports/predictions_log.csv`, and scores the played
     ones in `reports/live_scores.md`;
   - pulls Pinnacle's closing odds for finished Worlds matches (1 OddsPapi call; `--no-odds` to
     skip) and refreshes the model-vs-market comparison in `reports/market_benchmark.md`.
3. **Commit** the results file and the reports.
4. **Look** (optional): `uv run --group app streamlit run src/shyarly/app.py` opens a
   read-only page with title odds, Swiss record distributions, odds over time per team and the
   prediction log next to the results. It only reads `reports/` and `data/manual/`.

**Prediction log (the fresh holdout).** Every match gets two predictions: *pre-event* (model
trained on games before the stage started: Oct 23 for Swiss, Nov 3 for knockouts) and *live*
(games before the match's UTC day). A prediction is logged only after its cutoff has passed,
and the log is append-only: the first row per match, model and mode is the one scored, and each
row records the commit and a config hash. Production BT (`model = bt`) drives the odds; the
fitted shadow model is logged as `model = margin_bt` and only scored. Run `update` early on each match day (before 17:00 UTC)
so the live predictions are logged before the games start.

**Odds model for the bracket:** each 3-0 team meets a random 3-2 team, those two matches sit on
opposite halves, and the other four teams are paired at random (RULESET 4.3.4); every
knockout match is Bo5. Each run plays the Swiss Stage and the knockouts with the same strength
draw, so title odds carry the same uncertainty as qualify odds.

## Repo layout

```
config/            worlds2026.yaml (cited tournament facts), leagues.yaml, model.yaml, sources.yaml
data/              raw/ interim/ processed/ (gitignored) and manual/ (tracked: worlds2026_results.csv)
src/shyarly/       cli.py, paths.py, config.py, app.py (Streamlit page), dashboard.py (its loaders)
  data/            fetch.py (cache + manifest), leaguepedia.py, games.py (games/series tables),
                   manual.py (results file: loading, rule checks, replay into simulator inputs)
  models/          bt.py, elo.py, baselines.py, calibration.py, series.py, predictor.py
  eval/            metrics.py, backtest.py, live.py (prediction log + live scoring)
  sim/             swiss.py (Swiss simulator), knockout.py (bracket), run.py (tournament run + reports)
tests/             schema, leakage, models, metrics, calibration, series, Swiss invariants, predictor,
                   posterior, results file, knockout, prediction log, page loaders
reports/           backtest.md, swiss_odds.*, tournament_odds.*, daily/, predictions_log.csv, live_scores.md
```

## Data and licensing

`data/` is gitignored except the hand-entered files in `data/manual/`; run `shyarly ingest` to
download the match data yourself. Game statistics are the property of Riot Games. The match data
comes from [Oracle's Elixir](https://oracleselixir.com) (Tim Sevenhuysen), and some tournament
details come from [Leaguepedia](https://lol.fandom.com) (CC BY-SA 3.0). Betting odds (OddsPapi,
OddsPortal) are used locally as a benchmark and are not redistributed here.

To fetch odds yourself, copy `.env.example` to `.env` and add your own OddsPapi key. `.env` is
gitignored; never commit it.

The code is MIT-licensed ([LICENSE](LICENSE)); that covers the code only, not the data.

## Disclaimer

Shyarly isn't endorsed by Riot Games and doesn't reflect the views or opinions of Riot Games or
anyone officially involved in producing or managing Riot Games properties. Riot Games and all
associated properties are trademarks or registered trademarks of Riot Games, Inc.

The forecasts are a fan statistical project, not betting advice.
