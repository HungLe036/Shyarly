# Shyarly

Calibrated win probabilities and a Monte Carlo simulation of the League of Legends World Championship 2026.

Spec: [docs/02-refined-plan.md](docs/02-refined-plan.md). Tournament rules, teams and seeds: [config/worlds2026.yaml](config/worlds2026.yaml), with a source URL on every block.

## Setup (Windows)

```powershell
winget install --id astral-sh.uv -e
uv sync
```

## Commands

```
shyarly ingest [--refresh]                               # fetch/cache data, build data/processed/games.parquet
shyarly backtest                                         # tune, score holdout, write reports/backtest.md
shyarly predict match "<Team A>" "<Team B>" --bo 3 [--as-of 2026-10-25]
shyarly simulate --stage swiss --runs 50000 [--as-of 2026-10-25] [--seed N]   # reports/swiss_odds.md + .csv
```

Teams can be given as Worlds codes (`GEN`), Worlds names or Oracle's Elixir team names.
Run with `uv run shyarly ...`. Tests: `uv run pytest` (add `-m slow` for the 50k-run seed-stability check).

### Before and during the Swiss Stage
- **Play-In winner:** while `teams` has the pending `PIW` slot, `simulate` splits runs between the
  candidates by the model's Bo5 odds. Once decided, replace the `PIW` entry in `config/worlds2026.yaml`
  with the winner (and its source).
- **Round 1 draw:** fill `swiss.round1.pairings` with `[CODE, CODE]` pairs once published; until then
  the draw is simulated.
- Refresh data with `shyarly ingest --refresh` after new games.

## Data

- Oracle's Elixir yearly match CSVs (2022–2026), cached to `data/raw/oe/` with a sha256 manifest.
- Leaguepedia Cargo API (throttled; every response cached to `data/raw/leaguepedia/`).
- Hand-entered live results: `data/manual/worlds2026_results.csv`.

`data/` is gitignored except `data/manual/`. Game statistics are the property of Riot Games; OE data is provided by Tim Sevenhuysen (oracleselixir.com).
