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
shyarly ingest                                   # fetch and cache raw data
shyarly backtest                                 # M2/M3: writes reports/backtest.md
shyarly predict match "<Team A>" "<Team B>" --bo 3   # M4
shyarly simulate --stage swiss --runs 50000      # M4
```

Run with `uv run shyarly ...`. Tests: `uv run pytest`.

## Data

- Oracle's Elixir yearly match CSVs (2022–2026), cached to `data/raw/oe/` with a sha256 manifest.
- Leaguepedia Cargo API (throttled; every response cached to `data/raw/leaguepedia/`).
- Hand-entered live results: `data/manual/worlds2026_results.csv`.

`data/` is gitignored except `data/manual/`. Game statistics are the property of Riot Games; OE data is provided by Tim Sevenhuysen (oracleselixir.com).
