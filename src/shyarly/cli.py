"""Shyarly command line."""

from __future__ import annotations

import logging

import typer

from shyarly import paths
from shyarly.config import load_leagues, load_model_config, load_sources
from shyarly.data import fetch, games

app = typer.Typer(help="Worlds 2026 win probabilities and Swiss simulation.", no_args_is_help=True)


@app.callback()
def _main(verbose: bool = typer.Option(False, "--verbose", "-v")) -> None:
    logging.basicConfig(level=logging.DEBUG if verbose else logging.INFO, format="%(levelname)s %(message)s")


@app.command()
def ingest(refresh: bool = typer.Option(False, help="Re-download years listed in refresh_years.")) -> None:
    """Fetch and cache raw data, then build data/processed/games.parquet and series.parquet."""
    sources = load_sources()
    manifest = fetch.fetch_oracles_elixir(sources, refresh=refresh)
    fetch.fetch_rules(sources)
    for year, entry in sorted(manifest.items()):
        typer.echo(f"OE {year}: {entry['bytes'] / 1e6:6.1f} MB  sha256={entry['sha256'][:12]}  {entry['fetched_at_utc']}")

    g = games.write_processed(load_leagues())
    typer.echo(f"games.parquet: {len(g):,} games, {g['date_utc'].min():%Y-%m-%d} .. {g['date_utc'].max():%Y-%m-%d}")
    typer.echo(f"cross-region games: {int(g['cross_region'].sum()):,}; scored (Worlds/MSI main) games: {int(g['scored'].sum()):,}")


@app.command()
def backtest() -> None:
    """Tune on 2022–23 events, score Worlds/MSI holdout events, write reports/backtest.md."""
    import pandas as pd

    from shyarly.eval.backtest import run_backtest, write_report

    cfg = load_model_config()
    result = run_backtest(pd.read_parquet(paths.GAMES_PARQUET), cfg)
    result.predictions.to_parquet(paths.BACKTEST_PREDICTIONS, index=False)
    write_report(result, cfg)
    typer.echo(f"Wrote {paths.BACKTEST_REPORT.relative_to(paths.ROOT)} and {paths.BACKTEST_PREDICTIONS.relative_to(paths.ROOT)}")


if __name__ == "__main__":
    app()
