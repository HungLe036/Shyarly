"""Shyarly command line."""

from __future__ import annotations

import logging

import typer

from shyarly.config import load_sources
from shyarly.data import fetch

app = typer.Typer(help="Worlds 2026 win probabilities and Swiss simulation.", no_args_is_help=True)


@app.callback()
def _main(verbose: bool = typer.Option(False, "--verbose", "-v")) -> None:
    logging.basicConfig(level=logging.DEBUG if verbose else logging.INFO, format="%(levelname)s %(message)s")


@app.command()
def ingest(refresh: bool = typer.Option(False, help="Re-download years listed in refresh_years.")) -> None:
    """Fetch and cache raw data. (M1 adds building data/processed/games.parquet.)"""
    sources = load_sources()
    manifest = fetch.fetch_oracles_elixir(sources, refresh=refresh)
    fetch.fetch_rules(sources)
    for year, entry in sorted(manifest.items()):
        typer.echo(f"OE {year}: {entry['bytes'] / 1e6:6.1f} MB  sha256={entry['sha256'][:12]}  {entry['fetched_at_utc']}")


if __name__ == "__main__":
    app()
