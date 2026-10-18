"""Shyarly command line."""

from __future__ import annotations

import logging
import sys

import typer

from shyarly import paths
from shyarly.config import load_leagues, load_model_config, load_sources
from shyarly.data import fetch, games

app = typer.Typer(help="Worlds 2026 win probabilities and Swiss simulation.", no_args_is_help=True)


@app.callback()
def _main(verbose: bool = typer.Option(False, "--verbose", "-v")) -> None:
    for stream in (sys.stdout, sys.stderr):   # Windows consoles default to cp1252; reports use ≤, …, Ø
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
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
    write_report(result, cfg, pd.read_parquet(paths.SERIES_PARQUET))
    typer.echo(f"Wrote {paths.BACKTEST_REPORT.relative_to(paths.ROOT)} and {paths.BACKTEST_PREDICTIONS.relative_to(paths.ROOT)}")


def _as_of(value: str | None):
    import pandas as pd

    ts = pd.Timestamp(value) if value else pd.Timestamp.now(tz="UTC")
    return (ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")).floor("D")


predict_app = typer.Typer(help="Win probabilities for a matchup.", no_args_is_help=True)
app.add_typer(predict_app, name="predict")


@predict_app.command("match")
def predict_match(
    team_a: str = typer.Argument(..., help="Worlds code (GEN), Worlds name or Oracle's Elixir team name"),
    team_b: str = typer.Argument(...),
    bo: int = typer.Option(1, "--bo", help="Best-of: 1, 3 or 5"),
    as_of: str = typer.Option(None, "--as-of", help="Use only games before this UTC date (default: today)"),
) -> None:
    """P(team A beats team B) for one game and for a best-of series."""
    import pandas as pd

    from shyarly.config import load_worlds
    from shyarly.models.predictor import Predictor, resolve_team

    g = pd.read_parquet(paths.GAMES_PARQUET)
    cutoff = _as_of(as_of)
    worlds, leagues, cfg = load_worlds(), load_leagues(), load_model_config()
    a = resolve_team(team_a, worlds, leagues, g, cutoff)
    b = resolve_team(team_b, worlds, leagues, g, cutoff)
    pr = Predictor.from_config(g, cutoff, cfg)
    for t in (a, b):
        if not pr.has_data(t):
            typer.echo(f"warning: no {cutoff.year} games for {t.oe_name}; using its region strength only", err=True)
    typer.echo(f"{a.label} ({a.region}) vs {b.label} ({b.region}), games before {cutoff:%Y-%m-%d} UTC, "
               f"T={pr.temperature:.4f}")
    typer.echo(f"  one game:   {pr.p_game(a, b):6.1%}   (on blue {pr.p_blue(a, b):.1%}, on red {1 - pr.p_blue(b, a):.1%})")
    if bo != 1:
        typer.echo(f"  Bo{bo} series: {pr.p_series(a, b, bo):6.1%}")


@app.command()
def simulate(
    stage: str = typer.Option("swiss", "--stage", help="Only 'swiss' for now"),
    runs: int = typer.Option(None, "--runs", help="Default: simulation.runs in config/model.yaml"),
    as_of: str = typer.Option(None, "--as-of", help="Use only games before this UTC date (default: today)"),
    seed: int = typer.Option(None, "--seed", help="Default: seed in config/model.yaml"),
) -> None:
    """Monte Carlo of the Swiss Stage: qualify % and 3-0…0-3 per team."""
    import pandas as pd

    from shyarly.config import load_worlds
    from shyarly.models.predictor import Predictor
    from shyarly.sim.run import odds_markdown, simulate_worlds_swiss

    if stage != "swiss":
        raise typer.BadParameter("only --stage swiss is implemented (knockout is v2)")
    cfg = load_model_config()
    g = pd.read_parquet(paths.GAMES_PARQUET)
    cutoff = _as_of(as_of)
    pr = Predictor.from_config(g, cutoff, cfg)
    odds = simulate_worlds_swiss(load_worlds(), load_leagues(), pr,
                                 runs=runs or cfg["simulation"]["runs"], seed=cfg["seed"] if seed is None else seed)
    last_game = g.loc[g["date_utc"] < cutoff, "date_utc"].max()
    md = odds_markdown(odds, cutoff, pr.temperature, last_game)
    paths.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    paths.SWISS_ODDS_MD.write_text(md, encoding="utf-8")
    odds.table.to_csv(paths.SWISS_ODDS_CSV, index=False)
    typer.echo(md)
    typer.echo(f"Wrote {paths.SWISS_ODDS_MD.relative_to(paths.ROOT)} and {paths.SWISS_ODDS_CSV.relative_to(paths.ROOT)}")


if __name__ == "__main__":
    app()
