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
    _ingest(refresh)


def _ingest(refresh: bool) -> None:
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


@app.command()
def odds(
    tournament: list[int] = typer.Option(..., "--tournament", help="OddsPapi tournament id(s), e.g. 2549 = Worlds"),
    refresh: bool = typer.Option(False, "--refresh", help="Re-fetch the fixture list (1 billable call each)"),
) -> None:
    """Closing betting odds (Pinnacle) for finished matches -> data/manual/market_odds.csv."""
    from shyarly.config import _read_yaml
    from shyarly.data.market import MARKET_ODDS, append_market_rows, tournament_rows
    from shyarly.data.oddspapi import OddsPapi

    cfg = _read_yaml(paths.SOURCES_CONFIG)["oddspapi"]
    client = OddsPapi(cfg=cfg)
    markets = [cfg["match_winner_market"], cfg["first_map_market"]]
    for tid in tournament:
        for book in cfg["bookmakers"]:
            rows, missing = tournament_rows(client, tid, book, markets, refresh_fixtures=refresh)
            added = append_market_rows(MARKET_ODDS, rows)
            typer.echo(f"tournament {tid} / {book}: {len(rows)} closing quotes ({added} new), "
                       f"{missing} finished fixtures without {book} odds")
    typer.echo(f"OddsPapi billable calls this month: {client.billable_this_month()} "
               f"(budget {cfg['monthly_budget']}) -> {MARKET_ODDS.relative_to(paths.ROOT)}")


def _benchmark() -> str:
    """Model vs closing market odds -> reports/market_benchmark.md. Returns a one-line summary."""
    import pandas as pd

    from shyarly.config import _read_yaml
    from shyarly.data.market import MARKET_ODDS
    from shyarly.eval.benchmark import REPORT_NAME, add_model, benchmark_markdown, join_market, name_map, score_benchmark
    from shyarly.models.predictor import bt_params_from_config

    if not MARKET_ODDS.exists():
        return "Market benchmark: no market odds yet (run `shyarly odds`)."
    cfg, src = load_model_config(), _read_yaml(paths.SOURCES_CONFIG)["oddspapi"]
    aliases = src.get("team_aliases") or {}
    market = pd.read_csv(MARKET_ODDS)
    series, games = pd.read_parquet(paths.SERIES_PARQUET), pd.read_parquet(paths.GAMES_PARQUET)
    joined, unmatched = join_market(market, series, games, aliases)
    _, missing = name_map(sorted(set(market["team_1"]) | set(market["team_2"])),
                          set(series["team_a"]) | set(series["team_b"]), aliases)
    if len(joined):
        joined = add_model(joined, games, bt_params_from_config(cfg), cfg["production"]["temperature"])
    scores = score_benchmark(joined, cfg["backtest"]["bootstrap_resamples"], cfg["seed"]) if len(joined) else pd.DataFrame()
    md = benchmark_markdown(joined, scores, unmatched, missing)
    parts = [f"{s['target']}: n={s['n']}, model {s['model_log_loss']:.3f} vs market {s['market_log_loss']:.3f} "
             f"(Δ {s['delta']:+.3f} {s['ci95']})" for s in scores.to_dict("records")]
    summary = "Market benchmark (Pinnacle): " + ("; ".join(parts) or "no matched matches yet")

    op_summary = _oddsportal_benchmark(cfg, series, games)
    if op_summary:
        md += "\n" + op_summary[0]
        summary += "\nMarket benchmark (OddsPortal): " + op_summary[1]
    (paths.REPORTS_DIR / REPORT_NAME).write_text(md, encoding="utf-8")
    return summary + f"\n-> reports/{REPORT_NAME}"


def _oddsportal_benchmark(cfg, series, games) -> tuple[str, str] | None:
    """OddsPortal section, if the local scrape exists (data/raw/oddsportal/, not committed)."""
    import pandas as pd

    from shyarly.config import _read_yaml
    from shyarly.data.oddsportal import ODDSPORTAL_FILE, load_oddsportal
    from shyarly.eval.benchmark import add_model, join_quotes, make_resolver, oddsportal_markdown, score_benchmark
    from shyarly.models.predictor import bt_params_from_config

    if not ODDSPORTAL_FILE.exists():
        return None
    op = _read_yaml(paths.SOURCES_CONFIG)["oddsportal"]
    quotes, dropped = load_oddsportal(ODDSPORTAL_FILE)
    intl = quotes[quotes["tournament"].isin(op["international"])]
    resolve = make_resolver(set(series["team_a"]) | set(series["team_b"]), op.get("team_aliases") or {},
                            op.get("past_names") or {})
    # OddsPortal gives a date, not a time: allow the series to start up to 1.5 days either side.
    joined, unmatched = join_quotes(intl, series, games, resolve, window=pd.Timedelta(hours=36))
    joined = add_model(joined, games, bt_params_from_config(cfg), cfg["production"]["temperature"])
    split_of = {e: "tune" for e in cfg["backtest"]["tune_events"]} | {e: "holdout" for e in cfg["backtest"]["holdout_events"]}
    joined["split"] = joined["event"].map(split_of).fillna("other")
    args = (cfg["backtest"]["bootstrap_resamples"], cfg["seed"])
    splits = {"all": score_benchmark(joined, *args)} | {
        s: score_benchmark(joined[joined["split"] == s], *args) for s in ("tune", "holdout", "other")
        if (joined["split"] == s).any()}
    md = oddsportal_markdown(joined, splits, dropped, len(quotes), len(intl), len(unmatched))
    a = splits["all"].set_index("target").loc["series"]
    return md, (f"{int(a['n'])} international series, model {a['model_log_loss']:.3f} vs market "
                f"{a['market_log_loss']:.3f} (Δ {a['delta']:+.3f} {a['ci95']})")


@app.command()
def benchmark() -> None:
    """Compare the model with closing betting odds (data/manual/market_odds.csv)."""
    typer.echo(_benchmark())


@app.command()
def shadow() -> None:
    """Run the pre-registered shadow model margin_bt (config/model.yaml) and write reports/shadow_margin.md."""
    import pandas as pd

    from shyarly.eval.shadow import REPORT, run_shadow, shadow_markdown

    cfg = load_model_config()
    res = run_shadow(pd.read_parquet(paths.GAMES_PARQUET), cfg)
    md = shadow_markdown(res, cfg)
    REPORT.write_text(md, encoding="utf-8")
    typer.echo(md)
    typer.echo(f"Fitted (copy into shadow_models.margin_bt.fitted): margin_weight={res.selected_weight}, "
               f"margin_scale={res.scale['c_gold_per_logit']}, margin_sigma={res.scale['sigma_logit']}, "
               f"temperature={res.temperature:.4f}")
    typer.echo(f"Wrote {REPORT.relative_to(paths.ROOT)}")


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


def _results(as_of: str | None, worlds, cutoff, use: bool = True):
    """Rows from data/manual/worlds2026_results.csv. With an explicit --as-of, only matches
    played before that UTC day count as results (later ones are dropped)."""
    from shyarly.data.manual import ResultsError, load_results

    if not use:
        return []
    try:
        rows = load_results(paths.MANUAL_RESULTS, worlds)
    except ResultsError as e:
        raise typer.BadParameter(f"{paths.MANUAL_RESULTS.relative_to(paths.ROOT)}: {e}") from e
    return [r for r in rows if r.date_utc < cutoff] if as_of else rows


def _run_simulation(stage: str, runs: int | None, as_of: str | None, seed: int | None, posterior: bool | None,
                    use_results: bool, snapshot: bool = False):
    import pandas as pd

    from shyarly.config import load_worlds
    from shyarly.data.manual import ResultsError
    from shyarly.models.predictor import Predictor
    from shyarly.sim.run import odds_markdown, simulate_worlds, tournament_markdown

    if stage not in ("swiss", "tournament"):
        raise typer.BadParameter("--stage must be swiss or tournament")
    cfg, worlds = load_model_config(), load_worlds()
    g = pd.read_parquet(paths.GAMES_PARQUET)
    cutoff = _as_of(as_of)
    results = _results(as_of, worlds, cutoff, use_results)
    use_posterior = cfg["simulation"].get("posterior", False) if posterior is None else posterior
    pr = Predictor.from_config(g, cutoff, cfg, posterior=use_posterior)
    try:
        odds = simulate_worlds(worlds, load_leagues(), pr, runs=runs or cfg["simulation"]["runs"],
                               seed=cfg["seed"] if seed is None else seed, posterior=use_posterior,
                               results=results, knockout=stage == "tournament")
    except ResultsError as e:
        raise typer.BadParameter(f"{paths.MANUAL_RESULTS.relative_to(paths.ROOT)}: {e}") from e
    last_game = g.loc[g["date_utc"] < cutoff, "date_utc"].max()
    paths.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    swiss_cols = [c for c in odds.table.columns if c not in ("semis", "final", "champion")]
    md = odds_markdown(odds, cutoff, pr.temperature, last_game)
    paths.SWISS_ODDS_MD.write_text(md, encoding="utf-8")
    odds.table.sort_values(["qualify", "3-0"], ascending=False)[swiss_cols].to_csv(paths.SWISS_ODDS_CSV, index=False)
    written = [paths.SWISS_ODDS_MD, paths.SWISS_ODDS_CSV]
    if odds.knockout:
        md = tournament_markdown(odds, cutoff, pr.temperature, last_game)
        paths.TOURNAMENT_ODDS_MD.write_text(md, encoding="utf-8")
        odds.table.to_csv(paths.TOURNAMENT_ODDS_CSV, index=False)
        written += [paths.TOURNAMENT_ODDS_MD, paths.TOURNAMENT_ODDS_CSV]
    if snapshot:
        paths.DAILY_DIR.mkdir(parents=True, exist_ok=True)
        snap = paths.DAILY_DIR / f"{cutoff:%Y-%m-%d}_odds.csv"
        odds.table.assign(as_of=f"{cutoff:%Y-%m-%d}", runs=odds.runs, seed=odds.seed).to_csv(snap, index=False)
        written.append(snap)
    typer.echo(md)
    typer.echo("Wrote " + ", ".join(str(p.relative_to(paths.ROOT)) for p in written))
    return cfg, worlds, g, results


@app.command()
def simulate(
    stage: str = typer.Option("swiss", "--stage", help="swiss, or tournament (Swiss + knockout, champion %)"),
    runs: int = typer.Option(None, "--runs", help="Default: simulation.runs in config/model.yaml"),
    as_of: str = typer.Option(None, "--as-of", help="Use only games (and results) before this UTC date (default: today)"),
    seed: int = typer.Option(None, "--seed", help="Default: seed in config/model.yaml"),
    posterior: bool = typer.Option(None, "--posterior/--point",
                                   help="Draw strengths from the posterior each run (default: simulation.posterior)"),
    results: bool = typer.Option(True, "--results/--no-results",
                                 help="Condition on data/manual/worlds2026_results.csv"),
) -> None:
    """Monte Carlo of the Swiss Stage (qualify %, 3-0…0-3), or the whole tournament (champion %)."""
    _run_simulation(stage, runs, as_of, seed, posterior, results)


def _write_forecasts(g, cfg, worlds, results, now) -> int:
    """Current production forecast for drawn, unplayed matches -> reports/match_forecasts.csv."""
    from shyarly.data.manual import swiss_fixed
    from shyarly.eval.live import forecast_upcoming
    from shyarly.models.predictor import Predictor, worlds_team_ref

    leagues = load_leagues()
    fx = swiss_fixed(results, worlds, [t.code for t in worlds.teams])
    ref = lambda code: worlds_team_ref(code, worlds, leagues, now.year)  # noqa: E731
    out = forecast_upcoming(results, fx.best_of, lambda c: Predictor.from_config(g, c, cfg), ref, now)
    out.to_csv(paths.MATCH_FORECASTS, index=False)
    typer.echo(f"Match forecasts: {len(out)} upcoming matches -> {paths.MATCH_FORECASTS.relative_to(paths.ROOT)}")
    return len(out)


@app.command()
def forecast() -> None:
    """Current forecast for every drawn, unplayed match -> reports/match_forecasts.csv (shown on the page,
    not scored; the scored predictions are logged by `update` once each match day starts)."""
    import pandas as pd

    from shyarly.config import load_worlds
    from shyarly.data.manual import load_results

    worlds = load_worlds()
    results = load_results(paths.MANUAL_RESULTS, worlds)
    _write_forecasts(pd.read_parquet(paths.GAMES_PARQUET), load_model_config(), worlds, results,
                     pd.Timestamp.now(tz="UTC"))


@app.command()
def update(
    fetch: bool = typer.Option(True, "--fetch/--no-fetch", help="Re-download this year's Oracle's Elixir data first"),
    odds: bool = typer.Option(True, "--odds/--no-odds",
                              help="Refresh Worlds closing odds from OddsPapi (1 billable call) and the market benchmark"),
    runs: int = typer.Option(None, "--runs", help="Default: simulation.runs in config/model.yaml"),
    seed: int = typer.Option(None, "--seed", help="Default: seed in config/model.yaml"),
) -> None:
    """Daily during Worlds: refresh data, re-simulate the tournament with known results,
    log predictions for drawn matches and score the played ones."""
    import pandas as pd

    from shyarly.data.manual import swiss_fixed
    from shyarly.eval.live import (append_log, due_predictions, live_markdown, provenance, score_log,
                                   stage_starts)
    from shyarly.models.predictor import Predictor, shadow_params_from_config, worlds_team_ref

    if fetch:
        _ingest(refresh=True)
    cfg, worlds, g, results = _run_simulation("tournament", runs, None, seed, None, True, snapshot=True)

    leagues = load_leagues()
    codes = [t.code for t in worlds.teams]
    fx = swiss_fixed(results, worlds, codes)
    now = pd.Timestamp.now(tz="UTC")
    # Production BT drives the odds; fitted shadow models are only logged (config shadow_models).
    models = {"bt": lambda cutoff: Predictor.from_config(g, cutoff, cfg)}
    shadow = shadow_params_from_config(cfg)
    if shadow:
        models["margin_bt"] = lambda cutoff, s=shadow: Predictor(g, cutoff, s[0], s[1])
    ref = lambda code: worlds_team_ref(code, worlds, leagues, now.year)  # noqa: E731
    prov = provenance()
    added = 0
    for name, predictor_at in models.items():
        new = due_predictions(results, fx.best_of, predictor_at, ref, stage_starts(worlds), now, prov, model=name)
        added += append_log(paths.PREDICTIONS_LOG, new)
    log = pd.read_csv(paths.PREDICTIONS_LOG, dtype={"round": str})
    scores = score_log(log, results)
    paths.LIVE_SCORES_MD.write_text(live_markdown(scores, log, results), encoding="utf-8")
    typer.echo(f"Prediction log: {added} new rows ({len(log)} total) -> {paths.PREDICTIONS_LOG.relative_to(paths.ROOT)}")
    if scores.empty:
        typer.echo("Live scoring: no played matches with a logged prediction yet.")
    for s in scores.to_dict("records"):
        typer.echo(f"Live scoring [{s['model']} {s['mode']}]: {s['n_series']} series, log loss {s['series_log_loss']:.4f}; "
                   f"{s['n_games']} games, game log loss {s['game_log_loss']:.4f}")
    typer.echo(f"Wrote {paths.LIVE_SCORES_MD.relative_to(paths.ROOT)}")
    _write_forecasts(g, cfg, worlds, results, now)

    if odds:
        from shyarly.config import _read_yaml
        from shyarly.data.market import MARKET_ODDS, append_market_rows, tournament_rows
        from shyarly.data.oddspapi import OddsPapi, OddsPapiError

        src = _read_yaml(paths.SOURCES_CONFIG)["oddspapi"]
        try:
            client = OddsPapi(cfg=src)
            markets = [src["match_winner_market"], src["first_map_market"]]
            for book in src["bookmakers"]:
                rows, _ = tournament_rows(client, src["worlds_tournament_id"], book, markets, refresh_fixtures=True)
                added = append_market_rows(MARKET_ODDS, rows)
                typer.echo(f"Market odds: {added} new closing quotes ({book}); OddsPapi billable calls this month: "
                           f"{client.billable_this_month()} of {src['monthly_budget']}")
        except OddsPapiError as e:          # no key, budget used up, or the API is down: keep the update
            typer.echo(f"Market odds skipped: {e}", err=True)
        typer.echo(_benchmark())


if __name__ == "__main__":
    app()
