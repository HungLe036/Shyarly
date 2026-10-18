"""Rolling-origin backtest on international (Worlds/MSI main-event) games.

Modes:
  live      — each game is predicted from games before its UTC day (refit per match day)
  pre_event — every game of an event is predicted from games before the event's first day

Hyperparameters are tuned on the tune events only (config/model.yaml), then frozen.
Each model's outputs are then temperature-scaled with a rolling temperature: an
event's T is fitted on that model's predictions for all earlier events.
"""

from __future__ import annotations

import itertools
import json
import logging
import time
from dataclasses import dataclass

import numpy as np
import pandas as pd

from shyarly import paths
from shyarly.eval import metrics
from shyarly.models.baselines import CoinFlip
from shyarly.models.bt import BradleyTerry, BTParams
from shyarly.models.calibration import apply_temperature, rolling_temperatures
from shyarly.models.common import day_start
from shyarly.models.elo import Elo, EloParams
from shyarly.models.series import series_win_prob, side_averaged

log = logging.getLogger(__name__)

MODEL_KINDS = ["coinflip", "region_prior", "elo", "bt"]
SIDE_SWAP = {
    "blue_team": "red_team", "red_team": "blue_team",
    "blue_team_season_id": "red_team_season_id", "red_team_season_id": "blue_team_season_id",
    "blue_region": "red_region", "red_region": "blue_region",
}


def make_queries(games: pd.DataFrame, events: list[str], mode: str) -> pd.DataFrame:
    q = games[games["scored"] & games["event"].isin(events)].copy()
    if mode == "live":
        q["cutoff"] = day_start(q["date_utc"])
    elif mode == "pre_event":
        q["cutoff"] = day_start(q.groupby("event")["date_utc"].transform("min"))
    else:
        raise ValueError(mode)
    return q.reset_index(drop=True)


def build_model(kind: str, params: dict, cfg: dict):
    if kind == "coinflip":
        return CoinFlip()
    if kind == "elo":
        return Elo(EloParams(**params))
    section = cfg[kind]
    bt = BTParams(
        half_life_days=params["half_life_days"],
        lambda_team=params.get("lambda_team", 0.0),
        lambda_region_major=params["lambda_region_major"],
        lambda_region_minor=params["lambda_region_minor"],
        intl_weight=params["intl_weight"],
        lookback_days=section["lookback_days"],
        minor_regions=tuple(section["minor_regions"]),
        half_life_intl_days=params.get("half_life_intl_days"),
    )
    return BradleyTerry(bt, use_teams=(kind == "bt"))


def param_grid(cfg: dict, kind: str) -> list[dict]:
    if kind == "coinflip":
        return [{}]
    grid = cfg[kind]["grid"]
    keys = sorted(grid)
    return [dict(zip(keys, values)) for values in itertools.product(*(grid[k] for k in keys))]


def score(y, p, bins: int) -> dict:
    y, p = np.asarray(y), np.asarray(p)
    return {
        "n": int(len(y)),
        "log_loss": metrics.log_loss(y, p),
        "brier": metrics.brier(y, p),
        "ece": metrics.ece(y, p, bins),
        "slope": metrics.calibration_slope(y, p),
        "accuracy": float(np.mean((p > 0.5) == (y == 1))),
    }


def tune(kind: str, games: pd.DataFrame, cfg: dict) -> tuple[dict, pd.DataFrame]:
    bt_cfg = cfg["backtest"]
    q = make_queries(games, bt_cfg["tune_events"], bt_cfg["tune_mode"])
    grid = param_grid(cfg, kind)
    losses = [metrics.log_loss(q["blue_win"], build_model(kind, params, cfg).predict(games, q)) for params in grid]
    best = grid[int(np.argmin(losses))]            # first minimum: deterministic tie-break by grid order
    table = pd.DataFrame(grid).assign(log_loss=losses).sort_values("log_loss", kind="stable")
    log.info("%s: best %s (tune log loss %.4f, %d configs)", kind, best, min(losses), len(grid))
    return best, table


def event_order(games: pd.DataFrame, events: list[str]) -> list[str]:
    start = games[games["scored"] & games["event"].isin(events)].groupby("event")["date_utc"].min()
    return list(start.sort_values().index)


def calibrate(preds: pd.DataFrame, order: list[str]) -> pd.DataFrame:
    """Add rolling-temperature columns per (model, mode): temperature, p_cal, p_swap_cal."""
    out = []
    for _, d in preds.groupby(["model", "mode"], sort=False):
        temps = rolling_temperatures(d.assign(p=d["p_raw"]), order)
        T = d["event"].map(temps).to_numpy()
        out.append(d.assign(temperature=T, p_cal=apply_temperature(d["p_raw"], T),
                            p_swap_cal=apply_temperature(d["p_swap_raw"], T)))
    return pd.concat(out).sort_index()


@dataclass
class BacktestResult:
    predictions: pd.DataFrame
    best_params: dict[str, dict]
    tuning: dict[str, pd.DataFrame]


def run_backtest(games: pd.DataFrame, cfg: dict) -> BacktestResult:
    bt_cfg = cfg["backtest"]
    split_of = {e: "tune" for e in bt_cfg["tune_events"]} | {e: "holdout" for e in bt_cfg["holdout_events"]}
    events = list(split_of)
    best_params, tuning, preds = {}, {}, []
    for kind in MODEL_KINDS:
        t0 = time.perf_counter()
        best_params[kind], tuning[kind] = tune(kind, games, cfg)
        model = build_model(kind, best_params[kind], cfg)
        for mode in bt_cfg["modes"]:
            q = make_queries(games, events, mode)
            # One call for both orientations, so each cutoff is fitted once.
            both = model.predict(games, pd.concat([q, q.rename(columns=SIDE_SWAP)], ignore_index=True))
            preds.append(pd.DataFrame({
                "model": kind, "mode": mode, "split": q["event"].map(split_of), "event": q["event"],
                "game_id": q["game_id"], "series_id": q["series_id"], "date_utc": q["date_utc"],
                "cutoff": q["cutoff"], "blue_team": q["blue_team"], "red_team": q["red_team"],
                "y": q["blue_win"].astype(int), "p_raw": both[: len(q)], "p_swap_raw": both[len(q):],
            }))
        log.info("%s done in %.0fs", kind, time.perf_counter() - t0)
    preds = calibrate(pd.concat(preds, ignore_index=True), event_order(games, events))
    return BacktestResult(preds, best_params, tuning)


# ---------------------------------------------------------------- evaluation


def summary_table(preds: pd.DataFrame, split: str, mode: str, bins: int, col: str = "p_cal") -> pd.DataFrame:
    rows = []
    for kind in MODEL_KINDS:
        d = preds[(preds["model"] == kind) & (preds["split"] == split) & (preds["mode"] == mode)]
        rows.append({"model": kind, **score(d["y"], d[col], bins)})
    return pd.DataFrame(rows)


def per_event_table(preds: pd.DataFrame, mode: str, col: str = "p_cal") -> pd.DataFrame:
    d = preds[preds["mode"] == mode]
    t = d.groupby(["event", "model"]).apply(
        lambda g: metrics.log_loss(g["y"], g[col]), include_groups=False).unstack("model")[MODEL_KINDS]
    bt = d[d["model"] == "bt"].groupby("event")
    info = pd.DataFrame({"split": bt["split"].first(), "n": bt.size(), "T (bt)": bt["temperature"].first()})
    out = pd.concat([info, t], axis=1).reset_index()
    return out.sort_values(["split", "event"], ascending=[False, True]).reset_index(drop=True)


def _bootstrap(preds: pd.DataFrame, mode: str, cfg: dict, other: str, col: str, reference: str = "bt") -> dict:
    d = preds[(preds["split"] == "holdout") & (preds["mode"] == mode)]
    ref = d[d["model"] == reference].set_index("game_id")
    oth = d[d["model"] == other].set_index("game_id").loc[ref.index]
    return metrics.paired_bootstrap(
        metrics.game_log_loss(ref["y"], ref[col]), metrics.game_log_loss(oth["y"], oth[col]),
        ref["series_id"], resamples=cfg["backtest"]["bootstrap_resamples"], seed=cfg["seed"])


def bootstrap_table(preds: pd.DataFrame, mode: str, cfg: dict, col: str = "p_cal") -> pd.DataFrame:
    rows = []
    for kind in MODEL_KINDS[:-1]:
        res = _bootstrap(preds, mode, cfg, kind, col)
        rows.append({
            "comparison": f"bt − {kind}", "Δ log loss": res["mean"],
            "80% CI": f"[{res['ci80'][0]:+.4f}, {res['ci80'][1]:+.4f}]",
            "95% CI": f"[{res['ci95'][0]:+.4f}, {res['ci95'][1]:+.4f}]",
        })
    return pd.DataFrame(rows)


def evaluate_gates(preds: pd.DataFrame, cfg: dict, mode: str = "live", col: str = "p_cal") -> pd.DataFrame:
    """Spec promotion gates for BT on the holdout events."""
    g = cfg["gates"]
    d = preds[(preds["split"] == "holdout") & (preds["mode"] == mode) & (preds["model"] == "bt")]
    s = score(d["y"], d[col], cfg["backtest"]["ece_bins"])
    coin = _bootstrap(preds, mode, cfg, "coinflip", col)
    rp = _bootstrap(preds, mode, cfg, "region_prior", col)
    elo = _bootstrap(preds, mode, cfg, "elo", col)
    lo, hi = g["slope_range"]
    rows = [
        ("Beats 50/50", f"log loss {s['log_loss']:.4f}; Δ 95% CI [{coin['ci95'][0]:+.4f}, {coin['ci95'][1]:+.4f}]",
         f"< {g['coinflip_log_loss']}, CI < 0", s["log_loss"] < g["coinflip_log_loss"] and coin["ci95"][1] < 0),
        ("Beats region prior", f"Δ {rp['mean']:+.4f}; 95% CI [{rp['ci95'][0]:+.4f}, {rp['ci95'][1]:+.4f}]",
         f"Δ ≤ −{g['min_gain_vs_region_prior']}, CI < 0",
         rp["mean"] <= -g["min_gain_vs_region_prior"] and rp["ci95"][1] < 0),
        ("Not worse than Elo", f"Δ {elo['mean']:+.4f}; 80% CI upper {elo['ci80'][1]:+.4f}",
         f"Δ < 0, 80% upper ≤ +{g['max_ci80_upper_vs_elo']}",
         elo["mean"] < 0 and elo["ci80"][1] <= g["max_ci80_upper_vs_elo"]),
        ("ECE (5 bins)", f"{s['ece']:.4f}", f"≤ {g['max_ece']}", s["ece"] <= g["max_ece"]),
        ("Calibration slope", f"{s['slope']:.3f}", f"in [{lo}, {hi}]", lo <= s["slope"] <= hi),
    ]
    return pd.DataFrame(rows, columns=["gate", "measured", "threshold", "pass"]).assign(
        **{"pass": lambda t: t["pass"].map({True: "PASS", False: "FAIL"})})


def series_table(preds: pd.DataFrame, series: pd.DataFrame, mode: str = "live") -> pd.DataFrame:
    """Holdout series scored with the side-averaged calibrated game p of the series' first
    game and the closed-form best-of formula. 'bt, game p only' skips the formula, to show
    whether amplifying p for longer series helps."""
    d = preds[(preds["split"] == "holdout") & (preds["mode"] == mode)]
    first = d.sort_values("date_utc").drop_duplicates(["model", "series_id"])
    first = first.merge(series[["series_id", "winner_team", "inferred_best_of"]], on="series_id")
    first = first[first["inferred_best_of"].isin([1, 3, 5])]
    first["p_game"] = side_averaged(first["p_cal"], first["p_swap_cal"])   # P(blue team of game 1) per game
    first["y"] = (first["winner_team"] == first["blue_team"]).astype(int)
    first["p_series"] = [series_win_prob(p, int(bo)) for p, bo in zip(first["p_game"], first["inferred_best_of"])]
    rows = []
    for bo in (1, 3, 5):
        for kind in MODEL_KINDS:
            s = first[(first["model"] == kind) & (first["inferred_best_of"] == bo)]
            rows.append({"best_of": bo, "model": kind, "n_series": len(s),
                         "log_loss": metrics.log_loss(s["y"], s["p_series"]),
                         "brier": metrics.brier(s["y"], s["p_series"])})
        if bo > 1:
            s = first[(first["model"] == "bt") & (first["inferred_best_of"] == bo)]
            rows.append({"best_of": bo, "model": "bt, game p only", "n_series": len(s),
                         "log_loss": metrics.log_loss(s["y"], s["p_game"]),
                         "brier": metrics.brier(s["y"], s["p_game"])})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------- report


def _fmt_table(df: pd.DataFrame, floatfmt: str = ".4f") -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, row in df.iterrows():
        cells = [format(v, floatfmt) if isinstance(v, float) else str(v) for v in row]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def write_report(result: BacktestResult, cfg: dict, series: pd.DataFrame, path=paths.BACKTEST_REPORT) -> str:
    bt_cfg = cfg["backtest"]
    bins = bt_cfg["ece_bins"]
    preds = result.predictions
    manifest = json.loads((paths.OE_RAW_DIR / "manifest.json").read_text(encoding="utf-8"))
    snap = ", ".join(f"{y}: {m['sha256'][:10]} ({m['fetched_at_utc'][:10]})" for y, m in sorted(manifest.items()))
    gates = evaluate_gates(preds, cfg, "live")

    parts = [
        "# Shyarly backtest",
        "",
        f"Generated {pd.Timestamp.now(tz='UTC'):%Y-%m-%d %H:%M} UTC by `shyarly backtest`.",
        "",
        "## Promotion gates — BT, holdout, live mode, calibrated",
        _fmt_table(gates),
        "",
        "## Holdout use",
        "The holdout has been looked at during development, and the final setup was chosen after "
        f"those looks, so holdout scores are optimistically biased. Looks so far: {len(bt_cfg['holdout_looks'])}.",
        "",
        *[f"{i}. {look}" for i, look in enumerate(bt_cfg["holdout_looks"], start=1)],
        "",
        "## Setup",
        "- **Scored games:** Worlds/MSI main-event games (regional qualifiers excluded), blue-side win probability.",
        f"- **Tune events:** {', '.join(bt_cfg['tune_events'])}. **Holdout events:** {', '.join(bt_cfg['holdout_events'])}.",
        "- **Live mode:** each game uses only games before its UTC day. **Pre-event mode:** each event's games use only "
        "games before the event's first day.",
        f"- **Tuning:** grid search on the tune events in `{bt_cfg['tune_mode']}` mode, log loss; params then frozen.",
        "- **BT decay:** domestic games use `half_life_days` (60–180 per spec); cross-region games use their own "
        "`half_life_intl_days`, so region offsets can keep a longer memory than team form (added after M2).",
        "- **Calibration:** one temperature on the logit, no intercept. Each event's T is fitted on the same model's "
        "predictions for all earlier events (T = 1 for the first event), so it never sees the outcomes it is scored on.",
        f"- **Bootstrap:** {bt_cfg['bootstrap_resamples']} resamples of whole series (seed {cfg['seed']}).",
        f"- **Data snapshot (OE sha256):** {snap}.",
        "",
        "## Holdout — live mode, calibrated (primary)",
        _fmt_table(summary_table(preds, "holdout", "live", bins)),
        "",
        "## Holdout — live mode, raw (before temperature scaling)",
        _fmt_table(summary_table(preds, "holdout", "live", bins, col="p_raw")),
        "",
        "## Holdout — pre-event mode, calibrated",
        _fmt_table(summary_table(preds, "holdout", "pre_event", bins)),
        "",
        "## Tune events — live mode, calibrated",
        "The first tune event has T = 1, so these rows are close to raw.",
        "",
        _fmt_table(summary_table(preds, "tune", "live", bins)),
        "",
        "## Log loss per event — live mode, calibrated",
        _fmt_table(per_event_table(preds, "live")),
        "",
        "## Paired bootstrap on holdout, live mode, calibrated",
        "Δ = BT log loss − other model's log loss; negative means BT is better.",
        "",
        _fmt_table(bootstrap_table(preds, "live", cfg)),
        "",
        "## Series — holdout, live mode",
        "Series win probability from the first game's side-averaged calibrated p and the closed-form "
        "Bo1/Bo3/Bo5 formula (games treated as independent). Best-of is inferred from the winner's game count.",
        "",
        _fmt_table(series_table(preds, series, "live")),
        "",
        "## Tuned hyperparameters",
    ]
    for kind, params in result.best_params.items():
        tune_ll = result.tuning[kind]["log_loss"].iloc[0]
        parts.append(f"- **{kind}** (tune log loss {tune_ll:.4f}, {len(result.tuning[kind])} configs): `{params}`")
    parts.append("")
    text = "\n".join(parts)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return text
