"""Rolling-origin backtest on international (Worlds/MSI main-event) games.

Modes:
  live      — each game is predicted from games before its UTC day (refit per match day)
  pre_event — every game of an event is predicted from games before the event's first day

Hyperparameters are tuned on the tune events only (config/model.yaml), then frozen
for the holdout events.
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
from shyarly.models.common import day_start
from shyarly.models.elo import Elo, EloParams

log = logging.getLogger(__name__)

MODEL_KINDS = ["coinflip", "region_prior", "elo", "bt"]


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
            p = model.predict(games, q)
            preds.append(pd.DataFrame({
                "model": kind, "mode": mode, "split": q["event"].map(split_of), "event": q["event"],
                "game_id": q["game_id"], "series_id": q["series_id"], "date_utc": q["date_utc"],
                "cutoff": q["cutoff"], "y": q["blue_win"].astype(int), "p": p,
            }))
        log.info("%s done in %.0fs", kind, time.perf_counter() - t0)
    return BacktestResult(pd.concat(preds, ignore_index=True), best_params, tuning)


# --------------------------------------------------------------------- report


def _fmt_table(df: pd.DataFrame, floatfmt: str = ".4f") -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, row in df.iterrows():
        cells = [format(v, floatfmt) if isinstance(v, float) else str(v) for v in row]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def summary_table(preds: pd.DataFrame, split: str, mode: str, bins: int) -> pd.DataFrame:
    rows = []
    for kind in MODEL_KINDS:
        d = preds[(preds["model"] == kind) & (preds["split"] == split) & (preds["mode"] == mode)]
        rows.append({"model": kind, **score(d["y"], d["p"], bins)})
    return pd.DataFrame(rows)


def per_event_table(preds: pd.DataFrame, mode: str) -> pd.DataFrame:
    d = preds[preds["mode"] == mode]
    t = d.groupby(["event", "model"]).apply(
        lambda g: metrics.log_loss(g["y"], g["p"]), include_groups=False).unstack("model")[MODEL_KINDS]
    n = d[d["model"] == "bt"].groupby("event").size().rename("n")
    split = d.drop_duplicates("event").set_index("event")["split"]
    out = pd.concat([split, n, t], axis=1).reset_index()
    order = {e: i for i, e in enumerate(out.sort_values(["split", "event"], ascending=[False, True])["event"])}
    return out.sort_values("event", key=lambda s: s.map(order)).reset_index(drop=True)


def bootstrap_table(preds: pd.DataFrame, mode: str, cfg: dict, reference: str = "bt") -> pd.DataFrame:
    bt_cfg = cfg["backtest"]
    d = preds[(preds["split"] == "holdout") & (preds["mode"] == mode)]
    ref = d[d["model"] == reference].set_index("game_id")
    rows = []
    for kind in MODEL_KINDS:
        if kind == reference:
            continue
        other = d[d["model"] == kind].set_index("game_id").loc[ref.index]
        res = metrics.paired_bootstrap(
            metrics.game_log_loss(ref["y"], ref["p"]), metrics.game_log_loss(other["y"], other["p"]),
            ref["series_id"], resamples=bt_cfg["bootstrap_resamples"], seed=cfg["seed"])
        rows.append({
            "comparison": f"{reference} − {kind}", "Δ log loss": res["mean"],
            "80% CI": f"[{res['ci80'][0]:+.4f}, {res['ci80'][1]:+.4f}]",
            "95% CI": f"[{res['ci95'][0]:+.4f}, {res['ci95'][1]:+.4f}]",
        })
    return pd.DataFrame(rows)


def write_report(result: BacktestResult, cfg: dict, path=paths.BACKTEST_REPORT) -> str:
    bt_cfg = cfg["backtest"]
    bins = bt_cfg["ece_bins"]
    preds = result.predictions
    manifest = json.loads((paths.OE_RAW_DIR / "manifest.json").read_text(encoding="utf-8"))
    snap = ", ".join(f"{y}: {m['sha256'][:10]} ({m['fetched_at_utc'][:10]})" for y, m in sorted(manifest.items()))

    parts = [
        "# Shyarly backtest",
        "",
        f"Generated {pd.Timestamp.now(tz='UTC'):%Y-%m-%d %H:%M} UTC by `shyarly backtest`. Uncalibrated model "
        "outputs (temperature scaling and promotion gates are M3).",
        "",
        "## Setup",
        f"- **Scored games:** Worlds/MSI main-event games (regional qualifiers excluded), blue-side win probability.",
        f"- **Tune events:** {', '.join(bt_cfg['tune_events'])}. **Holdout events:** {', '.join(bt_cfg['holdout_events'])}.",
        "- **Live mode:** each game uses only games before its UTC day. **Pre-event mode:** each event's games use only "
        "games before the event's first day.",
        f"- **Tuning:** grid search on the tune events in `{bt_cfg['tune_mode']}` mode, log loss; params then frozen.",
        f"- **Bootstrap:** {bt_cfg['bootstrap_resamples']} resamples of whole series (seed {cfg['seed']}).",
        f"- **Data snapshot (OE sha256):** {snap}.",
        "",
        "## Holdout — live mode (primary)",
        _fmt_table(summary_table(preds, "holdout", "live", bins)),
        "",
        "## Holdout — pre-event mode",
        _fmt_table(summary_table(preds, "holdout", "pre_event", bins)),
        "",
        "## Tune events — live mode",
        _fmt_table(summary_table(preds, "tune", "live", bins)),
        "",
        "## Log loss per event — live mode",
        _fmt_table(per_event_table(preds, "live")),
        "",
        "## Paired bootstrap on holdout, live mode",
        "Δ = BT log loss − other model's log loss; negative means BT is better.",
        "",
        _fmt_table(bootstrap_table(preds, "live", cfg)),
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
