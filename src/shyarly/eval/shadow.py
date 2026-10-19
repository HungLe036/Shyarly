"""Shadow model margin_bt, run exactly as pre-registered in config/model.yaml `shadow_models`.

1. Scale: regress gd15 on the production BT logit over games before the first tune event
   (in-sample fit at that cutoff) -> c (gold per logit unit) and sigma; then frozen.
2. Backtest production BT and margin_bt for each margin_weight in the grid, both modes,
   all eight events, with the same rolling temperature calibration as the main backtest.
3. Select margin_weight on the tune events (pre_event, calibrated log loss).
4. Report tune-event Δ (decision input 1) and holdout Δ (report only); Worlds 2026 Swiss games
   (decision input 2) come from reports/predictions_log.csv after Round 5.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
import pandas as pd

from shyarly import paths
from shyarly.eval import metrics
from shyarly.eval.backtest import SIDE_SWAP, calibrate, event_order, make_queries, split_events
from shyarly.models.bt import BradleyTerry, BTParams
from shyarly.models.calibration import fit_temperature
from shyarly.models.common import day_start, training_view
from shyarly.models.predictor import bt_params_from_config

REPORT = paths.REPORTS_DIR / "shadow_margin.md"


def fit_margin_scale(games: pd.DataFrame, params: BTParams, before: pd.Timestamp) -> tuple[float, float, int]:
    """(c, sigma, n): gd15 ≈ c * z on games before `before`, z = in-sample production BT logit."""
    fit = BradleyTerry(replace(params, margin_weight=0.0)).fit(games, before)
    train = training_view(games, before, params.lookback_days)
    train = train[np.isfinite(train["gd15"].to_numpy(dtype=float))]
    z = np.array([fit.beta + fit.strength(bt, br) - fit.strength(rt, rr) for bt, br, rt, rr in
                  zip(train["blue_team_season_id"], train["blue_region"], train["red_team_season_id"], train["red_region"])])
    gd = train["gd15"].to_numpy(dtype=float)
    c = float(np.sum(gd * z) / np.sum(z * z))
    sigma = float(np.std(gd / c - z))
    return c, sigma, int(len(gd))


@dataclass
class ShadowResult:
    predictions: pd.DataFrame
    scale: dict
    selected_weight: float
    temperature: float
    table: pd.DataFrame          # per variant / split / mode: n, log loss
    deltas: pd.DataFrame         # selected margin_bt − bt, with bootstrap CIs


def _variant(prod: BTParams, weight: float, c: float, sigma: float) -> BTParams:
    return replace(prod, margin_weight=weight, margin_scale=c, margin_sigma=sigma)


def run_shadow(games: pd.DataFrame, cfg: dict) -> ShadowResult:
    reg = cfg["shadow_models"]["margin_bt"]
    bt_cfg = cfg["backtest"]
    prod = bt_params_from_config(cfg)
    events = split_events(cfg)
    order = event_order(games, events)
    first_tune = event_order(games, bt_cfg["tune_events"])[0]
    before = day_start(games.loc[games["scored"] & (games["event"] == first_tune), "date_utc"].min())
    c, sigma, n = fit_margin_scale(games, prod, before)
    scale = {"fit_before": f"{before:%Y-%m-%d}", "c_gold_per_logit": round(c, 1), "sigma_logit": round(sigma, 4),
             "n_games": n}

    split_of = {e: "tune" for e in bt_cfg["tune_events"]} | {e: "holdout" for e in bt_cfg["holdout_events"]}
    variants = {"bt": prod} | {f"margin_bt_w{w}": _variant(prod, w, c, sigma) for w in reg["grid"]["margin_weight"]}
    preds = []
    for name, params in variants.items():
        model = BradleyTerry(params)
        for mode in bt_cfg["modes"]:
            q = make_queries(games, events, mode)
            both = model.predict(games, pd.concat([q, q.rename(columns=SIDE_SWAP)], ignore_index=True))
            preds.append(pd.DataFrame({
                "model": name, "mode": mode, "split": q["event"].map(split_of), "event": q["event"],
                "game_id": q["game_id"], "series_id": q["series_id"], "y": q["blue_win"].astype(int),
                "p_raw": both[: len(q)], "p_swap_raw": both[len(q):],
            }))
    preds = calibrate(pd.concat(preds, ignore_index=True), order)

    def ll(model, split, mode):
        d = preds[(preds["model"] == model) & (preds["split"] == split) & (preds["mode"] == mode)]
        return metrics.log_loss(d["y"], d["p_cal"]), len(d)

    rows = [{"model": m, "split": s, "mode": md, "log_loss": ll(m, s, md)[0], "n": ll(m, s, md)[1]}
            for m in variants for s in ("tune", "holdout") for md in bt_cfg["modes"]]
    table = pd.DataFrame(rows)
    tune_pe = table[(table["split"] == "tune") & (table["mode"] == "pre_event") & (table["model"] != "bt")]
    best = tune_pe.loc[tune_pe["log_loss"].idxmin(), "model"]          # first minimum on ties
    weight = float(best.removeprefix("margin_bt_w"))

    deltas = []
    for split in ("tune", "holdout"):
        for mode in bt_cfg["modes"]:
            d = preds[(preds["split"] == split) & (preds["mode"] == mode)]
            ref = d[d["model"] == "bt"].set_index("game_id")
            oth = d[d["model"] == best].set_index("game_id").loc[ref.index]
            res = metrics.paired_bootstrap(metrics.game_log_loss(oth["y"], oth["p_cal"]),
                                           metrics.game_log_loss(ref["y"], ref["p_cal"]), ref["series_id"],
                                           resamples=bt_cfg["bootstrap_resamples"], seed=cfg["seed"])
            deltas.append({"split": split, "mode": mode, "n": len(ref), "Δ log loss (margin − bt)": res["mean"],
                           "95% CI": f"[{res['ci95'][0]:+.4f}, {res['ci95'][1]:+.4f}]"})
    live = preds[(preds["model"] == best) & (preds["mode"] == "live")]
    temperature = fit_temperature(live["y"], live["p_raw"])
    return ShadowResult(preds, scale, weight, temperature, table, pd.DataFrame(deltas))


def shadow_markdown(res: ShadowResult, cfg: dict) -> str:
    reg = cfg["shadow_models"]["margin_bt"]
    tune = res.deltas[(res.deltas["split"] == "tune") & (res.deltas["mode"] == "pre_event")].iloc[0]
    passed = tune["Δ log loss (margin − bt)"] < 0
    pivot = res.table.pivot_table(index="model", columns=["split", "mode"], values="log_loss")
    lines = [
        "# Shadow model margin_bt (gold difference at 15:00)", "",
        f"Pre-registered in `config/model.yaml` (`shadow_models.margin_bt`, {reg['registered']}).", "",
        f"- **Scale (frozen):** c = {res.scale['c_gold_per_logit']:.0f} gold per logit unit, sigma = "
        f"{res.scale['sigma_logit']:.3f}, fitted on {res.scale['n_games']:,} games before {res.scale['fit_before']}.",
        f"- **Selected margin_weight:** {res.selected_weight} (lower tune pre_event calibrated log loss).",
        f"- **Temperature for live use:** {res.temperature:.4f} (fitted like production on all eight events).",
        f"- **Decision input 1, tune events (pre_event, calibrated):** Δ = {tune['Δ log loss (margin − bt)']:+.4f} "
        f"{tune['95% CI']} -> {'passes' if passed else 'fails'} (needs Δ < 0).",
        "- **Decision input 2:** Worlds 2026 Swiss games (live), from `reports/predictions_log.csv`, after Round 5.",
        "- **Holdout rows are report only** (holdout look 4 in `config/model.yaml`).", "",
        "## Calibrated log loss", "", "| model | " + " | ".join(f"{s} {m}" for s, m in pivot.columns) + " |",
        "|---|" + "---|" * len(pivot.columns),
    ]
    for model, row in pivot.iterrows():
        lines.append(f"| {model} | " + " | ".join(f"{v:.4f}" for v in row) + " |")
    lines += ["", "## Selected margin_bt − production BT (paired bootstrap by series)", "",
              "| split | mode | games | Δ log loss | 95% CI |", "|---|---|---|---|---|"]
    for d in res.deltas.to_dict("records"):
        lines.append(f"| {d['split']} | {d['mode']} | {d['n']} | {d['Δ log loss (margin − bt)']:+.4f} | {d['95% CI']} |")
    lines.append("")
    return "\n".join(lines)
