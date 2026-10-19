# Shadow model margin_bt (gold difference at 15:00)

Pre-registered in `config/model.yaml` (`shadow_models.margin_bt`, 2026-10-19, before any Worlds 2026 Swiss game was played).

- **Scale (frozen):** c = 2451 gold per logit unit, sigma = 1.171, fitted on 3,050 games before 2022-09-29.
- **Selected margin_weight:** 0.5 (lower tune pre_event calibrated log loss).
- **Temperature for live use:** 0.7049 (fitted like production on all eight events).
- **Decision input 1, tune events (pre_event, calibrated):** Δ = -0.0024 [-0.0194, +0.0153] -> passes (needs Δ < 0).
- **Decision input 2:** Worlds 2026 Swiss games (live), from `reports/predictions_log.csv`, after Round 5.
- **Holdout rows are report only** (holdout look 4 in `config/model.yaml`).

## Calibrated log loss

| model | holdout live | holdout pre_event | tune live | tune pre_event |
|---|---|---|---|---|
| bt | 0.6199 | 0.6176 | 0.5691 | 0.5731 |
| margin_bt_w0.5 | 0.6184 | 0.6155 | 0.5692 | 0.5707 |
| margin_bt_w1.0 | 0.6188 | 0.6168 | 0.5756 | 0.5777 |

## Selected margin_bt − production BT (paired bootstrap by series)

| split | mode | games | Δ log loss | 95% CI |
|---|---|---|---|---|
| tune | live | 316 | +0.0001 | [-0.0174, +0.0181] |
| tune | pre_event | 316 | -0.0024 | [-0.0194, +0.0153] |
| holdout | live | 419 | -0.0015 | [-0.0094, +0.0064] |
| holdout | pre_event | 419 | -0.0021 | [-0.0100, +0.0057] |
