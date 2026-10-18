# Shyarly backtest

Generated 2026-10-18 02:53 UTC by `shyarly backtest`.

## Promotion gates — BT, holdout, live mode, calibrated
| gate | measured | threshold | pass |
|---|---|---|---|
| Beats 50/50 | log loss 0.6199; Δ 95% CI [-0.1131, -0.0347] | < 0.693, CI < 0 | PASS |
| Beats region prior | Δ -0.0046; 95% CI [-0.0091, -0.0001] | Δ < 0, 95% CI < 0 | PASS |
| Not worse than Elo | Δ -0.0240; 80% CI upper -0.0113 | Δ < 0, 80% upper ≤ +0.005 | PASS |
| ECE (5 bins) | 0.0441 | ≤ 0.05 | PASS |
| Calibration slope | 0.886 | in [0.8, 1.2] | PASS |

### Gate changes
- 2026-10-18: 'Beats region prior' revised from 'Δ ≤ −0.010 with 95% CI excluding 0' to 'Δ < 0 with 95% CI excluding 0, Δ reported'. Decided by the user after three holdout looks (see holdout_looks), on a review finding that team-level information adds about 0.005 log loss at best at international events, including on 308 non-holdout cross-region games.

## Production calibration
Temperature for new predictions (fitted on BT live predictions for all 8 backtest events): **T = 1.0502**. Copy into `config/model.yaml` `production.temperature` when it changes.

## Holdout use
The holdout has been looked at during development, and the final setup was chosen after those looks, so holdout scores are optimistically biased. Looks so far: 3.

1. M2 (38a6cba): one shared half-life. BT 0.6251 vs region prior 0.6237 (holdout live log loss).
2. e2fbbcb: separate cross-region half-life. BT 0.6189 vs region prior 0.6237.
3. Widened grids (lambda_team to 64, half-lives 30–1460 days, 4-year lookback), discarded: BT 0.6238 vs region prior 0.6252. The e2fbbcb setup was kept after seeing this.

## Setup
- **Scored games:** Worlds/MSI main-event games (regional qualifiers excluded), blue-side win probability.
- **Tune events:** WLDs 2022, MSI 2023, WLDs 2023. **Holdout events:** MSI 2024, WLDs 2024, MSI 2025, WLDs 2025, MSI 2026.
- **Live mode:** each game uses only games before its UTC day. **Pre-event mode:** each event's games use only games before the event's first day.
- **Tuning:** grid search on the tune events in `pre_event` mode, log loss; params then frozen.
- **BT decay:** domestic games use `half_life_days` (60–180 per spec); cross-region games use their own `half_life_intl_days`, so region offsets can keep a longer memory than team form (added after M2).
- **Calibration:** one temperature on the logit, no intercept. Each event's T is fitted on the same model's predictions for all earlier events (T = 1 for the first event), so it never sees the outcomes it is scored on.
- **Bootstrap:** 2000 resamples of whole series (seed 20261018).
- **Data snapshot (OE sha256):** 2022: 76b7cb8afc (2026-10-18), 2023: efe4d2abe5 (2026-10-18), 2024: a2c67680a5 (2026-10-18), 2025: 882245414e (2026-10-18), 2026: 73b398ab52 (2026-10-18).

## Holdout — live mode, calibrated (primary)
| model | n | log_loss | brier | ece | slope | accuracy |
|---|---|---|---|---|---|---|
| coinflip | 419 | 0.6931 | 0.2500 | 0.0656 | 1.0000 | 0.4344 |
| region_prior | 419 | 0.6246 | 0.2173 | 0.0270 | 0.8544 | 0.6539 |
| elo | 419 | 0.6440 | 0.2257 | 0.0495 | 0.6598 | 0.6444 |
| bt | 419 | 0.6199 | 0.2152 | 0.0441 | 0.8856 | 0.6444 |

## Holdout — live mode, raw (before temperature scaling)
| model | n | log_loss | brier | ece | slope | accuracy |
|---|---|---|---|---|---|---|
| coinflip | 419 | 0.6931 | 0.2500 | 0.0656 | 1.0000 | 0.4344 |
| region_prior | 419 | 0.6237 | 0.2171 | 0.0279 | 0.8268 | 0.6539 |
| elo | 419 | 0.6391 | 0.2242 | 0.0590 | 0.7024 | 0.6444 |
| bt | 419 | 0.6189 | 0.2150 | 0.0349 | 0.8590 | 0.6444 |

## Holdout — pre-event mode, calibrated
| model | n | log_loss | brier | ece | slope | accuracy |
|---|---|---|---|---|---|---|
| coinflip | 419 | 0.6931 | 0.2500 | 0.0656 | 1.0000 | 0.4344 |
| region_prior | 419 | 0.6213 | 0.2156 | 0.0355 | 0.8936 | 0.6492 |
| elo | 419 | 0.6310 | 0.2204 | 0.0582 | 0.8127 | 0.6659 |
| bt | 419 | 0.6176 | 0.2138 | 0.0440 | 0.9210 | 0.6539 |

## Tune events — live mode, calibrated
The first tune event has T = 1, so these rows are close to raw.

| model | n | log_loss | brier | ece | slope | accuracy |
|---|---|---|---|---|---|---|
| coinflip | 316 | 0.6931 | 0.2500 | 0.0411 | 1.0000 | 0.4589 |
| region_prior | 316 | 0.5677 | 0.1935 | 0.0225 | 1.0136 | 0.7089 |
| elo | 316 | 0.5952 | 0.2049 | 0.0210 | 0.9710 | 0.7120 |
| bt | 316 | 0.5691 | 0.1946 | 0.0142 | 1.0157 | 0.6709 |

## Log loss per event — live mode, calibrated
| event | split | n | T (bt) | coinflip | region_prior | elo | bt |
|---|---|---|---|---|---|---|---|
| MSI 2023 | tune | 76 | 1.0158 | 0.6931 | 0.5532 | 0.5952 | 0.5558 |
| WLDs 2022 | tune | 127 | 1.0000 | 0.6931 | 0.5397 | 0.5693 | 0.5420 |
| WLDs 2023 | tune | 113 | 0.9436 | 0.6931 | 0.6089 | 0.6244 | 0.6085 |
| MSI 2024 | holdout | 78 | 0.9641 | 0.6931 | 0.6243 | 0.6673 | 0.6220 |
| MSI 2025 | holdout | 80 | 1.1059 | 0.6931 | 0.6441 | 0.6452 | 0.6363 |
| MSI 2026 | holdout | 71 | 1.0555 | 0.6931 | 0.6112 | 0.6600 | 0.6033 |
| WLDs 2024 | holdout | 106 | 1.0701 | 0.6931 | 0.6438 | 0.6576 | 0.6409 |
| WLDs 2025 | holdout | 84 | 1.1080 | 0.6931 | 0.5933 | 0.5902 | 0.5900 |

## Paired bootstrap on holdout, live mode, calibrated
Δ = BT log loss − other model's log loss; negative means BT is better.

| comparison | Δ log loss | 80% CI | 95% CI |
|---|---|---|---|
| bt − coinflip | -0.0732 | [-0.0985, -0.0485] | [-0.1131, -0.0347] |
| bt − region_prior | -0.0046 | [-0.0077, -0.0018] | [-0.0091, -0.0001] |
| bt − elo | -0.0240 | [-0.0363, -0.0113] | [-0.0416, -0.0058] |

## Series — holdout, live mode
Series win probability from the first game's side-averaged calibrated p and the closed-form Bo1/Bo3/Bo5 formula (games treated as independent). Best-of is inferred from the winner's game count.

| best_of | model | n_series | log_loss | brier |
|---|---|---|---|---|
| 1 | coinflip | 40 | 0.6931 | 0.2500 |
| 1 | region_prior | 40 | 0.5357 | 0.1790 |
| 1 | elo | 40 | 0.5408 | 0.1829 |
| 1 | bt | 40 | 0.5185 | 0.1716 |
| 3 | coinflip | 46 | 0.6931 | 0.2500 |
| 3 | region_prior | 46 | 0.5134 | 0.1751 |
| 3 | elo | 46 | 0.4930 | 0.1683 |
| 3 | bt | 46 | 0.5001 | 0.1700 |
| 3 | bt, game p only | 46 | 0.5245 | 0.1758 |
| 5 | coinflip | 68 | 0.6931 | 0.2500 |
| 5 | region_prior | 68 | 0.5279 | 0.1759 |
| 5 | elo | 68 | 0.6037 | 0.2003 |
| 5 | bt | 68 | 0.5190 | 0.1737 |
| 5 | bt, game p only | 68 | 0.5436 | 0.1820 |

## Tuned hyperparameters
- **coinflip** (tune log loss 0.6931, 1 configs): `{}`
- **region_prior** (tune log loss 0.5711, 96 configs): `{'half_life_days': 730, 'intl_weight': 1.0, 'lambda_region_major': 0.1, 'lambda_region_minor': 2.0}`
- **elo** (tune log loss 0.6217, 120 configs): `{'k_region': 32, 'k_team': 4, 'season_carry': 0.25}`
- **bt** (tune log loss 0.5729, 144 configs): `{'half_life_days': 60, 'half_life_intl_days': 730, 'intl_weight': 1.0, 'lambda_region_major': 0.1, 'lambda_region_minor': 2.0, 'lambda_team': 16.0}`
