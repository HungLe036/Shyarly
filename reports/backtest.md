# Shyarly backtest

Generated 2026-10-18 02:26 UTC by `shyarly backtest`. Uncalibrated model outputs (temperature scaling and promotion gates are M3).

## Setup
- **Scored games:** Worlds/MSI main-event games (regional qualifiers excluded), blue-side win probability.
- **Tune events:** WLDs 2022, MSI 2023, WLDs 2023. **Holdout events:** MSI 2024, WLDs 2024, MSI 2025, WLDs 2025, MSI 2026.
- **Live mode:** each game uses only games before its UTC day. **Pre-event mode:** each event's games use only games before the event's first day.
- **Tuning:** grid search on the tune events in `pre_event` mode, log loss; params then frozen.
- **Bootstrap:** 2000 resamples of whole series (seed 20261018).
- **Data snapshot (OE sha256):** 2022: 76b7cb8afc (2026-10-18), 2023: efe4d2abe5 (2026-10-18), 2024: a2c67680a5 (2026-10-18), 2025: 882245414e (2026-10-18), 2026: 73b398ab52 (2026-10-18).

## Holdout — live mode (primary)
| model | n | log_loss | brier | ece | slope | accuracy |
|---|---|---|---|---|---|---|
| coinflip | 419 | 0.6931 | 0.2500 | 0.0656 | 1.0000 | 0.4344 |
| region_prior | 419 | 0.6237 | 0.2171 | 0.0279 | 0.8268 | 0.6539 |
| elo | 419 | 0.6391 | 0.2242 | 0.0590 | 0.7024 | 0.6444 |
| bt | 419 | 0.6251 | 0.2178 | 0.0351 | 0.8370 | 0.6539 |

## Holdout — pre-event mode
| model | n | log_loss | brier | ece | slope | accuracy |
|---|---|---|---|---|---|---|
| coinflip | 419 | 0.6931 | 0.2500 | 0.0656 | 1.0000 | 0.4344 |
| region_prior | 419 | 0.6207 | 0.2155 | 0.0258 | 0.8437 | 0.6492 |
| elo | 419 | 0.6347 | 0.2214 | 0.0725 | 0.7083 | 0.6659 |
| bt | 419 | 0.6189 | 0.2147 | 0.0479 | 0.8839 | 0.6635 |

## Tune events — live mode
| model | n | log_loss | brier | ece | slope | accuracy |
|---|---|---|---|---|---|---|
| coinflip | 316 | 0.6931 | 0.2500 | 0.0411 | 1.0000 | 0.4589 |
| region_prior | 316 | 0.5674 | 0.1934 | 0.0205 | 1.0357 | 0.7089 |
| elo | 316 | 0.5933 | 0.2044 | 0.0257 | 1.2432 | 0.7120 |
| bt | 316 | 0.5757 | 0.1969 | 0.0180 | 0.9801 | 0.6867 |

## Log loss per event — live mode
| event | split | n | coinflip | region_prior | elo | bt |
|---|---|---|---|---|---|---|
| MSI 2023 | tune | 76 | 0.6931 | 0.5530 | 0.5959 | 0.5718 |
| WLDs 2022 | tune | 127 | 0.6931 | 0.5397 | 0.5693 | 0.5410 |
| WLDs 2023 | tune | 113 | 0.6931 | 0.6083 | 0.6185 | 0.6173 |
| MSI 2024 | holdout | 78 | 0.6931 | 0.6206 | 0.6384 | 0.6099 |
| MSI 2025 | holdout | 80 | 0.6931 | 0.6457 | 0.6468 | 0.6380 |
| MSI 2026 | holdout | 71 | 0.6931 | 0.6111 | 0.6645 | 0.6185 |
| WLDs 2024 | holdout | 106 | 0.6931 | 0.6460 | 0.6563 | 0.6615 |
| WLDs 2025 | holdout | 84 | 0.6931 | 0.5883 | 0.5894 | 0.5863 |

## Paired bootstrap on holdout, live mode
Δ = BT log loss − other model's log loss; negative means BT is better.

| comparison | Δ log loss | 80% CI | 95% CI |
|---|---|---|---|
| bt − coinflip | -0.0681 | [-0.0945, -0.0437] | [-0.1074, -0.0307] |
| bt − region_prior | 0.0013 | [-0.0060, +0.0081] | [-0.0098, +0.0113] |
| bt − elo | -0.0141 | [-0.0241, -0.0037] | [-0.0291, +0.0012] |

## Tuned hyperparameters
- **coinflip** (tune log loss 0.6931, 1 configs): `{}`
- **region_prior** (tune log loss 0.5711, 96 configs): `{'half_life_days': 730, 'intl_weight': 1.0, 'lambda_region_major': 0.1, 'lambda_region_minor': 2.0}`
- **elo** (tune log loss 0.6217, 120 configs): `{'k_region': 32, 'k_team': 4, 'season_carry': 0.25}`
- **bt** (tune log loss 0.5851, 240 configs): `{'half_life_days': 180, 'intl_weight': 1.0, 'lambda_region_major': 0.1, 'lambda_region_minor': 0.1, 'lambda_team': 16.0}`
