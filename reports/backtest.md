# Shyarly backtest

Generated 2026-10-18 02:30 UTC by `shyarly backtest`. Uncalibrated model outputs (temperature scaling and promotion gates are M3).

## Setup
- **Scored games:** Worlds/MSI main-event games (regional qualifiers excluded), blue-side win probability.
- **Tune events:** WLDs 2022, MSI 2023, WLDs 2023. **Holdout events:** MSI 2024, WLDs 2024, MSI 2025, WLDs 2025, MSI 2026.
- **Live mode:** each game uses only games before its UTC day. **Pre-event mode:** each event's games use only games before the event's first day.
- **Tuning:** grid search on the tune events in `pre_event` mode, log loss; params then frozen.
- **BT decay:** domestic games use `half_life_days` (60–180 per spec); cross-region games use their own `half_life_intl_days`, so region offsets can keep a longer memory than team form (added after M2).
- **Bootstrap:** 2000 resamples of whole series (seed 20261018).
- **Data snapshot (OE sha256):** 2022: 76b7cb8afc (2026-10-18), 2023: efe4d2abe5 (2026-10-18), 2024: a2c67680a5 (2026-10-18), 2025: 882245414e (2026-10-18), 2026: 73b398ab52 (2026-10-18).

## Holdout — live mode (primary)
| model | n | log_loss | brier | ece | slope | accuracy |
|---|---|---|---|---|---|---|
| coinflip | 419 | 0.6931 | 0.2500 | 0.0656 | 1.0000 | 0.4344 |
| region_prior | 419 | 0.6237 | 0.2171 | 0.0279 | 0.8268 | 0.6539 |
| elo | 419 | 0.6391 | 0.2242 | 0.0590 | 0.7024 | 0.6444 |
| bt | 419 | 0.6189 | 0.2150 | 0.0349 | 0.8590 | 0.6444 |

## Holdout — pre-event mode
| model | n | log_loss | brier | ece | slope | accuracy |
|---|---|---|---|---|---|---|
| coinflip | 419 | 0.6931 | 0.2500 | 0.0656 | 1.0000 | 0.4344 |
| region_prior | 419 | 0.6207 | 0.2155 | 0.0258 | 0.8437 | 0.6492 |
| elo | 419 | 0.6347 | 0.2214 | 0.0725 | 0.7083 | 0.6659 |
| bt | 419 | 0.6170 | 0.2136 | 0.0395 | 0.8606 | 0.6539 |

## Tune events — live mode
| model | n | log_loss | brier | ece | slope | accuracy |
|---|---|---|---|---|---|---|
| coinflip | 316 | 0.6931 | 0.2500 | 0.0411 | 1.0000 | 0.4589 |
| region_prior | 316 | 0.5674 | 0.1934 | 0.0205 | 1.0357 | 0.7089 |
| elo | 316 | 0.5933 | 0.2044 | 0.0257 | 1.2432 | 0.7120 |
| bt | 316 | 0.5689 | 0.1945 | 0.0166 | 1.0315 | 0.6709 |

## Log loss per event — live mode
| event | split | n | coinflip | region_prior | elo | bt |
|---|---|---|---|---|---|---|
| MSI 2023 | tune | 76 | 0.6931 | 0.5530 | 0.5959 | 0.5553 |
| WLDs 2022 | tune | 127 | 0.6931 | 0.5397 | 0.5693 | 0.5420 |
| WLDs 2023 | tune | 113 | 0.6931 | 0.6083 | 0.6185 | 0.6082 |
| MSI 2024 | holdout | 78 | 0.6931 | 0.6206 | 0.6384 | 0.6188 |
| MSI 2025 | holdout | 80 | 0.6931 | 0.6457 | 0.6468 | 0.6371 |
| MSI 2026 | holdout | 71 | 0.6931 | 0.6111 | 0.6645 | 0.6031 |
| WLDs 2024 | holdout | 106 | 0.6931 | 0.6460 | 0.6563 | 0.6428 |
| WLDs 2025 | holdout | 84 | 0.6931 | 0.5883 | 0.5894 | 0.5850 |

## Paired bootstrap on holdout, live mode
Δ = BT log loss − other model's log loss; negative means BT is better.

| comparison | Δ log loss | 80% CI | 95% CI |
|---|---|---|---|
| bt − coinflip | -0.0742 | [-0.1001, -0.0491] | [-0.1150, -0.0350] |
| bt − region_prior | -0.0048 | [-0.0081, -0.0018] | [-0.0097, -0.0000] |
| bt − elo | -0.0202 | [-0.0324, -0.0070] | [-0.0380, -0.0016] |

## Tuned hyperparameters
- **coinflip** (tune log loss 0.6931, 1 configs): `{}`
- **region_prior** (tune log loss 0.5711, 96 configs): `{'half_life_days': 730, 'intl_weight': 1.0, 'lambda_region_major': 0.1, 'lambda_region_minor': 2.0}`
- **elo** (tune log loss 0.6217, 120 configs): `{'k_region': 32, 'k_team': 4, 'season_carry': 0.25}`
- **bt** (tune log loss 0.5729, 144 configs): `{'half_life_days': 60, 'half_life_intl_days': 730, 'intl_weight': 1.0, 'lambda_region_major': 0.1, 'lambda_region_minor': 2.0, 'lambda_team': 16.0}`
