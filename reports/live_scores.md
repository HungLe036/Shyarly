# Worlds 2026 live scoring

Predictions logged before each match (`reports/predictions_log.csv`, first row per match and mode counts), scored on played matches from `data/manual/worlds2026_results.csv`. pre_event = model trained on games before the stage started; live = before the match day. A coin flip scores 0.693 log loss and 0.25 Brier.

| model | mode | series | series log loss | series Brier | games | game log loss |
|---|---|---|---|---|---|---|
| bt | pre_event | 16 | 0.5766 | 0.1964 | 16 | 0.5766 |
| bt | live | 16 | 0.5805 | 0.1982 | 16 | 0.5805 |
| margin_bt | pre_event | 16 | 0.5960 | 0.2050 | 16 | 0.5960 |
| margin_bt | live | 16 | 0.5989 | 0.2064 | 16 | 0.5989 |

## Logged predictions

| match | Bo | model | mode | P(team_a wins) | result | logged at | commit |
|---|---|---|---|---|---|---|---|
| GEN vs IG (swiss 1) | 1 | bt | pre_event | 64.3% | GEN 1-0 | 2026-10-24T01:59 | inf |
| GEN vs IG (swiss 1) | 1 | bt | live | 64.3% | GEN 1-0 | 2026-10-24T01:59 | inf |
| HLE vs TES (swiss 1) | 1 | bt | pre_event | 64.3% | HLE 1-0 | 2026-10-24T01:59 | inf |
| HLE vs TES (swiss 1) | 1 | bt | live | 64.3% | HLE 1-0 | 2026-10-24T01:59 | inf |
| BLG vs MKOI (swiss 1) | 1 | bt | pre_event | 65.4% | MKOI 0-1 | 2026-10-24T01:59 | inf |
| BLG vs MKOI (swiss 1) | 1 | bt | live | 65.4% | MKOI 0-1 | 2026-10-24T01:59 | inf |
| AL vs KC (swiss 1) | 1 | bt | pre_event | 57.7% | KC 0-1 | 2026-10-24T01:59 | inf |
| AL vs KC (swiss 1) | 1 | bt | live | 57.7% | KC 0-1 | 2026-10-24T01:59 | inf |
| G2 vs DK (swiss 1) | 1 | bt | pre_event | 32.5% | DK 0-1 | 2026-10-24T01:59 | inf |
| G2 vs DK (swiss 1) | 1 | bt | live | 32.5% | DK 0-1 | 2026-10-24T01:59 | inf |
| TLAW vs CFO (swiss 1) | 1 | bt | pre_event | 63.8% | CFO 0-1 | 2026-10-24T01:59 | inf |
| TLAW vs CFO (swiss 1) | 1 | bt | live | 63.8% | CFO 0-1 | 2026-10-24T01:59 | inf |
| LOS vs T1 (swiss 1) | 1 | bt | pre_event | 16.4% | T1 0-1 | 2026-10-24T01:59 | inf |
| LOS vs T1 (swiss 1) | 1 | bt | live | 16.4% | T1 0-1 | 2026-10-24T01:59 | inf |
| TSW vs LYON (swiss 1) | 1 | bt | pre_event | 39.4% | LYON 0-1 | 2026-10-24T01:59 | inf |
| TSW vs LYON (swiss 1) | 1 | bt | live | 40.6% | LYON 0-1 | 2026-10-24T01:59 | inf |
| TES vs AL (swiss 2) | 1 | bt | pre_event | 48.1% | TES 1-0 | 2026-10-24T01:59 | inf |
| TES vs AL (swiss 2) | 1 | bt | live | 48.2% | TES 1-0 | 2026-10-24T01:59 | inf |
| CFO vs MKOI (swiss 2) | 1 | bt | pre_event | 31.1% | MKOI 0-1 | 2026-10-24T01:59 | inf |
| CFO vs MKOI (swiss 2) | 1 | bt | live | 31.3% | MKOI 0-1 | 2026-10-24T01:59 | inf |
| G2 vs LOS (swiss 2) | 1 | bt | pre_event | 71.2% | G2 1-0 | 2026-10-24T01:59 | inf |
| G2 vs LOS (swiss 2) | 1 | bt | live | 71.8% | G2 1-0 | 2026-10-24T01:59 | inf |
| LYON vs KC (swiss 2) | 1 | bt | pre_event | 40.8% | LYON 1-0 | 2026-10-24T01:59 | inf |
| LYON vs KC (swiss 2) | 1 | bt | live | 39.5% | LYON 1-0 | 2026-10-24T01:59 | inf |
| IG vs TSW (swiss 2) | 1 | bt | pre_event | 74.9% | IG 1-0 | 2026-10-24T01:59 | inf |
| IG vs TSW (swiss 2) | 1 | bt | live | 73.5% | IG 1-0 | 2026-10-24T01:59 | inf |
| HLE vs GEN (swiss 2) | 1 | bt | pre_event | 48.6% | GEN 0-1 | 2026-10-24T01:59 | inf |
| HLE vs GEN (swiss 2) | 1 | bt | live | 48.5% | GEN 0-1 | 2026-10-24T01:59 | inf |
| DK vs T1 (swiss 2) | 1 | bt | pre_event | 50.2% | DK 1-0 | 2026-10-24T01:59 | inf |
| DK vs T1 (swiss 2) | 1 | bt | live | 50.3% | DK 1-0 | 2026-10-24T01:59 | inf |
| TLAW vs BLG (swiss 2) | 1 | bt | pre_event | 29.6% | BLG 0-1 | 2026-10-24T01:59 | inf |
| GEN vs IG (swiss 1) | 1 | margin_bt | pre_event | 62.4% | GEN 1-0 | 2026-10-24T01:59 | inf |
| GEN vs IG (swiss 1) | 1 | margin_bt | live | 62.4% | GEN 1-0 | 2026-10-24T01:59 | inf |
| HLE vs TES (swiss 1) | 1 | margin_bt | pre_event | 66.8% | HLE 1-0 | 2026-10-24T01:59 | inf |
| HLE vs TES (swiss 1) | 1 | margin_bt | live | 66.8% | HLE 1-0 | 2026-10-24T01:59 | inf |
| BLG vs MKOI (swiss 1) | 1 | margin_bt | pre_event | 70.9% | MKOI 0-1 | 2026-10-24T01:59 | inf |
| BLG vs MKOI (swiss 1) | 1 | margin_bt | live | 70.9% | MKOI 0-1 | 2026-10-24T01:59 | inf |
| AL vs KC (swiss 1) | 1 | margin_bt | pre_event | 58.2% | KC 0-1 | 2026-10-24T01:59 | inf |
| AL vs KC (swiss 1) | 1 | margin_bt | live | 58.2% | KC 0-1 | 2026-10-24T01:59 | inf |
| G2 vs DK (swiss 1) | 1 | margin_bt | pre_event | 32.2% | DK 0-1 | 2026-10-24T01:59 | inf |
| G2 vs DK (swiss 1) | 1 | margin_bt | live | 32.2% | DK 0-1 | 2026-10-24T01:59 | inf |
| TLAW vs CFO (swiss 1) | 1 | margin_bt | pre_event | 64.5% | CFO 0-1 | 2026-10-24T01:59 | inf |
| TLAW vs CFO (swiss 1) | 1 | margin_bt | live | 64.5% | CFO 0-1 | 2026-10-24T01:59 | inf |
| LOS vs T1 (swiss 1) | 1 | margin_bt | pre_event | 17.3% | T1 0-1 | 2026-10-24T01:59 | inf |
| LOS vs T1 (swiss 1) | 1 | margin_bt | live | 17.3% | T1 0-1 | 2026-10-24T01:59 | inf |
| TSW vs LYON (swiss 1) | 1 | margin_bt | pre_event | 41.2% | LYON 0-1 | 2026-10-24T01:59 | inf |
| TSW vs LYON (swiss 1) | 1 | margin_bt | live | 42.0% | LYON 0-1 | 2026-10-24T01:59 | inf |
| TES vs AL (swiss 2) | 1 | margin_bt | pre_event | 45.9% | TES 1-0 | 2026-10-24T01:59 | inf |
| TES vs AL (swiss 2) | 1 | margin_bt | live | 46.0% | TES 1-0 | 2026-10-24T01:59 | inf |
| CFO vs MKOI (swiss 2) | 1 | margin_bt | pre_event | 33.6% | MKOI 0-1 | 2026-10-24T01:59 | inf |
| CFO vs MKOI (swiss 2) | 1 | margin_bt | live | 33.5% | MKOI 0-1 | 2026-10-24T01:59 | inf |
| G2 vs LOS (swiss 2) | 1 | margin_bt | pre_event | 70.1% | G2 1-0 | 2026-10-24T01:59 | inf |
| G2 vs LOS (swiss 2) | 1 | margin_bt | live | 70.9% | G2 1-0 | 2026-10-24T01:59 | inf |
| LYON vs KC (swiss 2) | 1 | margin_bt | pre_event | 42.6% | LYON 1-0 | 2026-10-24T01:59 | inf |
| LYON vs KC (swiss 2) | 1 | margin_bt | live | 41.5% | LYON 1-0 | 2026-10-24T01:59 | inf |
| IG vs TSW (swiss 2) | 1 | margin_bt | pre_event | 73.1% | IG 1-0 | 2026-10-24T01:59 | inf |
| IG vs TSW (swiss 2) | 1 | margin_bt | live | 72.1% | IG 1-0 | 2026-10-24T01:59 | inf |
| HLE vs GEN (swiss 2) | 1 | margin_bt | pre_event | 50.4% | GEN 0-1 | 2026-10-24T01:59 | inf |
| HLE vs GEN (swiss 2) | 1 | margin_bt | live | 50.5% | GEN 0-1 | 2026-10-24T01:59 | inf |
| DK vs T1 (swiss 2) | 1 | margin_bt | pre_event | 50.8% | DK 1-0 | 2026-10-24T01:59 | inf |
| DK vs T1 (swiss 2) | 1 | margin_bt | live | 50.6% | DK 1-0 | 2026-10-24T01:59 | inf |
| TLAW vs BLG (swiss 2) | 1 | margin_bt | pre_event | 27.4% | BLG 0-1 | 2026-10-24T01:59 | inf |
| TLAW vs BLG (swiss 2) | 1 | bt | live | 29.8% | BLG 0-1 | 2026-10-25T12:12 | ea97c2d |
| HLE vs IG (swiss 3) | 1 | bt | pre_event | 63.0% | — | 2026-10-25T12:12 | ea97c2d |
| HLE vs IG (swiss 3) | 1 | bt | live | 64.5% | — | 2026-10-25T12:12 | ea97c2d |
| TES vs G2 (swiss 3) | 1 | bt | pre_event | 54.4% | — | 2026-10-25T12:12 | ea97c2d |
| TES vs G2 (swiss 3) | 1 | bt | live | 53.0% | — | 2026-10-25T12:12 | ea97c2d |
| DK vs MKOI (swiss 3) | 3 | bt | pre_event | 79.9% | — | 2026-10-25T12:12 | ea97c2d |
| DK vs MKOI (swiss 3) | 3 | bt | live | 79.4% | — | 2026-10-25T12:12 | ea97c2d |
| GEN vs LYON (swiss 3) | 3 | bt | pre_event | 87.4% | — | 2026-10-25T12:12 | ea97c2d |
| GEN vs LYON (swiss 3) | 3 | bt | live | 88.1% | — | 2026-10-25T12:12 | ea97c2d |
| T1 vs CFO (swiss 3) | 1 | bt | pre_event | 84.5% | — | 2026-10-25T12:12 | ea97c2d |
| KC vs BLG (swiss 3) | 1 | bt | pre_event | 37.3% | — | 2026-10-25T12:12 | ea97c2d |
| LOS vs TSW (swiss 3) | 3 | bt | pre_event | 48.3% | — | 2026-10-25T12:12 | ea97c2d |
| TLAW vs AL (swiss 3) | 3 | bt | pre_event | 27.0% | — | 2026-10-25T12:12 | ea97c2d |
| TLAW vs BLG (swiss 2) | 1 | margin_bt | live | 27.5% | BLG 0-1 | 2026-10-25T12:12 | ea97c2d |
| HLE vs IG (swiss 3) | 1 | margin_bt | pre_event | 62.7% | — | 2026-10-25T12:12 | ea97c2d |
| HLE vs IG (swiss 3) | 1 | margin_bt | live | 63.5% | — | 2026-10-25T12:12 | ea97c2d |
| TES vs G2 (swiss 3) | 1 | margin_bt | pre_event | 53.3% | — | 2026-10-25T12:12 | ea97c2d |
| TES vs G2 (swiss 3) | 1 | margin_bt | live | 51.9% | — | 2026-10-25T12:12 | ea97c2d |
| DK vs MKOI (swiss 3) | 3 | margin_bt | pre_event | 81.6% | — | 2026-10-25T12:12 | ea97c2d |
| DK vs MKOI (swiss 3) | 3 | margin_bt | live | 80.2% | — | 2026-10-25T12:12 | ea97c2d |
| GEN vs LYON (swiss 3) | 3 | margin_bt | pre_event | 85.5% | — | 2026-10-25T12:12 | ea97c2d |
| GEN vs LYON (swiss 3) | 3 | margin_bt | live | 85.8% | — | 2026-10-25T12:12 | ea97c2d |
| T1 vs CFO (swiss 3) | 1 | margin_bt | pre_event | 83.6% | — | 2026-10-25T12:12 | ea97c2d |
| KC vs BLG (swiss 3) | 1 | margin_bt | pre_event | 33.3% | — | 2026-10-25T12:12 | ea97c2d |
| LOS vs TSW (swiss 3) | 3 | margin_bt | pre_event | 43.9% | — | 2026-10-25T12:12 | ea97c2d |
| TLAW vs AL (swiss 3) | 3 | margin_bt | pre_event | 28.5% | — | 2026-10-25T12:12 | ea97c2d |
