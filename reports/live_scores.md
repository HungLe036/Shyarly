# Worlds 2026 live scoring

Predictions logged before each match (`reports/predictions_log.csv`, first row per match and mode counts), scored on played matches from `data/manual/worlds2026_results.csv`. pre_event = model trained on games before the stage started; live = before the match day. A coin flip scores 0.693 log loss and 0.25 Brier.

| model | mode | series | series log loss | series Brier | games | game log loss |
|---|---|---|---|---|---|---|
| bt | pre_event | 8 | 0.6115 | 0.2137 | 8 | 0.6115 |
| bt | live | 8 | 0.6140 | 0.2149 | 8 | 0.6140 |
| margin_bt | pre_event | 8 | 0.6408 | 0.2265 | 8 | 0.6408 |
| margin_bt | live | 8 | 0.6426 | 0.2274 | 8 | 0.6426 |

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
| TES vs AL (swiss 2) | 1 | bt | pre_event | 48.1% | — | 2026-10-24T01:59 | inf |
| TES vs AL (swiss 2) | 1 | bt | live | 48.2% | — | 2026-10-24T01:59 | inf |
| CFO vs MKOI (swiss 2) | 1 | bt | pre_event | 31.1% | — | 2026-10-24T01:59 | inf |
| CFO vs MKOI (swiss 2) | 1 | bt | live | 31.3% | — | 2026-10-24T01:59 | inf |
| G2 vs LOS (swiss 2) | 1 | bt | pre_event | 71.2% | — | 2026-10-24T01:59 | inf |
| G2 vs LOS (swiss 2) | 1 | bt | live | 71.8% | — | 2026-10-24T01:59 | inf |
| LYON vs KC (swiss 2) | 1 | bt | pre_event | 40.8% | — | 2026-10-24T01:59 | inf |
| LYON vs KC (swiss 2) | 1 | bt | live | 39.5% | — | 2026-10-24T01:59 | inf |
| IG vs TSW (swiss 2) | 1 | bt | pre_event | 74.9% | — | 2026-10-24T01:59 | inf |
| IG vs TSW (swiss 2) | 1 | bt | live | 73.5% | — | 2026-10-24T01:59 | inf |
| HLE vs GEN (swiss 2) | 1 | bt | pre_event | 48.6% | — | 2026-10-24T01:59 | inf |
| HLE vs GEN (swiss 2) | 1 | bt | live | 48.5% | — | 2026-10-24T01:59 | inf |
| DK vs T1 (swiss 2) | 1 | bt | pre_event | 50.2% | — | 2026-10-24T01:59 | inf |
| DK vs T1 (swiss 2) | 1 | bt | live | 50.3% | — | 2026-10-24T01:59 | inf |
| TLAW vs BLG (swiss 2) | 1 | bt | pre_event | 29.6% | — | 2026-10-24T01:59 | inf |
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
| TES vs AL (swiss 2) | 1 | margin_bt | pre_event | 45.9% | — | 2026-10-24T01:59 | inf |
| TES vs AL (swiss 2) | 1 | margin_bt | live | 46.0% | — | 2026-10-24T01:59 | inf |
| CFO vs MKOI (swiss 2) | 1 | margin_bt | pre_event | 33.6% | — | 2026-10-24T01:59 | inf |
| CFO vs MKOI (swiss 2) | 1 | margin_bt | live | 33.5% | — | 2026-10-24T01:59 | inf |
| G2 vs LOS (swiss 2) | 1 | margin_bt | pre_event | 70.1% | — | 2026-10-24T01:59 | inf |
| G2 vs LOS (swiss 2) | 1 | margin_bt | live | 70.9% | — | 2026-10-24T01:59 | inf |
| LYON vs KC (swiss 2) | 1 | margin_bt | pre_event | 42.6% | — | 2026-10-24T01:59 | inf |
| LYON vs KC (swiss 2) | 1 | margin_bt | live | 41.5% | — | 2026-10-24T01:59 | inf |
| IG vs TSW (swiss 2) | 1 | margin_bt | pre_event | 73.1% | — | 2026-10-24T01:59 | inf |
| IG vs TSW (swiss 2) | 1 | margin_bt | live | 72.1% | — | 2026-10-24T01:59 | inf |
| HLE vs GEN (swiss 2) | 1 | margin_bt | pre_event | 50.4% | — | 2026-10-24T01:59 | inf |
| HLE vs GEN (swiss 2) | 1 | margin_bt | live | 50.5% | — | 2026-10-24T01:59 | inf |
| DK vs T1 (swiss 2) | 1 | margin_bt | pre_event | 50.8% | — | 2026-10-24T01:59 | inf |
| DK vs T1 (swiss 2) | 1 | margin_bt | live | 50.6% | — | 2026-10-24T01:59 | inf |
| TLAW vs BLG (swiss 2) | 1 | margin_bt | pre_event | 27.4% | — | 2026-10-24T01:59 | inf |
