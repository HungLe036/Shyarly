"""Project paths. Every file location in the codebase is derived from here."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

CONFIG_DIR = ROOT / "config"
WORLDS_CONFIG = CONFIG_DIR / "worlds2026.yaml"
SOURCES_CONFIG = CONFIG_DIR / "sources.yaml"
LEAGUES_CONFIG = CONFIG_DIR / "leagues.yaml"
MODEL_CONFIG = CONFIG_DIR / "model.yaml"

DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
OE_RAW_DIR = RAW_DIR / "oe"
LEAGUEPEDIA_RAW_DIR = RAW_DIR / "leaguepedia"
RULES_RAW_DIR = RAW_DIR / "rules"
INTERIM_DIR = DATA_DIR / "interim"
PROCESSED_DIR = DATA_DIR / "processed"
MANUAL_DIR = DATA_DIR / "manual"
MANUAL_RESULTS = MANUAL_DIR / "worlds2026_results.csv"
GAMES_PARQUET = PROCESSED_DIR / "games.parquet"
SERIES_PARQUET = PROCESSED_DIR / "series.parquet"
BACKTEST_PREDICTIONS = PROCESSED_DIR / "backtest_predictions.parquet"

REPORTS_DIR = ROOT / "reports"
BACKTEST_REPORT = REPORTS_DIR / "backtest.md"
SWISS_ODDS_MD = REPORTS_DIR / "swiss_odds.md"
SWISS_ODDS_CSV = REPORTS_DIR / "swiss_odds.csv"
TOURNAMENT_ODDS_MD = REPORTS_DIR / "tournament_odds.md"
TOURNAMENT_ODDS_CSV = REPORTS_DIR / "tournament_odds.csv"
DAILY_DIR = REPORTS_DIR / "daily"                 # one odds snapshot per `shyarly update` day
PREDICTIONS_LOG = REPORTS_DIR / "predictions_log.csv"
LIVE_SCORES_MD = REPORTS_DIR / "live_scores.md"
MATCH_FORECASTS = REPORTS_DIR / "match_forecasts.csv"   # current forecast for unplayed matches (not scored)
