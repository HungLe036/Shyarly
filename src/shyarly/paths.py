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
GAMES_PARQUET = PROCESSED_DIR / "games.parquet"
SERIES_PARQUET = PROCESSED_DIR / "series.parquet"
BACKTEST_PREDICTIONS = PROCESSED_DIR / "backtest_predictions.parquet"

REPORTS_DIR = ROOT / "reports"
BACKTEST_REPORT = REPORTS_DIR / "backtest.md"
