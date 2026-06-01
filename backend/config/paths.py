"""Central paths for the NBA prediction backend."""
from __future__ import annotations

from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = BACKEND_ROOT / "data"
PLAYER_DATA_DIR = DATA_DIR / "player_data"
TEAM_STATS_DIR = DATA_DIR / "team_stats"
# Chris Munch–style Kaggle bundle: processed/teams_boxscores.csv (game-level, merged advanced stats)
KAGGLE_TEAM_DATA_DIR = DATA_DIR / "Kaggle NBA Team Data"
KAGGLE_TEAMS_BOXSCORES_CSV = KAGGLE_TEAM_DATA_DIR / "processed" / "teams_boxscores.csv"
ADVANCED_DATA_DIR = DATA_DIR / "advanced_nba"
ADVANCED_SQLITE_DB = ADVANCED_DATA_DIR / "nba.sqlite"
AUXILIARY_DIR = DATA_DIR / "auxiliary"
PROCESSED_DIR = DATA_DIR / "processed"
ARTIFACTS_DIR = BACKEND_ROOT / "models" / "artifacts"

for d in (TEAM_STATS_DIR, AUXILIARY_DIR, PROCESSED_DIR, ARTIFACTS_DIR, ADVANCED_DATA_DIR):
    d.mkdir(parents=True, exist_ok=True)

TEAM_STATS_CSV = TEAM_STATS_DIR / "all_team_stats.csv"
TEAM_STATS_PARQUET = TEAM_STATS_DIR / "all_team_stats.parquet"
# Game-level team box (Kaggle); prior-game features derived with shift(1) in feature_engine
KAGGLE_TEAM_GAMES_PARQUET = PROCESSED_DIR / "kaggle_team_games.parquet"
INJURIES_PARQUET = AUXILIARY_DIR / "injuries.parquet"
DVP_PARQUET = AUXILIARY_DIR / "defense_vs_position.parquet"
ARCHETYPE_PIPELINE_PATH = PROCESSED_DIR / "archetype_gmm.joblib"
FEATURE_COLUMNS_PATH = ARTIFACTS_DIR / "feature_columns.json"
GLOBAL_MODEL_DIR = ARTIFACTS_DIR / "global"
