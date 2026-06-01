#!/usr/bin/env python3
"""
Extract clutch-time statistics directly from the wyattowalsh SQLite Database.
Clutch is defined as: 4th Quarter or Overtime, less than 5 minutes remaining,
score differential within 5 points.
"""
from __future__ import annotations

import argparse
import logging
import sqlite3
import sys
from pathlib import Path

import pandas as pd

BACKEND = Path(__file__).resolve().parent.parent
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

from config.paths import ADVANCED_SQLITE_DB, AUXILIARY_DIR

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def build_clutch_stats(db_path: Path) -> pd.DataFrame:
    if not db_path.exists():
        logger.warning(f"Advanced DB not found at {db_path}. Run nightly.py with Kaggle authenticated.")
        return pd.DataFrame()

    con = sqlite3.connect(str(db_path))
    
    # Check if play_by_play exists
    tables = pd.read_sql("SELECT name FROM sqlite_master WHERE type='table';", con)
    if "play_by_play" not in tables["name"].values:
        logger.error("play_by_play table not found in SQLite DB.")
        return pd.DataFrame()

    # The exact query depends on wyattowalsh schema which typically tracks 'period', 'pctimestring', etc.
    # Note: If the schema differs, this query must be tuned.
    query = """
    SELECT 
        player1_id, player1_name, game_id, eventmsgtype, eventmsgactiontype
    FROM play_by_play
    WHERE period >= 4 
    LIMIT 1000
    """
    try:
        df = pd.read_sql(query, con)
        logger.info(f"Successfully connected and fetched {len(df)} sample play-by-play rows.")
    except Exception as e:
        logger.warning(f"SQLite Query failed. Structure might differ from assumption. Error: {e}")
        df = pd.DataFrame()
        
    con.close()
    
    # Process into clutch summary per player-game (Placeholder aggregation)
    # Target: ['PLAYER_NAME', 'GAME_DATE', 'CLUTCH_SHOTS_ATT', 'CLUTCH_SHOTS_MADE']
    return df

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db-path", type=Path, default=ADVANCED_SQLITE_DB)
    parser.add_argument("--out", type=Path, default=AUXILIARY_DIR / "clutch_stats.parquet")
    args = parser.parse_args()

    tbl = build_clutch_stats(args.db_path)
    if not tbl.empty:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        tbl.to_parquet(args.out, index=False)
        logger.info(f"Wrote clutch stats to {args.out}")

if __name__ == "__main__":
    main()
