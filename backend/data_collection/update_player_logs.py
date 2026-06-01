#!/usr/bin/env python3
"""
Scrape Basketball-Reference for yesterday's box scores and append to player logs.
Designed to run automatically as part of the nightly job.
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
from datetime import datetime, timedelta
from io import StringIO
from pathlib import Path

import pandas as pd
import requests

BACKEND = Path(__file__).resolve().parent.parent
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

from config.paths import PLAYER_DATA_DIR

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def clean_slug(name: str) -> str:
    """e.g. 'LeBron James' -> 'LeBron_James'"""
    return name.replace(" ", "_")

def nba_season_label(d: datetime) -> str:
    y, m = d.year, d.month
    if m >= 10:
        return f"{y}-{str(y + 1)[-2:]}"
    return f"{y - 1}-{str(y)[-2:]}"

def parse_minutes(mp_str: str) -> int:
    try:
        parts = str(mp_str).split(":")
        if len(parts) == 2:
            return int(parts[0]) + round(int(parts[1]) / 60)
        return int(float(mp_str))
    except Exception:
        return 0

def fetch_bref_daily_leaders(dt: datetime) -> pd.DataFrame:
    url = f"https://www.basketball-reference.com/friv/dailyleaders.fcgi?month={dt.month}&day={dt.day}&year={dt.year}"
    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }
    logger.info("Fetching %s", url)
    resp = requests.get(url, headers=headers)
    resp.raise_for_status()
    try:
        tables = pd.read_html(StringIO(resp.text))
        if not tables:
            return pd.DataFrame()
        df = tables[0]
        # Filter out intermediate header rows
        df = df[df["Player"] != "Player"].copy()
        return df
    except ValueError: # No tables found
        return pd.DataFrame()

def update_player_logs(target_date: str = None) -> None:
    if target_date:
        dt = datetime.strptime(target_date, "%Y-%m-%d")
    else:
        # Default to yesterday
        dt = datetime.now() - timedelta(days=1)
    
    date_str = dt.strftime("%Y-%m-%d")
    season_str = nba_season_label(dt)
    
    logger.info("Updating player logs for date: %s", date_str)
    
    df = fetch_bref_daily_leaders(dt)
    if df.empty or len(df) == 0:
        logger.info("No games found for %s", date_str)
        return

    logger.info("Fetched %d player stat lines.", len(df))
    
    # Process and append to local CSVs
    updated = 0
    new_players = 0
    
    for _, row in df.iterrows():
        try:
            player_name = str(row.get("Player", "")).strip()
            if not player_name:
                continue
                
            slug = clean_slug(player_name)
            player_dir = PLAYER_DATA_DIR / slug
            csv_path = player_dir / f"{slug}_data.csv"
            
            is_new = False
            if not csv_path.exists():
                player_dir.mkdir(parents=True, exist_ok=True)
                is_new = True
                new_players += 1
            
            # Extract basic data
            team = str(row.get("Tm", ""))
            opp = str(row.get("Opp", ""))
            is_home = 0 if str(row.get("Unnamed: 3", "")) == "@" else 1
            matchup = f"{team} vs. {opp}" if is_home else f"{team} @ {opp}"
            
            new_row = {
                "GAME_DATE": date_str,
                "PLAYER_NAME": player_name,
                "PLAYER_TEAM": team,
                "OPPONENT_TEAM": opp,
                "IS_HOME": is_home,
                "MATCHUP": matchup,
                "SEASON": season_str,
                "MIN": parse_minutes(row.get("MP", 0)),
                "FGM": row.get("FG", 0),
                "FGA": row.get("FGA", 0),
                "FG_PCT": row.get("FG%", 0.0),
                "FG3M": row.get("3P", 0),
                "FG3A": row.get("3PA", 0),
                "FG3_PCT": row.get("3P%", 0.0),
                "FTM": row.get("FT", 0),
                "FTA": row.get("FTA", 0),
                "FT_PCT": row.get("FT%", 0.0),
                "OREB": row.get("ORB", 0),
                "DREB": row.get("DRB", 0),
                "REB": row.get("TRB", 0),
                "AST": row.get("AST", 0),
                "STL": row.get("STL", 0),
                "BLK": row.get("BLK", 0),
                "TOV": row.get("TOV", 0),
                "PF": row.get("PF", 0),
                "PTS": row.get("PTS", 0),
                "PLUS_MINUS": row.get("+/-", 0)
            }
            
            new_df = pd.DataFrame([new_row])
            
            if is_new:
                new_df.to_csv(csv_path, index=False)
            else:
                existing_df = pd.read_csv(csv_path)
                # Avoid duplicates
                if not ((existing_df["GAME_DATE"] == date_str) & (existing_df["PLAYER_NAME"] == player_name)).any():
                    # Keep all columns, default new ones to pd.NA
                    combined = pd.concat([existing_df, new_df], ignore_index=True)
                    combined.to_csv(csv_path, index=False)
                    updated += 1
        except Exception as e:
            logger.warning("Error processing %s: %s", player_name, str(e))
            
    logger.info("Successfully updated %d existing formats and created %d new players.", updated, new_players)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", type=str, help="YYYY-MM-DD (defaults to yesterday)")
    args = parser.parse_args()
    update_player_logs(args.date)
