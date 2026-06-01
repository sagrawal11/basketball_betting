#!/usr/bin/env python3
"""
Build Defense-vs-Position style aggregates from existing player game logs.
For each (defensive_team, season, position_group): mean opponent PTS/REB/AST per game
(attacker stats when facing that defense).
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

import sys

BACKEND = Path(__file__).resolve().parent.parent
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

from config.paths import DVP_PARQUET, PLAYER_DATA_DIR

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _position_group(pos: object) -> str:
    if pd.isna(pos):
        return "G"
    s = str(pos).upper()
    if "G" in s or "GUARD" in s:
        return "G"
    if "F" in s or "FORWARD" in s or "WING" in s:
        return "F"
    if "C" in s or "CENTER" in s:
        return "C"
    return "G"


def build_dvp_table(player_data_dir: Path, max_players: int | None = None) -> pd.DataFrame:
    player_data_dir = Path(player_data_dir)
    rows = []
    dirs = sorted([p for p in player_data_dir.iterdir() if p.is_dir()])
    if max_players:
        dirs = dirs[:max_players]
    for d in dirs:
        slug = d.name
        csv_path = d / f"{slug}_data.csv"
        if not csv_path.exists():
            continue
        try:
            df = pd.read_csv(csv_path)
        except Exception as e:
            logger.warning("Skip %s: %s", slug, e)
            continue
        need = {"OPPONENT_TEAM", "SEASON", "PTS"}
        if not need.issubset(df.columns):
            continue
        if "position" in df.columns:
            df["POSITION_GROUP"] = df["position"].apply(_position_group)
        else:
            df["POSITION_GROUP"] = "G"
        df["REB_T"] = df["OREB"].fillna(0) + df["DREB"].fillna(0) if "OREB" in df.columns else np.nan
        for _, r in df.iterrows():
            rows.append(
                {
                    "DEF_TEAM": r["OPPONENT_TEAM"],
                    "GAME_DATE": pd.to_datetime(r["GAME_DATE"]).normalize(),
                    "POSITION_GROUP": r["POSITION_GROUP"],
                    "SEASON": str(r["SEASON"]),
                    "MIN": float(r["MIN"]) if "MIN" in df.columns and pd.notna(r["MIN"]) else 0.0,
                    "PTS": float(r["PTS"]) if pd.notna(r["PTS"]) else 0.0,
                    "REB": float(r["REB_T"]) if pd.notna(r["REB_T"]) else 0.0,
                    "AST": float(r["AST"]) if "AST" in df.columns and pd.notna(r["AST"]) else 0.0,
                }
            )
    if not rows:
        return pd.DataFrame(
            columns=[
                "OPPONENT_TEAM",
                "SEASON",
                "POSITION_GROUP",
                "PTS_ALLOWED_PER36",
                "REB_ALLOWED_PER36",
                "AST_ALLOWED_PER36",
            ]
        )
    all_g = pd.DataFrame(rows)
    # Aggregate raw stats per GAME_DATE for each DEF_TEAM and POSITION_GROUP
    daily = all_g.groupby(["DEF_TEAM", "GAME_DATE", "POSITION_GROUP"], as_index=False).agg(
        TOT_MIN=("MIN", "sum"),
        TOT_PTS=("PTS", "sum"),
        TOT_REB=("REB", "sum"),
        TOT_AST=("AST", "sum"),
    )
    daily = daily.sort_values("GAME_DATE")
    
    # Calculate cumulative sum, then scale to per-36 min. Must shift(1) to avoid current game leakage.
    daily["CUM_MIN"] = daily.groupby(["DEF_TEAM", "POSITION_GROUP"])["TOT_MIN"].transform(lambda x: x.shift(1).cumsum())
    daily["CUM_PTS"] = daily.groupby(["DEF_TEAM", "POSITION_GROUP"])["TOT_PTS"].transform(lambda x: x.shift(1).cumsum())
    daily["CUM_REB"] = daily.groupby(["DEF_TEAM", "POSITION_GROUP"])["TOT_REB"].transform(lambda x: x.shift(1).cumsum())
    daily["CUM_AST"] = daily.groupby(["DEF_TEAM", "POSITION_GROUP"])["TOT_AST"].transform(lambda x: x.shift(1).cumsum())
    
    daily["PTS_ALLOWED_PER36"] = np.where(daily["CUM_MIN"] > 10, (daily["CUM_PTS"] / daily["CUM_MIN"]) * 36.0, np.nan)
    daily["REB_ALLOWED_PER36"] = np.where(daily["CUM_MIN"] > 10, (daily["CUM_REB"] / daily["CUM_MIN"]) * 36.0, np.nan)
    daily["AST_ALLOWED_PER36"] = np.where(daily["CUM_MIN"] > 10, (daily["CUM_AST"] / daily["CUM_MIN"]) * 36.0, np.nan)
    
    daily = daily.rename(columns={"DEF_TEAM": "OPPONENT_TEAM"})
    return daily


def main() -> None:
    ap = argparse.ArgumentParser(description="Build DvP table from player_data logs")
    ap.add_argument("--player-data", type=Path, default=PLAYER_DATA_DIR)
    ap.add_argument("--out", type=Path, default=DVP_PARQUET)
    ap.add_argument("--max-players", type=int, default=None)
    args = ap.parse_args()
    tbl = build_dvp_table(args.player_data, max_players=args.max_players)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    tbl.to_parquet(args.out, index=False)
    logger.info("Wrote %s rows to %s", len(tbl), args.out)


if __name__ == "__main__":
    main()
