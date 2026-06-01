#!/usr/bin/env python3
"""
Build team×season stats + parquet mirrors from processed Kaggle team game logs (teams_boxscores.csv).

Writes:
  - data/processed/kaggle_team_games.parquet — full game-level team box (canonical for merges)
  - data/team_stats/all_team_stats.{csv,parquet} — season means of pace/ratings

Aggregates game-level stats to team×season; derives SEASON from game_date (NBA Oct–Jun).
No external APIs — local CSV in repo only.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import pandas as pd

BACKEND = Path(__file__).resolve().parent.parent
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

from config.paths import (
    KAGGLE_TEAM_GAMES_PARQUET,
    KAGGLE_TEAMS_BOXSCORES_CSV,
    TEAM_STATS_DIR,
    TEAM_STATS_PARQUET,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

REQUIRED = ("game_date", "team", "pace", "off_rtg", "def_rtg", "net_rtg")


def nba_season_label(d: pd.Timestamp) -> str:
    y, m = d.year, d.month
    if m >= 10:
        return f"{y}-{str(y + 1)[-2:]}"
    return f"{y - 1}-{str(y)[-2:]}"


def aggregate(
    df: pd.DataFrame,
    *,
    regular_season_only: bool,
) -> pd.DataFrame:
    if regular_season_only and "is_playoff" in df.columns:
        before = len(df)
        df = df[df["is_playoff"].astype(int) == 0].copy()
        logger.info("Regular season only: %s -> %s rows", before, len(df))
    df["SEASON"] = df["game_date"].map(nba_season_label)
    g = (
        df.groupby(["team", "SEASON"], as_index=False)
        .agg(
            PACE=("pace", "mean"),
            OFF_RATING=("off_rtg", "mean"),
            DEF_RATING=("def_rtg", "mean"),
            NET_RATING=("net_rtg", "mean"),
            GP=("pace", "count"),
        )
        .rename(columns={"team": "TEAM_ABBREVIATION"})
    )
    for c in ("PACE", "OFF_RATING", "DEF_RATING", "NET_RATING"):
        g[c] = g[c].round(3)
    return g.sort_values(["SEASON", "TEAM_ABBREVIATION"]).reset_index(drop=True)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Kaggle teams_boxscores → all_team_stats.csv (team×season pace & ratings)"
    )
    ap.add_argument(
        "--input",
        type=Path,
        default=KAGGLE_TEAMS_BOXSCORES_CSV,
        help="processed/teams_boxscores.csv from the Kaggle bundle",
    )
    ap.add_argument("--out", type=Path, default=TEAM_STATS_DIR / "all_team_stats.csv")
    ap.add_argument(
        "--regular-season-only",
        action="store_true",
        help="Exclude playoff games (is_playoff=1)",
    )
    args = ap.parse_args()

    if not args.input.exists():
        raise SystemExit(f"Input not found: {args.input}")

    full = pd.read_csv(args.input)
    miss = [c for c in REQUIRED if c not in full.columns]
    if miss:
        raise SystemExit(f"Missing columns in {args.input}: {miss}")
    full["game_date"] = pd.to_datetime(full["game_date"])
    logger.info("Loaded %s game-level rows from %s", len(full), args.input)

    KAGGLE_TEAM_GAMES_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    full.to_parquet(KAGGLE_TEAM_GAMES_PARQUET, index=False)
    logger.info("Wrote game-level parquet %s", KAGGLE_TEAM_GAMES_PARQUET)

    table = aggregate(full, regular_season_only=args.regular_season_only)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.out, index=False)
    table.to_parquet(TEAM_STATS_PARQUET, index=False)
    logger.info("Wrote team×season parquet %s", TEAM_STATS_PARQUET)

    summary = {
        "source": "kaggle_teams_boxscores",
        "input_path": str(args.input.resolve()),
        "game_level_parquet": str(KAGGLE_TEAM_GAMES_PARQUET.resolve()),
        "team_stats_parquet": str(TEAM_STATS_PARQUET.resolve()),
        "regular_season_only": args.regular_season_only,
        "total_records": int(len(table)),
        "seasons_covered": sorted(table["SEASON"].astype(str).unique().tolist()),
        "note": "PACE/OFF/DEF/NET are means of game-level box advanced stats (not identical to official season dashboards).",
    }
    summary_path = args.out.parent / "team_stats_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    logger.info("Wrote %s rows to %s", len(table), args.out)
    logger.info("Summary -> %s", summary_path)


if __name__ == "__main__":
    main()
