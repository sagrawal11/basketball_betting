#!/usr/bin/env python3
"""
Build team-season OFF/DEF/NET rating and pace proxies entirely from local player game logs.
Does not call stats.nba.com — use this when the NBA API is flaky.

Method (standard approximations):
  - Aggregate box stats by (Game_ID, PLAYER_TEAM) → team game totals.
  - Pair teams per game → team_pts, opp_pts, team_poss_est.
  - poss_est = FGA - OREB + TOV + 0.44*FTA (team sum per game).
  - Season: OFF_RATING ≈ 100 * sum(pts) / sum(poss), DEF_RATING ≈ 100 * sum(opp_pts) / sum(team_poss),
            NET_RATING = OFF_RATING - DEF_RATING,
            PACE ≈ mean per game of (team_poss + opp_poss) / 2  (possessions scale ~ NBA pace).

Output matches feature_engine merge keys: TEAM_ABBREVIATION, SEASON, PACE, OFF_RATING, DEF_RATING, NET_RATING
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import List, Optional

import pandas as pd

BACKEND = Path(__file__).resolve().parent.parent
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

from config.paths import TEAM_STATS_DIR

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def _load_all_game_rows(player_data_dir: Path, max_files: Optional[int] = None) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    dirs = sorted([p for p in player_data_dir.iterdir() if p.is_dir()])
    if max_files:
        dirs = dirs[:max_files]
    for d in dirs:
        slug = d.name
        csv_path = d / f"{slug}_data.csv"
        if not csv_path.exists():
            continue
        try:
            df = pd.read_csv(csv_path)
            need = [
                "Game_ID",
                "PLAYER_TEAM",
                "SEASON",
                "PTS",
                "FGA",
                "FTA",
                "OREB",
                "TOV",
            ]
            if not all(c in df.columns for c in need):
                continue
            df = df[need].copy()
            if len(df) == 0:
                continue
            frames.append(df)
        except Exception as e:
            logger.warning("Skip %s: %s", slug, e)
    if not frames:
        raise RuntimeError("No player game logs found under %s" % player_data_dir)
    out = pd.concat(frames, ignore_index=True)
    out = out.dropna(subset=["Game_ID", "PLAYER_TEAM", "SEASON"])
    for c in ["PTS", "FGA", "FTA", "OREB", "TOV"]:
        out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0.0)
    return out


def _team_games(df: pd.DataFrame) -> pd.DataFrame:
    g = (
        df.groupby(["Game_ID", "PLAYER_TEAM", "SEASON"], as_index=False)
        .agg({"PTS": "sum", "FGA": "sum", "FTA": "sum", "OREB": "sum", "TOV": "sum"})
    )
    g["poss_est"] = g["FGA"] - g["OREB"] + g["TOV"] + 0.44 * g["FTA"]
    g["poss_est"] = g["poss_est"].clip(lower=20.0, upper=200.0)
    return g


def _expand_game_rows(team_games: pd.DataFrame) -> pd.DataFrame:
    """One row per team per game with opponent stats."""
    rows = []
    for gid, sub in team_games.groupby("Game_ID"):
        if len(sub) < 2:
            continue
        if len(sub) > 2:
            sub = sub.groupby("PLAYER_TEAM", as_index=False).sum(numeric_only=True)
        if len(sub) != 2:
            continue
        a, b = sub.iloc[0], sub.iloc[1]
        season = a["SEASON"] if pd.notna(a["SEASON"]) else b["SEASON"]

        def add_row(team_row, opp_row):
            rows.append(
                {
                    "Game_ID": gid,
                    "SEASON": season,
                    "TEAM_ABBREVIATION": team_row["PLAYER_TEAM"],
                    "team_pts": float(team_row["PTS"]),
                    "opp_pts": float(opp_row["PTS"]),
                    "team_poss": float(team_row["poss_est"]),
                    "opp_poss": float(opp_row["poss_est"]),
                }
            )

        add_row(a, b)
        add_row(b, a)
    return pd.DataFrame(rows)


def build_team_season_table(game_rows: pd.DataFrame) -> pd.DataFrame:
    records = []
    for (team, season), g in game_rows.groupby(["TEAM_ABBREVIATION", "SEASON"]):
        tp, tpos = g["team_pts"].sum(), g["team_poss"].sum()
        op = g["opp_pts"].sum()
        off = 100.0 * tp / max(tpos, 1.0)
        dff = 100.0 * op / max(tpos, 1.0)
        pace = float((g["team_poss"] + g["opp_poss"]).mean() / 2.0)
        records.append(
            {
                "TEAM_ABBREVIATION": team,
                "SEASON": season,
                "PACE": pace,
                "OFF_RATING": off,
                "DEF_RATING": dff,
                "NET_RATING": off - dff,
                "GP": len(g),
            }
        )
    return pd.DataFrame(records)


def main() -> None:
    ap = argparse.ArgumentParser(description="Build team stats CSV from player logs (no NBA API)")
    ap.add_argument(
        "--player-data",
        type=Path,
        default=BACKEND / "data" / "player_data",
        help="Directory containing one folder per player with *_data.csv",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=TEAM_STATS_DIR / "all_team_stats.csv",
        help="Output CSV path",
    )
    ap.add_argument("--max-players", type=int, default=None, help="Only read first N player folders (debug)")
    args = ap.parse_args()

    logger.info("Loading player logs from %s", args.player_data)
    raw = _load_all_game_rows(args.player_data, max_files=args.max_players)
    logger.info("Loaded %s player-game rows", len(raw))

    tg = _team_games(raw)
    logger.info("Aggregated to %s team-games", len(tg))

    games = _expand_game_rows(tg)
    logger.info("Expanded to %s team-side game rows", len(games))

    table = build_team_season_table(games)
    table = table.sort_values(["SEASON", "TEAM_ABBREVIATION"]).reset_index(drop=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.out, index=False)

    summary = {
        "source": "aggregated_from_player_logs",
        "total_records": int(len(table)),
        "seasons_covered": sorted(table["SEASON"].astype(str).unique().tolist()),
        "note": "PACE/OFF/DEF/NET are proxies, not official NBA.com team dashboards.",
    }
    summary_path = args.out.parent / "team_stats_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    logger.info("Wrote %s rows to %s", len(table), args.out)
    logger.info("Summary -> %s", summary_path)


if __name__ == "__main__":
    main()
