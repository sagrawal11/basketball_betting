#!/usr/bin/env python3
"""
Fetch NBA injury reports from the ESPN Injuries API into Parquet.

Source: https://site.api.espn.com/apis/site/v2/sports/basketball/nba/injuries
This is an unofficial ESPN endpoint — same risk profile as the scoreboard endpoint
we already use for live schedules.
"""
from __future__ import annotations

import argparse
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import requests

import sys

BACKEND = Path(__file__).resolve().parent.parent
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

from config.paths import INJURIES_PARQUET

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

ESPN_INJURIES_URL = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/injuries"

# ESPN abbreviation → our standard abbreviation (same map as app.py)
ESPN_TO_NBA_MAP = {
    "GS": "GSW", "NO": "NOP", "NY": "NYK",
    "SA": "SAS", "UTAH": "UTA", "WSH": "WAS",
}

SCHEMA_COLUMNS = [
    "team_abbrev", "player_name", "status", "injury_type",
    "detail", "side", "headshot_url", "position", "espn_id",
]


def _normalize_team_abbrev(abbrev: str) -> str:
    a = abbrev.upper().strip()
    return ESPN_TO_NBA_MAP.get(a, a)


def fetch_injuries_from_espn() -> pd.DataFrame:
    """Fetch all current NBA injuries from the ESPN API."""
    try:
        resp = requests.get(ESPN_INJURIES_URL, timeout=15)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        logger.warning("ESPN Injuries API request failed: %s", e)
        return pd.DataFrame(columns=SCHEMA_COLUMNS)

    rows: List[Dict[str, Any]] = []
    for team_block in data.get("injuries", []):
        team_name = team_block.get("displayName", "")
        for injury in team_block.get("injuries", []):
            athlete = injury.get("athlete", {})
            details = injury.get("details", {})
            team_info = athlete.get("team", {})
            headshot = athlete.get("headshot", {})
            position = athlete.get("position", {})

            raw_abbrev = team_info.get("abbreviation", "")

            # Map status string: "Out" / "Day-To-Day" → simplified
            raw_status = injury.get("status", "Unknown")
            # The type field has the canonical status
            type_info = injury.get("type", {})
            abbrev_status = type_info.get("abbreviation", "")  # "O", "DD", "DTD"

            if abbrev_status == "O":
                status = "Out"
            elif abbrev_status in ("DD", "DTD"):
                status = "Day-To-Day"
            elif raw_status.lower() == "out":
                status = "Out"
            elif "day" in raw_status.lower():
                status = "Day-To-Day"
            else:
                status = raw_status

            rows.append({
                "team_abbrev": _normalize_team_abbrev(raw_abbrev),
                "player_name": athlete.get("displayName", ""),
                "status": status,
                "injury_type": details.get("type", ""),
                "detail": details.get("detail", ""),
                "side": details.get("side", ""),
                "headshot_url": headshot.get("href", ""),
                "position": position.get("abbreviation", ""),
                "espn_id": str(athlete.get("links", [{}])[0].get("href", "")).split("/")[-1] if athlete.get("links") else "",
            })

    if not rows:
        return pd.DataFrame(columns=SCHEMA_COLUMNS)
    return pd.DataFrame(rows)


def fetch_team_injuries(team_abbrev: str) -> List[Dict[str, Any]]:
    """Return injury list for a specific team (used by the Flask API at request time)."""
    df = fetch_injuries_from_espn()
    if df.empty:
        return []
    team_df = df[df["team_abbrev"] == team_abbrev.upper()]
    return team_df.to_dict(orient="records")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=INJURIES_PARQUET)
    args = ap.parse_args()

    df = fetch_injuries_from_espn()
    df["ingested_at_utc"] = datetime.now(timezone.utc).isoformat()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.out, index=False)
    logger.info("Wrote injuries: %s rows -> %s", len(df), args.out)

    # Print summary
    if not df.empty:
        for status in df["status"].unique():
            count = len(df[df["status"] == status])
            logger.info("  %s: %d players", status, count)


if __name__ == "__main__":
    main()
