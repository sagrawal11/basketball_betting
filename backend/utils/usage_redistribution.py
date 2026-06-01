"""
Usage Redistribution Engine.

When key players are injured/out, remaining players get more minutes and usage.
This module computes a boost factor for each active player's predictions based on
who's sitting out.

Algorithm:
1. Look up recent average minutes (L10) for each "out" teammate.
2. Sum freed minutes.
3. Distribute freed minutes proportionally among remaining active players
   based on their own recent minutes share.
4. Compute boost_factor = projected_minutes / normal_minutes.
5. Scale predictions by this factor.

Constraints:
- Individual projected minutes capped at MAX_MINUTES (42).
- Boost factor capped at MAX_BOOST (1.35).
- Deep bench players (< MIN_THRESHOLD minutes) get reduced share.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

MAX_MINUTES = 42.0
MAX_BOOST = 1.35
MIN_THRESHOLD = 10.0  # Players averaging fewer minutes than this get reduced share
DEEP_BENCH_SHARE_FACTOR = 0.3  # Deep bench gets 30% of proportional share
L10_WINDOW = 10


def _get_player_recent_minutes(player_data_dir: Path, player_name: str) -> Optional[float]:
    """Get a player's average minutes over their last L10 games."""
    slug = player_name.replace(" ", "_")
    csv_path = player_data_dir / slug / f"{slug}_data.csv"
    if not csv_path.exists():
        return None
    try:
        df = pd.read_csv(csv_path)
        if "MIN" not in df.columns or len(df) == 0:
            return None
        return float(df["MIN"].tail(L10_WINDOW).mean())
    except Exception:
        return None


def compute_boost_factors(
    player_data_dir: Path,
    active_players: List[str],
    teammates_out: List[str],
) -> Dict[str, float]:
    """
    Compute per-player boost factors based on who's out.

    Parameters
    ----------
    player_data_dir : Path
        Path to player_data directory (contains slugged subdirectories).
    active_players : list[str]
        Names of players who ARE playing (will receive predictions).
    teammates_out : list[str]
        Names of players who are OUT (injured/excluded).

    Returns
    -------
    dict[str, float]
        Mapping of active player name → boost factor (>= 1.0).
    """
    if not teammates_out:
        return {p: 1.0 for p in active_players}

    # Step 1: Get average minutes for all out players
    freed_minutes = 0.0
    for name in teammates_out:
        mins = _get_player_recent_minutes(player_data_dir, name)
        if mins is not None and mins > 0:
            freed_minutes += mins
            logger.debug("OUT: %s frees %.1f minutes", name, mins)

    if freed_minutes <= 0:
        return {p: 1.0 for p in active_players}

    # Step 2: Get minutes for active players, compute shares
    active_minutes: Dict[str, float] = {}
    for name in active_players:
        mins = _get_player_recent_minutes(player_data_dir, name)
        active_minutes[name] = mins if mins is not None and mins > 0 else 12.0  # fallback

    total_active_mins = sum(active_minutes.values())
    if total_active_mins <= 0:
        return {p: 1.0 for p in active_players}

    # Step 3: Distribute freed minutes proportionally
    boost_factors: Dict[str, float] = {}
    for name in active_players:
        normal_mins = active_minutes[name]

        # Deep bench players get a reduced share of freed minutes
        if normal_mins < MIN_THRESHOLD:
            share = (normal_mins / total_active_mins) * DEEP_BENCH_SHARE_FACTOR
        else:
            share = normal_mins / total_active_mins

        added_mins = freed_minutes * share
        projected_mins = min(normal_mins + added_mins, MAX_MINUTES)
        raw_boost = projected_mins / max(normal_mins, 1.0)
        boost = min(raw_boost, MAX_BOOST)
        boost = max(boost, 1.0)  # Never reduce

        boost_factors[name] = round(boost, 4)
        if boost > 1.01:
            logger.debug(
                "BOOST: %s  %.1f → %.1f min  (factor=%.3f)",
                name, normal_mins, projected_mins, boost,
            )

    return boost_factors


def apply_boost_to_predictions(
    predictions: Dict[str, float],
    boost_factor: float,
) -> Dict[str, float]:
    """Scale all stat predictions by the boost factor."""
    if boost_factor <= 1.0:
        return predictions

    boosted = {}
    for stat, value in predictions.items():
        boosted[stat] = round(value * boost_factor, 4)
    return boosted
