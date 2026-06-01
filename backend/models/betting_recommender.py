"""Placeholder betting recommender (SportsGameOdds integration can be restored later)."""
from __future__ import annotations

from typing import Any, Dict, List, Optional


class BettingRecommender:
    def recommend_bet(
        self,
        player_name: str,
        stat_type: str,
        line: float,
        odds: int,
        prediction: float,
        std_estimate: float = 5.0,
    ) -> Dict[str, Any]:
        return {
            "player": player_name,
            "stat": stat_type,
            "line": line,
            "recommendation": "no_edge",
            "note": "BettingRecommender not configured",
        }

    def find_value_bets(self, predictions: List[Dict]) -> List[Dict]:
        return []
