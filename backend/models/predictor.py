"""
Inference: load global LightGBM models + unified FeatureEngine (same path as training).
"""
from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
import numpy as np
import pandas as pd

import sys

BACKEND = Path(__file__).resolve().parent.parent
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

from config.paths import DATA_DIR, GLOBAL_MODEL_DIR, PLAYER_DATA_DIR
from feature_engine import TARGET_STATS, load_feature_columns, default_engine
from utils.usage_redistribution import compute_boost_factors, apply_boost_to_predictions

logger = logging.getLogger(__name__)


class NBAPredictor:
    """Global-model predictor compatible with the Flask app expectations."""

    def __init__(
        self,
        data_dir: Optional[str] = None,
        models_dir: Optional[str] = None,
        artifacts_dir: Optional[str] = None,
    ):
        self.data_dir = Path(data_dir or DATA_DIR)
        self.models_dir = Path(models_dir or (self.data_dir / "player_data"))
        self.artifacts_dir = Path(artifacts_dir or GLOBAL_MODEL_DIR)
        self.feature_engine = default_engine()
        self.feature_columns: Optional[List[str]] = load_feature_columns()
        self.models: Dict[str, Any] = {}
        self.residual_models: Dict[str, Dict[str, Any]] = {}
        for t in TARGET_STATS:
            p = self.artifacts_dir / f"global_lgbm_{t}.joblib"
            if p.exists():
                blob = joblib.load(p)
                self.models[t] = blob.get("model", blob)
            
            p_res = self.artifacts_dir / f"residuals_{t}.joblib"
            if p_res.exists():
                self.residual_models[t] = joblib.load(p_res)

    def _get_current_season(self) -> str:
        now = datetime.now()
        year, month = now.year, now.month
        if month >= 10:
            return f"{year}-{str(year + 1)[-2:]}"
        return f"{year - 1}-{str(year)[-2:]}"

    def predict_player_stats(
        self,
        player_name: str,
        opponent_team: str,
        is_home: bool,
        game_date: str,
        season: str,
        injury_status: str = "Healthy",
        teammates_out: Optional[List[str]] = None,
    ) -> Optional[Dict[str, Any]]:
        # If the player themselves is marked as Out, return None
        if injury_status.lower() == "out":
            return None

        row = self.feature_engine.build_inference_row(
            player_name=player_name,
            opponent_team=opponent_team,
            is_home=is_home,
            game_date=game_date,
            season=season,
        )
        if row is None:
            return None
        if not self.models or not self.feature_columns:
            logger.warning("No trained global models or feature column list")
            return None
            
        # Reindex to absolute perfection vs the training columns
        sub = row.reindex(self.feature_columns)
        
        # Check for missing columns that are critical
        missing = [c for c in self.feature_columns if c not in row.index]
        if missing:
            logger.debug("Inference is missing %d managed columns (filling 0.0)", len(missing))

        # Convert to DataFrame with columns so LGBM is happy and doesn't warn about feature names
        X_df = pd.DataFrame([sub])
        X_df = X_df.apply(pd.to_numeric, errors="coerce").fillna(0.0)

        # Reasonable per-stat maximums (even the best game ever shouldn't exceed these)
        STAT_CAPS = {
            "PTS": 55, "AST": 20, "DREB": 18, "OREB": 8,
            "STL": 7, "BLK": 8, "TOV": 10, "FGM": 20,
            "FG3M": 12, "FTM": 20, "FGA": 35, "FG3A": 20, "FTA": 25,
        }
        # Max residual as fraction of base prediction (prevents wild extrapolation)
        MAX_RESIDUAL_ABS = 10.0  # Hard cap: residual can move prediction by at most ±10

        # Residual models use title-case names (from training data PLAYER_NAME column)
        title_name = player_name.replace("_", " ").title()

        preds: Dict[str, Any] = {}
        for t, model in self.models.items():
            try:
                base_pred = float(model.predict(X_df)[0])
                
                # Check if this specific player has a localized residual model for this stat
                res_dict = self.residual_models.get(t, {})
                res_key = title_name if title_name in res_dict else (player_name if player_name in res_dict else None)
                if res_key:
                    residual_model = res_dict[res_key]
                    res_pred = float(residual_model.predict(X_df)[0])
                    # Clamp residual to prevent wild Ridge extrapolation
                    res_pred = max(-MAX_RESIDUAL_ABS, min(MAX_RESIDUAL_ABS, res_pred))
                    base_pred += res_pred
                
                # Sanity: stats cannot be negative
                if base_pred < 0:
                    base_pred = 0.0
                
                # Apply per-stat cap
                cap = STAT_CAPS.get(t, 50)
                if base_pred > cap:
                    logger.info("Capping %s [%s] from %.1f to %d", player_name, t, base_pred, cap)
                    base_pred = float(cap)
                    
                preds[t] = base_pred
            except Exception as e:
                logger.warning("Predict %s failed: %s", t, e)

        if not preds:
            return None

        # Apply usage redistribution boost if teammates are out
        if teammates_out:
            boost_factors = compute_boost_factors(
                player_data_dir=self.models_dir,
                active_players=[player_name],
                teammates_out=teammates_out,
            )
            boost = boost_factors.get(player_name, 1.0)
            if boost > 1.0:
                logger.info(
                    "Boosting %s predictions by %.1f%% (%d teammates out)",
                    player_name, (boost - 1) * 100, len(teammates_out),
                )
                preds = apply_boost_to_predictions(preds, boost)

        return preds


def build_predictor() -> NBAPredictor:
    return NBAPredictor()
