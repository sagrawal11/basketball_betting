#!/usr/bin/env python3
"""
Train Hybrid Models: Global LightGBM regressors + Player-Specific Ridge residual models.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

BACKEND = Path(__file__).resolve().parent.parent
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

from config.paths import GLOBAL_MODEL_DIR, PLAYER_DATA_DIR
from feature_engine import FeatureEngine, TARGET_STATS, save_feature_columns, default_engine

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

ID_LIKE = {"Player_ID", "Game_ID", "SEASON_ID", "VIDEO_AVAILABLE", "ARCH_ARGMAX"}

def _load_or_build_features(engine: FeatureEngine, player_data_dir: Path, max_players: Optional[int]) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    dirs = sorted([p for p in player_data_dir.iterdir() if p.is_dir()])
    if max_players:
        dirs = dirs[:max_players]
    for d in dirs:
        slug = d.name
        name = slug.replace("_", " ")
        pq = d / f"{slug}_features.parquet"
        csv = d / f"{slug}_features.csv"
        
        # Load pre-built features if they exist AND contain the newly required ARCH elements
        loaded = False
        if pq.exists():
            try:
                df = pd.read_parquet(pq)
                if "ARCH_PROB_0" in df.columns:
                    frames.append(df)
                    loaded = True
            except Exception:
                pass
        
        if not loaded:
            built = engine.build_training_features(name, save_parquet=True)
            if built is not None and len(built) > 0:
                frames.append(built)
    if not frames:
        raise RuntimeError("No feature rows found. Run feature build or ensure player_data exists.")
    return pd.concat(frames, ignore_index=True)


def _feature_target_split(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str], List[str], pd.Series]:
    from feature_engine import default_engine
    engine = default_engine()
    
    df = df.copy()
    if "GAME_DATE" in df.columns:
        df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])
        df = df.sort_values("GAME_DATE").reset_index(drop=True)
    
    seasons = df["SEASON"] if "SEASON" in df.columns else pd.Series(["Unknown"] * len(df))
    
    # Use FeatureEngine to whitelist strictly managed columns
    df = engine._clean_output_columns(df)
    
    num = df.select_dtypes(include=[np.number])
    y_cols = [c for c in TARGET_STATS if c in num.columns]
    drop_y = [c for c in num.columns if c in TARGET_STATS]
    X_num = num.drop(columns=[c for c in drop_y if c in num.columns], errors="ignore")
    
    for c in list(X_num.columns):
        if c in ID_LIKE:
            X_num = X_num.drop(columns=[c])
            
    X_cols = list(X_num.columns)
    return X_num, X_cols, y_cols, seasons


def train_global_model(
    X: pd.DataFrame,
    y: pd.Series,
    seasons: pd.Series,
) -> Tuple[object, Dict[str, Any]]:
    try:
        from lightgbm import LGBMRegressor, early_stopping
    except ImportError as e:
        raise RuntimeError("lightgbm required") from e

    X = X.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    y = y.replace([np.inf, -np.inf], np.nan)
    valid = y.notna()
    X = X.loc[valid]
    y = y.loc[valid]
    seasons = seasons.loc[valid]
    if len(X) < 50:
        raise ValueError("Not enough rows to train global model")

    unique_seasons = sorted(seasons.unique().tolist())
    
    if len(unique_seasons) < 2:
        n = len(X)
        split = int(n * 0.8)
        split = max(split, 1)
        X_train, X_test = X.iloc[:split], X.iloc[split:]
        y_train, y_test = y.iloc[:split], y.iloc[split:]
        
        model = LGBMRegressor(
            n_estimators=1000,
            learning_rate=0.01,
            num_leaves=127,
            max_depth=10,
            min_child_samples=50,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1,
            verbose=-1,
        )
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_test, y_test)],
            callbacks=[early_stopping(50, verbose=False)],
        )
        pred = model.predict(X_test)
        mae = float(np.mean(np.abs(pred - y_test)))
        
        importance = pd.DataFrame({"feature": X.columns, "importance": model.feature_importances_}).sort_values("importance", ascending=False)
        
        return model, {"mae": mae, "n_train": int(len(X_train)), "n_test": int(len(X_test)), "top_features": importance.head(20).to_dict(orient="records")}
    else:
        cv_metrics = []
        for i in range(1, len(unique_seasons)):
            train_seasons = unique_seasons[:i]
            test_season = unique_seasons[i]
            
            train_idx = seasons.isin(train_seasons)
            test_idx = seasons == test_season
            
            X_train, y_train = X.loc[train_idx], y.loc[train_idx]
            X_test, y_test = X.loc[test_idx], y.loc[test_idx]
            
            if len(X_train) < 50 or len(X_test) < 10:
                continue
                
            model = LGBMRegressor(
                n_estimators=500,
                learning_rate=0.05,
                num_leaves=63,
                max_depth=-1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                n_jobs=-1,
                verbose=-1,
            )
            model.fit(
                X_train,
                y_train,
                eval_set=[(X_test, y_test)],
                callbacks=[early_stopping(50, verbose=False)],
            )
            pred = model.predict(X_test)
            mae = float(np.mean(np.abs(pred - y_test)))
            cv_metrics.append({"season": test_season, "mae": mae, "n_train": len(X_train), "n_test": len(X_test)})
            
        train_seasons = unique_seasons[:-1]
        test_season = unique_seasons[-1]
        train_idx = seasons.isin(train_seasons)
        test_idx = seasons == test_season
        
        X_train, y_train = X.loc[train_idx], y.loc[train_idx]
        X_test, y_test = X.loc[test_idx], y.loc[test_idx]
        
        final_model = LGBMRegressor(
            n_estimators=1000,
            learning_rate=0.01,
            num_leaves=127,
            max_depth=10,
            min_child_samples=50,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1,
            verbose=-1,
        )
        final_model.fit(
            X_train,
            y_train,
            eval_set=[(X_test, y_test)],
            callbacks=[early_stopping(50, verbose=False)],
        )
        pred = final_model.predict(X_test)
        final_mae = float(np.mean(np.abs(pred - y_test)))
        
        importance = pd.DataFrame({
            "feature": X.columns,
            "importance": final_model.feature_importances_
        }).sort_values("importance", ascending=False)
        
        return final_model, {
            "cv_metrics": cv_metrics,
            "final_test_season": test_season,
            "final_mae": final_mae,
            "n_train": len(X_train),
            "n_test": len(X_test),
            "top_features": importance.head(20).to_dict(orient="records")
        }


def train_residuals(
    X_full: pd.DataFrame, 
    y_full: pd.Series, 
    global_model: Any, 
    player_names: pd.Series
) -> Tuple[Dict[str, Any], int]:
    
    # Calculate global predictions on the valid rows only
    X_clean = X_full.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    y_clean = y_full.replace([np.inf, -np.inf], np.nan)
    valid = y_clean.notna()
    
    X = X_clean.loc[valid]
    y = y_clean.loc[valid]
    p_names = player_names.loc[valid]
    
    global_preds = global_model.predict(X)
    residuals = y - global_preds
    
    local_models = {}
    
    # Group by player
    df_combined = pd.DataFrame({"PLAYER_NAME": p_names, "residual": residuals})
    
    for name, group in df_combined.groupby("PLAYER_NAME"):
        if len(group) >= 150:
            player_X = X.loc[group.index]
            player_y = group["residual"]
            
            ridge = Ridge(alpha=1.0, random_state=42)
            ridge.fit(player_X, player_y)
            local_models[name] = ridge
            
    return local_models, len(local_models)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--player-data", type=Path, default=PLAYER_DATA_DIR)
    ap.add_argument("--out", type=Path, default=GLOBAL_MODEL_DIR)
    ap.add_argument("--max-players", type=int, default=None)
    ap.add_argument("--test-fraction", type=float, default=0.2)
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)

    engine = default_engine()
    logger.info("Loading/building feature matrix...")
    full = _load_or_build_features(engine, args.player_data, args.max_players)
    if "PLAYER_NAME" not in full.columns:
        raise RuntimeError("PLAYER_NAME column missing, cannot train residuals.")
    
    X_frame, feature_cols, y_cols, seasons = _feature_target_split(full)
    if not feature_cols:
        raise RuntimeError("No numeric feature columns after split")
    
    save_feature_columns(feature_cols)
    logger.info("Features: %d rows, %d X cols, targets=%s", len(full), len(feature_cols), y_cols)

    metrics_all: Dict[str, object] = {}
    for target in TARGET_STATS:
        if target not in full.columns:
            logger.warning("Skipping %s — not in data", target)
            continue
        try:
            y = full[target]
            
            # Step 1: Global Model
            model, m = train_global_model(X_frame[feature_cols], y, seasons)
            path = args.out / f"global_lgbm_{target}.joblib"
            joblib.dump({"model": model, "target": target, "metrics": m}, path)
            
            # Step 2: Residual Target Generation & Local Models
            local_models, residual_count = train_residuals(X_frame[feature_cols], y, model, full["PLAYER_NAME"])
            res_path = args.out / f"residuals_{target}.joblib"
            joblib.dump(local_models, res_path)
            
            m["residual_models_trained"] = residual_count
            metrics_all[target] = m
            
            logger.info("Trained %s Final MAE=%.3f [Residual Models: %d]", target, m.get("final_mae", m.get("mae", 0.0)), residual_count)
        except Exception as e:
            logger.exception("Failed training %s: %s", target, e)

    with open(args.out / "hybrid_training_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics_all, f, indent=2)
    logger.info("Done. Artifacts in %s", args.out)

if __name__ == "__main__":
    main()
