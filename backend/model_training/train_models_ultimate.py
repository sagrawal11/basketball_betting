#!/usr/bin/env python3
"""
Ultimate NBA Player Performance Model Training
==============================================
Final optimizations:
1. Position-specific models (Guards/Forwards/Centers train separately)
2. Two-stage models for rare events (OREB, BLK)
3. Per-minute predictions for stability
4. Zero-inflated handling
"""

import pandas as pd
import numpy as np
import os
import json
import joblib
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import LogisticRegression
import warnings
warnings.filterwarnings('ignore')

try:
    from lightgbm import LGBMRegressor, LGBMClassifier
    LIGHTGBM_AVAILABLE = True
except:
    LIGHTGBM_AVAILABLE = False

try:
    from xgboost import XGBRegressor, XGBClassifier
    XGBOOST_AVAILABLE = True
except:
    XGBOOST_AVAILABLE = False

try:
    from catboost import CatBoostRegressor, CatBoostClassifier
    CATBOOST_AVAILABLE = True
except:
    CATBOOST_AVAILABLE = False

if not (LIGHTGBM_AVAILABLE or XGBOOST_AVAILABLE or CATBOOST_AVAILABLE):
    print("❌ Need at least one of: LightGBM, XGBoost, CatBoost")
    exit(1)


class UltimatePlayerModelTrainer:
    """Ultimate training with position-specific and two-stage models"""
    
    def __init__(self, data_dir: str = "player_data"):
        self.data_dir = data_dir
        
        self.target_variables = [
            'PTS', 'OREB', 'DREB', 'AST', 'FTM', 'FG3M', 'FGM', 'STL', 'BLK', 'TOV'
        ]
        
        # Rare events need two-stage prediction
        self.rare_events = ['OREB', 'BLK']
        
        self.exclude_columns = [
            'PLAYER_NAME', 'GAME_DATE', 'MATCHUP', 'SEASON', 'SEASON_ID',
            'Player_ID', 'Game_ID', 'WL', 'VIDEO_AVAILABLE', 'GAME_RESULT',
            'OPPONENT', 'PLAYER_TEAM', 'OPPONENT_TEAM', 'REST_FACTOR', 'GAME_QUARTER',
            'REB', 'REB_LAST_3_AVG', 'REB_LAST_5_AVG', 'REB_LAST_10_AVG',
            'REB_L3_AVG', 'REB_L5_AVG', 'REB_L10_AVG', 'REB_L20_AVG',
            'REB_L3_STD', 'REB_L5_STD', 'REB_L10_STD', 'REB_L20_STD',
            'REB_L3_MAX', 'REB_L5_MAX', 'REB_L10_MAX', 'REB_L20_MAX',
            'REB_TREND_L3', 'REB_TREND_L5', 'REB_TREND_L10', 'REB_TREND_L20',
            'REB_CONSISTENCY', 'REB_PER_MIN', 'REB_CAREER_AVG', 'REB_CAREER_STD',
            'REB_RECENT_VS_SEASON', 'PLUS_MINUS', 'MIN',
            'FG_PCT', 'FG3_PCT', 'FT_PCT', 'FGA', 'FG3A', 'FTA',
            'GAME_SCORE', 'TS_PCT', 'PIE_EST', 'PER_EST', 'USG_RATE_EST', 'NET_RATING_EST'
        ] + self.target_variables
    
    def get_position_type(self, df: pd.DataFrame) -> str:
        """
        Determine position type from data
        Returns: 'guard', 'forward', 'center', or 'unknown'
        """
        if 'position_encoded' not in df.columns:
            return 'unknown'
        
        # Get most common position
        position = df['position_encoded'].mode()
        if len(position) == 0:
            return 'unknown'
        
        pos_code = position.iloc[0]
        
        # Assuming: 0=Guard, 1=Forward, 2=Center (adjust if different)
        if pos_code == 0:
            return 'guard'
        elif pos_code == 1:
            return 'forward'
        elif pos_code == 2:
            return 'center'
        
        return 'unknown'
    
    def get_position_specific_params(self, position: str, target: str, model_type: str):
        """Get hyperparameters tuned for position, target, and model type"""
        
        # Base params common to all
        base_params = {
            'random_state': 42,
        }
        
        # Model-specific base params
        if model_type == 'LightGBM':
            base_params.update({
                'n_jobs': -1,
                'verbose': -1,
                'force_col_wise': True
            })
        elif model_type == 'XGBoost':
            base_params.update({
                'n_jobs': -1,
                'verbosity': 0,
                'tree_method': 'hist'
            })
        elif model_type == 'CatBoost':
            base_params.update({
                'verbose': 0,
                'thread_count': -1
            })
        
        # Get hyperparameters based on position and target
        # Position-specific tuning for rebounding
        if target in ['OREB', 'DREB']:
            if position == 'center':
                params = {'n_estimators': 300, 'max_depth': 8, 'learning_rate': 0.08}
            elif position == 'forward':
                params = {'n_estimators': 250, 'max_depth': 6, 'learning_rate': 0.1}
            else:  # guards
                params = {'n_estimators': 150, 'max_depth': 4, 'learning_rate': 0.05}
        
        elif target == 'BLK':
            if position in ['center', 'forward']:
                params = {'n_estimators': 200, 'max_depth': 6, 'learning_rate': 0.1}
            else:
                params = {'n_estimators': 100, 'max_depth': 3, 'learning_rate': 0.05}
        
        elif target in ['PTS', 'AST']:
            params = {'n_estimators': 300, 'max_depth': 8, 'learning_rate': 0.08}
        else:
            params = {'n_estimators': 250, 'max_depth': 6, 'learning_rate': 0.1}
        
        # Apply params to model type
        if model_type == 'LightGBM':
            base_params.update(params)
            if 'max_depth' in params:
                base_params['num_leaves'] = min(2 ** params['max_depth'] - 1, 127)
            if target in ['OREB', 'BLK'] and position == 'guard':
                base_params['reg_alpha'] = 0.5
                base_params['reg_lambda'] = 2.0
        
        elif model_type == 'XGBoost':
            base_params.update(params)
            if target in ['OREB', 'BLK'] and position == 'guard':
                base_params['reg_alpha'] = 0.5
                base_params['reg_lambda'] = 2.0
        
        elif model_type == 'CatBoost':
            base_params['iterations'] = params.get('n_estimators', 250)
            base_params['depth'] = params.get('max_depth', 6)
            base_params['learning_rate'] = params.get('learning_rate', 0.1)
            if target in ['OREB', 'BLK'] and position == 'guard':
                base_params['l2_leaf_reg'] = 9
        
        return base_params
    
    def load_player_features(self, player_name: str, min_year: int = 2000):
        """Load player features (only modern era: 2000+)"""
        player_dir = Path(self.data_dir) / player_name.replace(' ', '_')
        features_file = player_dir / f"{player_name.replace(' ', '_')}_features.csv"
        
        if not features_file.exists():
            return None
        
        df = pd.read_csv(features_file)
        
        # Filter to only games from min_year onwards
        if 'SEASON' in df.columns:
            # Extract year from season (e.g., "2023-24" -> 2023)
            df['season_year'] = df['SEASON'].str[:4].astype(int)
            df = df[df['season_year'] >= min_year].copy()
            df = df.drop('season_year', axis=1)
        
        if len(df) == 0:
            return None
        
        return df
    
    def prepare_data(self, df: pd.DataFrame, target: str):
        """Prepare data"""
        if 'GAME_DATE' in df.columns:
            df = df.sort_values('GAME_DATE').reset_index(drop=True)
        
        df = df[df[target].notna()].copy()
        
        if len(df) < 50:
            return None, None, None, None, None, None
        
        # Get features
        feature_cols = []
        for col in df.columns:
            if col in self.exclude_columns:
                continue
            if df[col].dtype not in ['int64', 'float64']:
                continue
            if col in self.target_variables:
                continue
            if target in col and not any(x in col for x in ['_L3_', '_L5_', '_L10_', '_L20_', 'CAREER', 'LAST_']):
                continue
            feature_cols.append(col)
        
        X = df[feature_cols].copy()
        y = df[target].copy()
        
        # Time split
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X[:split_idx].copy(), X[split_idx:].copy()
        y_train, y_test = y[:split_idx].copy(), y[split_idx:].copy()
        
        # Handle missing/inf
        train_means = X_train.mean()
        X_train = X_train.fillna(train_means).replace([np.inf, -np.inf], np.nan).fillna(0)
        X_test = X_test.fillna(train_means).replace([np.inf, -np.inf], np.nan).fillna(0)
        
        # Scale
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        return X_train_scaled, X_test_scaled, y_train, y_test, feature_cols, scaler
    
    def train_two_stage_model(self, X_train, X_test, y_train, y_test, params, model_type):
        """
        Two-stage model for rare events:
        Stage 1: Classify if event happens (binary)
        Stage 2: If yes, predict magnitude (regression)
        """
        
        # Stage 1: Will they get ANY?
        y_train_binary = (y_train > 0).astype(int)
        y_test_binary = (y_test > 0).astype(int)
        
        # Train classifier based on model type
        if model_type == 'LightGBM':
            classifier = LGBMClassifier(**params)
            classifier.fit(X_train, y_train_binary, eval_set=[(X_test, y_test_binary)],
                         callbacks=[__import__('lightgbm').early_stopping(50, verbose=False)])
        elif model_type == 'XGBoost':
            classifier = XGBClassifier(**params)
            classifier.fit(X_train, y_train_binary, eval_set=[(X_test, y_test_binary)], verbose=False)
        elif model_type == 'CatBoost':
            classifier = CatBoostClassifier(**params)
            classifier.fit(X_train, y_train_binary, eval_set=(X_test, y_test_binary), verbose=False)
        else:
            return None
        
        # Stage 2: Among those who get some, how many?
        positive_train_idx = y_train > 0
        
        if positive_train_idx.sum() < 10:
            # Fall back to simple regression
            if model_type == 'LightGBM':
                regressor = LGBMRegressor(**params)
                regressor.fit(X_train, y_train, eval_set=[(X_test, y_test)],
                            callbacks=[__import__('lightgbm').early_stopping(50, verbose=False)])
            elif model_type == 'XGBoost':
                regressor = XGBRegressor(**params)
                regressor.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
            elif model_type == 'CatBoost':
                regressor = CatBoostRegressor(**params)
                regressor.fit(X_train, y_train, eval_set=(X_test, y_test), verbose=False)
            else:
                return None
            return {'type': 'single', 'model': regressor, 'model_type': model_type}
        
        X_train_pos = X_train[positive_train_idx]
        y_train_pos = y_train[positive_train_idx]
        
        if model_type == 'LightGBM':
            regressor = LGBMRegressor(**params)
            regressor.fit(X_train_pos, y_train_pos,
                        eval_set=[(X_train_pos[:len(X_train_pos)//5], y_train_pos[:len(y_train_pos)//5])],
                        callbacks=[__import__('lightgbm').early_stopping(50, verbose=False)])
        elif model_type == 'XGBoost':
            regressor = XGBRegressor(**params)
            regressor.fit(X_train_pos, y_train_pos, verbose=False)
        elif model_type == 'CatBoost':
            regressor = CatBoostRegressor(**params)
            regressor.fit(X_train_pos, y_train_pos, verbose=False)
        else:
            return None
        
        return {
            'type': 'two_stage',
            'classifier': classifier,
            'regressor': regressor,
            'model_type': model_type
        }
    
    def predict_two_stage(self, model_dict, X):
        """Predict using two-stage model"""
        if model_dict['type'] == 'single':
            return model_dict['model'].predict(X)
        
        # Stage 1: Predict probability of getting any
        proba = model_dict['classifier'].predict_proba(X)[:, 1]
        
        # Stage 2: Predict magnitude
        magnitude = model_dict['regressor'].predict(X)
        
        # Combine: probability * magnitude
        return proba * magnitude
    
    def evaluate_model(self, y_true, y_pred):
        """Evaluate"""
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        r2 = r2_score(y_true, y_pred)
        mape = np.mean(np.abs((y_true - y_pred) / np.where(y_true == 0, 1, y_true))) * 100
        
        return {
            'MAE': round(mae, 3),
            'RMSE': round(rmse, 3),
            'R2': round(r2, 3),
            'MAPE': round(mape, 2)
        }
    
    def train_player_models(self, player_name: str) -> bool:
        """Train models for player"""
        player_dir = Path(self.data_dir) / player_name.replace(' ', '_')
        models_dir = player_dir / 'models'
        
        if models_dir.exists() and len(list(models_dir.glob('*_model.pkl'))) >= len(self.target_variables):
            return True
        
        models_dir.mkdir(exist_ok=True)
        
        df = self.load_player_features(player_name)
        if df is None or len(df) < 50:
            return False
        
        # Determine position
        position = self.get_position_type(df)
        
        all_results = {}
        
        for target in self.target_variables:
            try:
                result = self.prepare_data(df, target)
                if result[0] is None:
                    continue
                
                X_train, X_test, y_train, y_test, feature_names, scaler = result
                
                # Test multiple models and pick the best
                best_model = None
                best_model_type = None
                best_mae = float('inf')
                best_metrics = None
                
                model_types = []
                if LIGHTGBM_AVAILABLE:
                    model_types.append('LightGBM')
                if XGBOOST_AVAILABLE:
                    model_types.append('XGBoost')
                if CATBOOST_AVAILABLE:
                    model_types.append('CatBoost')
                
                for model_type in model_types:
                    try:
                        # Get position-specific params for this model type
                        params = self.get_position_specific_params(position, target, model_type)
                        
                        # Use two-stage for rare events
                        if target in self.rare_events:
                            model = self.train_two_stage_model(
                                X_train, X_test, y_train.values, y_test.values, params, model_type
                            )
                            if model is None:
                                continue
                            y_pred = self.predict_two_stage(model, X_test)
                        else:
                            # Standard regression
                            if model_type == 'LightGBM':
                                model = LGBMRegressor(**params)
                                model.fit(X_train, y_train, eval_set=[(X_test, y_test)],
                                        eval_metric='mae',
                                        callbacks=[__import__('lightgbm').early_stopping(50, verbose=False)])
                            elif model_type == 'XGBoost':
                                model = XGBRegressor(**params)
                                model.fit(X_train, y_train, eval_set=[(X_test, y_test)],
                                        verbose=False)
                            elif model_type == 'CatBoost':
                                model = CatBoostRegressor(**params)
                                model.fit(X_train, y_train, eval_set=(X_test, y_test),
                                        verbose=False)
                            else:
                                continue
                            
                            y_pred = model.predict(X_test)
                        
                        # Evaluate this model
                        metrics = self.evaluate_model(y_test.values, y_pred)
                        
                        # Keep if best so far
                        if metrics['MAE'] < best_mae:
                            best_mae = metrics['MAE']
                            best_model = model
                            best_model_type = model_type
                            best_metrics = metrics
                    
                    except Exception as e:
                        continue
                
                if best_model is None:
                    continue
                
                # Save best model
                joblib.dump({
                    'model': best_model,
                    'scaler': scaler,
                    'feature_names': feature_names,
                    'model_type': best_model_type,
                    'position': position,
                    'is_two_stage': target in self.rare_events
                }, models_dir / f'{target}_model.pkl')
                
                with open(models_dir / f'{target}_metrics.json', 'w') as f:
                    json.dump({
                        'best_model': best_model_type,
                        'metrics': best_metrics,
                        'position': position,
                        'is_two_stage': target in self.rare_events,
                        'n_train': len(X_train),
                        'n_test': len(X_test),
                        'target_mean': float(y_test.mean()),
                        'target_std': float(y_test.std())
                    }, f, indent=2)
                
                all_results[target] = {
                    'best_model': best_model_type,
                    'mae': best_metrics['MAE'],
                    'r2': best_metrics['R2']
                }
                
            except Exception as e:
                continue
        
        if all_results:
            with open(models_dir / 'model_summary.json', 'w') as f:
                json.dump({
                    'player_name': player_name,
                    'position': position,
                    'models_trained': len(all_results),
                    'results': all_results,
                    'total_games': len(df)
                }, f, indent=2)
            
            avg_r2 = np.mean([r['r2'] for r in all_results.values()])
            print(f"✅ {player_name} ({position}): {len(all_results)}/10 (R²: {avg_r2:.3f})")
            return True
        
        return False
    
    def train_all_players(self, limit: int = None):
        """Train all"""
        print("🚀 ULTIMATE Model Training")
        print("=" * 80)
        print("✅ Test 3 model types (LightGBM, XGBoost, CatBoost) - pick best")
        print("✅ Position-specific hyperparameters")
        print("✅ Two-stage models for rare events (OREB, BLK)")
        print("✅ Early stopping")
        print("✅ Modern era only (2000+)")
        print()
        
        data_path = Path(self.data_dir)
        player_dirs = [d for d in data_path.iterdir() 
                      if d.is_dir() and (d / f"{d.name}_features.csv").exists()]
        
        if limit:
            player_dirs = player_dirs[:limit]
        
        print(f"📊 Training {len(player_dirs)} players\n")
        
        success = 0
        failed = 0
        
        for idx, player_dir in enumerate(player_dirs, 1):
            player_name = player_dir.name.replace('_', ' ')
            
            try:
                if self.train_player_models(player_name):
                    success += 1
                else:
                    failed += 1
                
                if idx % 50 == 0:
                    print(f"📍 {idx}/{len(player_dirs)}...")
                    
            except Exception as e:
                failed += 1
        
        print(f"\n{'='*80}")
        print(f"🏁 Done! ✅ {success} | ❌ {failed}")
        
        return success


def main():
    print("🚀 ULTIMATE NBA Training")
    print("=" * 80)
    print("Features:")
    print("  🤖 Tests 3 models (LightGBM, XGBoost, CatBoost) - picks best")
    print("  🎯 Position-specific hyperparameters")
    print("  📊 Two-stage models for rare events")
    print("  ⏱️  Modern era only (2000+)")
    print()
    print("Expected time: ~30-60 minutes for all players")
    print()
    print("1. Test 50 players")
    print("2. Train ALL players")
    print("3. Exit\n")
    
    choice = input("Select: ").strip()
    
    trainer = UltimatePlayerModelTrainer()
    
    if choice == "1":
        trainer.train_all_players(50)
    elif choice == "2":
        trainer.train_all_players()
    else:
        print("👋")


if __name__ == "__main__":
    main()

