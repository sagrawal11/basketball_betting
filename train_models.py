#!/usr/bin/env python3
"""
NBA Player Performance Model Training
====================================
Trains personalized models for each player to predict their performance stats
"""

import pandas as pd
import numpy as np
import os
import json
import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
import warnings
warnings.filterwarnings('ignore')

# Import gradient boosting libraries
try:
    from xgboost import XGBRegressor
    XGBOOST_AVAILABLE = True
except:
    XGBOOST_AVAILABLE = False
    print("⚠️  XGBoost not available")

try:
    from lightgbm import LGBMRegressor
    LIGHTGBM_AVAILABLE = True
except:
    LIGHTGBM_AVAILABLE = False
    print("⚠️  LightGBM not available")

try:
    from catboost import CatBoostRegressor
    CATBOOST_AVAILABLE = True
except:
    CATBOOST_AVAILABLE = False
    print("⚠️  CatBoost not available")


class PlayerModelTrainer:
    """Train prediction models for a single player"""
    
    def __init__(self, data_dir: str = "player_data"):
        self.data_dir = data_dir
        
        # Target variables to predict
        self.target_variables = [
            'PTS',      # Points
            'OREB',     # Offensive Rebounds
            'DREB',     # Defensive Rebounds
            'AST',      # Assists
            'FTM',      # Free Throws Made
            'FG3M',     # 3-Pointers Made
            'FGM',      # Field Goals Made
            'STL',      # Steals
            'BLK',      # Blocks
            'TOV'       # Turnovers
        ]
        
        # Columns to exclude from features
        self.exclude_columns = [
            'PLAYER_NAME', 'GAME_DATE', 'MATCHUP', 'SEASON', 'SEASON_ID',
            'Player_ID', 'Game_ID', 'WL', 'VIDEO_AVAILABLE', 'GAME_RESULT',
            'OPPONENT', 'PLAYER_TEAM', 'OPPONENT_TEAM', 'REST_FACTOR',
            'GAME_QUARTER'  # These are either IDs, dates, or already encoded
        ] + self.target_variables  # Don't use targets as features
    
    def get_models(self):
        """Get all models to test"""
        models = {}
        
        # Gradient Boosting Models (usually best for tabular data)
        if XGBOOST_AVAILABLE:
            models['XGBoost'] = XGBRegressor(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=42,
                n_jobs=-1
            )
        
        if LIGHTGBM_AVAILABLE:
            models['LightGBM'] = LGBMRegressor(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=42,
                n_jobs=-1,
                verbose=-1
            )
        
        if CATBOOST_AVAILABLE:
            models['CatBoost'] = CatBoostRegressor(
                iterations=100,
                depth=6,
                learning_rate=0.1,
                random_state=42,
                verbose=0
            )
        
        # Ensemble Methods
        models['RandomForest'] = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        )
        
        models['GradientBoosting'] = GradientBoostingRegressor(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            random_state=42
        )
        
        models['AdaBoost'] = AdaBoostRegressor(
            n_estimators=100,
            learning_rate=0.1,
            random_state=42
        )
        
        # Linear Models (fast, interpretable)
        models['Ridge'] = Ridge(alpha=1.0, random_state=42)
        models['Lasso'] = Lasso(alpha=1.0, random_state=42, max_iter=2000)
        models['ElasticNet'] = ElasticNet(alpha=1.0, random_state=42, max_iter=2000)
        
        # Neural Network
        models['NeuralNet'] = MLPRegressor(
            hidden_layer_sizes=(100, 50),
            max_iter=500,
            random_state=42,
            early_stopping=True
        )
        
        return models
    
    def load_player_features(self, player_name: str) -> pd.DataFrame:
        """Load player's feature data"""
        player_dir = Path(self.data_dir) / player_name.replace(' ', '_')
        features_file = player_dir / f"{player_name.replace(' ', '_')}_features.csv"
        
        if not features_file.exists():
            return None
        
        df = pd.read_csv(features_file)
        return df
    
    def prepare_data(self, df: pd.DataFrame, target: str):
        """
        Prepare data for training with time-based split
        
        Args:
            df: DataFrame with features
            target: Target variable to predict
            
        Returns:
            X_train, X_test, y_train, y_test, feature_names, scaler
        """
        # Sort by date to ensure proper time-based split
        if 'GAME_DATE' in df.columns:
            df = df.sort_values('GAME_DATE').reset_index(drop=True)
        
        # Remove rows where target is NaN
        df = df[df[target].notna()].copy()
        
        if len(df) < 50:  # Need minimum games for training
            return None, None, None, None, None, None
        
        # Get feature columns - exclude target-derived features to avoid leakage
        # For each target, exclude any feature that contains that target name
        # (except rolling averages which use past data)
        feature_cols = []
        for col in df.columns:
            # Skip if in exclude list
            if col in self.exclude_columns:
                continue
            
            # Skip if not numeric
            if df[col].dtype not in ['int64', 'float64']:
                continue
            
            # Skip if this column contains the current target (data leakage)
            # BUT allow rolling features (_L3_, _L5_, etc.) and career features
            if target in col and not any(x in col for x in ['_L3_', '_L5_', '_L10_', '_L20_', 'CAREER', 'LAST_']):
                continue
            
            feature_cols.append(col)
        
        X = df[feature_cols].copy()
        y = df[target].copy()
        
        # Time-based split: 80% train, 20% test (most recent)
        # IMPORTANT: Split BEFORE any data processing to avoid leakage
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X[:split_idx].copy(), X[split_idx:].copy()
        y_train, y_test = y[:split_idx].copy(), y[split_idx:].copy()
        
        # Handle missing values using TRAIN mean only (no leakage)
        train_means = X_train.mean()
        X_train = X_train.fillna(train_means)
        X_test = X_test.fillna(train_means)
        
        # Handle infinite values
        X_train = X_train.replace([np.inf, -np.inf], np.nan).fillna(0)
        X_test = X_test.replace([np.inf, -np.inf], np.nan).fillna(0)
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Debug: Print shapes
        # print(f"    Train shape: {X_train_scaled.shape}, Test shape: {X_test_scaled.shape}")
        # print(f"    y_train range: [{y_train.min():.1f}, {y_train.max():.1f}]")
        # print(f"    y_test range: [{y_test.min():.1f}, {y_test.max():.1f}]")
        
        return X_train_scaled, X_test_scaled, y_train, y_test, feature_cols, scaler
    
    def evaluate_model(self, y_true, y_pred):
        """Calculate evaluation metrics"""
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        r2 = r2_score(y_true, y_pred)
        
        # MAPE (handle zeros)
        mape = np.mean(np.abs((y_true - y_pred) / np.where(y_true == 0, 1, y_true))) * 100
        
        return {
            'MAE': round(mae, 3),
            'RMSE': round(rmse, 3),
            'R2': round(r2, 3),
            'MAPE': round(mape, 2)
        }
    
    def train_player_models(self, player_name: str) -> bool:
        """
        Train models for a single player
        
        Args:
            player_name: Name of the player
            
        Returns:
            True if successful, False otherwise
        """
        player_dir = Path(self.data_dir) / player_name.replace(' ', '_')
        models_dir = player_dir / 'models'
        
        # Check if already trained
        if models_dir.exists() and len(list(models_dir.glob('*_model.pkl'))) >= len(self.target_variables):
            print(f"⏭️  {player_name}: Models already exist, skipping...")
            return True
        
        # Create models directory
        models_dir.mkdir(exist_ok=True)
        
        # Load features
        df = self.load_player_features(player_name)
        if df is None or len(df) < 50:
            print(f"❌ {player_name}: Insufficient data (need 50+ games)")
            return False
        
        # Train models for each target
        all_results = {}
        
        for target in self.target_variables:
            try:
                # Prepare data
                X_train, X_test, y_train, y_test, feature_names, scaler = self.prepare_data(df, target)
                
                if X_train is None:
                    continue
                
                # Test all models
                models = self.get_models()
                results = {}
                trained_models = {}
                
                for model_name, model in models.items():
                    try:
                        # Train model
                        model.fit(X_train, y_train)
                        
                        # Predict on test set
                        y_pred = model.predict(X_test)
                        
                        # Evaluate
                        metrics = self.evaluate_model(y_test, y_pred)
                        results[model_name] = metrics
                        trained_models[model_name] = model
                        
                    except Exception as e:
                        print(f"⚠️  {player_name} - {target} - {model_name}: {e}")
                        continue
                
                if not results:
                    continue
                
                # Select best model based on MAE
                best_model_name = min(results, key=lambda x: results[x]['MAE'])
                best_model = trained_models[best_model_name]
                best_metrics = results[best_model_name]
                
                # Save best model
                model_path = models_dir / f'{target}_model.pkl'
                joblib.dump({
                    'model': best_model,
                    'scaler': scaler,
                    'feature_names': feature_names,
                    'model_type': best_model_name
                }, model_path)
                
                # Save metrics
                metrics_path = models_dir / f'{target}_metrics.json'
                with open(metrics_path, 'w') as f:
                    json.dump({
                        'best_model': best_model_name,
                        'metrics': best_metrics,
                        'all_models': results,
                        'n_train': len(X_train),
                        'n_test': len(X_test),
                        'target_mean': float(y_test.mean()),
                        'target_std': float(y_test.std())
                    }, f, indent=2)
                
                all_results[target] = {
                    'best_model': best_model_name,
                    'mae': best_metrics['MAE']
                }
                
            except Exception as e:
                print(f"❌ {player_name} - {target}: {e}")
                continue
        
        # Save summary
        if all_results:
            summary_path = models_dir / 'model_summary.json'
            with open(summary_path, 'w') as f:
                json.dump({
                    'player_name': player_name,
                    'models_trained': len(all_results),
                    'results': all_results,
                    'total_games': len(df)
                }, f, indent=2)
            
            print(f"✅ {player_name}: Trained {len(all_results)}/{len(self.target_variables)} models")
            return True
        else:
            print(f"❌ {player_name}: No models trained successfully")
            return False
    
    def train_all_players(self, limit: int = None):
        """
        Train models for all players
        
        Args:
            limit: Limit number of players to process (for testing)
        """
        print("🏀 Starting Model Training for All Players")
        print("=" * 60)
        
        data_path = Path(self.data_dir)
        player_dirs = [d for d in data_path.iterdir() 
                      if d.is_dir() and d.name != 'college']
        
        # Filter to only players with features
        player_dirs = [d for d in player_dirs 
                      if (d / f"{d.name}_features.csv").exists()]
        
        if limit:
            player_dirs = player_dirs[:limit]
        
        print(f"📊 Processing {len(player_dirs)} players...")
        print(f"🎯 Training {len(self.target_variables)} models per player")
        print(f"🤖 Testing {len(self.get_models())} model types")
        print()
        
        success_count = 0
        failed_count = 0
        skipped_count = 0
        
        for idx, player_dir in enumerate(player_dirs, 1):
            player_name = player_dir.name.replace('_', ' ')
            
            try:
                # Check if already trained
                models_dir = player_dir / 'models'
                if models_dir.exists() and len(list(models_dir.glob('*_model.pkl'))) >= len(self.target_variables):
                    skipped_count += 1
                    if idx % 50 == 0:
                        print(f"⏭️  Processed {idx}/{len(player_dirs)} players (skipped: {skipped_count})...")
                    continue
                
                if self.train_player_models(player_name):
                    success_count += 1
                else:
                    failed_count += 1
                
                if idx % 50 == 0:
                    print(f"✅ Processed {idx}/{len(player_dirs)} players...")
                    
            except Exception as e:
                print(f"❌ Error training {player_name}: {e}")
                failed_count += 1
        
        print()
        print("=" * 60)
        print("🏁 Model Training Complete!")
        print(f"✅ Successfully trained: {success_count} players")
        print(f"⏭️  Skipped (already exist): {skipped_count} players")
        print(f"❌ Failed: {failed_count} players")
        print(f"📁 Models saved to: data2/[Player_Name]/models/")
        
        return success_count


def main():
    """Main execution"""
    print("🏀 NBA Player Performance Model Training System")
    print("=" * 60)
    print("1. Test with single player")
    print("2. Test with limited players")
    print("3. Train all players")
    print("4. Exit")
    print()
    
    choice = input("Select option (1-4): ").strip()
    
    trainer = PlayerModelTrainer()
    
    if choice == "1":
        player_name = input("Enter player name (e.g., Kevin Durant): ").strip()
        if trainer.train_player_models(player_name):
            print(f"\n✅ Successfully trained models for {player_name}")
            
            # Show results
            player_dir = Path('data2') / player_name.replace(' ', '_')
            summary_file = player_dir / 'models' / 'model_summary.json'
            if summary_file.exists():
                with open(summary_file, 'r') as f:
                    summary = json.load(f)
                    print("\n📊 Model Summary:")
                    for target, info in summary['results'].items():
                        print(f"  {target}: {info['best_model']} (MAE: {info['mae']})")
        else:
            print(f"\n❌ Failed to train models for {player_name}")
    
    elif choice == "2":
        limit = int(input("How many players to process? ") or "10")
        trainer.train_all_players(limit=limit)
    
    elif choice == "3":
        confirm = input("Train ALL players? This will take several hours. (yes/no): ").strip().lower()
        if confirm == 'yes':
            trainer.train_all_players()
        else:
            print("❌ Training cancelled")
    
    elif choice == "4":
        print("👋 Goodbye!")
    
    else:
        print("❌ Invalid choice")


if __name__ == "__main__":
    main()

