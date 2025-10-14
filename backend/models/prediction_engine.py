#!/usr/bin/env python3
"""
NBA Betting Prediction Engine
=============================
Make real-time predictions for player performance in upcoming games
"""

import pandas as pd
import numpy as np
import joblib
import json
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')


class NBAPredictionEngine:
    """Generate predictions for NBA player performance"""
    
    def __init__(self, data_dir: str = "player_data"):
        self.data_dir = data_dir
        
        # Target stats we can predict
        self.target_stats = ['PTS', 'OREB', 'DREB', 'AST', 'FTM', 'FG3M', 'FGM', 'STL', 'BLK', 'TOV']
    
    def load_player_model(self, player_name: str, stat: str):
        """Load a trained model for a player and stat"""
        player_dir = Path(self.data_dir) / player_name.replace(' ', '_')
        model_file = player_dir / 'models' / f'{stat}_model.pkl'
        metrics_file = player_dir / 'models' / f'{stat}_metrics.json'
        
        if not model_file.exists():
            return None, None
        
        # Load model
        model_data = joblib.load(model_file)
        
        # Load metrics
        metrics = None
        if metrics_file.exists():
            with open(metrics_file, 'r') as f:
                metrics = json.load(f)
        
        return model_data, metrics
    
    def get_player_recent_stats(self, player_name: str, n_games: int = 10):
        """Get player's recent game statistics"""
        player_dir = Path(self.data_dir) / player_name.replace(' ', '_')
        features_file = player_dir / f"{player_name.replace(' ', '_')}_features.csv"
        
        if not features_file.exists():
            return None
        
        # Load features
        df = pd.read_csv(features_file)
        
        # Get most recent games
        if 'GAME_DATE' in df.columns:
            df['GAME_DATE'] = pd.to_datetime(df['GAME_DATE'])
            df = df.sort_values('GAME_DATE', ascending=False)
        
        return df.head(n_games)
    
    def predict_stat(self, player_name: str, stat: str, game_context: dict = None):
        """
        Predict a single stat for a player
        
        Args:
            player_name: Name of the player
            stat: Stat to predict (PTS, REB, AST, etc.)
            game_context: Optional game context (opponent, is_home, rest_days, etc.)
            
        Returns:
            Dictionary with prediction and confidence
        """
        # Load model
        model_data, metrics = self.load_player_model(player_name, stat)
        
        if model_data is None:
            return {
                'player': player_name,
                'stat': stat,
                'prediction': None,
                'confidence': None,
                'error': 'No model available'
            }
        
        # Get recent stats to build features
        recent_stats = self.get_player_recent_stats(player_name, n_games=20)
        
        if recent_stats is None or len(recent_stats) == 0:
            return {
                'player': player_name,
                'stat': stat,
                'prediction': None,
                'confidence': None,
                'error': 'No recent data available'
            }
        
        # Build feature vector for prediction
        feature_vector = self._build_feature_vector(recent_stats, game_context, model_data['feature_names'])
        
        if feature_vector is None:
            return {
                'player': player_name,
                'stat': stat,
                'prediction': None,
                'confidence': None,
                'error': 'Could not build feature vector'
            }
        
        # Make prediction
        try:
            # Scale features
            feature_vector_scaled = model_data['scaler'].transform([feature_vector])
            
            # Predict
            prediction = model_data['model'].predict(feature_vector_scaled)[0]
            
            # Get confidence from model metrics
            confidence = self._calculate_confidence(prediction, metrics)
            
            return {
                'player': player_name,
                'stat': stat,
                'prediction': round(prediction, 2),
                'confidence_interval': confidence,
                'mae': metrics['metrics']['MAE'] if metrics else None,
                'model_type': model_data['model_type'],
                'based_on_games': len(recent_stats)
            }
            
        except Exception as e:
            return {
                'player': player_name,
                'stat': stat,
                'prediction': None,
                'confidence': None,
                'error': str(e)
            }
    
    def _build_feature_vector(self, recent_stats: pd.DataFrame, game_context: dict, feature_names: list):
        """Build feature vector for prediction from recent stats and game context"""
        try:
            # Use AVERAGE of recent games as base (more stable than single game)
            # For rolling features, use most recent game
            # For other features, average last 5 games
            
            latest_game = recent_stats.iloc[0]
            recent_5 = recent_stats.head(5)
            
            # Create feature dictionary
            features = {}
            
            # Add features
            for feat in feature_names:
                if feat in latest_game.index:
                    # Use most recent for rolling/temporal features
                    if any(x in feat for x in ['_L3_', '_L5_', '_L10_', '_L20_', 'CAREER', 'LAST', 
                                                'ARCHETYPE', 'STRENGTH', 'ENCODED', 'SEASON']):
                        features[feat] = latest_game[feat]
                    # Average recent games for other features
                    elif feat in recent_5.columns:
                        features[feat] = recent_5[feat].mean()
                    else:
                        features[feat] = latest_game[feat]
                else:
                    features[feat] = 0
            
            # Update with game context if provided
            if game_context:
                # Update contextual features
                if 'is_home' in game_context:
                    if 'IS_HOME' in features:
                        features['IS_HOME'] = 1 if game_context['is_home'] else 0
                
                if 'rest_days' in game_context:
                    if 'DAYS_SINCE_LAST_GAME' in features:
                        features['DAYS_SINCE_LAST_GAME'] = game_context['rest_days']
                    if 'REST_ENCODED' in features:
                        if game_context['rest_days'] == 0:
                            features['REST_ENCODED'] = 0
                        elif game_context['rest_days'] == 1:
                            features['REST_ENCODED'] = 1
                        else:
                            features['REST_ENCODED'] = 2
                    if 'IS_BACK_TO_BACK' in features:
                        features['IS_BACK_TO_BACK'] = 1 if game_context['rest_days'] <= 1 else 0
                    if 'IS_WELL_RESTED' in features:
                        features['IS_WELL_RESTED'] = 1 if game_context['rest_days'] >= 3 else 0
                
                if 'opponent' in game_context:
                    # Opponent encoding would go here
                    pass
            
            # Convert to array in correct order
            feature_vector = [features.get(feat, 0) for feat in feature_names]
            
            # Handle NaN and inf
            feature_vector = np.array(feature_vector)
            feature_vector = np.nan_to_num(feature_vector, nan=0.0, posinf=0.0, neginf=0.0)
            
            return feature_vector
            
        except Exception as e:
            print(f"Error building feature vector: {e}")
            return None
    
    def _calculate_confidence(self, prediction: float, metrics: dict):
        """Calculate confidence interval for prediction"""
        if metrics is None:
            return None
        
        mae = metrics['metrics']['MAE']
        
        # 68% confidence interval (±1 MAE)
        # 95% confidence interval (±2 MAE)
        return {
            '68%': [round(prediction - mae, 2), round(prediction + mae, 2)],
            '95%': [round(prediction - 2*mae, 2), round(prediction + 2*mae, 2)]
        }
    
    def predict_all_stats(self, player_name: str, game_context: dict = None):
        """
        Predict all available stats for a player
        
        Args:
            player_name: Name of the player
            game_context: Game context (opponent, home/away, rest, etc.)
            
        Returns:
            Dictionary with all predictions
        """
        print(f"🏀 Generating Predictions for {player_name}")
        print("=" * 60)
        
        predictions = {}
        
        for stat in self.target_stats:
            result = self.predict_stat(player_name, stat, game_context)
            predictions[stat] = result
        
        return predictions
    
    def print_prediction_summary(self, predictions: dict, betting_lines: dict = None):
        """Print formatted prediction summary"""
        player_name = next(iter(predictions.values()))['player']
        
        print(f"\n📊 Prediction Summary for {player_name}")
        print("=" * 70)
        
        for stat, pred in predictions.items():
            if pred['prediction'] is not None:
                print(f"\n{stat}:")
                print(f"  Prediction: {pred['prediction']:.1f}")
                if pred['confidence_interval']:
                    print(f"  68% CI: {pred['confidence_interval']['68%'][0]:.1f} - {pred['confidence_interval']['68%'][1]:.1f}")
                    print(f"  95% CI: {pred['confidence_interval']['95%'][0]:.1f} - {pred['confidence_interval']['95%'][1]:.1f}")
                print(f"  Model: {pred.get('model_type', 'Unknown')} (MAE: {pred.get('mae', 'N/A')})")
                
                # Compare to betting line if provided
                if betting_lines and stat in betting_lines:
                    line = betting_lines[stat]
                    diff = pred['prediction'] - line
                    recommendation = "OVER" if diff > pred.get('mae', 0) else "UNDER" if diff < -pred.get('mae', 0) else "SKIP"
                    print(f"  Betting Line: {line}")
                    print(f"  Recommendation: {recommendation} ({diff:+.1f})")
            else:
                print(f"\n{stat}: ❌ {pred.get('error', 'Unknown error')}")
    
    def compare_players(self, player1: str, player2: str, stat: str, game_context: dict = None):
        """Compare predicted performance of two players"""
        pred1 = self.predict_stat(player1, stat, game_context)
        pred2 = self.predict_stat(player2, stat, game_context)
        
        print(f"\n🔍 Head-to-Head Comparison: {stat}")
        print("=" * 60)
        print(f"{player1:30s}: {pred1['prediction']:.1f} ± {pred1.get('mae', 0):.2f}")
        print(f"{player2:30s}: {pred2['prediction']:.1f} ± {pred2.get('mae', 0):.2f}")
        
        if pred1['prediction'] and pred2['prediction']:
            diff = pred1['prediction'] - pred2['prediction']
            print(f"\nDifference: {diff:+.1f} (in favor of {player1 if diff > 0 else player2})")


def main():
    """Main execution"""
    print("🏀 NBA Betting Prediction Engine")
    print("=" * 60)
    print("1. Predict single player stats")
    print("2. Predict with game context")
    print("3. Compare two players")
    print("4. Test prediction accuracy")
    print("5. Exit")
    print()
    
    choice = input("Select option (1-5): ").strip()
    
    engine = NBAPredictionEngine()
    
    if choice == "1":
        player_name = input("Enter player name: ").strip()
        predictions = engine.predict_all_stats(player_name)
        engine.print_prediction_summary(predictions)
    
    elif choice == "2":
        player_name = input("Enter player name: ").strip()
        is_home = input("Home game? (yes/no): ").strip().lower() == 'yes'
        rest_days = int(input("Days of rest: ") or "1")
        
        game_context = {
            'is_home': is_home,
            'rest_days': rest_days
        }
        
        predictions = engine.predict_all_stats(player_name, game_context)
        engine.print_prediction_summary(predictions)
    
    elif choice == "3":
        player1 = input("Enter first player name: ").strip()
        player2 = input("Enter second player name: ").strip()
        stat = input("Enter stat to compare (PTS/REB/AST/etc.): ").strip().upper()
        
        engine.compare_players(player1, player2, stat)
    
    elif choice == "4":
        print("\n🧪 Testing prediction accuracy...")
        print("(This would test predictions on recent games)")
        print("Coming soon!")
    
    elif choice == "5":
        print("👋 Goodbye!")
    
    else:
        print("❌ Invalid choice")


if __name__ == "__main__":
    main()

