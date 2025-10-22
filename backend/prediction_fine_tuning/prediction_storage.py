#!/usr/bin/env python3
"""
Prediction Storage System
=========================
Saves and loads predictions for historical comparison
"""

import json
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional


class PredictionStorage:
    """Store and retrieve predictions for historical analysis"""
    
    def __init__(self, storage_dir: str = "backend/prediction_fine_tuning/predictions"):
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        
        # Also create CSV for easy viewing
        self.csv_file = self.storage_dir / "all_predictions.csv"
        
    def save_game_prediction(self, game_id: str, game_date: str, 
                            home_team: str, away_team: str,
                            home_predicted_score: float, away_predicted_score: float,
                            player_predictions: List[Dict]):
        """Save predictions for a game"""
        
        prediction_data = {
            'game_id': game_id,
            'game_date': game_date,
            'timestamp': datetime.now().isoformat(),
            'home_team': home_team,
            'away_team': away_team,
            'home_predicted_score': home_predicted_score,
            'away_predicted_score': away_predicted_score,
            'player_predictions': player_predictions
        }
        
        # Save as JSON
        json_file = self.storage_dir / f"{game_id}.json"
        with open(json_file, 'w') as f:
            json.dump(prediction_data, f, indent=2)
        
        # Also append to CSV for easy analysis
        self._append_to_csv(prediction_data)
        
        print(f"💾 Saved predictions for game {game_id}")
        
    def _append_to_csv(self, prediction_data: Dict):
        """Append prediction to CSV file"""
        
        # Flatten player predictions
        rows = []
        for player in prediction_data['player_predictions']:
            row = {
                'game_id': prediction_data['game_id'],
                'game_date': prediction_data['game_date'],
                'timestamp': prediction_data['timestamp'],
                'home_team': prediction_data['home_team'],
                'away_team': prediction_data['away_team'],
                'home_predicted_score': prediction_data['home_predicted_score'],
                'away_predicted_score': prediction_data['away_predicted_score'],
                'player_name': player['name'],
                'player_team': player.get('team', ''),
                'is_home': player.get('is_home', False),
            }
            
            # Add all predicted stats
            if 'stats' in player:
                for stat, value in player['stats'].items():
                    row[f'predicted_{stat}'] = value
            
            rows.append(row)
        
        # Append to CSV
        df = pd.DataFrame(rows)
        if self.csv_file.exists():
            df.to_csv(self.csv_file, mode='a', header=False, index=False)
        else:
            df.to_csv(self.csv_file, index=False)
    
    def get_game_prediction(self, game_id: str) -> Optional[Dict]:
        """Load saved prediction for a game"""
        
        json_file = self.storage_dir / f"{game_id}.json"
        if not json_file.exists():
            return None
        
        with open(json_file, 'r') as f:
            return json.load(f)
    
    def get_all_predictions(self) -> List[Dict]:
        """Get all saved predictions"""
        
        predictions = []
        for json_file in self.storage_dir.glob("*.json"):
            with open(json_file, 'r') as f:
                predictions.append(json.load(f))
        
        return predictions
    
    def update_with_actual_results(self, game_id: str, 
                                   home_actual_score: int, 
                                   away_actual_score: int,
                                   player_actual_stats: List[Dict]):
        """Update prediction with actual results for comparison"""
        
        prediction = self.get_game_prediction(game_id)
        if not prediction:
            print(f"⚠️  No prediction found for game {game_id}")
            return
        
        # Add actual results
        prediction['home_actual_score'] = home_actual_score
        prediction['away_actual_score'] = away_actual_score
        prediction['player_actual_stats'] = player_actual_stats
        
        # Calculate comprehensive stat accuracy
        stat_accuracies = []
        
        # 1. Team Score Accuracy (weighted 30%)
        if home_actual_score > 0 and away_actual_score > 0:
            home_score_error = abs(prediction['home_predicted_score'] - home_actual_score)
            away_score_error = abs(prediction['away_predicted_score'] - away_actual_score)
            
            # Percentage accuracy for each team
            home_acc = max(0, 100 * (1 - home_score_error / home_actual_score))
            away_acc = max(0, 100 * (1 - away_score_error / away_actual_score))
            team_score_acc = (home_acc + away_acc) / 2
            
            stat_accuracies.append(('team_score', team_score_acc, 0.3))  # 30% weight
        
        # 2. Player Points Accuracy (weighted 40%)
        player_points_accuracies = []
        player_all_stats_accuracies = []
        
        if player_actual_stats:
            for actual in player_actual_stats:
                predicted = next(
                    (p for p in prediction.get('player_predictions', []) 
                     if p['name'] == actual['player_name']),
                    None
                )
                
                if predicted and 'stats' in predicted:
                    # Points accuracy
                    pred_pts = predicted['stats'].get('points', 0)
                    actual_pts = actual.get('points', 0)
                    
                    if actual_pts > 0:
                        pts_error = abs(pred_pts - actual_pts)
                        pts_acc = max(0, 100 * (1 - pts_error / actual_pts))
                        player_points_accuracies.append(pts_acc)
                    
                    # All stats accuracy (rebounds, assists, etc.) - weighted 30%
                    stat_accs = []
                    
                    for pred_key, actual_key in [
                        ('rebounds', 'rebounds'),
                        ('assists', 'assists'),
                        ('steals', 'steals'),
                        ('blocks', 'blocks')
                    ]:
                        pred_val = predicted['stats'].get(pred_key, 0)
                        actual_val = actual.get(actual_key, 0)
                        
                        if actual_val > 0:
                            error = abs(pred_val - actual_val)
                            acc = max(0, 100 * (1 - error / actual_val))
                            stat_accs.append(acc)
                    
                    if stat_accs:
                        player_all_stats_accuracies.append(sum(stat_accs) / len(stat_accs))
        
        if player_points_accuracies:
            avg_player_pts = sum(player_points_accuracies) / len(player_points_accuracies)
            stat_accuracies.append(('player_points', avg_player_pts, 0.4))  # 40% weight
        
        if player_all_stats_accuracies:
            avg_other_stats = sum(player_all_stats_accuracies) / len(player_all_stats_accuracies)
            stat_accuracies.append(('other_stats', avg_other_stats, 0.3))  # 30% weight
        
        # Calculate weighted overall accuracy
        if stat_accuracies:
            weighted_sum = sum(acc * weight for _, acc, weight in stat_accuracies)
            total_weight = sum(weight for _, _, weight in stat_accuracies)
            overall_accuracy = round(weighted_sum / total_weight, 1)
        else:
            overall_accuracy = None
        
        prediction['stat_accuracy'] = overall_accuracy
        prediction['score_accuracy'] = overall_accuracy  # Keep for backwards compatibility
        
        # Calculate average points difference
        if player_actual_stats:
            prediction['avg_points_diff'] = round(sum([
                abs(
                    next((p['stats'].get('points', 0) for p in prediction.get('player_predictions', []) 
                          if p['name'] == a['player_name']), 0) - a.get('points', 0)
                )
                for a in player_actual_stats
            ]) / len(player_actual_stats), 1)
        else:
            prediction['avg_points_diff'] = None
        
        # Store breakdown for debugging
        prediction['accuracy_breakdown'] = {
            'team_score_accuracy': stat_accuracies[0][1] if len(stat_accuracies) > 0 else None,
            'player_points_accuracy': stat_accuracies[1][1] if len(stat_accuracies) > 1 else None,
            'other_stats_accuracy': stat_accuracies[2][1] if len(stat_accuracies) > 2 else None
        }
        
        # Save updated prediction
        json_file = self.storage_dir / f"{game_id}.json"
        with open(json_file, 'w') as f:
            json.dump(prediction, f, indent=2)
        
        print(f"✅ Updated game {game_id} with actual results")
        print(f"   Overall Stat Accuracy: {prediction['stat_accuracy']:.1f}%")
        
        # Show breakdown
        breakdown = prediction.get('accuracy_breakdown', {})
        if breakdown.get('team_score_accuracy'):
            print(f"   ├─ Team Score: {breakdown['team_score_accuracy']:.1f}%")
        if breakdown.get('player_points_accuracy'):
            print(f"   ├─ Player Points: {breakdown['player_points_accuracy']:.1f}%")
        if breakdown.get('other_stats_accuracy'):
            print(f"   └─ Other Stats: {breakdown['other_stats_accuracy']:.1f}%")
        
        if prediction.get('avg_points_diff'):
            print(f"   Avg points diff: ±{prediction['avg_points_diff']:.1f}")
    
    def _calculate_score_accuracy(self, home_pred: float, away_pred: float,
                                  home_actual: int, away_actual: int) -> float:
        """Calculate prediction accuracy percentage"""
        
        total_predicted = home_pred + away_pred
        total_actual = home_actual + away_actual
        
        if total_actual == 0:
            return 0.0
        
        # Calculate percentage accuracy (inverse of error rate)
        error = abs(total_predicted - total_actual)
        error_rate = error / total_actual
        accuracy = max(0, 100 * (1 - error_rate))
        
        return round(accuracy, 1)


if __name__ == "__main__":
    # Test the storage system
    storage = PredictionStorage()
    
    # Example: Save a prediction
    storage.save_game_prediction(
        game_id="0022500002",
        game_date="2025-10-22",
        home_team="LAL",
        away_team="GSW",
        home_predicted_score=110.5,
        away_predicted_score=108.2,
        player_predictions=[
            {
                'name': 'LeBron James',
                'team': 'LAL',
                'is_home': True,
                'stats': {
                    'points': 24.5,
                    'rebounds': 7.2,
                    'assists': 8.1
                }
            },
            {
                'name': 'Stephen Curry',
                'team': 'GSW',
                'is_home': False,
                'stats': {
                    'points': 28.3,
                    'rebounds': 5.1,
                    'assists': 6.4
                }
            }
        ]
    )
    
    print("\n✅ Test complete - check backend/prediction_fine_tuning/predictions/")

