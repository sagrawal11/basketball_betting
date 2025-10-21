#!/usr/bin/env python3
"""
Scoring Pattern Analyzer
=========================
Analyzes completed NBA games to learn:
- How scoring is distributed across players
- Typical team totals
- Bench contribution patterns
- Optimal scaling factors for team score predictions

This DOES NOT modify player models - only tunes the prediction engine's
team score calculation logic.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
from collections import defaultdict

class ScoringPatternAnalyzer:
    """Analyze NBA games to understand scoring distribution"""
    
    def __init__(self, data_dir: str = "backend/data"):
        self.data_dir = Path(data_dir)
        self.player_data_dir = self.data_dir / "player_data"
        
    def analyze_2024_25_season(self):
        """Analyze all games from 2024-25 season"""
        print("🔍 Analyzing 2024-25 Season Scoring Patterns")
        print("=" * 80)
        
        # Collect all games from 2024-25 season
        games_by_game_id = defaultdict(list)
        
        # Get all players with data
        player_dirs = [d for d in self.player_data_dir.iterdir() if d.is_dir()]
        
        print(f"📊 Scanning all {len(player_dirs)} players... (this may take a minute)")
        
        for i, player_dir in enumerate(player_dirs):
            if i % 500 == 0:
                print(f"   Progress: {i}/{len(player_dirs)} players scanned...")
            player_csv = player_dir / f"{player_dir.name}_data.csv"
            
            if not player_csv.exists():
                continue
            
            try:
                df = pd.read_csv(player_csv)
                
                # Filter to 2024-25 season only
                if 'SEASON' in df.columns:
                    df_2425 = df[df['SEASON'] == '2024-25'].copy()
                    
                    if len(df_2425) == 0:
                        continue
                    
                    # Group by game
                    for _, game in df_2425.iterrows():
                        game_id = game.get('Game_ID')
                        team = game.get('PLAYER_TEAM')
                        player_name = game.get('PLAYER_NAME', player_dir.name.replace('_', ' '))
                        
                        if pd.notna(game_id) and pd.notna(team):
                            games_by_game_id[game_id].append({
                                'player': player_name,
                                'team': team,
                                'points': game.get('PTS', 0),
                                'minutes': game.get('MIN', 0),
                                'is_home': game.get('IS_HOME', 0),
                                'rebounds': game.get('REB', 0),
                                'assists': game.get('AST', 0)
                            })
            except Exception as e:
                continue
        
        print(f"\n✅ Collected {len(games_by_game_id)} games from 2024-25 season")
        
        # Analyze patterns
        return self.calculate_scoring_patterns(games_by_game_id)
    
    def calculate_scoring_patterns(self, games_by_game_id):
        """Calculate scoring distribution patterns"""
        print("\n📈 Calculating Scoring Patterns...")
        print("-" * 80)
        
        team_totals = []
        top_5_percentages = []
        top_8_percentages = []
        bench_contributions = []
        
        games_processed = 0
        games_skipped = 0
        
        for game_id, players in games_by_game_id.items():
            if len(players) < 5:  # Need at least 5 players (lowered threshold)
                games_skipped += 1
                continue
            
            # Group by team
            teams = {}
            for player in players:
                team = player['team']
                if team not in teams:
                    teams[team] = []
                teams[team].append(player)
            
            # Analyze each team
            for team, team_players in teams.items():
                if len(team_players) < 5:  # Lowered from 8 to 5
                    continue
                
                games_processed += 1
                
                # Sort by points
                sorted_players = sorted(team_players, key=lambda x: x['points'], reverse=True)
                
                # Calculate totals
                team_total = sum([p['points'] for p in sorted_players])
                top_5_total = sum([p['points'] for p in sorted_players[:5]])
                top_8_total = sum([p['points'] for p in sorted_players[:8]])
                bench_total = team_total - top_8_total
                
                if team_total > 0:  # Valid game
                    team_totals.append(team_total)
                    top_5_percentages.append(top_5_total / team_total)
                    top_8_percentages.append(top_8_total / team_total)
                    bench_contributions.append(bench_total)
        
        # Check if we have valid data
        print(f"\n📊 Processing Summary:")
        print(f"   Games collected: {len(games_by_game_id)}")
        print(f"   Team performances analyzed: {games_processed}")
        print(f"   Games skipped (insufficient data): {games_skipped}")
        
        if len(team_totals) == 0:
            print("\n❌ No valid team data found!")
            print("   Tip: Make sure player CSVs have 2024-25 season data")
            return None
        
        # Calculate statistics
        results = {
            'avg_team_score': np.mean(team_totals),
            'median_team_score': np.median(team_totals),
            'std_team_score': np.std(team_totals),
            'min_team_score': np.min(team_totals),
            'max_team_score': np.max(team_totals),
            
            'top_5_percentage': np.mean(top_5_percentages),
            'top_5_std': np.std(top_5_percentages),
            
            'top_8_percentage': np.mean(top_8_percentages),
            'top_8_std': np.std(top_8_percentages),
            
            'avg_bench_contribution': np.mean(bench_contributions),
            'median_bench_contribution': np.median(bench_contributions),
            'std_bench_contribution': np.std(bench_contributions),
            
            'games_analyzed': len(team_totals)
        }
        
        # Print results
        print(f"\n✅ Analysis Complete!")
        print(f"   Games analyzed: {results['games_analyzed']}")
        print(f"\n📊 Team Scoring:")
        print(f"   Average: {results['avg_team_score']:.1f} points")
        print(f"   Median:  {results['median_team_score']:.1f} points")
        print(f"   Range:   {results['min_team_score']:.0f} - {results['max_team_score']:.0f} points")
        print(f"\n🎯 Scoring Distribution:")
        print(f"   Top 5 players: {results['top_5_percentage']*100:.1f}% of team score")
        print(f"   Top 8 players: {results['top_8_percentage']*100:.1f}% of team score")
        print(f"   Bench (rest):  {(1-results['top_8_percentage'])*100:.1f}% of team score")
        print(f"\n💡 Bench Contribution:")
        print(f"   Average: {results['avg_bench_contribution']:.1f} points")
        print(f"   Median:  {results['median_bench_contribution']:.1f} points")
        
        return results
    
    def save_learned_parameters(self, results):
        """Save learned parameters to config file"""
        output_file = Path("backend/prediction_fine_tuning/scoring_parameters.json")
        
        config = {
            'season_analyzed': '2024-25',
            'games_analyzed': results['games_analyzed'],
            'avg_team_score': round(results['avg_team_score'], 1),
            'top_5_scoring_percentage': round(results['top_5_percentage'], 3),
            'top_8_scoring_percentage': round(results['top_8_percentage'], 3),
            'bench_contribution_avg': round(results['avg_bench_contribution'], 1),
            'bench_contribution_median': round(results['median_bench_contribution'], 1),
            
            # Recommended formula
            'recommended_formula': {
                'description': 'For predicting team scores from individual predictions',
                'if_have_top_5': 'team_score = top_5_total / {:.3f} OR top_5_total + {:.1f} bench'.format(
                    results['top_5_percentage'], results['avg_bench_contribution']
                ),
                'if_have_top_8': 'team_score = top_8_total / {:.3f} OR top_8_total + {:.1f} bench'.format(
                    results['top_8_percentage'], (1 - results['top_8_percentage']) * results['avg_team_score']
                ),
                'if_have_all_rotation': 'team_score = all_players_total (no scaling needed)'
            }
        }
        
        with open(output_file, 'w') as f:
            json.dump(config, f, indent=2)
        
        print(f"\n💾 Saved parameters to: {output_file}")
        print(f"\n✅ RECOMMENDED FORMULA:")
        print(f"   Team Score = (Top 5 Predicted Points) + {config['bench_contribution_median']:.0f} bench")
        print(f"   OR")
        print(f"   Team Score = (Top 5 Predicted Points) / {config['top_5_scoring_percentage']:.3f}")
        
        return config


def main():
    """Run scoring pattern analysis"""
    analyzer = ScoringPatternAnalyzer()
    
    # Analyze 2024-25 season
    results = analyzer.analyze_2024_25_season()
    
    if results is None:
        print("\n❌ Analysis failed - not enough data")
        return
    
    # Save learned parameters
    config = analyzer.save_learned_parameters(results)
    
    print("\n" + "=" * 80)
    print("🎯 Next Steps:")
    print("   1. Review scoring_parameters.json")
    print("   2. Update backend/web/app.py to use learned bench contribution")
    print("   3. Test with tonight's games")
    print("=" * 80)


if __name__ == "__main__":
    main()

