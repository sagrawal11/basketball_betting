#!/usr/bin/env python3
"""
Daily Update Script
===================
Comprehensive daily update for the NBA betting system.

Runs after each game day to:
1. Collect new game data from completed games
2. Update player CSV files
3. Regenerate features
4. Update player archetypes
5. Retrain models with new data
6. Update historical predictions with actual results

Usage:
    python backend/prediction_fine_tuning/daily_update.py

Run this once per day after games finish (or next morning).
"""

import sys
from pathlib import Path
from datetime import datetime, timedelta
import time
import pandas as pd

# Add project paths
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(Path(__file__).parent.parent))

print("\n" + "="*80)
print("🏀 NBA DAILY UPDATE SYSTEM")
print("="*80)
print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("="*80 + "\n")


# ============================================================================
# STEP 1: COLLECT NEW GAME DATA
# ============================================================================

def get_last_update_date():
    """Get the date of the last update"""
    import pytz
    backend_dir = Path(__file__).parent.parent
    update_file = backend_dir / "prediction_fine_tuning" / "last_update.txt"
    
    eastern = pytz.timezone('US/Eastern')
    
    if update_file.exists():
        try:
            with open(update_file, 'r') as f:
                date_str = f.read().strip()
                naive_date = datetime.strptime(date_str, '%Y-%m-%d')
                return eastern.localize(naive_date)
        except:
            pass
    
    # Default to Oct 21, 2025 (opening night)
    return eastern.localize(datetime(2025, 10, 21))


def save_last_update_date(date):
    """Save the last update date"""
    backend_dir = Path(__file__).parent.parent
    update_file = backend_dir / "prediction_fine_tuning" / "last_update.txt"
    
    with open(update_file, 'w') as f:
        f.write(date.strftime('%Y-%m-%d'))


def collect_new_game_data():
    """Collect game data from all days since last update"""
    print("\n📥 STEP 1: Collecting New Game Data")
    print("-" * 80)
    
    from nba_api.live.nba.endpoints import scoreboard, boxscore
    from nba_api.stats.static import teams as nba_teams
    import pytz
    
    # Get date range to check
    eastern = pytz.timezone('US/Eastern')
    today = datetime.now(eastern)
    last_update = get_last_update_date()
    
    # Generate all dates from last update through today
    dates_to_check = []
    current = last_update
    while current <= today:
        dates_to_check.append(current)
        current += timedelta(days=1)
    
    print(f"Checking from {last_update.strftime('%Y-%m-%d')} through {today.strftime('%Y-%m-%d')}")
    print(f"Total days to check: {len(dates_to_check)}")
    
    new_games_data = {}
    
    # First, check all games we have predictions for
    from prediction_fine_tuning.prediction_storage import PredictionStorage
    backend_dir = Path(__file__).parent.parent
    predictions_dir = backend_dir / "prediction_fine_tuning" / "predictions"
    storage = PredictionStorage(storage_dir=str(predictions_dir))
    
    all_predictions = storage.get_all_predictions()
    games_to_check = []
    
    for pred in all_predictions:
        # Check all predictions (we'll verify CSV status later)
        games_to_check.append(pred['game_id'])
    
    print(f"Found {len(games_to_check)} games to check")
    
    # Fetch each game directly
    for game_id in games_to_check:
        try:
            game_data = boxscore.BoxScore(game_id=game_id)
            full_game = game_data.get_dict()
            
            if 'game' not in full_game:
                continue
            
            game_info = full_game['game']
            status = game_info.get('gameStatusText', '')
            
            if 'Final' not in status:
                print(f"  ⏭️  {game_id}: Not finished ({status})")
                continue
            
            home_team = game_info.get('homeTeam', {})
            away_team = game_info.get('awayTeam', {})
            
            print(f"  ✅ {away_team.get('teamTricode')} {away_team.get('score')} @ {home_team.get('teamTricode')} {home_team.get('score')} - {status}")
            
            new_games_data[game_id] = full_game
            time.sleep(0.5)
            
        except Exception as e:
            print(f"  ⚠️  Error fetching {game_id}: {e}")
    
    print(f"\n✅ Collected {len(new_games_data)} completed games")
    return new_games_data


# ============================================================================
# STEP 2: UPDATE PLAYER CSV FILES
# ============================================================================

def update_player_csvs(games_data: dict):
    """Append new game data to player CSV files - using live API data formatted to match CSV"""
    print("\n📝 STEP 2: Updating Player CSV Files")
    print("-" * 80)
    
    backend_dir = Path(__file__).parent.parent
    player_data_dir = backend_dir / "data" / "player_data"
    
    players_updated = set()
    total_rows_added = 0
    
    for game_id, game_data in games_data.items():
        if 'game' not in game_data:
            continue
        
        game_info = game_data['game']
        game_date_str = game_info.get('gameTimeLocal', '').split('T')[0]  # YYYY-MM-DD
        
        home_team = game_info.get('homeTeam', {})
        away_team = game_info.get('awayTeam', {})
        home_abbrev = home_team.get('teamTricode')
        away_abbrev = away_team.get('teamTricode')
        
        print(f"\n  Processing game {game_id}: {away_abbrev} @ {home_abbrev}")
        
        # Process both teams
        for team_key in ['homeTeam', 'awayTeam']:
            team_data = game_info.get(team_key, {})
            team_abbrev = team_data.get('teamTricode')
            is_home = (team_key == 'homeTeam')
            opponent = away_abbrev if is_home else home_abbrev
            
            for player_data in team_data.get('players', []):
                # Skip DNPs
                if player_data.get('played') != '1':
                    continue
                
                # Get player name
                first_name = player_data.get('firstName', '')
                last_name = player_data.get('familyName', '')
                full_name = f"{first_name} {last_name}".strip()
                
                # Find player directory
                player_dir_name = full_name.replace(' ', '_').replace('.', '').replace("'", "")
                player_dir = player_data_dir / player_dir_name
                
                if not player_dir.exists():
                    continue
                
                csv_file = player_dir / f"{player_dir_name}_data.csv"
                
                if not csv_file.exists():
                    continue
                
                # Read existing data
                df = pd.read_csv(csv_file)
                
                # Check if game already exists
                if 'Game_ID' in df.columns and game_id in df['Game_ID'].values:
                    continue
                
                # Get stats from live API
                stats = player_data.get('statistics', {})
                
                # Get player info from last row of CSV (reuse existing metadata)
                last_row = df.iloc[-1] if len(df) > 0 else {}
                
                # Create MATCHUP string
                matchup = f"{team_abbrev} @ {opponent}" if not is_home else f"{team_abbrev} vs. {opponent}"
                
                # Create new row matching NBA stats API format
                new_row = {
                    'SEASON_ID': last_row.get('SEASON_ID', '22025'),
                    'Player_ID': last_row.get('Player_ID', player_data.get('personId')),
                    'Game_ID': game_id,
                    'GAME_DATE': game_date_str,
                    'MATCHUP': matchup,
                    'WL': '',  # Will be filled by feature engineering
                    'MIN': stats.get('minutes', '0'),
                    'FGM': stats.get('fieldGoalsMade', 0),
                    'FGA': stats.get('fieldGoalsAttempted', 0),
                    'FG_PCT': stats.get('fieldGoalsPercentage', 0) / 100 if stats.get('fieldGoalsPercentage') else 0,
                    'FG3M': stats.get('threePointersMade', 0),
                    'FG3A': stats.get('threePointersAttempted', 0),
                    'FG3_PCT': stats.get('threePointersPercentage', 0) / 100 if stats.get('threePointersPercentage') else 0,
                    'FTM': stats.get('freeThrowsMade', 0),
                    'FTA': stats.get('freeThrowsAttempted', 0),
                    'FT_PCT': stats.get('freeThrowsPercentage', 0) / 100 if stats.get('freeThrowsPercentage') else 0,
                    'OREB': stats.get('reboundsOffensive', 0),
                    'DREB': stats.get('reboundsDefensive', 0),
                    'REB': stats.get('reboundsTotal', 0),
                    'AST': stats.get('assists', 0),
                    'STL': stats.get('steals', 0),
                    'BLK': stats.get('blocks', 0),
                    'TOV': stats.get('turnovers', 0),
                    'PF': stats.get('foulsPersonal', 0),
                    'PTS': stats.get('points', 0),
                    'PLUS_MINUS': stats.get('plusMinusPoints', 0),
                    'VIDEO_AVAILABLE': 0,
                    'PLAYER_NAME': full_name,
                    'SEASON': '2025-26',
                    'height': last_row.get('height', ''),
                    'weight': last_row.get('weight', ''),
                    'position': last_row.get('position', ''),
                    'draft_year': last_row.get('draft_year', ''),
                    'season_exp': last_row.get('season_exp', 0),
                    'PLAYER_TEAM': team_abbrev,
                    'OPPONENT_TEAM': opponent,
                    'IS_HOME': 1 if is_home else 0
                }
                
                # Append to CSV
                df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
                df.to_csv(csv_file, index=False)
                
                players_updated.add(full_name)
                total_rows_added += 1
                print(f"    ✅ Added game for {full_name}")
        
        time.sleep(0.3)
    
    print(f"\n✅ Updated {len(players_updated)} players with {total_rows_added} new games")
    return list(players_updated)


# ============================================================================
# STEP 3: FEATURE ENGINEERING
# ============================================================================

def run_feature_engineering(player_names: list):
    """Run feature engineering for updated players"""
    print("\n🔧 STEP 3: Feature Engineering")
    print("-" * 80)
    
    from feature_engineering.feature_engineering import NBAFeatureEngineering
    
    backend_dir = Path(__file__).parent.parent
    data_dir = backend_dir / "data" / "player_data"
    
    feature_eng = NBAFeatureEngineering(data_dir=str(data_dir))
    
    success_count = 0
    for player_name in player_names:
        try:
            result_df = feature_eng.process_player_data(player_name)
            if result_df is not None:
                success_count += 1
                print(f"  ✅ {player_name}")
        except Exception as e:
            print(f"  ❌ {player_name}: {e}")
    
    print(f"\n✅ Feature engineering complete for {success_count}/{len(player_names)} players")
    return success_count


# ============================================================================
# STEP 4: PLAYER ARCHETYPING
# ============================================================================

def run_archetyping(player_names: list):
    """Run archetyping for updated players"""
    print("\n🎭 STEP 4: Player Archetyping")
    print("-" * 80)
    
    from feature_engineering.player_archetyping import PlayerArchetyping
    
    backend_dir = Path(__file__).parent.parent
    data_dir = backend_dir / "data" / "player_data"
    
    archetyper = PlayerArchetyping(data_dir=str(data_dir))
    
    success_count = 0
    for player_name in player_names:
        try:
            archetyper.process_player(player_name)
            success_count += 1
            print(f"  ✅ {player_name}")
        except Exception as e:
            print(f"  ❌ {player_name}: {e}")
    
    print(f"\n✅ Archetyping complete for {success_count}/{len(player_names)} players")
    return success_count


# ============================================================================
# STEP 5: MODEL TRAINING
# ============================================================================

def retrain_models(player_names: list):
    """Retrain models for updated players"""
    print("\n🤖 STEP 5: Model Training")
    print("-" * 80)
    
    from model_training.train_models_ultimate import UltimatePlayerModelTrainer
    
    backend_dir = Path(__file__).parent.parent
    data_dir = backend_dir / "data" / "player_data"
    
    trainer = UltimatePlayerModelTrainer(data_dir=str(data_dir))
    
    success_count = 0
    for player_name in player_names:
        try:
            result = trainer.train_player_models(player_name)
            if result:
                success_count += 1
                print(f"  ✅ {player_name}")
        except Exception as e:
            print(f"  ❌ {player_name}: {e}")
    
    print(f"\n✅ Model training complete for {success_count}/{len(player_names)} players")
    return success_count


# ============================================================================
# STEP 6: UPDATE HISTORICAL RECORDS
# ============================================================================

def update_historical_records():
    """Update predictions with actual results"""
    print("\n📊 STEP 6: Updating Historical Records")
    print("-" * 80)
    
    from nba_api.live.nba.endpoints import boxscore as live_boxscore
    from prediction_fine_tuning.prediction_storage import PredictionStorage
    
    backend_dir = Path(__file__).parent.parent
    predictions_dir = backend_dir / "prediction_fine_tuning" / "predictions"
    storage = PredictionStorage(storage_dir=str(predictions_dir))
    
    # Get all saved predictions
    all_predictions = storage.get_all_predictions()
    
    games_updated = 0
    
    for prediction in all_predictions:
        game_id = prediction.get('game_id')
        
        # Skip if already has actual results
        if 'home_actual_score' in prediction and prediction['home_actual_score'] > 0:
            continue
        
        # Try to fetch actual results
        try:
            game_data = live_boxscore.BoxScore(game_id=game_id)
            data = game_data.get_dict()
            
            if 'game' not in data:
                continue
            
            game = data['game']
            status = game.get('gameStatusText', '')
            
            if 'Final' not in status:
                continue
            
            home_team = game.get('homeTeam', {})
            away_team = game.get('awayTeam', {})
            
            home_score = home_team.get('score', 0)
            away_score = away_team.get('score', 0)
            
            # Get player stats
            player_actual_stats = []
            predicted_players = [p['name'] for p in prediction.get('player_predictions', [])]
            
            for team_key in ['homeTeam', 'awayTeam']:
                if team_key in game and 'players' in game[team_key]:
                    team_abbrev = game[team_key].get('teamTricode', '')
                    
                    for player_info in game[team_key]['players']:
                        first_name = player_info.get('firstName', '')
                        last_name = player_info.get('familyName', '')
                        full_name = f"{first_name} {last_name}".strip()
                        
                        if full_name not in predicted_players:
                            continue
                        
                        stats = player_info.get('statistics', {})
                        
                        player_actual_stats.append({
                            'player_name': full_name,
                            'team': team_abbrev,
                            'points': stats.get('points', 0),
                            'rebounds': stats.get('reboundsTotal', 0),
                            'assists': stats.get('assists', 0),
                            'steals': stats.get('steals', 0),
                            'blocks': stats.get('blocks', 0),
                            'fg_pct': stats.get('fieldGoalsPercentage', 0),
                            'fg3_pct': stats.get('threePointersPercentage', 0),
                            'ft_pct': stats.get('freeThrowsPercentage', 0),
                            'minutes': stats.get('minutes', '0')
                        })
            
            # Update prediction with actual results
            storage.update_with_actual_results(
                game_id=game_id,
                home_actual_score=home_score,
                away_actual_score=away_score,
                player_actual_stats=player_actual_stats
            )
            
            games_updated += 1
            time.sleep(0.5)
            
        except Exception as e:
            print(f"  ⚠️  Error updating {game_id}: {e}")
    
    print(f"\n✅ Updated {games_updated} historical records")
    return games_updated


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Run complete daily update workflow"""
    
    start_time = time.time()
    
    try:
        # Step 1: Collect new game data
        new_games = collect_new_game_data()
        
        if len(new_games) == 0:
            print("\n⚠️  No new completed games found. Exiting.")
            return
        
        # Step 2: Update player CSVs
        updated_players = update_player_csvs(new_games)
        
        if len(updated_players) == 0:
            print("\n⚠️  No players updated. Skipping feature/model updates.")
        else:
            # Step 3: Feature Engineering
            run_feature_engineering(updated_players)
            
            # Step 4: Player Archetyping
            run_archetyping(updated_players)
            
            # Step 5: Model Training
            print("\n🤖 Starting model training (this may take a while)...")
            retrain_models(updated_players)
        
        # Step 6: Update Historical Records (always run)
        update_historical_records()
        
        # Save the last update date
        import pytz
        eastern = pytz.timezone('US/Eastern')
        today = datetime.now(eastern)
        save_last_update_date(today)
        
        # Summary
        elapsed = time.time() - start_time
        minutes = int(elapsed // 60)
        seconds = int(elapsed % 60)
        
        print("\n" + "="*80)
        print("🎉 DAILY UPDATE COMPLETE!")
        print("="*80)
        print(f"Games processed: {len(new_games)}")
        print(f"Players updated: {len(updated_players)}")
        print(f"Time elapsed: {minutes}m {seconds}s")
        print(f"Last updated: {today.strftime('%Y-%m-%d %H:%M %Z')}")
        print("="*80)
        print("\n✅ Your system is now up to date!")
        print("   - Player data includes latest games")
        print("   - Features regenerated")
        print("   - Models retrained with fresh data")
        print("   - History page updated with actual results")
        print("\n💡 Refresh your web app to see the latest predictions!")
        print("="*80 + "\n")
        
    except Exception as e:
        print(f"\n❌ Error during daily update: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
