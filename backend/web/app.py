#!/usr/bin/env python3
"""
NBA Betting Assistant - Web Application
========================================
Flask web app for making data-driven betting decisions
"""

from flask import Flask, render_template, jsonify, request
from flask_cors import CORS
from datetime import datetime, timedelta
import sys
from pathlib import Path
import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.prediction_engine import NBAGamePredictor
from models.betting_recommender import BettingRecommender

# Add prediction storage
sys.path.insert(0, str(Path(__file__).parent.parent / "prediction_fine_tuning"))
from prediction_storage import PredictionStorage

app = Flask(__name__)
app.config['SECRET_KEY'] = 'nba-betting-secret-key-change-in-production'

# Enable CORS for React frontend (allow all origins during development)
CORS(app)

# Get absolute paths
backend_dir = Path(__file__).parent.parent
data_dir = backend_dir / "data"
models_dir = data_dir / "player_data"

# Initialize prediction engine, recommender, and storage with absolute paths
predictor = NBAGamePredictor(data_dir=str(data_dir), models_dir=str(models_dir))
recommender = BettingRecommender()
predictions_dir = backend_dir / "prediction_fine_tuning" / "predictions"
storage = PredictionStorage(storage_dir=str(predictions_dir))


@app.route('/')
def index():
    """Home page - shows today's games"""
    return render_template('index.html')


@app.route('/game/<game_id>')
def game_page(game_id):
    """Game page - shows players and predictions"""
    return render_template('game.html')


@app.route('/api/games/today')
def get_todays_games():
    """API endpoint to fetch today's NBA games"""
    try:
        # Get today's games from NBA API
        from nba_api.stats.endpoints import scoreboardv2
        from nba_api.stats.static import teams as nba_teams
        import pytz
        
        # Get current date in US Eastern Time (NBA's timezone)
        eastern = pytz.timezone('US/Eastern')
        today = datetime.now(eastern)
        date_str = today.strftime('%m/%d/%Y')  # Format: MM/DD/YYYY for NBA API
        
        print(f"📅 Fetching games for: {today.strftime('%Y-%m-%d %H:%M %Z')} (API format: {date_str})")
        
        # Get all NBA teams for ID lookup
        all_teams = nba_teams.get_teams()
        team_lookup = {team['id']: team for team in all_teams}
        
        # Fetch scoreboard for specific date
        scoreboard = scoreboardv2.ScoreboardV2(game_date=date_str)
        games_data = scoreboard.get_normalized_dict()
        
        formatted_games = []
        
        # Parse games from the response
        if 'GameHeader' in games_data:
            for game in games_data['GameHeader']:
                game_id = game.get('GAME_ID')
                home_team_id = game.get('HOME_TEAM_ID')
                visitor_team_id = game.get('VISITOR_TEAM_ID')
                
                # Get team info from static teams data
                home_team_data = team_lookup.get(home_team_id, {})
                away_team_data = team_lookup.get(visitor_team_id, {})
                
                home_team = {
                    'name': home_team_data.get('full_name', 'Home Team'),
                    'abbrev': home_team_data.get('abbreviation', 'HOME'),
                    'score': 0
                }
                
                away_team = {
                    'name': away_team_data.get('full_name', 'Away Team'),
                    'abbrev': away_team_data.get('abbreviation', 'AWAY'),
                    'score': 0
                }
                
                # Check LineScore for live scores (if game started)
                if 'LineScore' in games_data:
                    for team in games_data['LineScore']:
                        if team.get('GAME_ID') == game_id:
                            if team.get('TEAM_ID') == home_team_id:
                                home_team['score'] = team.get('PTS') or 0
                            elif team.get('TEAM_ID') == visitor_team_id:
                                away_team['score'] = team.get('PTS') or 0
                
                # Format to match React component expectations
                formatted_games.append({
                    'id': game_id,
                    'homeTeam': {
                        'name': home_team['name'],
                        'abbrev': home_team['abbrev'],
                        'logo': f"https://cdn.nba.com/logos/nba/{home_team_id}/global/L/logo.svg"
                    },
                    'awayTeam': {
                        'name': away_team['name'],
                        'abbrev': away_team['abbrev'],
                        'logo': f"https://cdn.nba.com/logos/nba/{visitor_team_id}/global/L/logo.svg"
                    },
                    'time': game.get('GAME_STATUS_TEXT', 'TBD'),
                    'date': today.strftime('%B %d, %Y'),
                    'location': 'NBA Arena'  # Can enhance this later
                })
        
        print(f"✅ Found {len(formatted_games)} games")
        
        return jsonify({
            'success': True,
            'games': formatted_games,
            'date': today.strftime('%Y-%m-%d'),
            'count': len(formatted_games)
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/game/<game_id>/players')
def get_game_players(game_id):
    """Get all players for a specific game with PRE-GENERATED predictions"""
    try:
        from nba_api.stats.endpoints import commonteamroster
        import pytz
        
        # Get team abbreviations from query params
        home_abbrev = request.args.get('home')
        away_abbrev = request.args.get('away')
        
        if not home_abbrev or not away_abbrev:
            return jsonify({'success': False, 'error': 'Missing team info'}), 400
        
        print(f"📋 Fetching rosters for {away_abbrev} @ {home_abbrev}")
        
        # Get current season and date
        eastern = pytz.timezone('US/Eastern')
        today = datetime.now(eastern)
        season = predictor._get_current_season()
        game_date = today.strftime('%Y-%m-%d')
        
        # Get team IDs
        from nba_api.stats.static import teams as nba_teams
        all_teams = nba_teams.get_teams()
        team_lookup = {team['abbreviation']: team['id'] for team in all_teams}
        
        home_team_id = team_lookup.get(home_abbrev)
        away_team_id = team_lookup.get(away_abbrev)
        
        players_with_predictions = []
        
        # Fetch rosters for both teams
        for team_id, team_abbrev, is_home_team in [(home_team_id, home_abbrev, True), (away_team_id, away_abbrev, False)]:
            if not team_id:
                continue
            
            try:
                # Get roster (this gets current season roster)
                roster = commonteamroster.CommonTeamRoster(team_id=team_id, season=season)
                roster_data = roster.get_normalized_dict()
                
                if 'CommonTeamRoster' in roster_data:
                    # Get top 10 players by experience (starters + key bench)
                    # Handle 'R' for rookies
                    def get_exp(player):
                        exp = player.get('EXP', 0)
                        if exp == 'R':
                            return 0
                        try:
                            return int(exp)
                        except:
                            return 0
                    
                    players = sorted(roster_data['CommonTeamRoster'], 
                                   key=get_exp, 
                                   reverse=True)[:10]
                    
                    for player in players:
                        player_name = player.get('PLAYER')
                        
                        # Check if we have a model for this player
                        player_model_dir = predictor.models_dir / player_name.replace(' ', '_')
                        if not player_model_dir.exists():
                            continue  # Skip if no trained model
                        
                        # Auto-detect injury status and teammates out
                        # TODO: Implement NBA injury API fetch
                        # For now, default to Healthy with no teammates out
                        injury_status = 'Healthy'
                        teammates_out = []
                        
                        # Generate prediction
                        opponent_abbrev = away_abbrev if is_home_team else home_abbrev
                        preds = predictor.predict_player_stats(
                            player_name=player_name,
                            opponent_team=opponent_abbrev,
                            is_home=is_home_team,
                            game_date=game_date,
                            season=season,
                            injury_status=injury_status,
                            teammates_out=teammates_out
                        )
                        
                        if preds:
                            # Get player ID for headshot
                            from nba_api.stats.static import players as nba_players
                            all_players = nba_players.get_players()
                            player_info = next((p for p in all_players if p['full_name'] == player_name), None)
                            player_id = player_info['id'] if player_info else None
                            
                            # Get position
                            position = player.get('POSITION', 'G')
                            
                            # Format stats to match React component - ALL rounded to 1 decimal
                            player_stats = {
                                'points': round(float(preds.get('PTS', 0)), 1),
                                'rebounds': round(float(preds.get('DREB', 0)) + float(preds.get('OREB', 0)), 1),
                                'assists': round(float(preds.get('AST', 0)), 1),
                                'steals': round(float(preds.get('STL', 0)), 1),
                                'blocks': round(float(preds.get('BLK', 0)), 1),
                                'fg': f"{float(preds.get('FGM', 0)) / (float(preds.get('FGM', 0)) / 0.47 + 0.1) * 100:.1f}",  # Estimate FG%
                                'threePt': f"{float(preds.get('FG3M', 0)) / (float(preds.get('FG3M', 0)) / 0.37 + 0.1) * 100:.1f}",  # Estimate 3P%
                                'ft': f"{float(preds.get('FTM', 0)) / (float(preds.get('FTM', 0)) / 0.80 + 0.1) * 100:.1f}"  # Estimate FT%
                            }
                            
                            players_with_predictions.append({
                                'name': player_name,
                                'position': position,
                                'image': f"https://cdn.nba.com/headshots/nba/latest/1040x760/{player_id}.png" if player_id else "https://via.placeholder.com/260x190?text=No+Image",
                                'stats': player_stats,
                                'team': team_abbrev,
                                'is_home': is_home_team
                            })
                            print(f"  ✅ {player_name}")
                        
            except Exception as e:
                print(f"  ⚠️  Error fetching roster for {team_abbrev}: {e}")
                continue
        
        print(f"✅ Generated predictions for {len(players_with_predictions)} players")
        
        # Separate into home and away players
        home_players = [p for p in players_with_predictions if p['is_home']]
        away_players = [p for p in players_with_predictions if not p['is_home']]
        
        # Calculate predicted team scores using LEARNED parameters from 2024-25 analysis
        # Analysis showed: Top 5 = 82.8%, Top 8 = 97.7% of team score
        home_score = sum([p['stats']['points'] for p in home_players]) if home_players else 0
        away_score = sum([p['stats']['points'] for p in away_players]) if away_players else 0
        
        # Determine which scaling to use based on number of players
        if len(home_players) >= 8:
            # Have rotation players (top 8-10) = ~97.7% of score
            # Add small deep bench contribution (learned: 2.5 points)
            home_final = home_score + 3
            away_final = away_score + 3
        elif len(home_players) >= 5:
            # Have top 5 starters = ~82.8% of score
            # Scale up: total = top_5 / 0.828
            home_final = home_score / 0.828
            away_final = away_score / 0.828
        else:
            # Not enough data, use average (103.9 points)
            home_final = 104
            away_final = 104
        
        # Round all player stats to 1 decimal place
        for player in players_with_predictions:
            if 'stats' in player:
                for stat_key in player['stats']:
                    if isinstance(player['stats'][stat_key], (int, float)):
                        player['stats'][stat_key] = round(player['stats'][stat_key], 1)
        
        # Save predictions for historical comparison
        try:
            storage.save_game_prediction(
                game_id=game_id,
                game_date=today.strftime('%Y-%m-%d'),
                home_team=home_abbrev,
                away_team=away_abbrev,
                home_predicted_score=round(home_final, 1),
                away_predicted_score=round(away_final, 1),
                player_predictions=players_with_predictions
            )
        except Exception as save_error:
            print(f"⚠️  Could not save prediction: {save_error}")
        
        # Format to match React expectations
        return jsonify({
            'success': True,
            'homeTeam': {
                'name': home_players[0]['team'] if home_players else home_abbrev,
                'logo': f"https://cdn.nba.com/logos/nba/{home_team_id}/global/L/logo.svg",
                'predictedScore': round(home_final)
            },
            'awayTeam': {
                'name': away_players[0]['team'] if away_players else away_abbrev,
                'logo': f"https://cdn.nba.com/logos/nba/{away_team_id}/global/L/logo.svg",
                'predictedScore': round(away_final)
            },
            'date': today.strftime('%B %d, %Y'),
            'time': '7:30 PM ET',  # Can enhance with actual game time
            'location': 'NBA Arena',
            'homePlayers': [{'name': p['name'], 'position': p['position'], 'image': p['image'], 'stats': p['stats']} for p in home_players],
            'awayPlayers': [{'name': p['name'], 'position': p['position'], 'image': p['image'], 'stats': p['stats']} for p in away_players]
        })
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/predict', methods=['POST'])
def get_prediction():
    """Generate prediction for a specific player"""
    try:
        data = request.json
        
        player_name = data.get('player_name')
        opponent = data.get('opponent')
        is_home = data.get('is_home', True)
        game_date = data.get('game_date', datetime.now().strftime('%Y-%m-%d'))
        season = data.get('season', predictor._get_current_season())
        injury_status = data.get('injury_status', 'Healthy')
        teammates_out = data.get('teammates_out', [])
        
        print(f"🎯 Predicting for {player_name} vs {opponent} (home={is_home})")
        
        # Generate prediction
        predictions = predictor.predict_player_stats(
            player_name=player_name,
            opponent_team=opponent,
            is_home=is_home,
            game_date=game_date,
            season=season,
            injury_status=injury_status,
            teammates_out=teammates_out
        )
        
        if predictions:
            print(f"  ✅ Generated predictions for {player_name}")
            return jsonify({
                'success': True,
                'player': player_name,
                'predictions': predictions
            })
        else:
            print(f"  ❌ No model found for {player_name}")
            return jsonify({
                'success': False,
                'error': f'No trained model found for {player_name}'
            }), 404
            
    except Exception as e:
        print(f"  ❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/generate-all-predictions', methods=['POST'])
def generate_all_predictions():
    """Auto-generate predictions for ALL of today's games"""
    try:
        from nba_api.stats.endpoints import scoreboardv2
        from nba_api.stats.static import teams as nba_teams
        import pytz
        
        eastern = pytz.timezone('US/Eastern')
        today = datetime.now(eastern)
        date_str = today.strftime('%m/%d/%Y')
        
        print(f"🤖 Auto-generating predictions for all games on {date_str}")
        
        # Fetch today's games
        scoreboard = scoreboardv2.ScoreboardV2(game_date=date_str)
        games_data = scoreboard.get_normalized_dict()
        
        all_teams = nba_teams.get_teams()
        team_lookup = {team['id']: team for team in all_teams}
        
        generated_count = 0
        skipped_count = 0
        
        if 'GameHeader' in games_data:
            for game in games_data['GameHeader']:
                game_id = game.get('GAME_ID')
                home_team_id = game.get('HOME_TEAM_ID')
                visitor_team_id = game.get('VISITOR_TEAM_ID')
                
                # Check if we already have predictions for this game
                existing = storage.get_game_prediction(game_id)
                if existing:
                    print(f"  ⏭️  Skipping {game_id} (already have predictions)")
                    skipped_count += 1
                    continue
                
                # Get team abbreviations
                home_team = team_lookup.get(home_team_id, {})
                away_team = team_lookup.get(visitor_team_id, {})
                home_abbrev = home_team.get('abbreviation', 'HOME')
                away_abbrev = away_team.get('abbreviation', 'AWAY')
                
                print(f"  🔮 Generating: {away_abbrev} @ {home_abbrev} ({game_id})")
                
                # Generate predictions by calling the existing endpoint logic
                # (We'll extract this to a helper function to avoid duplication)
                try:
                    # Make internal request to game players endpoint
                    import requests
                    response = requests.get(
                        f'http://localhost:5001/api/game/{game_id}/players?home={home_abbrev}&away={away_abbrev}',
                        timeout=30
                    )
                    if response.status_code == 200:
                        generated_count += 1
                        print(f"  ✅ Generated predictions for {game_id}")
                    else:
                        print(f"  ⚠️  Failed to generate for {game_id}")
                except Exception as e:
                    print(f"  ❌ Error generating {game_id}: {e}")
        
        return jsonify({
            'success': True,
            'generated': generated_count,
            'skipped': skipped_count,
            'total': generated_count + skipped_count
        })
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/history/games')
def get_historical_games():
    """Get completed games with actual scores for history"""
    try:
        from nba_api.stats.endpoints import scoreboardv2, leaguegamefinder
        import pytz
        
        # Get yesterday's games (Oct 21, 2025)
        eastern = pytz.timezone('US/Eastern')
        yesterday = datetime.now(eastern) - timedelta(days=1)
        date_str = yesterday.strftime('%m/%d/%Y')
        
        print(f"📅 Fetching completed games for: {date_str}")
        
        # Fetch scoreboard for yesterday
        scoreboard = scoreboardv2.ScoreboardV2(game_date=date_str)
        games_data = scoreboard.get_normalized_dict()
        
        # Get all NBA teams for ID lookup
        from nba_api.stats.static import teams as nba_teams
        all_teams = nba_teams.get_teams()
        team_lookup = {team['id']: team for team in all_teams}
        
        historical_games = []
        
        # Parse games with actual scores
        if 'GameHeader' in games_data:
            for game in games_data['GameHeader']:
                game_id = game.get('GAME_ID')
                home_team_id = game.get('HOME_TEAM_ID')
                visitor_team_id = game.get('VISITOR_TEAM_ID')
                
                home_team_data = team_lookup.get(home_team_id, {})
                away_team_data = team_lookup.get(visitor_team_id, {})
                
                # Get actual scores from LineScore
                home_score = 0
                away_score = 0
                
                if 'LineScore' in games_data:
                    for team in games_data['LineScore']:
                        if team.get('GAME_ID') == game_id:
                            if team.get('TEAM_ID') == home_team_id:
                                home_score = team.get('PTS') or 0
                            elif team.get('TEAM_ID') == visitor_team_id:
                                away_score = team.get('PTS') or 0
                
                # Try to load saved predictions for this game
                saved_prediction = storage.get_game_prediction(game_id)
                home_pred_score = None
                away_pred_score = None
                score_accuracy = None
                avg_points_diff = None
                player_stats_accuracy = None
                
                if saved_prediction:
                    home_pred_score = saved_prediction.get('home_predicted_score')
                    away_pred_score = saved_prediction.get('away_predicted_score')
                    
                    # Use actual scores from saved prediction if available (more reliable than API)
                    saved_home_actual = saved_prediction.get('home_actual_score')
                    saved_away_actual = saved_prediction.get('away_actual_score')
                    
                    if saved_home_actual is not None and saved_home_actual > 0:
                        home_score = saved_home_actual
                    if saved_away_actual is not None and saved_away_actual > 0:
                        away_score = saved_away_actual
                    
                    # Get pre-calculated accuracy from storage
                    score_accuracy = saved_prediction.get('score_accuracy')
                    player_stats_accuracy = saved_prediction.get('player_stats_accuracy')
                    avg_points_diff = saved_prediction.get('avg_points_diff')
                
                historical_games.append({
                    'id': game_id,
                    'homeTeam': {
                        'name': home_team_data.get('full_name', 'Home Team'),
                        'abbrev': home_team_data.get('abbreviation', 'HOME'),
                        'logo': f"https://cdn.nba.com/logos/nba/{home_team_id}/global/L/logo.svg",
                        'predictedScore': home_pred_score,
                        'actualScore': home_score
                    },
                    'awayTeam': {
                        'name': away_team_data.get('full_name', 'Away Team'),
                        'abbrev': away_team_data.get('abbreviation', 'AWAY'),
                        'logo': f"https://cdn.nba.com/logos/nba/{visitor_team_id}/global/L/logo.svg",
                        'predictedScore': away_pred_score,
                        'actualScore': away_score
                    },
                    'time': game.get('GAME_STATUS_TEXT', 'Final'),
                    'date': yesterday.strftime('%B %d, %Y'),
                    'location': 'NBA Arena',
                    'accuracy': {
                        'scoreAccuracy': score_accuracy,
                        'playerStatsAccuracy': player_stats_accuracy,
                        'avgPointsDiff': avg_points_diff
                    }
                })
        
        print(f"✅ Found {len(historical_games)} completed games")
        
        return jsonify({
            'success': True,
            'games': historical_games
        })
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/analyze-bet', methods=['POST'])
def analyze_bet():
    """Analyze a betting opportunity"""
    try:
        data = request.json
        
        # Get recommendation
        rec = recommender.recommend_bet(
            player=data.get('player'),
            opponent=data.get('opponent'),
            is_home=data.get('is_home', True),
            game_date=data.get('game_date', datetime.now().strftime('%Y-%m-%d')),
            season=data.get('season', predictor._get_current_season()),
            stat=data.get('stat'),
            line=float(data.get('line')),
            over_odds=int(data.get('over_odds')),
            injury_status=data.get('injury_status', 'Healthy'),
            teammates_out=data.get('teammates_out', [])
        )
        
        return jsonify({
            'success': True,
            'recommendation': rec
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


if __name__ == '__main__':
    print("🏀 NBA Betting Assistant")
    print("=" * 80)
    print("Starting server at http://localhost:5001")
    print("Press Ctrl+C to stop")
    print("Note: Using port 5001 (port 5000 is taken by AirPlay)")
    print()
    
    app.run(debug=True, port=5001, host='127.0.0.1')

