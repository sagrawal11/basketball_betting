#!/usr/bin/env python3
"""
NBA Betting Assistant - Web Application
========================================
Flask web app for making data-driven betting decisions
"""

from flask import Flask, render_template, jsonify, request
from datetime import datetime, timedelta
import sys
from pathlib import Path
import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.prediction_engine import NBAGamePredictor
from models.betting_recommender import BettingRecommender

app = Flask(__name__)
app.config['SECRET_KEY'] = 'nba-betting-secret-key-change-in-production'

# Get absolute paths
backend_dir = Path(__file__).parent.parent
data_dir = backend_dir / "data"
models_dir = data_dir / "player_data"

# Initialize prediction engine and recommender with absolute paths
predictor = NBAGamePredictor(data_dir=str(data_dir), models_dir=str(models_dir))
recommender = BettingRecommender()


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
                
                formatted_games.append({
                    'game_id': game_id,
                    'home_team': home_team,
                    'away_team': away_team,
                    'game_time': game.get('GAME_DATE_EST'),
                    'status': game.get('GAME_STATUS_TEXT', 'Scheduled')
                })
        
        print(f"✅ Found {len(formatted_games)} games")
        
        return jsonify({
            'success': True,
            'games': formatted_games,
            'date': today.strftime('%Y-%m-%d')
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
                            # Convert numpy types to Python native types for JSON serialization
                            clean_preds = {}
                            for key, value in preds.items():
                                if isinstance(value, (np.integer, np.floating)):
                                    clean_preds[key] = float(value)
                                else:
                                    clean_preds[key] = value
                            
                            players_with_predictions.append({
                                'name': player_name,
                                'team': team_abbrev,
                                'is_home': is_home_team,
                                'predictions': clean_preds
                            })
                            print(f"  ✅ {player_name}")
                        
            except Exception as e:
                print(f"  ⚠️  Error fetching roster for {team_abbrev}: {e}")
                continue
        
        print(f"✅ Generated predictions for {len(players_with_predictions)} players")
        
        return jsonify({
            'success': True,
            'game_id': game_id,
            'players': players_with_predictions
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
    print("Starting server at http://localhost:5000")
    print("Press Ctrl+C to stop")
    print()
    
    app.run(debug=True, port=5000)

