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
        from nba_api.live.nba.endpoints import scoreboard
        import pytz
        
        # Get current date in US Eastern Time (NBA's timezone)
        eastern = pytz.timezone('US/Eastern')
        today = datetime.now(eastern)
        
        print(f"📅 Fetching games for: {today.strftime('%Y-%m-%d %H:%M %Z')}")
        
        games = scoreboard.ScoreBoard()
        games_data = games.get_dict()
        
        formatted_games = []
        for game in games_data.get('scoreboard', {}).get('games', []):
            home_team = game.get('homeTeam', {})
            away_team = game.get('awayTeam', {})
            
            formatted_games.append({
                'game_id': game.get('gameId'),
                'home_team': {
                    'name': home_team.get('teamName'),
                    'abbrev': home_team.get('teamTricode'),
                    'score': home_team.get('score', 0)
                },
                'away_team': {
                    'name': away_team.get('teamName'),
                    'abbrev': away_team.get('teamTricode'),
                    'score': away_team.get('score', 0)
                },
                'game_time': game.get('gameTimeUTC'),
                'status': game.get('gameStatusText', 'Scheduled')
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
    """Get all players for a specific game with predictions"""
    try:
        # This will fetch rosters and generate predictions
        # For now, return mock data - we'll implement full logic next
        
        return jsonify({
            'success': True,
            'game_id': game_id,
            'players': []  # Will implement in next step
        })
        
    except Exception as e:
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
            return jsonify({
                'success': True,
                'player': player_name,
                'predictions': predictions
            })
        else:
            return jsonify({
                'success': False,
                'error': 'Could not generate predictions'
            }), 400
            
    except Exception as e:
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

