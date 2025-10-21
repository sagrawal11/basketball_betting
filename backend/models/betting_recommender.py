#!/usr/bin/env python3
"""
Betting Recommender
===================
Integrates with SportsGameOdds API and recommends profitable bets
"""

import requests
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
import json
import sys
from scipy.stats import norm

# Add backend/models to path for imports
sys.path.insert(0, str(Path(__file__).parent))
from prediction_engine import NBAGamePredictor


class BettingRecommender:
    """Find +EV betting opportunities by comparing predictions to odds"""
    
    def __init__(self, api_key: str = None):
        """
        Initialize betting recommender
        
        Args:
            api_key: SportsGameOdds API key
        """
        self.api_key = api_key or self._load_api_key()
        self.predictor = NBAGamePredictor()
        self.base_url = "https://api.sportsgameodds.com/v1"
        
        # Stat mappings (SportsGameOdds names → our stat names)
        self.stat_mapping = {
            'points': 'PTS',
            'assists': 'AST',
            'rebounds': 'REB',  # Note: They combine OREB+DREB
            'threes': 'FG3M',
            'steals': 'STL',
            'blocks': 'BLK',
            'turnovers': 'TOV',
            'pts_reb_ast': 'PTS_REB_AST',  # Combined prop
        }
    
    def _load_api_key(self):
        """Load API key from config file"""
        config_file = Path('backend/config/api_keys.json')
        
        if config_file.exists():
            with open(config_file, 'r') as f:
                config = json.load(f)
                return config.get('sportsgameodds_api_key')
        
        return None
    
    def fetch_player_props(self, date: str = None):
        """
        Fetch player prop odds from SportsGameOdds API
        
        Args:
            date: Date in YYYY-MM-DD format (default: today)
        
        Returns:
            DataFrame with player props and odds
        """
        if not self.api_key:
            print("❌ No API key found! Set in backend/config/api_keys.json")
            return None
        
        if date is None:
            date = datetime.now().strftime('%Y-%m-%d')
        
        try:
            print(f"📡 Fetching player props for {date}...")
            
            # Endpoint for NBA player props
            endpoint = f"{self.base_url}/nba/player-props"
            
            headers = {
                'Authorization': f'Bearer {self.api_key}',
                'Content-Type': 'application/json'
            }
            
            params = {
                'date': date,
                'market': 'player_props'  # Filter to player props only
            }
            
            response = requests.get(endpoint, headers=headers, params=params, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                
                # Parse response into DataFrame
                props = self._parse_props_response(data)
                print(f"✅ Fetched {len(props)} player props")
                
                return props
            else:
                print(f"❌ API Error {response.status_code}: {response.text}")
                return None
                
        except Exception as e:
            print(f"❌ Error fetching odds: {e}")
            return None
    
    def _parse_props_response(self, data: dict):
        """Parse SportsGameOdds response into clean DataFrame"""
        props = []
        
        # Note: This is a placeholder - actual API structure may differ
        # Adjust based on real API response format
        
        if 'props' in data:
            for prop in data['props']:
                props.append({
                    'player': prop.get('player_name'),
                    'team': prop.get('team'),
                    'opponent': prop.get('opponent'),
                    'stat': self.stat_mapping.get(prop.get('market'), prop.get('market')),
                    'line': float(prop.get('line', 0)),
                    'over_odds': int(prop.get('over_odds', -110)),
                    'under_odds': int(prop.get('under_odds', -110)),
                    'book': prop.get('sportsbook', 'Unknown'),
                    'game_time': prop.get('game_time'),
                })
        
        return pd.DataFrame(props)
    
    def calculate_implied_probability(self, american_odds: int):
        """
        Convert American odds to implied probability
        
        Args:
            american_odds: e.g., -110, +150
        
        Returns:
            Implied probability (0-1)
        """
        if american_odds < 0:
            # Favorite
            return abs(american_odds) / (abs(american_odds) + 100)
        else:
            # Underdog
            return 100 / (american_odds + 100)
    
    def calculate_expected_value(self, prediction: float, line: float, odds: int):
        """
        Calculate expected value (EV) of a bet
        
        Args:
            prediction: Our predicted value (e.g., 25.5 points)
            line: Betting line (e.g., 24.5)
            odds: American odds (e.g., -110)
        
        Returns:
            Expected value as percentage
        """
        # Calculate our win probability
        # Simple approach: probability increases with distance from line
        # More sophisticated: use prediction uncertainty
        
        # Assume normal distribution around our prediction with std of 3
        std = 3.0  # Could use model's RMSE for more accuracy
        
        # Probability of going OVER the line
        prob_over = 1 - norm.cdf(line, loc=prediction, scale=std)
        
        # Implied probability from odds
        implied_prob = self.calculate_implied_probability(odds)
        
        # Expected value
        if odds < 0:
            # Betting on favorite
            payout = 100 / abs(odds)
            ev = (prob_over * payout) - ((1 - prob_over) * 1)
        else:
            # Betting on underdog
            payout = odds / 100
            ev = (prob_over * payout) - ((1 - prob_over) * 1)
        
        return ev * 100  # Return as percentage
    
    def find_value_bets(self, date: str = None, min_ev: float = 5.0, min_confidence: str = 'Medium'):
        """
        Find +EV betting opportunities
        
        Args:
            date: Date to analyze (default: today)
            min_ev: Minimum expected value percentage (default: 5%)
            min_confidence: Minimum prediction confidence ('High', 'Medium', 'Low')
        
        Returns:
            DataFrame with recommended bets sorted by EV
        """
        print("🎰 Finding Value Bets")
        print("=" * 80)
        
        # Fetch odds
        odds = self.fetch_player_props(date)
        
        if odds is None or len(odds) == 0:
            print("❌ No odds data available")
            return None
        
        # Generate predictions for all players in odds
        recommendations = []
        
        confidence_order = {'High': 3, 'Medium': 2, 'Low': 1}
        min_conf_val = confidence_order.get(min_confidence, 2)
        
        for idx, prop in odds.iterrows():
            player = prop['player']
            stat = prop['stat']
            line = prop['line']
            over_odds = prop['over_odds']
            under_odds = prop['under_odds']
            
            # Get our prediction
            # Simplified: would need to know opponent, home/away from odds data
            # For now, using manual prediction
            
            print(f"  Analyzing {player} - {stat} (Line: {line})...")
            
            # Would call predictor here with game context
            # prediction = self.predictor.predict_player_stats(...)
            
            # Placeholder for now
            continue
        
        print(f"\n✅ Found {len(recommendations)} value bets")
        
        if recommendations:
            rec_df = pd.DataFrame(recommendations)
            rec_df = rec_df.sort_values('ev', ascending=False)
            return rec_df
        
        return None
    
    def recommend_bet(self, player: str, opponent: str, is_home: bool, 
                     game_date: str, season: str, stat: str, line: float, 
                     over_odds: int, injury_status: str = 'Healthy',
                     teammates_out: list = None):
        """
        Get recommendation for a specific bet
        
        Args:
            player: Player name
            opponent: Opponent team
            is_home: Home game?
            game_date: Game date
            season: Season
            stat: Stat to bet on (PTS, AST, etc.)
            line: Betting line
            over_odds: Odds for OVER
            injury_status: Player injury status
            teammates_out: List of teammates out
        
        Returns:
            Dict with recommendation
        """
        # Get prediction
        preds = self.predictor.predict_player_stats(
            player, opponent, is_home, game_date, season,
            injury_status, teammates_out
        )
        
        if not preds or preds.get(stat) is None:
            return {'recommendation': 'SKIP', 'reason': 'No prediction available'}
        
        prediction = preds[stat]
        confidence = preds.get(f'{stat}_confidence', 'Medium')
        
        # Calculate EV
        ev_over = self.calculate_expected_value(prediction, line, over_odds)
        
        # Recommendation logic
        if ev_over >= 5.0 and confidence in ['High', 'Medium']:
            return {
                'recommendation': 'BET OVER',
                'player': player,
                'stat': stat,
                'prediction': prediction,
                'line': line,
                'odds': over_odds,
                'ev': round(ev_over, 2),
                'confidence': confidence,
                'edge': round(prediction - line, 2)
            }
        elif ev_over <= -5.0 and confidence in ['High', 'Medium']:
            # Under might be good (negative EV on over = positive EV on under)
            under_odds_calc = under_odds if 'under_odds' in locals() else -110
            return {
                'recommendation': 'BET UNDER',
                'player': player,
                'stat': stat,
                'prediction': prediction,
                'line': line,
                'odds': under_odds_calc,
                'ev': round(abs(ev_over), 2),
                'confidence': confidence,
                'edge': round(line - prediction, 2)
            }
        else:
            return {
                'recommendation': 'SKIP',
                'reason': f'No edge (EV: {ev_over:.1f}%, Pred: {prediction:.1f}, Line: {line})'
            }


def main():
    """Demo betting recommender"""
    print("🎰 NBA Betting Recommender")
    print("=" * 80)
    print()
    
    # Check for API key
    recommender = BettingRecommender()
    
    if not recommender.api_key:
        print("⚠️  No SportsGameOdds API key found!")
        print("   Create: backend/config/api_keys.json")
        print("   Format: {\"sportsgameodds_api_key\": \"your_key_here\"}")
        print()
    
    print("Options:")
    print("1. Analyze specific bet")
    print("2. Find all value bets for today")
    print("3. Exit")
    print()
    
    choice = input("Select (1-3): ").strip()
    
    if choice == "1":
        player = input("Player name (e.g., LeBron James): ").strip()
        opponent = input("Opponent (e.g., CHI): ").strip()
        stat = input("Stat (PTS/AST/REB/etc.): ").strip().upper()
        line = float(input("Betting line (e.g., 24.5): ").strip())
        over_odds = int(input("Over odds (e.g., -110): ").strip())
        is_home_input = input("Home game? (yes/no): ").strip().lower()
        is_home = is_home_input == 'yes'
        
        injury = input("Injury status (Healthy/Probable/Questionable, default=Healthy): ").strip() or 'Healthy'
        teammates = input("Teammates out (comma-separated, or Enter to skip): ").strip()
        teammates_out = [t.strip() for t in teammates.split(',')] if teammates else []
        
        game_date = input("Game date (YYYY-MM-DD, or Enter for today): ").strip()
        if not game_date:
            game_date = datetime.now().strftime('%Y-%m-%d')
        
        season = recommender.predictor._get_current_season()
        
        rec = recommender.recommend_bet(
            player, opponent, is_home, game_date, season,
            stat, line, over_odds, injury, teammates_out
        )
        
        print("\n" + "=" * 80)
        print(f"📊 BETTING RECOMMENDATION")
        print("=" * 80)
        print(f"Player: {player} vs {opponent}")
        print(f"Bet: {stat} {'OVER' if 'OVER' in rec['recommendation'] else 'UNDER'} {line}")
        print(f"Prediction: {rec.get('prediction', 'N/A')}")
        print(f"Confidence: {rec.get('confidence', 'N/A')}")
        print(f"Expected Value: {rec.get('ev', 0):.2f}%")
        print(f"\n🎯 RECOMMENDATION: {rec['recommendation']}")
        
        if 'reason' in rec:
            print(f"   Reason: {rec['reason']}")
    
    elif choice == "2":
        print("\n🔍 Scanning for value bets...")
        value_bets = recommender.find_value_bets(min_ev=5.0)
        
        if value_bets is not None and len(value_bets) > 0:
            print(f"\n✅ Found {len(value_bets)} value bets:")
            print(value_bets[['player', 'stat', 'line', 'prediction', 'ev', 'recommendation']].head(20))
        else:
            print("❌ No value bets found today")
    
    elif choice == "3":
        print("👋 Goodbye!")
    
    else:
        print("❌ Invalid choice")


if __name__ == "__main__":
    main()

