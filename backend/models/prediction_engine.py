#!/usr/bin/env python3
"""
NBA Prediction Engine
====================
Fetches upcoming games and generates predictions using trained models
"""

import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from datetime import datetime, timedelta
from nba_api.stats.endpoints import scoreboardv2, leaguegamefinder, playergamelogs, commonteamroster
from nba_api.stats.static import teams, players
import time
import warnings
warnings.filterwarnings('ignore')


class NBAGamePredictor:
    """Predict player performance for upcoming games"""
    
    def __init__(self, data_dir: str = "backend/data", models_dir: str = "backend/data/player_data"):
        self.data_dir = Path(data_dir)
        self.models_dir = Path(models_dir)
        self.team_stats_file = self.data_dir / "team_stats" / "all_team_stats.csv"
        
        # Load team stats
        if self.team_stats_file.exists():
            self.team_stats = pd.read_csv(self.team_stats_file)
            # Deduplicate
            self.team_stats = self.team_stats.drop_duplicates(subset=['TEAM_ABBREVIATION', 'SEASON'], keep='first')
        else:
            self.team_stats = None
            print("⚠️  Warning: Team stats not found")
        
        # Stats we predict
        self.target_variables = [
            'PTS', 'OREB', 'DREB', 'AST', 'FTM', 'FG3M', 'FGM', 'STL', 'BLK', 'TOV'
        ]
    
    def get_upcoming_games(self, days_ahead: int = 1):
        """
        Fetch upcoming games
        
        Args:
            days_ahead: Number of days ahead to fetch (default: today + tomorrow)
        
        Returns:
            DataFrame with game information
        """
        games = []
        
        for day_offset in range(days_ahead):
            target_date = datetime.now() + timedelta(days=day_offset)
            date_str = target_date.strftime('%Y-%m-%d')
            
            try:
                print(f"📅 Fetching games for {date_str}...")
                scoreboard = scoreboardv2.ScoreboardV2(game_date=date_str)
                game_header = scoreboard.get_data_frames()[0]
                
                if len(game_header) == 0:
                    print(f"  No games scheduled for {date_str}")
                    continue
                
                for idx, game in game_header.iterrows():
                    games.append({
                        'game_date': date_str,
                        'game_id': game['GAME_ID'],
                        'home_team': game['HOME_TEAM_ID'],
                        'away_team': game['VISITOR_TEAM_ID'],
                        'home_team_abbrev': self._get_team_abbrev(game['HOME_TEAM_ID']),
                        'away_team_abbrev': self._get_team_abbrev(game['VISITOR_TEAM_ID']),
                        'season': self._get_current_season()
                    })
                
                print(f"  Found {len(game_header)} games")
                time.sleep(0.6)  # Rate limit
                
            except Exception as e:
                print(f"❌ Error fetching games for {date_str}: {e}")
                continue
        
        return pd.DataFrame(games)
    
    def get_team_roster(self, team_id: int, season: str):
        """Get current roster for a team"""
        try:
            roster = commonteamroster.CommonTeamRoster(team_id=team_id, season=season)
            roster_df = roster.get_data_frames()[0]
            return roster_df['PLAYER'].tolist()
        except Exception as e:
            print(f"⚠️  Error fetching roster for team {team_id}: {e}")
            return []
    
    def get_player_recent_games(self, player_name: str, n_games: int = 20):
        """Get player's most recent games for feature generation"""
        player_dir = self.models_dir / player_name.replace(' ', '_')
        data_file = player_dir / f"{player_name.replace(' ', '_')}_data.csv"
        
        if not data_file.exists():
            return None
        
        df = pd.read_csv(data_file)
        df['GAME_DATE'] = pd.to_datetime(df['GAME_DATE'])
        df = df.sort_values('GAME_DATE', ascending=False)
        
        # Get most recent games
        recent = df.head(n_games).copy()
        return recent.sort_values('GAME_DATE').reset_index(drop=True)
    
    def generate_prediction_features(self, player_name: str, opponent_team: str, 
                                    is_home: bool, game_date: str, season: str,
                                    injury_status: str = "Healthy",
                                    teammates_out: list = None):
        """
        Generate features for upcoming game prediction
        
        Args:
            player_name: Player to predict
            opponent_team: Opponent abbreviation (e.g., 'CHI')
            is_home: True if home game
            game_date: Date of game
            season: NBA season (e.g., '2024-25')
            injury_status: 'Healthy', 'Probable', 'Questionable', 'Doubtful' (optional)
            teammates_out: List of teammate names who are out (optional)
        
        This mimics the feature engineering process but for a future game
        """
        if teammates_out is None:
            teammates_out = []
        # Get recent games
        recent_games = self.get_player_recent_games(player_name, n_games=20)
        
        if recent_games is None or len(recent_games) == 0:
            return None
        
        # Create feature dictionary
        features = {}
        
        # 1. ROLLING AVERAGES (from recent games)
        for window in [3, 5, 10, 20]:
            games_for_window = recent_games.tail(min(window, len(recent_games)))
            
            for stat in self.target_variables:
                if stat in games_for_window.columns:
                    features[f'{stat}_L{window}_AVG'] = games_for_window[stat].mean()
                    features[f'{stat}_L{window}_STD'] = games_for_window[stat].std()
                    features[f'{stat}_L{window}_MAX'] = games_for_window[stat].max()
        
        # 2. CAREER STATS (expanding averages)
        for stat in self.target_variables:
            if stat in recent_games.columns:
                features[f'{stat}_CAREER_AVG'] = recent_games[stat].mean()
                features[f'{stat}_CAREER_STD'] = recent_games[stat].std()
        
        # 3. CONTEXTUAL FEATURES
        features['IS_HOME'] = 1 if is_home else 0
        
        # Opponent encoding (use numerical hash for now)
        features['OPPONENT_ENCODED'] = hash(opponent_team) % 1000
        
        # 4. REST FEATURES
        if 'GAME_DATE' in recent_games.columns:
            last_game_date = recent_games['GAME_DATE'].iloc[-1]
            days_rest = (pd.to_datetime(game_date) - last_game_date).days
            features['DAYS_SINCE_LAST_GAME'] = days_rest
            features['IS_BACK_TO_BACK'] = 1 if days_rest <= 1 else 0
            features['IS_WELL_RESTED'] = 1 if days_rest >= 3 else 0
            features['REST_ENCODED'] = 2 if days_rest >= 2 else (1 if days_rest == 1 else 0)
        
        # 5. PLAYER INFO
        if 'height' in recent_games.columns:
            height = recent_games['height'].iloc[-1]
            features['height_inches'] = self._height_to_inches(height) if pd.notna(height) else 75
        else:
            features['height_inches'] = 75  # Default
        
        if 'position_encoded' in recent_games.columns:
            features['position_encoded'] = recent_games['position_encoded'].iloc[-1]
        else:
            features['position_encoded'] = 0  # Default to guard
        
        if 'season_exp' in recent_games.columns:
            features['season_exp'] = recent_games['season_exp'].iloc[-1]
            features['YEARS_EXPERIENCE'] = int(features['season_exp'])
            features['IS_ROOKIE'] = 1 if features['YEARS_EXPERIENCE'] == 0 else 0
            features['IS_VETERAN'] = 1 if features['YEARS_EXPERIENCE'] >= 5 else 0
        
        # 6. TEAM STATS & OPPONENT FEATURES
        player_team = recent_games['PLAYER_TEAM'].iloc[-1] if 'PLAYER_TEAM' in recent_games.columns else None
        
        if self.team_stats is not None and player_team and opponent_team:
            # Get team stats
            team_data = self._get_team_stats(player_team, season)
            opp_data = self._get_team_stats(opponent_team, season)
            
            if team_data is not None:
                for key, val in team_data.items():
                    features[f'TEAM_{key}'] = val
            
            if opp_data is not None:
                for key, val in opp_data.items():
                    features[f'OPP_{key}'] = val
            
            # Calculate derived features
            if team_data and opp_data:
                self._add_matchup_features(features, team_data, opp_data)
        
        # 7. TEMPORAL FEATURES
        game_dt = pd.to_datetime(game_date)
        features['GAME_DAY_OF_WEEK'] = game_dt.dayofweek
        features['GAME_MONTH'] = game_dt.month
        features['DAY_SIN'] = np.sin(2 * np.pi * features['GAME_DAY_OF_WEEK'] / 7)
        features['DAY_COS'] = np.cos(2 * np.pi * features['GAME_DAY_OF_WEEK'] / 7)
        features['MONTH_SIN'] = np.sin(2 * np.pi * features['GAME_MONTH'] / 12)
        features['MONTH_COS'] = np.cos(2 * np.pi * features['GAME_MONTH'] / 12)
        
        # 8. CAREER PROGRESSION
        features['CAREER_GAME_NUM'] = len(recent_games)
        
        # ====================================================================
        # NEW: INJURY STATUS & TEAMMATES OUT (HIGH IMPACT INPUTS)
        # ====================================================================
        
        # 9. INJURY STATUS
        injury_multiplier = {
            'Healthy': 1.0,
            'Probable': 0.95,
            'Questionable': 0.85,
            'Doubtful': 0.70
        }
        features['INJURY_MULTIPLIER'] = injury_multiplier.get(injury_status, 1.0)
        features['IS_INJURED'] = 1 if injury_status in ['Questionable', 'Doubtful'] else 0
        
        # 10. TEAMMATES OUT (Usage Rate Boost)
        features['TEAMMATES_OUT_COUNT'] = len(teammates_out)
        features['STAR_TEAMMATE_OUT'] = 0  # Will be set below
        
        # Detect if a star teammate is out (increases usage)
        # Star = player who averages 15+ PPG (simplified)
        star_threshold_ppg = 15
        
        for teammate in teammates_out:
            teammate_dir = self.models_dir / teammate.replace(' ', '_')
            teammate_data_file = teammate_dir / f"{teammate.replace(' ', '_')}_data.csv"
            
            if teammate_data_file.exists():
                try:
                    tm_df = pd.read_csv(teammate_data_file)
                    if 'PTS' in tm_df.columns and tm_df['PTS'].mean() > star_threshold_ppg:
                        features['STAR_TEAMMATE_OUT'] = 1
                        break  # Found at least one star out
                except:
                    pass
        
        # Usage boost when teammates are out
        if features['TEAMMATES_OUT_COUNT'] > 0:
            # Boost scoring stats prediction (more shots, more usage)
            for stat in ['PTS', 'AST', 'FGM', 'FG3M', 'FTM']:
                avg_key = f'{stat}_L10_AVG'
                if avg_key in features:
                    # 10% boost per teammate out, max 30%
                    boost_factor = min(1.3, 1 + (0.1 * features['TEAMMATES_OUT_COUNT']))
                    features[f'{stat}_USAGE_BOOST'] = features[avg_key] * (boost_factor - 1)
        
        # Apply injury multiplier to prediction features (if injured, lower expectations)
        if features['INJURY_MULTIPLIER'] < 1.0:
            for stat in self.target_variables:
                avg_key = f'{stat}_L5_AVG'
                if avg_key in features:
                    features[f'{stat}_INJURY_ADJUSTED'] = features[avg_key] * features['INJURY_MULTIPLIER']
        
        return features
    
    def _height_to_inches(self, height_str):
        """Convert height string to inches"""
        try:
            if isinstance(height_str, str) and '-' in height_str:
                feet, inches = height_str.split('-')
                return int(feet) * 12 + int(inches)
        except:
            pass
        return 75  # Default
    
    def _get_team_abbrev(self, team_id):
        """Get team abbreviation from ID"""
        all_teams = teams.get_teams()
        for team in all_teams:
            if team['id'] == team_id:
                return team['abbreviation']
        return 'UNK'
    
    def _get_current_season(self):
        """Get current NBA season string (e.g., '2024-25')"""
        now = datetime.now()
        year = now.year
        month = now.month
        
        # NBA season starts in October
        if month >= 10:
            return f"{year}-{str(year+1)[-2:]}"
        else:
            return f"{year-1}-{str(year)[-2:]}"
    
    def _get_team_stats(self, team_abbrev: str, season: str):
        """Get team stats for a specific season"""
        if self.team_stats is None:
            return None
        
        team_data = self.team_stats[
            (self.team_stats['TEAM_ABBREVIATION'] == team_abbrev) &
            (self.team_stats['SEASON'] == season)
        ]
        
        if len(team_data) == 0:
            return None
        
        # Return dict of key stats
        row = team_data.iloc[0]
        return {
            'OFF_RATING': row.get('OFF_RATING', 110),
            'DEF_RATING': row.get('DEF_RATING', 110),
            'PACE': row.get('PACE', 98),
            'NET_RATING': row.get('NET_RATING', 0),
            'W_PCT': row.get('W_PCT', 0.5),
            'OREB': row.get('OREB', 10),
            'DREB': row.get('DREB', 35),
            'REB': row.get('REB', 45),
            'FGA': row.get('FGA', 85),
            'FG_PCT': row.get('FG_PCT', 0.46),
            'FG3A': row.get('FG3A', 30),
            'AST': row.get('AST', 24),
            'TOV': row.get('TOV', 14),
            'STL': row.get('STL', 7),
            'BLK': row.get('BLK', 5),
            'OREB_PCT': row.get('OREB_PCT', 0.25),
            'DREB_PCT': row.get('DREB_PCT', 0.75),
        }
    
    def _add_matchup_features(self, features: dict, team_data: dict, opp_data: dict):
        """Add matchup-specific features"""
        # Rebounding opportunities
        if 'FGA' in team_data and 'FG_PCT' in team_data:
            features['TEAM_MISSED_SHOTS'] = team_data['FGA'] * (1 - team_data['FG_PCT'])
        
        if 'FGA' in opp_data and 'FG_PCT' in opp_data:
            features['OPP_MISSED_SHOTS'] = opp_data['FGA'] * (1 - opp_data['FG_PCT'])
            features['TOTAL_REBOUND_OPP'] = team_data.get('FGA', 85) + opp_data.get('FGA', 85)
        
        # Pace
        features['GAME_PACE_EST'] = (team_data.get('PACE', 98) + opp_data.get('PACE', 98)) / 2
        features['IS_FAST_PACED'] = 1 if features['GAME_PACE_EST'] > 100 else 0
        
        # Team quality
        features['TEAM_QUALITY_DIFF'] = team_data.get('NET_RATING', 0) - opp_data.get('NET_RATING', 0)
        features['IS_UNDERDOG'] = 1 if features['TEAM_QUALITY_DIFF'] < -5 else 0
        features['IS_FAVORITE'] = 1 if features['TEAM_QUALITY_DIFF'] > 5 else 0
        
        # Defense
        features['VS_STRONG_DEFENSE'] = 1 if opp_data.get('DEF_RATING', 110) < 105 else 0
        features['DEF_RATING_IMPACT'] = 115 - opp_data.get('DEF_RATING', 110)
        
        # Opponent weaknesses
        features['OPP_WEAK_DREB'] = 1 if opp_data.get('DREB_PCT', 0.75) < 0.73 else 0
        features['OPP_WEAK_OREB'] = 1 if opp_data.get('OREB_PCT', 0.25) < 0.25 else 0
        
        # Scoring pace
        if 'PTS' in team_data and 'PTS' in opp_data:
            features['GAME_SCORING_PACE'] = (team_data.get('PTS', 110) + opp_data.get('PTS', 110)) / 2
            features['IS_HIGH_SCORING_GAME'] = 1 if features['GAME_SCORING_PACE'] > 110 else 0
        
        # Opponent foul rate
        if 'FTA' in opp_data:
            features['OPP_FOUL_RATE'] = opp_data['FTA']
            features['VS_FOUL_PRONE_TEAM'] = 1 if opp_data['FTA'] > 25 else 0
        
        # Rebounding strength
        if 'DREB_PCT' in opp_data:
            features['OPP_DREB_STRENGTH'] = opp_data['DREB_PCT']
        if 'OREB_PCT' in opp_data:
            features['OPP_OREB_STRENGTH'] = opp_data['OREB_PCT']
        
        # Height interactions
        if 'height_inches' in features and 'TOTAL_REBOUND_OPP' in features:
            height_norm = (features['height_inches'] - 66) / 24
            features['HEIGHT_NORM'] = height_norm
            features['HEIGHT_REB_OPP'] = height_norm * features['TOTAL_REBOUND_OPP']
    
    def load_player_model(self, player_name: str, stat: str):
        """Load trained model for player and stat"""
        player_dir = self.models_dir / player_name.replace(' ', '_')
        model_file = player_dir / 'models' / f'{stat}_model.pkl'
        
        if not model_file.exists():
            return None
        
        try:
            model_data = joblib.load(model_file)
            return model_data
        except Exception as e:
            print(f"⚠️  Error loading model for {player_name} - {stat}: {e}")
            return None
    
    def predict_player_stats(self, player_name: str, opponent_team: str, 
                            is_home: bool, game_date: str, season: str,
                            injury_status: str = "Healthy",
                            teammates_out: list = None):
        """
        Predict all stats for a player in upcoming game
        
        Args:
            player_name: Player to predict
            opponent_team: Opponent team abbreviation
            is_home: True if home game
            game_date: Date of game
            season: Season (e.g., '2024-25')
            injury_status: Player's injury status ('Healthy', 'Probable', 'Questionable', 'Doubtful')
            teammates_out: List of teammate names who are out (e.g., ['Anthony Davis'])
        
        Returns:
            Dictionary with predictions for all stats
        """
        if teammates_out is None:
            teammates_out = []
        
        # Generate features
        features = self.generate_prediction_features(
            player_name, opponent_team, is_home, game_date, season,
            injury_status, teammates_out
        )
        
        if features is None:
            return None
        
        # Convert to DataFrame
        feature_df = pd.DataFrame([features])
        
        # Make predictions for each stat
        predictions = {
                'player': player_name,
            'opponent': opponent_team,
            'is_home': is_home,
            'game_date': game_date
        }
        
        for stat in self.target_variables:
            model_data = self.load_player_model(player_name, stat)
            
            if model_data is None:
                predictions[stat] = None
                predictions[f'{stat}_confidence'] = None
                continue
            
            try:
                # Get model components
                model = model_data['model']
                scaler = model_data['scaler']
                feature_names = model_data['feature_names']
                
                # Align features (use only features the model was trained on)
                aligned_features = []
                for feat in feature_names:
                    if feat in feature_df.columns:
                        aligned_features.append(feature_df[feat].iloc[0])
                    else:
                        aligned_features.append(0)  # Missing feature = 0
                
                # Scale
                X = np.array(aligned_features).reshape(1, -1)
                X_scaled = scaler.transform(X)
                
                # Predict
                if model_data.get('is_two_stage', False):
                    # Two-stage model
                    prediction = self._predict_two_stage(model, X_scaled)
                else:
                    # Regular model
                    prediction = model.predict(X_scaled)[0]
                
                # Round to reasonable precision
                base_prediction = round(max(0, prediction), 2)
                
                # POST-PROCESS ADJUSTMENTS (injury, teammates)
                adjusted_prediction = base_prediction
                
                # Injury adjustment (reduces all stats)
                if features.get('INJURY_MULTIPLIER', 1.0) < 1.0:
                    adjusted_prediction *= features['INJURY_MULTIPLIER']
                
                # Teammates out adjustment (increases usage for scoring stats)
                if features.get('TEAMMATES_OUT_COUNT', 0) > 0 and stat in ['PTS', 'AST', 'FGM', 'FG3M', 'FTM', 'TOV']:
                    # 8% boost per star teammate out for scoring/playmaking
                    if features.get('STAR_TEAMMATE_OUT', 0) == 1:
                        boost = min(1.25, 1 + (0.08 * features['TEAMMATES_OUT_COUNT']))
                        adjusted_prediction *= boost
                
                # Fatigue adjustment (back-to-back games)
                if features.get('IS_BACK_TO_BACK', 0) == 1:
                    # Back-to-back games: 3-5% decrease in performance
                    fatigue_penalty = 0.96  # 4% reduction
                    adjusted_prediction *= fatigue_penalty
                elif features.get('DAYS_SINCE_LAST_GAME', 3) >= 4:
                    # Well rested (4+ days): 2-3% boost
                    rest_boost = 1.02
                    adjusted_prediction *= rest_boost
                
                predictions[stat] = round(max(0, adjusted_prediction), 2)
                predictions[f'{stat}_base'] = base_prediction  # Save unadjusted
                
                # Confidence based on recent consistency
                std_key = f'{stat}_L5_STD'
                if std_key in features:
                    # Lower std = higher confidence
                    std = features[std_key]
                    # Reduce confidence if injured
                    if features.get('IS_INJURED', 0) == 1:
                        predictions[f'{stat}_confidence'] = 'Low'
                    else:
                        predictions[f'{stat}_confidence'] = 'High' if std < 2 else ('Medium' if std < 4 else 'Low')
                    predictions[f'{stat}_confidence'] = 'Medium' if features.get('IS_INJURED', 0) == 0 else 'Low'
                
            except Exception as e:
                print(f"⚠️  Error predicting {stat} for {player_name}: {e}")
                predictions[stat] = None
                predictions[f'{stat}_confidence'] = None
        
        return predictions
    
    def _predict_two_stage(self, model_dict, X):
        """Handle two-stage model predictions"""
        if isinstance(model_dict, dict) and model_dict.get('type') == 'two_stage':
            # Probability of getting any
            proba = model_dict['classifier'].predict_proba(X)[0, 1]
            # Magnitude if getting some
            magnitude = model_dict['regressor'].predict(X)[0]
            return proba * magnitude
        else:
            # Single model
            if isinstance(model_dict, dict) and 'model' in model_dict:
                return model_dict['model'].predict(X)[0]
            return model_dict.predict(X)[0]
    
    def predict_game(self, home_team: str, away_team: str, game_date: str, season: str):
        """
        Predict all player stats for a specific game
        
        Returns:
            DataFrame with predictions for all players
        """
        print(f"\n🏀 Predicting: {away_team} @ {home_team} on {game_date}")
        print("=" * 80)
        
        # Get rosters (simplified - use players with models)
        home_players = self._get_players_with_models(home_team)
        away_players = self._get_players_with_models(away_team)
        
        all_predictions = []
        
        # Predict home team
        for player in home_players[:15]:  # Top 15 players with models
            preds = self.predict_player_stats(player, away_team, True, game_date, season)
            if preds:
                all_predictions.append(preds)
        
        # Predict away team
        for player in away_players[:15]:
            preds = self.predict_player_stats(player, home_team, False, game_date, season)
            if preds:
                all_predictions.append(preds)
        
        return pd.DataFrame(all_predictions)
    
    def _get_players_with_models(self, team_abbrev: str):
        """Get list of players who have trained models (approximation)"""
        # This is a simplified version - in production, you'd maintain a player-team mapping
        players_with_models = []
        
        for player_dir in self.models_dir.iterdir():
            if not player_dir.is_dir():
                continue
            
            models_dir = player_dir / 'models'
            if models_dir.exists():
                player_name = player_dir.name.replace('_', ' ')
                # Check if player recently played for this team (simplified)
                players_with_models.append(player_name)
        
        return players_with_models[:50]  # Return first 50
    
    def predict_player_next_game(self, player_name: str, auto_fetch_context: bool = True):
        """
        SMART PREDICTION: Fetch all context automatically and predict
        
        Args:
            player_name: Player to predict (e.g., 'LeBron James')
            auto_fetch_context: If True, fetch opponent, injury status, etc. from NBA
        
        Returns:
            Predictions dict with all auto-fetched context
        """
        # 1. Find player's next game
        print(f"🔍 Finding next game for {player_name}...")
        
        # Get player's team (from most recent game)
        recent_games = self.get_player_recent_games(player_name, n_games=1)
        if recent_games is None or len(recent_games) == 0:
            print(f"❌ No data found for {player_name}")
            return None
        
        player_team = recent_games['PLAYER_TEAM'].iloc[0] if 'PLAYER_TEAM' in recent_games.columns else None
        
        if not player_team:
            print(f"❌ Could not determine team for {player_name}")
            return None
        
        print(f"  Player team: {player_team}")
        
        # 2. Get upcoming games for player's team
        upcoming_games = self.get_upcoming_games(days_ahead=7)  # Next week
        
        if upcoming_games is None or len(upcoming_games) == 0:
            print("❌ No upcoming games found")
            return None
    
        # Find this team's next game
        team_game = upcoming_games[
            (upcoming_games['home_team_abbrev'] == player_team) | 
            (upcoming_games['away_team_abbrev'] == player_team)
        ]
        
        if len(team_game) == 0:
            print(f"❌ No upcoming games found for {player_team}")
            return None
        
        next_game = team_game.iloc[0]
        
        # 3. Determine game context
        is_home = next_game['home_team_abbrev'] == player_team
        opponent = next_game['away_team_abbrev'] if is_home else next_game['home_team_abbrev']
        game_date = next_game['game_date']
        season = next_game['season']
        
        print(f"  Next game: {player_team} vs {opponent} on {game_date}")
        print(f"  Location: {'Home' if is_home else 'Away'}")
        
        # 4. Auto-detect injury status (placeholder - would use NBA injury API)
        injury_status = 'Healthy'  # TODO: Fetch from NBA injury report
        
        # 5. Auto-detect teammates out (placeholder - would use NBA injury API)
        teammates_out = []  # TODO: Fetch from NBA injury report
        
        # 6. Detect if starter (based on recent games)
        is_starter = self._is_likely_starter(player_name)
        print(f"  Role: {'Starter' if is_starter else 'Bench'}")
        
        # 7. Make prediction
        print(f"\n🎯 Generating predictions...")
        predictions = self.predict_player_stats(
            player_name=player_name,
            opponent_team=opponent,
            is_home=is_home,
            game_date=game_date,
            season=season,
            injury_status=injury_status,
            teammates_out=teammates_out
        )
        
        if predictions:
            # Add context to predictions
            predictions['auto_fetched'] = True
            predictions['is_starter'] = is_starter
            predictions['player_team'] = player_team
            
        return predictions
    
    def _is_likely_starter(self, player_name: str):
        """Determine if player is likely a starter based on recent minutes"""
        recent_games = self.get_player_recent_games(player_name, n_games=5)
        
        if recent_games is None or len(recent_games) == 0:
            return True  # Default to starter
        
        if 'MIN' in recent_games.columns:
            avg_minutes = pd.to_numeric(recent_games['MIN'], errors='coerce').mean()
            # If averaging 28+ minutes, likely a starter
            return avg_minutes >= 28
        
        return True  # Default
    
    def predict_all_upcoming_games(self, days_ahead: int = 1):
        """
        Predict stats for all upcoming games
        
        Args:
            days_ahead: How many days ahead to predict
            
        Returns:
            DataFrame with all predictions
        """
        print("🎯 NBA Prediction Engine")
        print("=" * 80)
        
        # Get upcoming games
        games = self.get_upcoming_games(days_ahead)
        
        if len(games) == 0:
            print("❌ No upcoming games found!")
            return None
        
        print(f"\n✅ Found {len(games)} upcoming games\n")
        
        all_predictions = []
        
        for idx, game in games.iterrows():
            predictions = self.predict_game(
                game['home_team_abbrev'],
                game['away_team_abbrev'],
                game['game_date'],
                game['season']
            )
            
            if predictions is not None and len(predictions) > 0:
                predictions['game_id'] = game['game_id']
                all_predictions.append(predictions)
        
        if all_predictions:
            final_df = pd.concat(all_predictions, ignore_index=True)
            
            # Save predictions
            output_file = Path('backend/outputs') / f"predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            final_df.to_csv(output_file, index=False)
            print(f"\n💾 Saved predictions to: {output_file}")
            
            return final_df
        
        return None


def main():
    """Demo: Predict upcoming games"""
    predictor = NBAGamePredictor()
    
    print("🏀 NBA Prediction Engine")
    print("=" * 80)
    print()
    print("Options:")
    print("1. SMART: Auto-predict player's next game (fetches all context)")
    print("2. Manual: Predict specific game with custom inputs")
    print("3. Predict all today's games")
    print("4. Exit")
    print()
    
    choice = input("Select option (1-4): ").strip()
    
    if choice == "1":
        # SMART AUTO-FETCH MODE
        player = input("Enter player name (e.g., LeBron James): ").strip()
        
        preds = predictor.predict_player_next_game(player, auto_fetch_context=True)
        
        if preds:
            print(f"\n📊 AUTO-FETCHED PREDICTIONS:")
            print("=" * 80)
            print(f"Player: {preds['player']}")
            print(f"Opponent: {preds['opponent']} ({'Home' if preds['is_home'] else 'Away'})")
            print(f"Date: {preds['game_date']}")
            print(f"Role: {'Starter' if preds.get('is_starter') else 'Bench'}")
            print()
            print("Predicted Stats:")
            for stat in predictor.target_variables:
                if preds.get(stat) is not None:
                    conf = preds.get(f'{stat}_confidence', 'N/A')
                    base = preds.get(f'{stat}_base', preds[stat])
                    print(f"  {stat:<6} {preds[stat]:>6.1f}  (Confidence: {conf})")
        else:
            print(f"❌ Could not generate predictions")
    
    elif choice == "2":
        # MANUAL MODE
        player = input("Enter player name (e.g., LeBron James): ").strip()
        opponent = input("Enter opponent team (e.g., CHI): ").strip()
        is_home = input("Home game? (yes/no): ").strip().lower() == 'yes'
        game_date = input("Game date (YYYY-MM-DD): ").strip() or datetime.now().strftime('%Y-%m-%d')
        injury = input("Injury status (Healthy/Probable/Questionable/Doubtful): ").strip() or 'Healthy'
        teammates = input("Teammates out (comma-separated, or press Enter): ").strip()
        
        teammates_out = [t.strip() for t in teammates.split(',')] if teammates else []
        season = predictor._get_current_season()
        
        preds = predictor.predict_player_stats(
            player, opponent, is_home, game_date, season, injury, teammates_out
        )
        
        if preds:
            print(f"\n🎯 Predictions for {player} vs {opponent}:")
            print("=" * 80)
            for stat in predictor.target_variables:
                if preds.get(stat) is not None:
                    conf = preds.get(f'{stat}_confidence', 'N/A')
                    base = preds.get(f'{stat}_base', preds[stat])
                    if base != preds[stat]:
                        print(f"  {stat:<6} {preds[stat]:>6.1f} (was {base:.1f}, adjusted)  [{conf}]")
                    else:
                        print(f"  {stat:<6} {preds[stat]:>6.1f}  [{conf}]")
        else:
            print(f"❌ Could not generate predictions")
    
    elif choice == "3":
        # ALL GAMES
        predictions = predictor.predict_all_upcoming_games(days_ahead=1)
        if predictions is not None:
            print("\n📊 Predictions Summary:")
            print(predictions[['player', 'opponent', 'is_home', 'PTS', 'AST']].head(20))
    
    elif choice == "4":
        print("👋 Goodbye!")
    
    else:
        print("❌ Invalid choice")


if __name__ == "__main__":
    main()
