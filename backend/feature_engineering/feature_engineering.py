#!/usr/bin/env python3
"""
Feature Engineering for NBA Player Performance Prediction
=========================================================
Creates comprehensive features for predicting:
- Points (PTS)
- Offensive Rebounds (OREB)
- Defensive Rebounds (DREB)
- Assists (AST)
- Free Throws Made (FTM)
- 3-Pointers Made (FG3M)
- Field Goals Made (FGM)
- Steals (STL)
- Blocks (BLK)
- Turnovers (TOV)
"""

import pandas as pd
import numpy as np
import os
import logging
from pathlib import Path
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class NBAFeatureEngineering:
    """Create features for NBA player performance prediction"""
    
    def __init__(self, data_dir: str = "player_data", output_dir: str = "processed_data"):
        self.data_dir = data_dir
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Label encoders for categorical variables
        self.encoders = {}
        
        # Load team stats once for all players (now in root directory)
        team_stats_file = Path('team_stats') / 'all_team_stats.csv'
        if team_stats_file.exists():
            self.team_stats = pd.read_csv(team_stats_file)
            
            # CRITICAL: Deduplicate team stats to prevent cartesian product on merge
            # Keep only one row per (TEAM_ABBREVIATION, SEASON) combination
            original_len = len(self.team_stats)
            self.team_stats = self.team_stats.drop_duplicates(subset=['TEAM_ABBREVIATION', 'SEASON'], keep='first')
            
            logger.info(f"✅ Loaded {len(self.team_stats)} unique team-season records (removed {original_len - len(self.team_stats)} duplicates)")
        else:
            self.team_stats = None
            logger.warning("⚠️  Team stats not found - opponent strength features will be limited")
        
        # Target variables we want to predict
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
        
        # Columns to exclude from ML training (but keep in data for analysis)
        self.exclude_columns = [
            # IDs and dates (not meaningful for prediction)
            'SEASON_ID', 'Player_ID', 'Game_ID', 'GAME_DATE',
            
            # Player/team identifiers (use encoded versions instead)
            'PLAYER_NAME', 'MATCHUP', 'OPPONENT', 'PLAYER_TEAM', 'OPPONENT_TEAM',
            
            # Categorical variables (use encoded versions instead)
            'SEASON', 'WL', 'GAME_RESULT', 'REST_FACTOR', 'height', 'position',
            
            # Non-numeric or unreliable
            'VIDEO_AVAILABLE', 'GAME_QUARTER',
            
            # Archetype categorical columns (ALL will be dropped, only encoded versions kept)
            'ARCHETYPE_PRIMARY_ROLE', 'ARCHETYPE_SHOOTING_STYLE', 
            'ARCHETYPE_PLAYMAKING', 'ARCHETYPE_REBOUNDING',
            'ARCHETYPE_DEFENSE', 'ARCHETYPE_EFFICIENCY',
            'ARCHETYPE_USAGE', 'ARCHETYPE_PLAY_STYLE',
            'ARCHETYPE_PLAYING_TIME', 'ARCHETYPE_CONSISTENCY',
            'ARCHETYPE_FREE_THROW', 'ARCHETYPE_VERSATILITY',
            'COMPOSITE_ARCHETYPE',
            
            # Target variables (don't use targets to predict targets)
        ] + self.target_variables
    
    def process_player_data(self, player_name: str, save_to_file: bool = True) -> pd.DataFrame:
        """
        Process a single player's data with comprehensive feature engineering
        
        Args:
            player_name: Name of the player
            save_to_file: If True, save features to player's directory
            
        Returns:
            DataFrame with engineered features
        """
        # Load player data
        player_dir = Path(self.data_dir) / player_name.replace(' ', '_')
        nba_file = player_dir / f"{player_name.replace(' ', '_')}_data.csv"
        college_file = player_dir / f"{player_name.replace(' ', '_')}_college_data.csv"
        archetype_file = player_dir / f"{player_name.replace(' ', '_')}_archetype.csv"
        features_file = player_dir / f"{player_name.replace(' ', '_')}_features.csv"
        
        if not nba_file.exists():
            return None
        
        # Check if features already exist (checkpointing enabled)
        if features_file.exists() and save_to_file:
            # Skip if features already exist (allows resuming)
            return None
        
        df = pd.read_csv(nba_file)
        
        # Clean data - replace empty strings with NaN
        df = df.replace('', np.nan)
        df = df.replace(' ', np.nan)
        
        # Encode categorical columns from raw data
        if 'WL' in df.columns:
            df['WIN'] = (df['WL'] == 'W').astype(int)
            df = df.drop('WL', axis=1)
        
        if 'height' in df.columns:
            # Convert height to inches (e.g., "6-7" -> 79)
            df['height_inches'] = df['height'].apply(lambda x: self._height_to_inches(x) if pd.notna(x) else np.nan)
            df = df.drop('height', axis=1)
        
        if 'position' in df.columns:
            df['position_encoded'] = pd.Categorical(df['position']).codes
            df = df.drop('position', axis=1)
        
        if 'REST_FACTOR' in df.columns:
            # Encode rest factor
            rest_mapping = {'No Rest': 0, '1 Day': 1, '2+ Days': 2, 'Unknown': -1}
            df['REST_ENCODED'] = df['REST_FACTOR'].map(rest_mapping).fillna(-1).astype(int)
            df = df.drop('REST_FACTOR', axis=1)
        
        # Convert date column
        df['GAME_DATE'] = pd.to_datetime(df['GAME_DATE'])
        df = df.sort_values('GAME_DATE').reset_index(drop=True)
        
        # Add college data if available
        if college_file.exists():
            college_df = pd.read_csv(college_file)
            df = self._add_college_features(df, college_df)
        
        # Add archetype data if available
        if archetype_file.exists():
            archetype_df = pd.read_csv(archetype_file)
            df = self._add_archetype_features(df, archetype_df)
        
        # 1. STATISTICAL FEATURES
        df = self._create_statistical_features(df)
        
        # 2. ROLLING AVERAGES (Form/Momentum)
        df = self._create_rolling_features(df)
        
        # 3. CONTEXTUAL FEATURES (Encoded)
        df = self._create_contextual_features(df)
        
        # 4. TEMPORAL FEATURES
        df = self._create_temporal_features(df)
        
        # 5. OPPONENT FEATURES
        df = self._create_opponent_features(df)
        
        # 6. REST AND FATIGUE FEATURES
        df = self._create_rest_features(df)
        
        # 7. CAREER PROGRESSION FEATURES
        df = self._create_career_features(df)
        
        # 8. INTERACTION FEATURES
        df = self._create_interaction_features(df)
        
        # 9. ARCHETYPE-BASED MATCHUP FEATURES (if archetypes available)
        df = self._create_archetype_matchup_features(df)
        
        # Add player identifier
        df['PLAYER_NAME'] = player_name
        
        # CRITICAL: Remove leakage columns (but KEEP target variables for training)
        # We need targets in the CSV to train on, but we remove derived leakage features
        columns_to_remove = [
            # Total rebounds (REB = OREB + DREB, causes leakage when predicting OREB/DREB)
            'REB', 'REB_LAST_3_AVG', 'REB_LAST_5_AVG', 'REB_LAST_10_AVG', 
            'REB_L3_AVG', 'REB_L5_AVG', 'REB_L10_AVG', 'REB_L20_AVG',
            'REB_L3_STD', 'REB_L5_STD', 'REB_L10_STD', 'REB_L20_STD',
            'REB_L3_MAX', 'REB_L5_MAX', 'REB_L10_MAX', 'REB_L20_MAX',
            'REB_TREND_L3', 'REB_TREND_L5', 'REB_TREND_L10', 'REB_TREND_L20',
            'REB_CONSISTENCY', 'REB_PER_MIN', 'REB_CAREER_AVG', 'REB_CAREER_STD',
            'REB_RECENT_VS_SEASON',
            
            # Plus-minus from current game (heavily correlated with all stats)
            'PLUS_MINUS',
            
            # Current game minutes (strong indicator of all stats in that game)
            'MIN',
            
            # Shooting percentages from current game (derived from makes/attempts)
            'FG_PCT', 'FG3_PCT', 'FT_PCT',
            
            # Attempts from current game (when predicting makes)
            'FGA', 'FG3A', 'FTA',
            
            # Useless columns
            'VIDEO_AVAILABLE'
        ]
        
        # Remove leakage columns (KEEP target variables!)
        df = df.drop(columns=[c for c in columns_to_remove if c in df.columns], errors='ignore')
        
        # Save to player's directory if requested
        if save_to_file:
            df.to_csv(features_file, index=False)
            print(f"💾 Saved features: {player_name}")
        
        return df
    
    def _height_to_inches(self, height_str):
        """Convert height string like '6-7' to inches"""
        try:
            if isinstance(height_str, str) and '-' in height_str:
                feet, inches = height_str.split('-')
                return int(feet) * 12 + int(inches)
        except:
            pass
        return np.nan
    
    def _add_college_features(self, df: pd.DataFrame, college_df: pd.DataFrame) -> pd.DataFrame:
        """Add college statistics as features"""
        if len(college_df) > 0:
            college_stats = college_df.iloc[0]
            df['COLLEGE_PPG'] = college_stats.get('college_ppg', 0)
            df['COLLEGE_RPG'] = college_stats.get('college_rpg', 0)
            df['COLLEGE_APG'] = college_stats.get('college_apg', 0)
            df['COLLEGE_FG_PCT'] = college_stats.get('college_fg_pct', 0)
            df['COLLEGE_3P_PCT'] = college_stats.get('college_3p_pct', 0)
            df['COLLEGE_FT_PCT'] = college_stats.get('college_ft_pct', 0)
            df['COLLEGE_SEASONS'] = college_stats.get('college_seasons', 0)
        else:
            df['COLLEGE_PPG'] = 0
            df['COLLEGE_RPG'] = 0
            df['COLLEGE_APG'] = 0
            df['COLLEGE_FG_PCT'] = 0
            df['COLLEGE_3P_PCT'] = 0
            df['COLLEGE_FT_PCT'] = 0
            df['COLLEGE_SEASONS'] = 0
        
        return df
    
    def _add_archetype_features(self, df: pd.DataFrame, archetype_df: pd.DataFrame) -> pd.DataFrame:
        """Add archetype features to main dataframe"""
        # Merge archetypes by game date
        if 'GAME_DATE' in df.columns and 'GAME_DATE' in archetype_df.columns:
            df['GAME_DATE_MERGE'] = pd.to_datetime(df['GAME_DATE'])
            archetype_df['GAME_DATE_MERGE'] = pd.to_datetime(archetype_df['GAME_DATE'])
            
            # Select archetype columns to merge
            archetype_cols = ['GAME_DATE_MERGE'] + [col for col in archetype_df.columns 
                            if 'ARCHETYPE' in col or 'STRENGTH' in col]
            
            # Merge
            df = df.merge(archetype_df[archetype_cols], on='GAME_DATE_MERGE', how='left', suffixes=('', '_ARCH'))
            df = df.drop('GAME_DATE_MERGE', axis=1)
            
            # Encode ALL categorical archetypes
            archetype_categorical = [col for col in df.columns if col.startswith('ARCHETYPE_')]
            
            for col in archetype_categorical:
                # Label encode each archetype dimension
                if df[col].dtype == 'object':
                    # Create encoded version
                    df[f'{col}_ENCODED'] = pd.Categorical(df[col]).codes
                    # Also create one-hot encoding for important archetypes
                    if col in ['ARCHETYPE_PRIMARY_ROLE', 'ARCHETYPE_SHOOTING_STYLE', 
                              'ARCHETYPE_PLAYMAKING', 'ARCHETYPE_DEFENSE']:
                        # Create binary columns for each archetype value
                        dummies = pd.get_dummies(df[col], prefix=col, drop_first=True)
                        df = pd.concat([df, dummies], axis=1)
                    
                    # DROP the original categorical column to save space
                    df = df.drop(col, axis=1)
            
            # Drop COMPOSITE_ARCHETYPE (it's too complex to encode meaningfully)
            if 'COMPOSITE_ARCHETYPE' in df.columns:
                df = df.drop('COMPOSITE_ARCHETYPE', axis=1)
        
        return df
    
    def _create_statistical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create basic statistical features"""
        
        # Ensure numeric types
        numeric_cols = ['FGM', 'FGA', 'FG3M', 'FG3A', 'FTM', 'FTA', 'OREB', 'DREB', 
                       'AST', 'STL', 'BLK', 'TOV', 'PTS', 'MIN']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Shooting efficiency
        df['FG_PCT'] = df['FGM'] / df['FGA'].replace(0, np.nan)
        df['FG3_PCT'] = df['FG3M'] / df['FG3A'].replace(0, np.nan)
        df['FT_PCT'] = df['FTM'] / df['FTA'].replace(0, np.nan)
        
        # Total rebounds
        df['REB'] = df['OREB'] + df['DREB']
        
        # Advanced metrics (already in data but ensuring they exist)
        if 'TS_PCT' not in df.columns:
            df['TS_PCT'] = df['PTS'] / (2 * (df['FGA'] + 0.44 * df['FTA'])).replace(0, np.nan)
        
        # Assist-to-turnover ratio - will be calculated after rolling features
        
        # Rebound rate (per minute)
        df['REB_PER_MIN'] = df['REB'] / df['MIN'].replace(0, np.nan)
        
        # Stock (Steals + Blocks)
        df['STOCK'] = df['STL'] + df['BLK']
        
        # Fantasy points - will be calculated after rolling features
        
        return df
    
    def _create_rolling_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create rolling averages for momentum/form"""
        
        windows = [3, 5, 10, 20]  # Last 3, 5, 10, 20 games
        
        for window in windows:
            for stat in self.target_variables:
                if stat in df.columns:
                    # CRITICAL FIX: Shift by 1 to exclude current game from rolling average
                    # This prevents data leakage - we can't use current game to predict current game!
                    stat_shifted = df[stat].shift(1)
                    
                    # Rolling mean (of PREVIOUS games only)
                    df[f'{stat}_L{window}_AVG'] = stat_shifted.rolling(window=window, min_periods=1).mean()
                    
                    # Rolling std (consistency)
                    df[f'{stat}_L{window}_STD'] = stat_shifted.rolling(window=window, min_periods=1).std()
                    
                    # Rolling max
                    df[f'{stat}_L{window}_MAX'] = stat_shifted.rolling(window=window, min_periods=1).max()
                    
                    # Trend (change in rolling average, NOT difference from current game)
                    # Use shift to compare current rolling avg to previous rolling avg
                    df[f'{stat}_TREND_L{window}'] = df[f'{stat}_L{window}_AVG'] - df[f'{stat}_L{window}_AVG'].shift(1)
        
        # Recent vs long-term form comparison
        for stat in self.target_variables:
            if stat in df.columns:
                df[f'{stat}_RECENT_VS_SEASON'] = df[f'{stat}_L5_AVG'] / df[f'{stat}_L20_AVG'].replace(0, np.nan)
        
        # NOW create derived features using rolling averages (no leakage)
        # Assist-to-turnover ratio from rolling averages
        if 'AST_L5_AVG' in df.columns and 'TOV_L5_AVG' in df.columns:
            df['AST_TO_RATIO'] = df['AST_L5_AVG'] / df['TOV_L5_AVG'].replace(0, np.nan)
        
        # Points per FGA from rolling average
        if 'PTS_L5_AVG' in df.columns and 'FGA' in df.columns:
            df['PTS_PER_FGA'] = df['PTS_L5_AVG'] / pd.to_numeric(df['FGA'], errors='coerce').replace(0, np.nan)
        
        # Fantasy points from rolling averages
        if all(f'{stat}_L5_AVG' in df.columns for stat in ['PTS', 'AST', 'STL', 'BLK', 'TOV']):
            if 'OREB_L5_AVG' in df.columns and 'DREB_L5_AVG' in df.columns:
                df['FANTASY_PTS'] = (
                    df['PTS_L5_AVG'] + 
                    1.2 * (df['OREB_L5_AVG'] + df['DREB_L5_AVG']) + 
                    1.5 * df['AST_L5_AVG'] + 
                    3 * df['STL_L5_AVG'] + 
                    3 * df['BLK_L5_AVG'] - 
                    df['TOV_L5_AVG']
                )
        
        return df
    
    def _create_contextual_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create and encode contextual features"""
        
        # 1. Home/Away (already binary)
        df['IS_HOME'] = df['IS_HOME'].fillna(0).astype(int)
        
        # 2. Opponent encoding (Label encoding)
        if 'OPPONENT' in df.columns:
            df['OPPONENT'] = df['OPPONENT'].fillna('UNKNOWN')
            if 'OPPONENT' not in self.encoders:
                self.encoders['OPPONENT'] = LabelEncoder()
                df['OPPONENT_ENCODED'] = self.encoders['OPPONENT'].fit_transform(df['OPPONENT'])
            else:
                # Handle new opponents
                known_opponents = set(self.encoders['OPPONENT'].classes_)
                df['OPPONENT_ENCODED'] = df['OPPONENT'].apply(
                    lambda x: self.encoders['OPPONENT'].transform([x])[0] if x in known_opponents else -1
                )
        
        # 3. Rest factor encoding (already done at data loading, skip here)
        
        # 4. Day of week encoding (already encoded in GAME_DAY_OF_WEEK)
        
        # 5. Month encoding (already in GAME_MONTH)
        
        # 6. Game result (Win/Loss) as binary
        if 'GAME_RESULT' in df.columns:
            df['WIN'] = (df['GAME_RESULT'] == 'W').astype(int)
        
        return df
    
    def _create_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create time-based features"""
        
        # Season progression (0-1 scale)
        if 'GAME_NUMBER_IN_SEASON' in df.columns:
            df['GAME_NUMBER_IN_SEASON'] = pd.to_numeric(df['GAME_NUMBER_IN_SEASON'], errors='coerce')
            df['SEASON_PROGRESS'] = df['GAME_NUMBER_IN_SEASON'] / 82.0
            
            # Early, mid, late season indicators
            df['IS_EARLY_SEASON'] = (df['GAME_NUMBER_IN_SEASON'] <= 20).astype(int)
            df['IS_MID_SEASON'] = ((df['GAME_NUMBER_IN_SEASON'] > 20) & (df['GAME_NUMBER_IN_SEASON'] <= 60)).astype(int)
            df['IS_LATE_SEASON'] = (df['GAME_NUMBER_IN_SEASON'] > 60).astype(int)
        
        # Day of week features (cyclical encoding)
        if 'GAME_DAY_OF_WEEK' in df.columns:
            df['GAME_DAY_OF_WEEK'] = pd.to_numeric(df['GAME_DAY_OF_WEEK'], errors='coerce')
            df['DAY_SIN'] = np.sin(2 * np.pi * df['GAME_DAY_OF_WEEK'] / 7)
            df['DAY_COS'] = np.cos(2 * np.pi * df['GAME_DAY_OF_WEEK'] / 7)
        
        # Month features (cyclical encoding)
        if 'GAME_MONTH' in df.columns:
            df['GAME_MONTH'] = pd.to_numeric(df['GAME_MONTH'], errors='coerce')
            df['MONTH_SIN'] = np.sin(2 * np.pi * df['GAME_MONTH'] / 12)
            df['MONTH_COS'] = np.cos(2 * np.pi * df['GAME_MONTH'] / 12)
        
        return df
    
    def _create_opponent_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create opponent-specific features using team stats"""
        
        # Add team-level stats if available
        if self.team_stats is not None and 'OPPONENT_TEAM' in df.columns and 'SEASON' in df.columns:
            # Prepare opponent team stats with ALL available data
            opp_cols = ['SEASON', 'TEAM_ABBREVIATION', 'DEF_RATING', 'OFF_RATING', 'PACE', 'NET_RATING', 'W_PCT']
            
            # Add rebounding columns if available
            if 'OREB' in self.team_stats.columns:
                opp_cols.extend(['OREB', 'DREB', 'REB', 'OREB_PCT', 'DREB_PCT'])
            
            # Add shot volume
            if 'FGA' in self.team_stats.columns:
                opp_cols.extend(['FGA', 'FG3A', 'FG_PCT', 'FG3_PCT'])
            
            # Add ALL counting stats for better predictions
            if 'AST' in self.team_stats.columns:
                opp_cols.extend(['AST', 'TOV', 'STL', 'BLK', 'PTS'])
            
            # Add free throw data
            if 'FTA' in self.team_stats.columns:
                opp_cols.append('FTA')
            
            # Filter to available columns
            opp_cols = [col for col in opp_cols if col in self.team_stats.columns]
            
            opp_team_stats = self.team_stats[opp_cols].copy()
            opp_team_stats = opp_team_stats.rename(columns={
                'TEAM_ABBREVIATION': 'OPPONENT_TEAM',
                'DEF_RATING': 'OPP_DEF_RATING',
                'OFF_RATING': 'OPP_OFF_RATING',
                'PACE': 'OPP_PACE',
                'NET_RATING': 'OPP_NET_RATING',
                'W_PCT': 'OPP_WIN_PCT',
                # Rebounding
                'OREB': 'OPP_OREB',
                'DREB': 'OPP_DREB',
                'REB': 'OPP_REB',
                'OREB_PCT': 'OPP_OREB_PCT',
                'DREB_PCT': 'OPP_DREB_PCT',
                # Shooting
                'FGA': 'OPP_FGA',
                'FG3A': 'OPP_FG3A',
                'FG_PCT': 'OPP_FG_PCT',
                'FG3_PCT': 'OPP_FG3_PCT',
                'FTA': 'OPP_FTA',
                # Counting stats (how they perform, tells us about matchups)
                'AST': 'OPP_AST',
                'TOV': 'OPP_TOV',  # Teams with high TOV = more steal opportunities for us
                'STL': 'OPP_STL',  # Teams with high STL = more pressure on our turnovers
                'BLK': 'OPP_BLK',  # Teams with high BLK = harder to score inside
                'PTS': 'OPP_PTS'
            })
            
            df = df.merge(opp_team_stats, on=['OPPONENT_TEAM', 'SEASON'], how='left')
            
            # Merge player's own team stats with ALL data
            player_cols = ['SEASON', 'TEAM_ABBREVIATION', 'DEF_RATING', 'OFF_RATING', 'PACE', 'NET_RATING', 'W_PCT']
            
            if 'OREB' in self.team_stats.columns:
                player_cols.extend(['OREB', 'DREB', 'REB', 'OREB_PCT', 'DREB_PCT'])
            
            if 'FGA' in self.team_stats.columns:
                player_cols.extend(['FGA', 'FG3A', 'FG_PCT', 'FG3_PCT'])
            
            if 'AST' in self.team_stats.columns:
                player_cols.extend(['AST', 'TOV', 'STL', 'BLK', 'PTS'])
            
            if 'FTA' in self.team_stats.columns:
                player_cols.append('FTA')
            
            player_cols = [col for col in player_cols if col in self.team_stats.columns]
            
            player_team_stats = self.team_stats[player_cols].copy()
            player_team_stats = player_team_stats.rename(columns={
                'TEAM_ABBREVIATION': 'PLAYER_TEAM',
                'DEF_RATING': 'TEAM_DEF_RATING',
                'OFF_RATING': 'TEAM_OFF_RATING',
                'PACE': 'TEAM_PACE',
                'NET_RATING': 'TEAM_NET_RATING',
                'W_PCT': 'TEAM_WIN_PCT',
                # Rebounding
                'OREB': 'TEAM_OREB',
                'DREB': 'TEAM_DREB',
                'REB': 'TEAM_REB',
                'OREB_PCT': 'TEAM_OREB_PCT',
                'DREB_PCT': 'TEAM_DREB_PCT',
                # Shooting
                'FGA': 'TEAM_FGA',
                'FG3A': 'TEAM_FG3A',
                'FG_PCT': 'TEAM_FG_PCT',
                'FG3_PCT': 'TEAM_FG3_PCT',
                'FTA': 'TEAM_FTA',
                # Counting stats (context for player's role)
                'AST': 'TEAM_AST',
                'TOV': 'TEAM_TOV',
                'STL': 'TEAM_STL',
                'BLK': 'TEAM_BLK',
                'PTS': 'TEAM_PTS'
            })
            
            df = df.merge(player_team_stats, on=['PLAYER_TEAM', 'SEASON'], how='left')
            
            # =================================================================
            # NEW REBOUNDING OPPORTUNITY FEATURES
            # =================================================================
            
            # 1. Rebounding opportunities based on shot volume (NEW FEATURE #4)
            if 'OPP_FGA' in df.columns and 'TEAM_FGA' in df.columns:
                # Estimate total missed shots in a game (= rebounding opportunities)
                # Opponent's FGA * (1 - Opponent's FG%) = opponent's missed shots = our DREB opportunities
                if 'OPP_FG_PCT' in df.columns:
                    df['OPP_MISSED_SHOTS'] = df['OPP_FGA'] * (1 - df['OPP_FG_PCT'])  # DREB opportunities
                
                # Our team's FGA * (1 - Team FG%) = our missed shots = our OREB opportunities
                if 'TEAM_FG_PCT' in df.columns:
                    df['TEAM_MISSED_SHOTS'] = df['TEAM_FGA'] * (1 - df['TEAM_FG_PCT'])  # OREB opportunities
                
                # Total rebounding opportunities in a game
                df['TOTAL_REBOUND_OPP'] = df['OPP_FGA'] + df['TEAM_FGA']
            
            # 2. Opponent rebounding strength (NEW FEATURE #1)
            if 'OPP_DREB_PCT' in df.columns:
                # High opponent DREB% = they grab more defensive rebounds = less for us to get offensively
                df['OPP_DREB_STRENGTH'] = df['OPP_DREB_PCT']  # Higher = harder to get OREB
                df['OPP_WEAK_DREB'] = (df['OPP_DREB_PCT'] < 0.73).astype(int)  # Weak defensive rebounding
            
            if 'OPP_OREB_PCT' in df.columns:
                # High opponent OREB% = they're good at offensive rebounding = less DREB for us
                df['OPP_OREB_STRENGTH'] = df['OPP_OREB_PCT']
                df['OPP_WEAK_OREB'] = (df['OPP_OREB_PCT'] < 0.25).astype(int)
            
            # 3. Pace-adjusted rebounding opportunities (NEW FEATURE #3)
            if 'GAME_PACE_EST' in df.columns or ('TEAM_PACE' in df.columns and 'OPP_PACE' in df.columns):
                if 'GAME_PACE_EST' not in df.columns:
                    df['GAME_PACE_EST'] = (df['TEAM_PACE'] + df['OPP_PACE']) / 2
                
                df['IS_FAST_PACED'] = (df['GAME_PACE_EST'] > 100).astype(int)
                
                # Faster pace = more possessions = more shots = more rebounds
                if 'TOTAL_REBOUND_OPP' in df.columns:
                    df['PACE_ADJ_REB_OPP'] = df['TOTAL_REBOUND_OPP'] * (df['GAME_PACE_EST'] / 95.0)  # Normalize to league avg
            
            # 4. Existing features (defensive strength, team quality)
            if 'OPP_DEF_RATING' in df.columns:
                df['VS_STRONG_DEFENSE'] = (df['OPP_DEF_RATING'] < 105).astype(int)
                df['DEF_RATING_IMPACT'] = 115 - df['OPP_DEF_RATING']
            
            if 'TEAM_NET_RATING' in df.columns and 'OPP_NET_RATING' in df.columns:
                df['TEAM_QUALITY_DIFF'] = df['TEAM_NET_RATING'] - df['OPP_NET_RATING']
                df['IS_UNDERDOG'] = (df['TEAM_QUALITY_DIFF'] < -5).astype(int)
                df['IS_FAVORITE'] = (df['TEAM_QUALITY_DIFF'] > 5).astype(int)
            
            # =================================================================
            # STAT-SPECIFIC MATCHUP FEATURES (FOR IMPROVING ALL PREDICTIONS)
            # =================================================================
            
            # ASSIST FEATURES
            if 'OPP_AST' in df.columns and 'TEAM_AST' in df.columns:
                # Playing on a high-assist team = more assist opportunities
                df['TEAM_AST_RATE'] = df['TEAM_AST']
                df['IS_HIGH_AST_TEAM'] = (df['TEAM_AST'] > 24).astype(int)  # League avg ~24
                
                # Opponent allows assists
                df['OPP_AST_ALLOWED'] = df['OPP_AST']
            
            # TURNOVER FEATURES
            if 'OPP_STL' in df.columns:
                # Opponent steals = defensive pressure = more turnovers
                df['OPP_DEFENSIVE_PRESSURE'] = df['OPP_STL']
                df['VS_HIGH_PRESSURE_DEF'] = (df['OPP_STL'] > 8).astype(int)
            
            if 'OPP_TOV' in df.columns:
                # High opponent TOV = sloppy team = good for our steals
                df['OPP_TURNOVER_PRONE'] = df['OPP_TOV']
                df['VS_SLOPPY_TEAM'] = (df['OPP_TOV'] > 15).astype(int)
            
            # STEAL FEATURES
            # Already handled above with OPP_TOV
            
            # BLOCK FEATURES
            if 'OPP_BLK' in df.columns:
                # High opponent blocks = harder to score inside
                df['OPP_RIM_PROTECTION'] = df['OPP_BLK']
                df['VS_STRONG_RIM_PROTECT'] = (df['OPP_BLK'] > 5).astype(int)
            
            if 'OPP_FGA' in df.columns and 'OPP_FG3A' in df.columns:
                # Opponent's inside shots = block opportunities for us
                df['OPP_INSIDE_SHOTS'] = df['OPP_FGA'] - df['OPP_FG3A']
            
            # FREE THROW FEATURES
            if 'OPP_FTA' in df.columns:
                # High opponent FTA = they foul a lot = more FT opportunities
                df['OPP_FOUL_RATE'] = df['OPP_FTA']
                df['VS_FOUL_PRONE_TEAM'] = (df['OPP_FTA'] > 25).astype(int)
            
            # FIELD GOAL / 3-POINTER FEATURES
            if 'OPP_FG_PCT' in df.columns:
                # Worse opponent defense = easier to score
                df['OPP_FG_DEF_QUALITY'] = 1 - df['OPP_FG_PCT']  # Inverse (higher = better for us)
            
            if 'OPP_FG3_PCT' in df.columns:
                # Opponent 3P defense
                df['OPP_3P_DEF_QUALITY'] = 1 - df['OPP_FG3_PCT']
                df['VS_WEAK_3P_DEF'] = (df['OPP_FG3_PCT'] > 0.365).astype(int)
            
            # POINTS SCORING CONTEXT
            if 'TEAM_PTS' in df.columns and 'OPP_PTS' in df.columns:
                # High-scoring teams/games = more points for everyone
                df['GAME_SCORING_PACE'] = (df['TEAM_PTS'] + df['OPP_PTS']) / 2
                df['IS_HIGH_SCORING_GAME'] = (df['GAME_SCORING_PACE'] > 110).astype(int)
        
        # Historical performance vs opponent (player-specific)
        if 'OPPONENT_STRENGTH' not in df.columns:
            if 'OPPONENT' in df.columns:
                opponent_stats = df.groupby('OPPONENT')[self.target_variables].mean()
                
                for stat in self.target_variables:
                    if stat in opponent_stats.columns:
                        df[f'AVG_{stat}_VS_OPP'] = df['OPPONENT'].map(opponent_stats[stat])
        
        return df
    
    def _create_rest_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create rest and fatigue features"""
        
        # Days since last game (already in DAYS_SINCE_LAST_GAME)
        
        # Back-to-back indicator
        if 'DAYS_SINCE_LAST_GAME' in df.columns:
            df['DAYS_SINCE_LAST_GAME'] = pd.to_numeric(df['DAYS_SINCE_LAST_GAME'], errors='coerce')
            df['IS_BACK_TO_BACK'] = (df['DAYS_SINCE_LAST_GAME'] <= 1).astype(int)
            df['IS_WELL_RESTED'] = (df['DAYS_SINCE_LAST_GAME'] >= 3).astype(int)
        
        # Games in last 7 days (fatigue indicator)
        try:
            df['GAMES_LAST_7_DAYS'] = df.set_index('GAME_DATE').rolling('7D').size().reset_index(drop=True)
        except:
            df['GAMES_LAST_7_DAYS'] = 0
        
        # Minutes played recently (fatigue)
        if 'MIN' in df.columns:
            df['MIN'] = pd.to_numeric(df['MIN'], errors='coerce')
            df['MIN_L3_TOTAL'] = df['MIN'].rolling(window=3, min_periods=1).sum()
            df['MIN_L5_TOTAL'] = df['MIN'].rolling(window=5, min_periods=1).sum()
        
        return df
    
    def _create_career_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create career progression features"""
        
        # Career game number
        df['CAREER_GAME_NUM'] = range(1, len(df) + 1)
        
        # Career averages (expanding window) - EXCLUDE current game
        for stat in self.target_variables:
            if stat in df.columns:
                stat_shifted = df[stat].shift(1)
                df[f'{stat}_CAREER_AVG'] = stat_shifted.expanding().mean()
                df[f'{stat}_CAREER_STD'] = stat_shifted.expanding().std()
        
        # Experience level
        df['YEARS_EXPERIENCE'] = (df['CAREER_GAME_NUM'] / 82).astype(int)
        df['IS_ROOKIE'] = (df['YEARS_EXPERIENCE'] == 0).astype(int)
        df['IS_VETERAN'] = (df['YEARS_EXPERIENCE'] >= 5).astype(int)
        
        return df
    
    def _create_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create interaction features"""
        
        # Home advantage for different stats
        if 'IS_HOME' in df.columns:
            df['HOME_PTS_BOOST'] = df['IS_HOME'] * df['PTS_L5_AVG']
            df['HOME_AST_BOOST'] = df['IS_HOME'] * df['AST_L5_AVG']
        
        # Rest impact on performance
        if 'REST_ENCODED' in df.columns:
            df['REST_PTS_IMPACT'] = df['REST_ENCODED'] * df['PTS_L10_AVG']
            df['REST_MIN_IMPACT'] = df['REST_ENCODED'] * df.get('MIN_L5_TOTAL', 0)
        
        # College to NBA translation (for rookies/young players)
        if 'COLLEGE_PPG' in df.columns and 'IS_ROOKIE' in df.columns:
            df['COLLEGE_NBA_SCORING_RATIO'] = df['COLLEGE_PPG'] / df['PTS_L10_AVG'].replace(0, np.nan)
        
        # =================================================================
        # REBOUNDING-SPECIFIC INTERACTION FEATURES (NEW FOR REBOUNDING IMPROVEMENT)
        # =================================================================
        
        # 1. Height x Rebounding Opportunities (taller players benefit more from opportunities)
        if 'height_inches' in df.columns and 'TOTAL_REBOUND_OPP' in df.columns:
            # Normalize height (0-1 scale, 66" to 90")
            df['HEIGHT_NORM'] = (df['height_inches'] - 66) / 24
            df['HEIGHT_REB_OPP'] = df['HEIGHT_NORM'] * df['TOTAL_REBOUND_OPP']
            
            # Height advantage for specific rebounds
            if 'OPP_MISSED_SHOTS' in df.columns:
                df['HEIGHT_DREB_OPP'] = df['HEIGHT_NORM'] * df['OPP_MISSED_SHOTS']
            if 'TEAM_MISSED_SHOTS' in df.columns:
                df['HEIGHT_OREB_OPP'] = df['HEIGHT_NORM'] * df['TEAM_MISSED_SHOTS']
        
        # 2. Minutes x Rebounding Opportunities (NEW FEATURE #5: More minutes = more opportunities)
        if 'MIN_L5_TOTAL' in df.columns:
            # Average minutes per game (last 5)
            df['AVG_MIN_L5'] = df['MIN_L5_TOTAL'] / 5
            
            if 'TOTAL_REBOUND_OPP' in df.columns:
                df['MIN_REB_OPP'] = df['AVG_MIN_L5'] * df['TOTAL_REBOUND_OPP'] / 48  # Normalize to 48 min game
            
            # More minutes when well-rested = better rebounding
            if 'IS_WELL_RESTED' in df.columns:
                df['RESTED_MIN_BOOST'] = df['IS_WELL_RESTED'] * df['AVG_MIN_L5']
        
        # 3. Position x Rebounding Context (forwards/centers should rebound more)
        if 'position_encoded' in df.columns:
            # Create position-based rebounding multipliers
            # Assuming encoding: 0=Guard, 1=Forward, 2=Center (adjust if different)
            df['POSITION_REB_WEIGHT'] = df['position_encoded'] / 2  # 0 for guards, 0.5 for forwards, 1 for centers
            
            if 'TOTAL_REBOUND_OPP' in df.columns:
                df['POSITION_REB_OPP'] = df['POSITION_REB_WEIGHT'] * df['TOTAL_REBOUND_OPP']
        
        # 4. Opponent Rebounding Weakness x Player Rebounding Skill
        if 'OPP_WEAK_DREB' in df.columns and 'DREB_L10_AVG' in df.columns:
            # If opponent is weak at DREB and player is good at OREB, boost
            if 'OREB_L10_AVG' in df.columns:
                df['OREB_MATCHUP_ADVANTAGE'] = df['OPP_WEAK_DREB'] * df['OREB_L10_AVG']
        
        if 'OPP_WEAK_OREB' in df.columns and 'DREB_L10_AVG' in df.columns:
            # If opponent is weak at OREB and player is good at DREB, boost  
            df['DREB_MATCHUP_ADVANTAGE'] = df['OPP_WEAK_OREB'] * df['DREB_L10_AVG']
        
        # 5. Pace x Player's Rebounding Rate
        if 'GAME_PACE_EST' in df.columns:
            if 'OREB_L10_AVG' in df.columns:
                df['PACE_OREB_BOOST'] = (df['GAME_PACE_EST'] / 95) * df['OREB_L10_AVG']
            if 'DREB_L10_AVG' in df.columns:
                df['PACE_DREB_BOOST'] = (df['GAME_PACE_EST'] / 95) * df['DREB_L10_AVG']
        
        return df
    
    def _create_archetype_matchup_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create features based on archetypes and matchups"""
        
        # Archetype strength interactions with context
        if 'SCORER_STRENGTH' in df.columns:
            # Home scoring boost for scorers
            if 'IS_HOME' in df.columns:
                df['SCORER_HOME_BOOST'] = df['SCORER_STRENGTH'] * df['IS_HOME']
            
            # Rest impact on scorers
            if 'REST_ENCODED' in df.columns:
                df['SCORER_REST_BOOST'] = df['SCORER_STRENGTH'] * df['REST_ENCODED']
        
        # Playmaker interactions
        if 'PLAYMAKER_STRENGTH' in df.columns:
            # Playmakers need energy
            if 'IS_BACK_TO_BACK' in df.columns:
                df['PLAYMAKER_FATIGUE_PENALTY'] = df['PLAYMAKER_STRENGTH'] * df['IS_BACK_TO_BACK']
        
        # Defender strength vs usage
        if 'DEFENDER_STRENGTH' in df.columns and 'MIN' in df.columns:
            df['DEFENDER_MINUTES_INTERACTION'] = df['DEFENDER_STRENGTH'] * pd.to_numeric(df['MIN'], errors='coerce')
        
        # Shooter performance in different situations
        if 'SHOOTER_STRENGTH' in df.columns:
            if 'IS_HOME' in df.columns:
                df['SHOOTER_HOME_ADVANTAGE'] = df['SHOOTER_STRENGTH'] * df['IS_HOME']
        
        # Archetype vs opponent strength
        if all(col in df.columns for col in ['SCORER_STRENGTH', 'OPPONENT_STRENGTH']):
            df['SCORER_VS_OPPONENT'] = df['SCORER_STRENGTH'] * (1 - df['OPPONENT_STRENGTH'])
        
        # Experience interactions with archetypes
        if 'season_exp' in df.columns:
            exp = pd.to_numeric(df['season_exp'], errors='coerce')
            
            if 'SCORER_STRENGTH' in df.columns:
                df['SCORER_EXPERIENCE_BOOST'] = df['SCORER_STRENGTH'] * (exp / 10.0)  # Normalize experience
            
            if 'PLAYMAKER_STRENGTH' in df.columns:
                df['PLAYMAKER_EXPERIENCE_BOOST'] = df['PLAYMAKER_STRENGTH'] * (exp / 10.0)
        
        # Archetype consistency features
        if 'ARCHETYPE_PRIMARY_ROLE' in df.columns:
            # How many games in current archetype (stability)
            df['ARCHETYPE_STABILITY'] = (df['ARCHETYPE_PRIMARY_ROLE'] == df['ARCHETYPE_PRIMARY_ROLE'].shift(1)).astype(int).rolling(10).sum()
        
        # Ensure ALL archetype columns are encoded (double-check)
        all_archetype_cols = [col for col in df.columns if col.startswith('ARCHETYPE_') and '_ENCODED' not in col]
        
        for col in all_archetype_cols:
            if col in df.columns and f'{col}_ENCODED' not in df.columns:
                if df[col].dtype == 'object':
                    df[f'{col}_ENCODED'] = pd.Categorical(df[col]).codes
        
        return df
    
    def process_all_players(self, limit: int = None):
        """
        Process all players and save features to individual files
        
        Args:
            limit: Limit number of players to process (for testing)
        """
        print("🏀 Starting Feature Engineering for All Players")
        print("=" * 60)
        
        data_path = Path(self.data_dir)
        player_dirs = [d for d in data_path.iterdir() if d.is_dir() and d.name != 'college']
        
        if limit:
            player_dirs = player_dirs[:limit]
        
        print(f"📊 Processing {len(player_dirs)} players...")
        print(f"💾 Saving features to individual player directories")
        print()
        
        success_count = 0
        failed_count = 0
        skipped_count = 0
        
        for idx, player_dir in enumerate(player_dirs, 1):
            player_name = player_dir.name.replace('_', ' ')
            
            try:
                # Check if already processed
                features_file = player_dir / f"{player_dir.name}_features.csv"
                if features_file.exists():
                    skipped_count += 1
                    if idx % 100 == 0:
                        print(f"⏭️  Processed {idx}/{len(player_dirs)} players (skipped: {skipped_count})...")
                    continue
                
                df = self.process_player_data(player_name, save_to_file=True)
                
                if df is not None and len(df) > 0:
                    success_count += 1
                    
                    if idx % 100 == 0:
                        print(f"✅ Processed {idx}/{len(player_dirs)} players...")
                else:
                    failed_count += 1
                    
            except Exception as e:
                print(f"❌ Error processing {player_name}: {e}")
                failed_count += 1
        
        print()
        print("=" * 60)
        print("🏁 Feature Engineering Complete!")
        print(f"✅ Successfully processed: {success_count} players")
        print(f"⏭️  Skipped (already exist): {skipped_count} players")
        print(f"❌ Failed: {failed_count} players")
        print(f"📁 Features saved to: data2/[Player_Name]/[Player_Name]_features.csv")
        
        return success_count
    
    def get_feature_summary(self, df: pd.DataFrame):
        """Print summary of features created"""
        print("\n📊 Feature Categories:")
        print("=" * 60)
        
        categories = {
            'Target Variables': self.target_variables,
            'Statistical Features': [col for col in df.columns if any(x in col for x in ['PCT', 'RATIO', 'PER_MIN', 'STOCK', 'FANTASY'])],
            'Rolling Features': [col for col in df.columns if any(x in col for x in ['_L3_', '_L5_', '_L10_', '_L20_', 'TREND', 'RECENT_VS'])],
            'Contextual Features': [col for col in df.columns if any(x in col for x in ['IS_HOME', 'OPPONENT', 'REST', 'WIN'])],
            'Temporal Features': [col for col in df.columns if any(x in col for x in ['SEASON', 'DAY_', 'MONTH_', 'EARLY', 'MID', 'LATE'])],
            'Rest/Fatigue Features': [col for col in df.columns if any(x in col for x in ['BACK_TO_BACK', 'WELL_RESTED', 'GAMES_LAST', 'MIN_L'])],
            'Career Features': [col for col in df.columns if any(x in col for x in ['CAREER', 'EXPERIENCE', 'ROOKIE', 'VETERAN'])],
            'College Features': [col for col in df.columns if 'COLLEGE' in col],
        }
        
        for category, features in categories.items():
            available_features = [f for f in features if f in df.columns]
            print(f"\n{category}: {len(available_features)} features")
            if len(available_features) <= 10:
                for feat in available_features:
                    print(f"  - {feat}")
            else:
                print(f"  - {available_features[0]}")
                print(f"  - {available_features[1]}")
                print(f"  ... and {len(available_features) - 2} more")

def main():
    """Main execution"""
    print("🏀 NBA Feature Engineering System")
    print("=" * 50)
    print("1. Process all players")
    print("2. Process limited players (testing)")
    print("3. Exit")
    print()
    
    choice = input("Select option (1-3): ").strip()
    
    engineer = NBAFeatureEngineering()
    
    if choice == "1":
        df = engineer.process_all_players()
        if df is not None:
            engineer.get_feature_summary(df)
    
    elif choice == "2":
        limit = int(input("How many players to process? ") or "10")
        df = engineer.process_all_players(limit=limit)
        if df is not None:
            engineer.get_feature_summary(df)
    
    elif choice == "3":
        print("👋 Goodbye!")
    
    else:
        print("❌ Invalid choice")

if __name__ == "__main__":
    main()

