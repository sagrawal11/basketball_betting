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
from pathlib import Path
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

class NBAFeatureEngineering:
    """Create features for NBA player performance prediction"""
    
    def __init__(self, data_dir: str = "data2", output_dir: str = "processed_data"):
        self.data_dir = data_dir
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Label encoders for categorical variables
        self.encoders = {}
        
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
            'SEASON', 'WL', 'GAME_RESULT', 'REST_FACTOR',
            
            # Non-numeric or unreliable
            'VIDEO_AVAILABLE', 'GAME_QUARTER',
            
            # Archetype categorical columns (use encoded versions)
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
        
        # Check if features already exist (skip only if we have archetype-enhanced features)
        # We'll regenerate all features since we're adding archetype features
        # if features_file.exists() and save_to_file:
        #     print(f"⏭️  {player_name}: Features already exist, skipping...")
        #     return None
        
        df = pd.read_csv(nba_file)
        
        # Clean data - replace empty strings with NaN
        df = df.replace('', np.nan)
        df = df.replace(' ', np.nan)
        
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
        
        # Save to player's directory if requested
        if save_to_file:
            df.to_csv(features_file, index=False)
            print(f"💾 Saved features: {player_name}")
        
        return df
    
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
            
            # Encode ALL categorical archetypes (don't skip any)
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
        
        # Assist-to-turnover ratio
        df['AST_TO_RATIO'] = df['AST'] / df['TOV'].replace(0, np.nan)
        
        # Points per field goal attempt
        df['PTS_PER_FGA'] = df['PTS'] / df['FGA'].replace(0, np.nan)
        
        # Rebound rate (per minute)
        df['REB_PER_MIN'] = df['REB'] / df['MIN'].replace(0, np.nan)
        
        # Stock (Steals + Blocks)
        df['STOCK'] = df['STL'] + df['BLK']
        
        # Fantasy points (common fantasy scoring)
        df['FANTASY_PTS'] = (
            df['PTS'] + 
            1.2 * df['REB'] + 
            1.5 * df['AST'] + 
            3 * df['STL'] + 
            3 * df['BLK'] - 
            df['TOV']
        )
        
        return df
    
    def _create_rolling_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create rolling averages for momentum/form"""
        
        windows = [3, 5, 10, 20]  # Last 3, 5, 10, 20 games
        
        for window in windows:
            for stat in self.target_variables:
                if stat in df.columns:
                    # Rolling mean
                    df[f'{stat}_L{window}_AVG'] = df[stat].rolling(window=window, min_periods=1).mean()
                    
                    # Rolling std (consistency)
                    df[f'{stat}_L{window}_STD'] = df[stat].rolling(window=window, min_periods=1).std()
                    
                    # Rolling max
                    df[f'{stat}_L{window}_MAX'] = df[stat].rolling(window=window, min_periods=1).max()
                    
                    # Trend (difference from rolling average)
                    df[f'{stat}_TREND_L{window}'] = df[stat] - df[f'{stat}_L{window}_AVG']
        
        # Recent vs long-term form comparison
        for stat in self.target_variables:
            if stat in df.columns:
                df[f'{stat}_RECENT_VS_SEASON'] = df[f'{stat}_L5_AVG'] / df[f'{stat}_L20_AVG'].replace(0, np.nan)
        
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
        
        # 3. Rest factor encoding
        if 'REST_FACTOR' in df.columns:
            rest_mapping = {
                'No Rest': 0,
                '1 Day': 1,
                '2+ Days': 2,
                'Unknown': -1
            }
            df['REST_ENCODED'] = df['REST_FACTOR'].map(rest_mapping).fillna(-1).astype(int)
        
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
        """Create opponent-specific features"""
        
        if 'OPPONENT_STRENGTH' not in df.columns:
            # Create opponent strength based on historical performance against them
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
        
        # Career averages (expanding window)
        for stat in self.target_variables:
            if stat in df.columns:
                df[f'{stat}_CAREER_AVG'] = df[stat].expanding().mean()
                df[f'{stat}_CAREER_STD'] = df[stat].expanding().std()
        
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

