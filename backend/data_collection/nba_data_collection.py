#!/usr/bin/env python3
"""
NBA Data Collection System
=========================
Collect comprehensive game-by-game data for NBA players optimized for betting predictions
"""

import pandas as pd
import numpy as np
import time
import warnings
import json
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Set
from tqdm import tqdm
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging

# NBA API imports
try:
    from nba_api.stats.endpoints import playergamelog, commonplayerinfo
    from nba_api.stats.library.parameters import Season, SeasonType
    from nba_api.stats.static import players
    NBA_API_AVAILABLE = True
except ImportError:
    NBA_API_AVAILABLE = False
    print("⚠️ NBA API not available. Install with: pip install nba_api")

warnings.filterwarnings('ignore')

class NBADataCollector:
    """Collect comprehensive game-by-game data optimized for betting predictions"""
    
    def __init__(self, data_dir: str = None, max_workers: int = 2):
        """
        Initialize the data collector
        
        Args:
            data_dir: Directory to store data (defaults to player_data)
            max_workers: Maximum number of concurrent workers
        """
        if not NBA_API_AVAILABLE:
            raise ImportError("NBA API is required. Install with: pip install nba_api")
        
        # Set up directories
        if data_dir is None:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            self.data_dir = os.path.join(current_dir, 'player_data')
        else:
            self.data_dir = data_dir
        
        # Ensure player_data directory exists
        os.makedirs(self.data_dir, exist_ok=True)
        
        # Paths
        self.player_list_path = os.path.join(current_dir, 'nba', 'data', 'NBA-COMPLETE-playerlist.csv')
        self.collection_log_path = os.path.join(self.data_dir, 'nba_collection_log.json')
        self.checkpoint_file = os.path.join(self.data_dir, 'nba_collection_checkpoint.json')
        
        # Collection state
        self.processed_players = set()
        self.failed_players = set()
        self.no_data_players = set()
        self.total_games_collected = 0
        self.start_time = None
        
        # Threading and rate limiting
        self.max_workers = max_workers
        self.request_delay = 10.0  # Very respectful rate limiting
        self.max_retries = 3
        
        # Thread safety
        self.lock = threading.Lock()
        
        # Setup logging
        self.setup_logging()
        
        # Load checkpoint
        self.load_checkpoint()
    
    def setup_logging(self):
        """Setup logging for collection process"""
        log_file = os.path.join(self.data_dir, 'nba_collection.log')
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        
        self.logger = logging.getLogger(__name__)
    
    def load_checkpoint(self):
        """Load collection checkpoint if exists"""
        if os.path.exists(self.checkpoint_file):
            try:
                with open(self.checkpoint_file, 'r') as f:
                    checkpoint = json.load(f)
                    self.processed_players = set(checkpoint.get('processed_players', []))
                    self.failed_players = set(checkpoint.get('failed_players', []))
                    self.no_data_players = set(checkpoint.get('no_data_players', []))
                    self.total_games_collected = checkpoint.get('total_games_collected', 0)
                    self.logger.info(f"📋 Loaded checkpoint: {len(self.processed_players)} processed, {len(self.failed_players)} failed")
            except Exception as e:
                self.logger.warning(f"⚠️ Could not load checkpoint: {e}")
    
    def save_checkpoint(self):
        """Save collection checkpoint"""
        checkpoint = {
            'processed_players': list(self.processed_players),
            'failed_players': list(self.failed_players),
            'no_data_players': list(self.no_data_players),
            'total_games_collected': self.total_games_collected,
            'timestamp': datetime.now().isoformat()
        }
        
        try:
            with open(self.checkpoint_file, 'w') as f:
                json.dump(checkpoint, f, indent=2)
        except Exception as e:
            self.logger.warning(f"⚠️ Could not save checkpoint: {e}")
    
    def load_player_list(self) -> pd.DataFrame:
        """Load the complete player list"""
        if not os.path.exists(self.player_list_path):
            raise FileNotFoundError(f"Player list not found: {self.player_list_path}")
        
        df = pd.read_csv(self.player_list_path)
        self.logger.info(f"📋 Loaded {len(df)} players from {self.player_list_path}")
        return df
    
    def get_player_nba_id(self, player_name: str) -> Optional[int]:
        """Get NBA API player ID for a player name"""
        try:
            # Try to find player by name
            player_matches = players.find_players_by_full_name(player_name)
            if player_matches:
                return player_matches[0]['id']
            
            # Try with first and last name separately
            name_parts = player_name.split()
            if len(name_parts) >= 2:
                first_name = name_parts[0]
                last_name = ' '.join(name_parts[1:])
                
                # Try exact match
                for player in players.get_players():
                    if (player['first_name'].lower() == first_name.lower() and 
                        player['last_name'].lower() == last_name.lower()):
                        return player['id']
                
                # Try partial match
                for player in players.get_players():
                    if (first_name.lower() in player['first_name'].lower() and 
                        last_name.lower() in player['last_name'].lower()):
                        return player['id']
            
            return None
            
        except Exception as e:
            self.logger.warning(f"⚠️ Error finding NBA ID for {player_name}: {e}")
            return None
    
    def get_player_info(self, player_id: int) -> Dict:
        """Get essential player information"""
        try:
            player_info = commonplayerinfo.CommonPlayerInfo(player_id=player_id)
            info_df = player_info.get_data_frames()[0]
            
            if not info_df.empty:
                return {
                    'height': info_df['HEIGHT'].iloc[0],
                    'weight': info_df['WEIGHT'].iloc[0],
                    'position': info_df['POSITION'].iloc[0],
                    'draft_year': info_df['DRAFT_YEAR'].iloc[0],
                    'season_exp': info_df['SEASON_EXP'].iloc[0]
                }
        except Exception as e:
            self.logger.warning(f"⚠️ Could not get player info for {player_id}: {e}")
        
        return {}
    
    def get_player_career_span(self, player_id: int) -> List[str]:
        """Get career span for a player"""
        try:
            # Get player info to determine career span
            player_info = commonplayerinfo.CommonPlayerInfo(player_id=player_id)
            info_df = player_info.get_data_frames()[0]
            
            if not info_df.empty:
                from_year = info_df['FROM_YEAR'].iloc[0]
                to_year = info_df['TO_YEAR'].iloc[0]
                
                # Generate season list
                seasons = []
                for year in range(from_year, to_year + 1):
                    seasons.append(f"{year}-{str(year + 1)[-2:]}")
                
                return seasons
        except Exception as e:
            self.logger.warning(f"⚠️ Could not get career span for player {player_id}: {e}")
        
        # Fallback: use 1990-2025 range
        return [f"{year}-{str(year + 1)[-2:]}" for year in range(1990, 2026)]
    
    def collect_player_data(self, player_name: str, player_id: int) -> Dict:
        """Collect comprehensive data for a single player"""
        result = {
            'player_name': player_name,
            'player_id': player_id,
            'success': False,
            'games_collected': 0,
            'error': None
        }
        
        try:
            # Create player directory
            player_dir = os.path.join(self.data_dir, player_name.replace(' ', '_').replace('.', '').replace("'", ""))
            os.makedirs(player_dir, exist_ok=True)
            
            # Get player info
            player_info = self.get_player_info(player_id)
            
            # Get career span
            seasons = self.get_player_career_span(player_id)
            
            all_games = []
            
            for season in seasons:
                try:
                    # Get game log for this season
                    game_log = playergamelog.PlayerGameLog(
                        player_id=player_id,
                        season=season,
                        season_type_all_star=SeasonType.regular
                    )
                    
                    df = game_log.get_data_frames()[0]
                    
                    if not df.empty:
                        # Add metadata
                        df['PLAYER_NAME'] = player_name
                        df['SEASON'] = season
                        df['GAME_DATE'] = pd.to_datetime(df['GAME_DATE'], format='mixed')
                        
                        # Add essential player info
                        for key, value in player_info.items():
                            df[key] = value
                        
                        # Parse matchup to get player team and opponent
                        df = self._parse_matchup(df)
                        
                        # Calculate advanced metrics
                        df = self._calculate_advanced_metrics(df)
                        
                        # Calculate contextual features
                        df = self._calculate_contextual_features(df)
                        
                        # Calculate betting-relevant features
                        df = self._calculate_betting_features(df)
                        
                        all_games.append(df)
                        
                    # Rate limiting - be more respectful to NBA API
                    time.sleep(self.request_delay)
                    
                except Exception as e:
                    self.logger.warning(f"⚠️ Error collecting {season} for {player_name}: {e}")
                    # Additional delay if we hit timeouts
                    if 'timeout' in str(e).lower():
                        time.sleep(30)  # Extra delay for timeout recovery
                    continue
            
            if all_games:
                # Combine all seasons
                combined_df = pd.concat(all_games, ignore_index=True)
                combined_df = combined_df.sort_values('GAME_DATE')
                
                # Save to CSV with firstname_lastname_data.csv format
                name_parts = player_name.split()
                if len(name_parts) >= 2:
                    csv_filename = f"{name_parts[0]}_{name_parts[-1]}_data.csv"
                else:
                    csv_filename = f"{player_name.replace(' ', '_')}_data.csv"
                csv_path = os.path.join(player_dir, csv_filename)
                combined_df.to_csv(csv_path, index=False)
                
                result['success'] = True
                result['games_collected'] = len(combined_df)
                result['file_path'] = csv_path
                
                self.logger.info(f"✅ {player_name}: {len(combined_df)} games saved to {csv_path}")
            else:
                result['error'] = "No game data found"
                self.logger.warning(f"⚠️ {player_name}: No game data found")
                
        except Exception as e:
            result['error'] = str(e)
            self.logger.error(f"❌ {player_name}: {e}")
        
        return result
    
    def _parse_matchup(self, df: pd.DataFrame) -> pd.DataFrame:
        """Parse matchup to extract player team and opponent team"""
        
        # Extract player team and opponent from MATCHUP
        # Format: "TEAM @ OPPONENT" or "TEAM vs. OPPONENT"
        df['PLAYER_TEAM'] = df['MATCHUP'].str.extract(r'^(\w+)')
        df['OPPONENT_TEAM'] = df['MATCHUP'].str.extract(r'@\s*(\w+)|vs\.\s*(\w+)')[0].fillna(
            df['MATCHUP'].str.extract(r'@\s*(\w+)|vs\.\s*(\w+)')[1]
        )
        
        # Create binary home/away indicator
        df['IS_HOME'] = df['MATCHUP'].str.contains('vs.').astype(int)
        
        return df
    
    def _calculate_advanced_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate advanced metrics for each game"""
        
        # True Shooting Percentage
        df['TS_PCT'] = df['PTS'] / (2 * (df['FGA'] + 0.44 * df['FTA'])).replace(0, np.nan)
        
        # Game Score (John Hollinger's formula)
        df['GAME_SCORE'] = (
            df['PTS'] + 0.4 * df['FGM'] - 0.7 * df['FGA'] - 0.4 * (df['FTA'] - df['FTM']) +
            0.7 * df['OREB'] + 0.3 * df['DREB'] + df['STL'] + 0.7 * df['AST'] + 0.7 * df['BLK'] -
            0.4 * df['PF'] - df['TOV']
        )
        
        # Estimated PIE (Player Impact Estimate)
        df['PIE_EST'] = (
            (df['PTS'] + df['AST'] + df['REB'] + df['STL'] + df['BLK']) /
            (df['PTS'] + df['AST'] + df['REB'] + df['STL'] + df['BLK'] + df['TOV'] + df['PF']).replace(0, np.nan)
        )
        
        # Estimated PER (Player Efficiency Rating approximation)
        df['PER_EST'] = (
            df['PTS'] * 1.0 + df['REB'] * 1.2 + df['AST'] * 1.5 + df['STL'] * 2.0 + 
            df['BLK'] * 2.0 - df['TOV'] * 1.0 - df['PF'] * 0.5
        ) / df['MIN'].replace(0, 1) * 48
        
        # Usage Rate estimation
        df['USG_RATE_EST'] = (
            (df['FGA'] + 0.44 * df['FTA'] + df['TOV']) / df['MIN'].replace(0, 1) * 48
        )
        
        # Net Rating estimation
        df['NET_RATING_EST'] = df['GAME_SCORE'] / df['MIN'].replace(0, 1) * 48
        
        return df
    
    def _calculate_contextual_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate contextual features for each game"""
        
        # Game day of week
        df['GAME_DAY_OF_WEEK'] = df['GAME_DATE'].dt.day_name()
        
        # Game month
        df['GAME_MONTH'] = df['GAME_DATE'].dt.month
        
        # Game time of year (1-4 quarters)
        df['GAME_QUARTER'] = ((df['GAME_DATE'].dt.month - 1) // 3) + 1
        
        # Days since last game (calculate correctly by sorting by date first)
        df = df.sort_values('GAME_DATE')
        df['DAYS_SINCE_LAST_GAME'] = df['GAME_DATE'].diff().dt.days.fillna(0)
        
        return df
    
    def _calculate_betting_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate features specifically useful for betting predictions"""
        
        # Rolling averages (last 3, 5, 10 games)
        for window in [3, 5, 10]:
            for col in ['PTS', 'REB', 'AST', 'STL', 'BLK', 'TOV', 'PF', 'FGA', 'FGM', 'FG3A', 'FG3M', 'FTA', 'FTM']:
                if col in df.columns:
                    df[f'{col}_LAST_{window}_AVG'] = df[col].rolling(window=window, min_periods=1).mean()
        
        # Performance trends (last 5 games vs season average)
        for col in ['PTS', 'REB', 'AST']:
            if col in df.columns:
                season_avg = df[col].mean()
                df[f'{col}_TREND'] = df[f'{col}_LAST_5_AVG'] - season_avg
        
        # Consistency metrics (standard deviation of last 10 games)
        for col in ['PTS', 'REB', 'AST']:
            if col in df.columns:
                df[f'{col}_CONSISTENCY'] = df[col].rolling(window=10, min_periods=1).std()
        
        # Rest factor (days since last game)
        df['REST_FACTOR'] = df['DAYS_SINCE_LAST_GAME'].apply(lambda x: 'No Rest' if x == 0 else '1 Day' if x == 1 else '2+ Days' if x >= 2 else 'Unknown')
        
        # Season progression (game number in season)
        df['GAME_NUMBER_IN_SEASON'] = df.groupby('SEASON').cumcount() + 1
        
        # Performance vs opponent (simplified - would need historical data for full implementation)
        df['OPPONENT_STRENGTH'] = 0.5  # Placeholder
        
        return df
    
    def collect_all_players(self, start_from: int = 0, max_players: int = None):
        """Collect data for all players in the list"""
        
        # Load player list
        player_df = self.load_player_list()
        
        if max_players:
            player_df = player_df.iloc[:max_players]
        
        if start_from > 0:
            player_df = player_df.iloc[start_from:]
        
        self.start_time = datetime.now()
        self.logger.info(f"🚀 Starting NBA data collection for {len(player_df)} players")
        self.logger.info(f"📊 Processed: {len(self.processed_players)}, Failed: {len(self.failed_players)}")
        
        # Process players
        for idx, row in tqdm(player_df.iterrows(), total=len(player_df), desc="Collecting NBA player data"):
            player_name = row['name']
            
            # Skip if already processed
            if player_name in self.processed_players:
                continue
            
            try:
                # Get NBA API player ID
                player_id = self.get_player_nba_id(player_name)
                
                if player_id is None:
                    self.logger.warning(f"⚠️ Could not find NBA ID for {player_name}")
                    self.failed_players.add(player_name)
                    continue
                
                # Collect data
                result = self.collect_player_data(player_name, player_id)
                
                with self.lock:
                    if result['success']:
                        self.processed_players.add(player_name)
                        self.total_games_collected += result['games_collected']
                    else:
                        if result['error'] == "No game data found":
                            self.no_data_players.add(player_name)
                        else:
                            self.failed_players.add(player_name)
                
                # Save checkpoint every 10 players
                if (idx + 1) % 10 == 0:
                    self.save_checkpoint()
                    self.logger.info(f"💾 Checkpoint saved: {len(self.processed_players)} processed, {len(self.failed_players)} failed")
                
            except Exception as e:
                self.logger.error(f"❌ Unexpected error processing {player_name}: {e}")
                with self.lock:
                    self.failed_players.add(player_name)
        
        # Final save
        self.save_checkpoint()
        
        # Final report
        duration = datetime.now() - self.start_time
        self.logger.info(f"🏁 Collection complete!")
        self.logger.info(f"⏱️ Duration: {duration}")
        self.logger.info(f"✅ Processed: {len(self.processed_players)}")
        self.logger.info(f"❌ Failed: {len(self.failed_players)}")
        self.logger.info(f"📊 No data: {len(self.no_data_players)}")
        self.logger.info(f"🎮 Total games: {self.total_games_collected}")
        
        # Save final report
        self.save_final_report()
    
    def save_final_report(self):
        """Save final collection report"""
        report = {
            'collection_summary': {
                'total_players_attempted': len(self.processed_players) + len(self.failed_players) + len(self.no_data_players),
                'successful_players': len(self.processed_players),
                'failed_players': len(self.failed_players),
                'no_data_players': len(self.no_data_players),
                'total_games_collected': self.total_games_collected,
                'collection_duration': str(datetime.now() - self.start_time) if self.start_time else None
            },
            'processed_players': list(self.processed_players),
            'failed_players': list(self.failed_players),
            'no_data_players': list(self.no_data_players),
            'timestamp': datetime.now().isoformat()
        }
        
        report_path = os.path.join(self.data_dir, 'nba_collection_report.json')
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        self.logger.info(f"📋 Final report saved to {report_path}")

def main():
    """Main execution function"""
    print("🏀 NBA Data Collection System")
    print("=" * 40)
    print("1. Test with sample players")
    print("2. Collect data for ALL players")
    print("3. Resume from checkpoint")
    print("4. Exit")
    print()
    
    choice = input("Select option (1-4): ").strip()
    
    if choice == "1":
        print("\n🧪 Testing with sample players...")
        collector = NBADataCollector()
        test_players = ["LeBron James", "Stephen Curry", "Kevin Durant"]
        
        for player_name in test_players:
            print(f"🔍 Testing {player_name}...")
            player_id = collector.get_player_nba_id(player_name)
            if player_id:
                result = collector.collect_player_data(player_name, player_id)
                if result['success']:
                    print(f"✅ {player_name}: {result['games_collected']} games collected")
                else:
                    print(f"❌ {player_name}: {result['error']}")
    
    elif choice == "2":
        print("\n🚀 Starting full collection...")
        collector = NBADataCollector()
        collector.collect_all_players()
    
    elif choice == "3":
        print("🔄 Resuming from checkpoint...")
        collector = NBADataCollector()
        collector.collect_all_players()
    
    elif choice == "4":
        print("👋 Goodbye!")
    
    else:
        print("❌ Invalid choice")

if __name__ == "__main__":
    main()
