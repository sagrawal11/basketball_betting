#!/usr/bin/env python3
"""
College Basketball Data Collector
=================================
Collect college basketball data for NBA rookies and draft prospects
"""

import pandas as pd
import numpy as np
import requests
import time
import json
import os
from datetime import datetime
from typing import Dict, List, Optional
import warnings
from bs4 import BeautifulSoup
import re
warnings.filterwarnings('ignore')

class CollegeBasketballCollector:
    """Collect college basketball data for NBA prospects"""
    
    def __init__(self, data_dir: str = None):
        """
        Initialize the college basketball collector
        
        Args:
            data_dir: Directory to store data (defaults to data2/college)
        """
        if data_dir is None:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            self.data_dir = os.path.join(current_dir, 'data2', 'college')
        else:
            self.data_dir = data_dir
        
        # Ensure directory exists
        os.makedirs(self.data_dir, exist_ok=True)
        
        # Collection state
        self.collected_players = set()
        self.failed_players = set()
        
        # Checkpointing
        self.checkpoint_file = os.path.join(self.data_dir, 'college_collection_checkpoint.json')
        self.college_data = []
        
        # Rate limiting
        self.request_delay = 2.0
        
    def collect_nba_rookies_college_data(self, start_year: int = 2005, end_year: int = 2024):
        """
        Collect college data for NBA rookies from specified years
        
        Args:
            start_year: Starting draft year
            end_year: Ending draft year
        """
        print(f"🏀 Collecting College Data for NBA Rookies ({start_year}-{end_year})")
        print("=" * 60)
        
        # Load NBA player data to get rookies
        nba_data_dir = os.path.join(os.path.dirname(self.data_dir), '..', 'nba', 'data')
        player_list_path = os.path.join(nba_data_dir, 'NBA-COMPLETE-playerlist.csv')
        
        if not os.path.exists(player_list_path):
            print(f"❌ NBA player list not found: {player_list_path}")
            return
        
        # Load NBA players
        nba_players = pd.read_csv(player_list_path)
        
        # Filter for recent players (approximate rookie years)
        recent_players = nba_players[
            (nba_players['from_year'] >= start_year) & 
            (nba_players['from_year'] <= end_year)
        ]
        
        print(f"📊 Found {len(recent_players)} recent NBA players")
        
        # Load checkpoint if exists
        self._load_checkpoint()
        
        # Collect college data for each player
        college_data = self.college_data.copy()
        
        for idx, player in recent_players.iterrows():
            player_name = player['name']
            draft_year = int(player['from_year']) if pd.notna(player['from_year']) else None
            
            print(f"🔍 Collecting college data for {player_name} (Draft: {draft_year})")
            
            try:
                # Collect college data
                player_college_data = self.collect_player_college_data(player_name, draft_year)
                
                if player_college_data and player_college_data.get('college') != 'Unknown':
                    college_data.append(player_college_data)
                    self.collected_players.add(player_name)
                    
                    # Save individual player college data
                    self._save_player_college_data(player_name, player_college_data)
                    
                    print(f"✅ {player_name}: College data collected")
                else:
                    self.failed_players.add(player_name)
                    print(f"❌ {player_name}: No college data found")
                
                # Save checkpoint every 10 players
                if len(college_data) % 10 == 0:
                    self._save_checkpoint(college_data)
                    print(f"💾 Checkpoint saved: {len(college_data)} players collected")
                
                # Rate limiting
                time.sleep(self.request_delay)
                
            except Exception as e:
                print(f"❌ Error collecting {player_name}: {e}")
                self.failed_players.add(player_name)
        
        # Save collected data
        if college_data:
            df = pd.DataFrame(college_data)
            output_path = os.path.join(self.data_dir, f'nba_rookies_college_data_{start_year}_{end_year}.csv')
            df.to_csv(output_path, index=False)
            print(f"✅ Saved {len(college_data)} players' college data to {output_path}")
        
        # Final save
        self._save_checkpoint(college_data)
        
        print(f"\n🏁 Collection complete!")
        print(f"✅ Successful: {len(self.collected_players)}")
        print(f"❌ Failed: {len(self.failed_players)}")
        
        # Clean up checkpoint file
        if os.path.exists(self.checkpoint_file):
            os.remove(self.checkpoint_file)
            print("🧹 Checkpoint file cleaned up")
    
    def collect_current_college_players(self, year: int = 2024):
        """
        Collect data for current college basketball players
        
        Args:
            year: Current year
        """
        print(f"🏀 Collecting Current College Players ({year})")
        print("=" * 50)
        
        # This would collect data for current college players
        # Implementation would depend on available data sources
        
        # For now, we'll create a placeholder
        print("📊 This would collect data for current college players")
        print("🔍 Sources: ESPN, NCAA, Sports Reference")
        print("📈 Data: Current season stats, team performance, etc.")
        
        pass
    
    def collect_player_college_data(self, player_name: str, draft_year: int = None) -> Optional[Dict]:
        """
        Collect college data for a specific player from Sports Reference
        
        Args:
            player_name: Name of the player
            draft_year: Draft year (for context)
            
        Returns:
            Dictionary with college data or None if not found
        """
        try:
            # Search for player on Sports Reference
            search_url = f"https://www.sports-reference.com/cbb/players/{self._format_name_for_url(player_name)}-1.html"
            
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            
            response = requests.get(search_url, headers=headers, timeout=10)
            
            if response.status_code == 200:
                soup = BeautifulSoup(response.content, 'html.parser')
                college_data = self._parse_sports_reference_page(soup, player_name, draft_year)
                return college_data
            else:
                # Try alternative search methods
                return self._search_alternative_sources(player_name, draft_year)
                
        except Exception as e:
            print(f"⚠️ Error collecting college data for {player_name}: {e}")
            return None
    
    def _format_name_for_url(self, name: str) -> str:
        """Format player name for Sports Reference URL"""
        # Convert "LeBron James" to "lebron-james"
        return name.lower().replace(' ', '-').replace('.', '').replace("'", "")
    
    def _parse_sports_reference_page(self, soup: BeautifulSoup, player_name: str, draft_year: int) -> Dict:
        """Parse Sports Reference page for player data"""
        
        college_data = {
            'player_name': player_name,
            'draft_year': draft_year,
            'college': 'Unknown',
            'conference': 'Unknown',
            'height': None,
            'weight': None,
            'position': 'Unknown',
            'college_seasons': 0,
            'college_ppg': None,
            'college_rpg': None,
            'college_apg': None,
            'college_fg_pct': None,
            'college_3p_pct': None,
            'college_ft_pct': None,
            'college_per': None,
            'college_usage_rate': None,
            'college_ts_pct': None,
            'team_record': None,
            'conference_strength': None,
            'tournament_appearances': 0,
            'draft_position': None,
            'pre_draft_rank': None,
            'scouting_notes': None,
            'data_source': 'sports_reference'
        }
        
        try:
            # Extract basic info
            name_element = soup.find('h1')
            if name_element:
                college_data['player_name'] = name_element.get_text().strip()
            
            # Extract college and conference
            college_info = soup.find('div', {'id': 'meta'})
            if college_info:
                # Look for college name
                college_links = college_info.find_all('a', href=re.compile(r'/cbb/schools/'))
                if college_links:
                    college_data['college'] = college_links[0].get_text().strip()
                
                # Look for conference
                conf_links = college_info.find_all('a', href=re.compile(r'/cbb/conferences/'))
                if conf_links:
                    college_data['conference'] = conf_links[0].get_text().strip()
            
            # Debug: find all tables (commented out for production)
            # all_tables = soup.find_all('table')
            # print(f"🔍 Found {len(all_tables)} tables on page")
            # for i, table in enumerate(all_tables):
            #     table_id = table.get('id', 'no-id')
            #     print(f"  Table {i}: id='{table_id}'")
            
            # Extract stats from career summary table
            stats_table = soup.find('table', {'id': 'players_per_game'})
            if stats_table:
                rows = stats_table.find_all('tr')
                if len(rows) > 1:  # Skip header row
                    # For players with only one season, use the first data row (not header)
                    # For players with multiple seasons, we could use career totals or individual seasons
                    if len(rows) > 1:
                        # Use the first data row (index 1, after header)
                        career_row = rows[1]
                        # print(f"🔍 Using first season row with {len(career_row.find_all('td'))} cells")
                    else:
                        career_row = rows[0]
                    
                    cells = career_row.find_all('td')
                    
                    if len(cells) >= 10:
                        # Debug: print all cell values from first season (commented out for production)
                        # print(f"🔍 Debug - First season row with {len(cells)} cells:")
                        # for i, cell in enumerate(cells):
                        #     print(f"  Cell {i}: '{cell.get_text().strip()}'")
                        
                        # Based on debug output, the structure is:
                        # Cell 0: 'Texas' (School)
                        # Cell 1: 'Big 12' (Conference)
                        # Cell 2: 'FR' (Class)
                        # Cell 3: 'F' (Position)
                        # Cell 4: '35' (Games)
                        # Cell 5: '35' (Games Started)
                        # Cell 6: '35.9' (Minutes Per Game)
                        # Cell 7: '8.7' (FG)
                        # Cell 8: '18.5' (FGA)
                        # Cell 9: '.473' (FG%)
                        # Cell 10: '2.3' (3P)
                        # Cell 11: '5.8' (3PA)
                        # Cell 12: '.404' (3P%)
                        # Cell 13: '6.4' (FT)
                        # Cell 14: '12.7' (FTA)
                        # Cell 15: '.505' (FT%)
                        # Cell 16: '.536' (ORB)
                        # Cell 17: '6.0' (DRB)
                        # Cell 18: '7.3' (TRB)
                        # Cell 19: '.816' (AST)
                        # Cell 20: '3.0' (STL)
                        # Cell 21: '8.1' (BLK)
                        # Cell 22: '11.1' (TOV)
                        # Cell 23: '1.3' (PF)
                        # Cell 24: '1.9' (PTS)
                        # Cell 25: '1.9' (PTS)
                        # Cell 26: '2.8' (PTS)
                        # Cell 27: '2.0' (PTS)
                        # Cell 28: '25.8' (PTS)
                        
                        college_data['college_ppg'] = self._safe_float(cells[28].get_text())  # PTS (25.8)
                        college_data['college_rpg'] = self._safe_float(cells[22].get_text())  # TRB (11.1)
                        college_data['college_apg'] = self._safe_float(cells[23].get_text())  # AST (1.3)
                        college_data['college_fg_pct'] = self._safe_float(cells[9].get_text())  # FG% (0.473)
                        college_data['college_3p_pct'] = self._safe_float(cells[12].get_text())  # 3P% (0.404)
                        college_data['college_ft_pct'] = self._safe_float(cells[15].get_text())  # FT% (0.505)
            
            # Extract advanced stats if available
            advanced_table = soup.find('table', {'id': 'players_advanced'})
            if advanced_table:
                rows = advanced_table.find_all('tr')
                if len(rows) > 1:
                    career_row = rows[-1]
                    cells = career_row.find_all('td')
                    
                    if len(cells) >= 5:
                        college_data['college_per'] = self._safe_float(cells[1].get_text())
                        college_data['college_ts_pct'] = self._safe_float(cells[2].get_text())
                        college_data['college_usage_rate'] = self._safe_float(cells[3].get_text())
            
            # Count seasons
            seasons_table = soup.find('table', {'id': 'players_per_game'})
            if seasons_table:
                rows = seasons_table.find_all('tr')
                college_data['college_seasons'] = len(rows) - 1  # Subtract header row
            
        except Exception as e:
            print(f"⚠️ Error parsing Sports Reference data for {player_name}: {e}")
        
        return college_data
    
    def _safe_float(self, value: str) -> Optional[float]:
        """Safely convert string to float"""
        try:
            if value and value.strip() and value != '':
                return float(value.strip())
        except (ValueError, TypeError):
            pass
        return None
    
    def _search_alternative_sources(self, player_name: str, draft_year: int) -> Optional[Dict]:
        """Search alternative sources if Sports Reference fails"""
        try:
            # Try ESPN search
            return self._search_espn(player_name, draft_year)
        except Exception as e:
            print(f"⚠️ Alternative search failed for {player_name}: {e}")
            return None
    
    def _search_espn(self, player_name: str, draft_year: int) -> Optional[Dict]:
        """Search ESPN for player data"""
        # This would implement ESPN search
        # For now, return basic structure
        return {
            'player_name': player_name,
            'draft_year': draft_year,
            'college': 'Unknown',
            'conference': 'Unknown',
            'data_source': 'espn_placeholder'
        }
    
    def collect_draft_prospects_data(self, year: int = 2024):
        """
        Collect data for current draft prospects
        
        Args:
            year: Draft year
        """
        print(f"🏀 Collecting Data for {year} Draft Prospects")
        print("=" * 40)
        
        # This would collect data for current draft prospects
        # Implementation would depend on available data sources
        
        pass
    
    def create_rookie_archetypes(self):
        """
        Create rookie archetypes based on college performance patterns
        """
        print("🎯 Creating Rookie Archetypes")
        print("=" * 30)
        
        # Load college data
        college_files = [f for f in os.listdir(self.data_dir) if f.endswith('.csv')]
        
        if not college_files:
            print("❌ No college data found. Run data collection first.")
            return
        
        # Load and analyze college data
        all_college_data = []
        for file in college_files:
            df = pd.read_csv(os.path.join(self.data_dir, file))
            all_college_data.append(df)
        
        if all_college_data:
            combined_df = pd.concat(all_college_data, ignore_index=True)
            
            # Create archetypes based on college performance patterns
            archetypes = self._analyze_college_patterns(combined_df)
            
            # Save archetypes
            archetypes_path = os.path.join(self.data_dir, 'rookie_archetypes.json')
            with open(archetypes_path, 'w') as f:
                json.dump(archetypes, f, indent=2)
            
            print(f"✅ Rookie archetypes saved to {archetypes_path}")
    
    def _analyze_college_patterns(self, df: pd.DataFrame) -> Dict:
        """
        Analyze college performance patterns to create archetypes
        
        Args:
            df: DataFrame with college data
            
        Returns:
            Dictionary with archetype definitions
        """
        # This would analyze patterns in college data to create archetypes
        # For example:
        # - High-scoring guards
        # - Rebounding bigs
        # - 3-and-D wings
        # - Playmaking guards
        # etc.
        
        archetypes = {
            'high_scoring_guard': {
                'description': 'High-scoring guards with good shooting',
                'criteria': {
                    'college_ppg': '> 20',
                    'college_3p_pct': '> 0.35',
                    'position': 'Guard'
                }
            },
            'rebounding_big': {
                'description': 'Dominant rebounding big men',
                'criteria': {
                    'college_rpg': '> 10',
                    'college_fg_pct': '> 0.55',
                    'position': 'Forward/Center'
                }
            },
            'playmaking_guard': {
                'description': 'High-assist guards with good court vision',
                'criteria': {
                    'college_apg': '> 6',
                    'college_ast_to_ratio': '> 2.0',
                    'position': 'Guard'
                }
            }
        }
        
        return archetypes
    
    def _load_checkpoint(self):
        """Load checkpoint data if exists"""
        if os.path.exists(self.checkpoint_file):
            try:
                with open(self.checkpoint_file, 'r') as f:
                    checkpoint_data = json.load(f)
                    self.college_data = checkpoint_data.get('college_data', [])
                    self.collected_players = set(checkpoint_data.get('collected_players', []))
                    self.failed_players = set(checkpoint_data.get('failed_players', []))
                    print(f"🔄 Loaded checkpoint: {len(self.college_data)} players already collected")
            except Exception as e:
                print(f"⚠️ Error loading checkpoint: {e}")
                self.college_data = []
                self.collected_players = set()
                self.failed_players = set()
    
    def _save_checkpoint(self, college_data):
        """Save checkpoint data"""
        try:
            checkpoint_data = {
                'college_data': college_data,
                'collected_players': list(self.collected_players),
                'failed_players': list(self.failed_players),
                'timestamp': datetime.now().isoformat()
            }
            with open(self.checkpoint_file, 'w') as f:
                json.dump(checkpoint_data, f, indent=2)
        except Exception as e:
            print(f"⚠️ Error saving checkpoint: {e}")
    
    def _save_player_college_data(self, player_name: str, college_data: Dict):
        """Save individual player's college data to their directory"""
        try:
            # Create player directory path (same as NBA data structure)
            # data_dir is data2/college, so we go up one level to get data2
            base_data_dir = os.path.dirname(self.data_dir)
            player_dir = os.path.join(base_data_dir, player_name.replace(' ', '_'))
            os.makedirs(player_dir, exist_ok=True)
            
            # Create college data CSV
            college_df = pd.DataFrame([college_data])
            college_file = os.path.join(player_dir, f'{player_name.replace(" ", "_")}_college_data.csv')
            college_df.to_csv(college_file, index=False)
            
            print(f"💾 Saved college data to: {college_file}")
            
        except Exception as e:
            print(f"⚠️ Error saving college data for {player_name}: {e}")

def test_single_player():
    """Test the collector with a single player"""
    print("🧪 Testing College Data Collection with Kevin Durant")
    print("=" * 55)
    
    collector = CollegeBasketballCollector()
    
    # Test with Kevin Durant (played 1 year at Texas)
    test_player = "Kevin Durant"
    draft_year = 2007
    
    print(f"🔍 Testing with: {test_player} (Draft: {draft_year})")
    
    try:
        college_data = collector.collect_player_college_data(test_player, draft_year)
        
        if college_data:
            print(f"✅ Successfully collected data for {test_player}")
            print(f"📊 College: {college_data.get('college', 'Unknown')}")
            print(f"📊 Conference: {college_data.get('conference', 'Unknown')}")
            print(f"📊 PPG: {college_data.get('college_ppg', 'N/A')}")
            print(f"📊 RPG: {college_data.get('college_rpg', 'N/A')}")
            print(f"📊 APG: {college_data.get('college_apg', 'N/A')}")
            print(f"📊 FG%: {college_data.get('college_fg_pct', 'N/A')}")
            print(f"📊 3P%: {college_data.get('college_3p_pct', 'N/A')}")
            print(f"📊 FT%: {college_data.get('college_ft_pct', 'N/A')}")
            print(f"📊 PER: {college_data.get('college_per', 'N/A')}")
            print(f"📊 Seasons: {college_data.get('college_seasons', 'N/A')}")
            
            # Save individual player college data
            collector._save_player_college_data(test_player, college_data)
        else:
            print(f"❌ Failed to collect data for {test_player}")
            
    except Exception as e:
        print(f"❌ Error testing {test_player}: {e}")

def main():
    """Main execution function"""
    print("🏀 College Basketball Data Collection System")
    print("=" * 50)
    print("1. Test with single player (Kevin Durant)")
    print("2. Collect NBA rookies college data (2005-2024)")
    print("3. Collect current draft prospects")
    print("4. Create rookie archetypes")
    print("5. Exit")
    print()
    
    choice = input("Select option (1-5): ").strip()
    
    collector = CollegeBasketballCollector()
    
    if choice == "1":
        test_single_player()
    
    elif choice == "2":
        start_year = int(input("Start year (default 2005): ") or "2005")
        end_year = int(input("End year (default 2024): ") or "2024")
        collector.collect_nba_rookies_college_data(start_year, end_year)
    
    elif choice == "3":
        year = int(input("Draft year (default 2024): ") or "2024")
        collector.collect_draft_prospects_data(year)
    
    elif choice == "4":
        collector.create_rookie_archetypes()
    
    elif choice == "5":
        print("👋 Goodbye!")
    
    else:
        print("❌ Invalid choice")

if __name__ == "__main__":
    main()
