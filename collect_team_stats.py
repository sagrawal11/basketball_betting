#!/usr/bin/env python3
"""
NBA Team Stats Collector
========================
Collects team-level statistics (defensive rating, pace, offensive rating)
for all seasons to enhance player predictions
"""

import pandas as pd
import time
import logging
from pathlib import Path
from nba_api.stats.endpoints import leaguedashteamstats
import json

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class TeamStatsCollector:
    """Collect team-level statistics across all seasons"""
    
    def __init__(self, output_dir: str = "team_stats"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.request_delay = 2.0  # Be nice to NBA API
        
    def collect_season_team_stats(self, season: str):
        """
        Collect team stats for a specific season
        
        Args:
            season: Season in format '2023-24'
        """
        try:
            logger.info(f"📊 Collecting team stats for {season}...")
            
            # Get team stats - Basic (has abbreviations)
            basic_stats = leaguedashteamstats.LeagueDashTeamStats(
                season=season,
                measure_type_detailed_defense='Base',
                per_mode_detailed='PerGame'
            )
            basic_df = basic_stats.get_data_frames()[0]
            
            time.sleep(self.request_delay)
            
            # Get team stats - Advanced (includes PACE, OFF_RATING, DEF_RATING)
            advanced_stats = leaguedashteamstats.LeagueDashTeamStats(
                season=season,
                measure_type_detailed_defense='Advanced',
                per_mode_detailed='PerGame'
            )
            advanced_df = advanced_stats.get_data_frames()[0]
            
            # Merge to get both basic info and advanced stats
            team_stats = basic_df[['TEAM_ID', 'TEAM_NAME']].merge(
                advanced_df[['TEAM_ID', 'OFF_RATING', 'DEF_RATING', 'NET_RATING', 'PACE', 'W_PCT']],
                on='TEAM_ID',
                how='left'
            )
            
            # Create abbreviation from team name
            # Map full names to abbreviations
            abbrev_map = {
                # Current teams
                'Atlanta Hawks': 'ATL', 'Boston Celtics': 'BOS', 'Brooklyn Nets': 'BKN',
                'Charlotte Hornets': 'CHA', 'Chicago Bulls': 'CHI', 'Cleveland Cavaliers': 'CLE',
                'Dallas Mavericks': 'DAL', 'Denver Nuggets': 'DEN', 'Detroit Pistons': 'DET',
                'Golden State Warriors': 'GSW', 'Houston Rockets': 'HOU', 'Indiana Pacers': 'IND',
                'LA Clippers': 'LAC', 'Los Angeles Clippers': 'LAC',  # Both variations
                'Los Angeles Lakers': 'LAL', 'Memphis Grizzlies': 'MEM',
                'Miami Heat': 'MIA', 'Milwaukee Bucks': 'MIL', 'Minnesota Timberwolves': 'MIN',
                'New Orleans Pelicans': 'NOP', 'New York Knicks': 'NYK', 'Oklahoma City Thunder': 'OKC',
                'Orlando Magic': 'ORL', 'Philadelphia 76ers': 'PHI', 'Phoenix Suns': 'PHX',
                'Portland Trail Blazers': 'POR', 'Sacramento Kings': 'SAC', 'San Antonio Spurs': 'SAS',
                'Toronto Raptors': 'TOR', 'Utah Jazz': 'UTA', 'Washington Wizards': 'WAS',
                # Historical NBA teams
                'New Jersey Nets': 'NJN', 'Seattle SuperSonics': 'SEA', 'Charlotte Bobcats': 'CHA',
                'New Orleans Hornets': 'NOH', 'New Orleans/Oklahoma City Hornets': 'NOK',
                'Vancouver Grizzlies': 'VAN', 'Washington Bullets': 'WSB'
            }
            
            team_stats['TEAM_ABBREVIATION'] = team_stats['TEAM_NAME'].map(abbrev_map)
            
            # Filter out WNBA teams (no abbreviation mapping)
            team_stats = team_stats[team_stats['TEAM_ABBREVIATION'].notna()]
            
            # Add season column
            team_stats['SEASON'] = season
            
            return team_stats
            
        except Exception as e:
            logger.error(f"❌ Error collecting {season}: {e}")
            return None
    
    def collect_all_seasons(self, start_year: int = 1996, end_year: int = 2024):
        """
        Collect team stats for all seasons from start_year to end_year
        
        Args:
            start_year: First year to collect (default 1996, when advanced stats became reliable)
            end_year: Last year to collect
        """
        all_team_stats = []
        
        # Create list of seasons
        seasons = []
        for year in range(start_year, end_year + 1):
            next_year = str(year + 1)[-2:]
            season = f"{year}-{next_year}"
            seasons.append(season)
        
        logger.info(f"🏀 Collecting team stats for {len(seasons)} seasons ({start_year}-{end_year})")
        
        for idx, season in enumerate(seasons, 1):
            season_stats = self.collect_season_team_stats(season)
            
            if season_stats is not None:
                all_team_stats.append(season_stats)
                logger.info(f"✅ [{idx}/{len(seasons)}] {season}: {len(season_stats)} teams")
            
            time.sleep(self.request_delay)
        
        # Combine all seasons
        if all_team_stats:
            combined_df = pd.concat(all_team_stats, ignore_index=True)
            
            # Save to CSV
            output_file = self.output_dir / "all_team_stats.csv"
            combined_df.to_csv(output_file, index=False)
            
            logger.info(f"💾 Saved {len(combined_df)} team-season records to {output_file}")
            
            # Create summary
            summary = {
                'total_records': len(combined_df),
                'seasons': len(seasons),
                'teams_per_season': int(len(combined_df) / len(seasons)),
                'seasons_covered': [s for s in seasons if any(combined_df['SEASON'] == s)],
                'key_columns': ['TEAM_ABBREVIATION', 'OFF_RATING', 'DEF_RATING', 'PACE', 'NET_RATING', 'W_PCT']
            }
            
            summary_file = self.output_dir / "team_stats_summary.json"
            with open(summary_file, 'w') as f:
                json.dump(summary, f, indent=2)
            
            return combined_df
        
        return None


def main():
    """Main execution"""
    collector = TeamStatsCollector()
    
    print("🏀 NBA Team Stats Collection")
    print("=" * 70)
    print()
    print("Collecting team-level stats for all seasons (1996-2024):")
    print("  📊 Offensive Rating")
    print("  🛡️  Defensive Rating")
    print("  ⚡ Pace")
    print("  📈 Net Rating")
    print("  🏆 Win Percentage")
    print()
    print("⏱️  Estimated time: 5 minutes")
    print("📁 Saves to: team_stats/all_team_stats.csv")
    print("✅ Won't interfere with player data!")
    print()
    print("=" * 70)
    print()
    
    # Collect all seasons
    df = collector.collect_all_seasons(start_year=1996, end_year=2024)
    
    if df is not None:
        print()
        print("=" * 70)
        print("✅ Team Stats Collection Complete!")
        print(f"📊 Total records: {len(df)}")
        print(f"📁 Saved to: team_stats/all_team_stats.csv")
        print()
        print("Sample team stats:")
        print(df[['SEASON', 'TEAM_NAME', 'W_PCT', 'OFF_RATING', 'DEF_RATING', 'PACE']].head(10))
        print()
        print("🚀 Ready to merge with player data!")


if __name__ == "__main__":
    main()

