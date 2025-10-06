#!/usr/bin/env python3
"""
Cleanup College-Only Players
============================
Remove college data for players who don't have NBA game data
"""

import os
import pandas as pd
from pathlib import Path

def cleanup_college_only_players(data_dir: str = "data2"):
    """
    Remove college data CSVs for players who don't have NBA data
    
    Args:
        data_dir: Directory containing player folders
    """
    print("🧹 Cleaning up college-only player data")
    print("=" * 50)
    
    removed_count = 0
    kept_count = 0
    no_college_data = 0
    
    # Get all player directories
    data_path = Path(data_dir)
    player_dirs = [d for d in data_path.iterdir() if d.is_dir() and d.name != 'college']
    
    print(f"📂 Checking {len(player_dirs)} player directories...")
    print()
    
    for player_dir in player_dirs:
        player_name = player_dir.name.replace('_', ' ')
        
        # Check for NBA data file
        nba_data_file = player_dir / f"{player_dir.name}_data.csv"
        college_data_file = player_dir / f"{player_dir.name}_college_data.csv"
        
        # If college data exists
        if college_data_file.exists():
            # Check if NBA data exists
            if nba_data_file.exists():
                # Check if NBA data file has actual game data (not just header)
                try:
                    nba_df = pd.read_csv(nba_data_file)
                    if len(nba_df) > 0:
                        kept_count += 1
                        print(f"✅ {player_name}: Has NBA data ({len(nba_df)} games)")
                    else:
                        # No NBA games, remove college data
                        college_data_file.unlink()
                        removed_count += 1
                        print(f"🗑️  {player_name}: No NBA games, removed college data")
                except Exception as e:
                    print(f"⚠️  {player_name}: Error reading NBA data: {e}")
            else:
                # No NBA data file at all, remove college data
                college_data_file.unlink()
                removed_count += 1
                print(f"🗑️  {player_name}: No NBA data file, removed college data")
        else:
            no_college_data += 1
    
    print()
    print("=" * 50)
    print("🏁 Cleanup Complete!")
    print(f"✅ Kept: {kept_count} players with both NBA and college data")
    print(f"🗑️  Removed: {removed_count} college-only players")
    print(f"📊 No college data: {no_college_data} players")

if __name__ == "__main__":
    cleanup_college_only_players()

