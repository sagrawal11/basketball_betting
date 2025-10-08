#!/usr/bin/env python3
"""
Fix Season Experience Column
============================
Corrects the season_exp column to properly increment with each season
"""

import pandas as pd
import os
from pathlib import Path
from tqdm import tqdm

def fix_player_experience(player_dir: Path) -> bool:
    """
    Fix season experience for a single player
    
    Args:
        player_dir: Path to player directory
        
    Returns:
        True if successful, False otherwise
    """
    player_name = player_dir.name
    nba_file = player_dir / f"{player_name}_data.csv"
    
    if not nba_file.exists():
        return False
    
    try:
        # Read data
        df = pd.read_csv(nba_file)
        
        if 'SEASON' not in df.columns:
            return False
        
        # Sort by game date to ensure correct order
        df['GAME_DATE'] = pd.to_datetime(df['GAME_DATE'])
        df = df.sort_values('GAME_DATE').reset_index(drop=True)
        
        # Get unique seasons in order
        df['season_year'] = df['SEASON'].str[:4].astype(int)
        unique_seasons = df['season_year'].unique()
        unique_seasons.sort()
        
        # Create a mapping of season to experience
        season_to_exp = {season: idx for idx, season in enumerate(unique_seasons)}
        
        # Apply the mapping
        df['season_exp'] = df['season_year'].map(season_to_exp)
        
        # Drop temporary column
        df = df.drop('season_year', axis=1)
        
        # Save back to file
        df.to_csv(nba_file, index=False)
        
        return True
        
    except Exception as e:
        print(f"❌ Error fixing {player_name}: {e}")
        return False

def fix_all_players(data_dir: str = "player_data"):
    """
    Fix season experience for all players
    
    Args:
        data_dir: Directory containing player folders
    """
    print("🔧 Fixing Season Experience for All Players")
    print("=" * 60)
    
    data_path = Path(data_dir)
    player_dirs = [d for d in data_path.iterdir() if d.is_dir() and d.name != 'college']
    
    print(f"📊 Processing {len(player_dirs)} players...")
    print()
    
    success_count = 0
    failed_count = 0
    
    for player_dir in tqdm(player_dirs, desc="Fixing players"):
        if fix_player_experience(player_dir):
            success_count += 1
        else:
            failed_count += 1
    
    print()
    print("=" * 60)
    print("🏁 Complete!")
    print(f"✅ Successfully fixed: {success_count} players")
    print(f"❌ Failed: {failed_count} players")

if __name__ == "__main__":
    fix_all_players()

