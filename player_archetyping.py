#!/usr/bin/env python3
"""
NBA Player Archetyping System
=============================
Creates dynamic archetypes for each player throughout their career
"""

import pandas as pd
import numpy as np
import os
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings('ignore')


class PlayerArchetyping:
    """Create comprehensive archetypes for NBA players"""
    
    def __init__(self, data_dir: str = "player_data"):
        self.data_dir = data_dir
        
        # Archetype dimensions and thresholds
        self.archetypes = self._define_archetypes()
    
    def _define_archetypes(self):
        """Define comprehensive archetype categories"""
        return {
            # PRIMARY ROLE ARCHETYPES
            'primary_role': {
                'Elite Scorer': {
                    'criteria': lambda df: (df['PTS'] >= 25) & (df['USG_RATE_EST'] >= 25)
                },
                'Primary Scorer': {
                    'criteria': lambda df: (df['PTS'] >= 18) & (df['PTS'] < 25) & (df['USG_RATE_EST'] >= 20)
                },
                'Secondary Scorer': {
                    'criteria': lambda df: (df['PTS'] >= 12) & (df['PTS'] < 18)
                },
                'Role Player': {
                    'criteria': lambda df: (df['PTS'] >= 6) & (df['PTS'] < 12)
                },
                'Bench Player': {
                    'criteria': lambda df: (df['PTS'] >= 0) & (df['PTS'] < 6)
                }
            },
            
            # SHOOTING ARCHETYPES
            'shooting_style': {
                'Elite Shooter': {
                    'criteria': lambda df: (df['FG3M'] >= 2.5) & (df['FG3_PCT'] >= 0.38)
                },
                'Volume 3PT Shooter': {
                    'criteria': lambda df: (df['FG3M'] >= 2.0) & (df['FG3_PCT'] >= 0.35)
                },
                'Spot-Up Shooter': {
                    'criteria': lambda df: (df['FG3M'] >= 1.0) & (df['FG3M'] < 2.0) & (df['FG3_PCT'] >= 0.36)
                },
                'Mid-Range Specialist': {
                    'criteria': lambda df: (df['FGM'] >= 6) & (df['FG3M'] < 1.5) & (df['FG_PCT'] >= 0.45)
                },
                'Paint Scorer': {
                    'criteria': lambda df: (df['FGM'] >= 5) & (df['FG3M'] < 1.0) & (df['FG_PCT'] >= 0.50)
                },
                'Non-Shooter': {
                    'criteria': lambda df: (df['FG3M'] < 0.5)
                }
            },
            
            # PLAYMAKING ARCHETYPES
            'playmaking': {
                'Elite Playmaker': {
                    'criteria': lambda df: (df['AST'] >= 8) & (df['AST_TO_RATIO'] >= 2.0)
                },
                'Primary Playmaker': {
                    'criteria': lambda df: (df['AST'] >= 5) & (df['AST'] < 8) & (df['AST_TO_RATIO'] >= 1.5)
                },
                'Secondary Playmaker': {
                    'criteria': lambda df: (df['AST'] >= 3) & (df['AST'] < 5)
                },
                'Limited Playmaker': {
                    'criteria': lambda df: (df['AST'] >= 1) & (df['AST'] < 3)
                },
                'Non-Playmaker': {
                    'criteria': lambda df: df['AST'] < 1
                }
            },
            
            # REBOUNDING ARCHETYPES
            'rebounding': {
                'Elite Rebounder': {
                    'criteria': lambda df: df['REB'] >= 12
                },
                'Strong Rebounder': {
                    'criteria': lambda df: (df['REB'] >= 8) & (df['REB'] < 12)
                },
                'Solid Rebounder': {
                    'criteria': lambda df: (df['REB'] >= 5) & (df['REB'] < 8)
                },
                'Average Rebounder': {
                    'criteria': lambda df: (df['REB'] >= 3) & (df['REB'] < 5)
                },
                'Weak Rebounder': {
                    'criteria': lambda df: df['REB'] < 3
                },
                'Offensive Glass Specialist': {
                    'criteria': lambda df: (df['OREB'] >= 2.5) & (df['OREB'] > df['DREB'])
                },
                'Defensive Glass Specialist': {
                    'criteria': lambda df: (df['DREB'] >= 7) & (df['DREB'] > df['OREB'] * 2)
                }
            },
            
            # DEFENSIVE ARCHETYPES
            'defense': {
                'Elite Defender': {
                    'criteria': lambda df: (df['STL'] + df['BLK']) >= 3
                },
                'Perimeter Defender': {
                    'criteria': lambda df: (df['STL'] >= 1.5) & (df['STL'] > df['BLK'])
                },
                'Rim Protector': {
                    'criteria': lambda df: (df['BLK'] >= 2.0) & (df['BLK'] > df['STL'])
                },
                'Switchable Defender': {
                    'criteria': lambda df: (df['STL'] >= 1.0) & (df['BLK'] >= 1.0)
                },
                'Weak Defender': {
                    'criteria': lambda df: (df['STL'] + df['BLK']) < 1
                }
            },
            
            # EFFICIENCY ARCHETYPES
            'efficiency': {
                'Elite Efficiency': {
                    'criteria': lambda df: (df['TS_PCT'] >= 0.60) & (df['PTS'] >= 15)
                },
                'High Efficiency': {
                    'criteria': lambda df: (df['TS_PCT'] >= 0.55) & (df['TS_PCT'] < 0.60)
                },
                'Average Efficiency': {
                    'criteria': lambda df: (df['TS_PCT'] >= 0.50) & (df['TS_PCT'] < 0.55)
                },
                'Low Efficiency': {
                    'criteria': lambda df: df['TS_PCT'] < 0.50
                },
                'Volume Scorer': {
                    'criteria': lambda df: (df['FGA'] >= 15) & (df['USG_RATE_EST'] >= 25)
                },
                'Efficient Role Player': {
                    'criteria': lambda df: (df['TS_PCT'] >= 0.58) & (df['PTS'] < 15)
                }
            },
            
            # USAGE/ROLE ARCHETYPES
            'usage': {
                'Superstar Usage': {
                    'criteria': lambda df: df['USG_RATE_EST'] >= 30
                },
                'Star Usage': {
                    'criteria': lambda df: (df['USG_RATE_EST'] >= 25) & (df['USG_RATE_EST'] < 30)
                },
                'High Usage': {
                    'criteria': lambda df: (df['USG_RATE_EST'] >= 20) & (df['USG_RATE_EST'] < 25)
                },
                'Medium Usage': {
                    'criteria': lambda df: (df['USG_RATE_EST'] >= 15) & (df['USG_RATE_EST'] < 20)
                },
                'Low Usage': {
                    'criteria': lambda df: df['USG_RATE_EST'] < 15
                }
            },
            
            # PLAY STYLE ARCHETYPES
            'play_style': {
                '3-and-D Wing': {
                    'criteria': lambda df: (df['FG3M'] >= 1.5) & (df['STL'] + df['BLK'] >= 1.5) & (df['PTS'] < 15)
                },
                'Stretch Big': {
                    'criteria': lambda df: (df['FG3M'] >= 1.0) & (df['REB'] >= 6) & (df['BLK'] >= 0.8)
                },
                'Point Forward': {
                    'criteria': lambda df: (df['AST'] >= 4) & (df['REB'] >= 5) & (df['PTS'] >= 12)
                },
                'Traditional Big': {
                    'criteria': lambda df: (df['REB'] >= 8) & (df['BLK'] >= 1.5) & (df['FG3M'] < 0.5)
                },
                'Floor General': {
                    'criteria': lambda df: (df['AST'] >= 7) & (df['TOV'] >= 2) & (df['PTS'] >= 10)
                },
                'Combo Guard': {
                    'criteria': lambda df: (df['PTS'] >= 15) & (df['AST'] >= 3) & (df['AST'] < 7)
                },
                'Shot Creator': {
                    'criteria': lambda df: (df['PTS'] >= 18) & (df['FGA'] >= 14) & (df['AST'] < 5)
                },
                'Energy Big': {
                    'criteria': lambda df: (df['OREB'] >= 2) & (df['PF'] >= 3) & (df['MIN'] < 28)
                },
                'Glue Guy': {
                    'criteria': lambda df: (df['REB'] >= 4) & (df['AST'] >= 2) & (df['STL'] >= 0.8) & (df['PTS'] < 12)
                }
            },
            
            # MINUTES/ROLE ARCHETYPES
            'playing_time': {
                'Starter': {
                    'criteria': lambda df: df['MIN'] >= 30
                },
                'Sixth Man': {
                    'criteria': lambda df: (df['MIN'] >= 20) & (df['MIN'] < 30)
                },
                'Rotation Player': {
                    'criteria': lambda df: (df['MIN'] >= 15) & (df['MIN'] < 20)
                },
                'Deep Bench': {
                    'criteria': lambda df: (df['MIN'] >= 5) & (df['MIN'] < 15)
                },
                'Garbage Time': {
                    'criteria': lambda df: df['MIN'] < 5
                }
            },
            
            # CONSISTENCY ARCHETYPES
            'consistency': {
                'Highly Consistent': {
                    'criteria': lambda df: (df['PTS_CONSISTENCY'] <= df['PTS'].mean() * 0.25) if 'PTS_CONSISTENCY' in df.columns else False
                },
                'Moderately Consistent': {
                    'criteria': lambda df: (df['PTS_CONSISTENCY'] > df['PTS'].mean() * 0.25) & (df['PTS_CONSISTENCY'] <= df['PTS'].mean() * 0.4) if 'PTS_CONSISTENCY' in df.columns else False
                },
                'Inconsistent': {
                    'criteria': lambda df: (df['PTS_CONSISTENCY'] > df['PTS'].mean() * 0.4) if 'PTS_CONSISTENCY' in df.columns else False
                }
            },
            
            # FREE THROW ARCHETYPES
            'free_throw': {
                'Free Throw Machine': {
                    'criteria': lambda df: (df['FTM'] >= 6) & (df['FT_PCT'] >= 0.85)
                },
                'Gets to the Line': {
                    'criteria': lambda df: (df['FTM'] >= 4) & (df['FTM'] < 6)
                },
                'Rarely at Line': {
                    'criteria': lambda df: df['FTM'] < 2
                },
                'Poor FT Shooter': {
                    'criteria': lambda df: (df['FT_PCT'] < 0.65) & (df['FTM'] >= 2)
                }
            },
            
            # VERSATILITY ARCHETYPES
            'versatility': {
                'All-Around Star': {
                    'criteria': lambda df: (df['PTS'] >= 20) & (df['REB'] >= 6) & (df['AST'] >= 4)
                },
                'Triple-Double Threat': {
                    'criteria': lambda df: (df['PTS'] >= 15) & (df['REB'] >= 7) & (df['AST'] >= 7)
                },
                'Two-Way Player': {
                    'criteria': lambda df: (df['PTS'] >= 12) & (df['STL'] + df['BLK'] >= 2)
                },
                'Specialist': {
                    'criteria': lambda df: ((df['FG3M'] >= 2) | (df['BLK'] >= 2) | (df['OREB'] >= 3)) & (df['PTS'] < 12)
                }
            }
        }
    
    def calculate_rolling_stats(self, df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
        """Calculate rolling averages for archetype determination"""
        stats_to_roll = ['PTS', 'REB', 'AST', 'STL', 'BLK', 'FGM', 'FGA', 'FG3M', 
                         'FG3A', 'FTM', 'FTA', 'OREB', 'DREB', 'TOV', 'PF', 'MIN']
        
        for stat in stats_to_roll:
            if stat in df.columns:
                df[f'{stat}_ROLL'] = df[stat].rolling(window=window, min_periods=5).mean()
        
        # Calculate derived stats from rolling averages
        if 'FGM_ROLL' in df.columns and 'FGA_ROLL' in df.columns:
            df['FG_PCT'] = df['FGM_ROLL'] / df['FGA_ROLL'].replace(0, np.nan)
        
        if 'FG3M_ROLL' in df.columns and 'FG3A_ROLL' in df.columns:
            df['FG3_PCT'] = df['FG3M_ROLL'] / df['FG3A_ROLL'].replace(0, np.nan)
        
        if 'FTM_ROLL' in df.columns and 'FTA_ROLL' in df.columns:
            df['FT_PCT'] = df['FTM_ROLL'] / df['FTA_ROLL'].replace(0, np.nan)
        
        if 'AST_ROLL' in df.columns and 'TOV_ROLL' in df.columns:
            df['AST_TO_RATIO'] = df['AST_ROLL'] / df['TOV_ROLL'].replace(0, np.nan)
        
        # True Shooting % from rolling
        if all(col in df.columns for col in ['PTS_ROLL', 'FGA_ROLL', 'FTA_ROLL']):
            df['TS_PCT'] = df['PTS_ROLL'] / (2 * (df['FGA_ROLL'] + 0.44 * df['FTA_ROLL'])).replace(0, np.nan)
        
        # Usage Rate Estimate from rolling
        if all(col in df.columns for col in ['FGA_ROLL', 'FTA_ROLL', 'TOV_ROLL', 'MIN_ROLL']):
            df['USG_RATE_EST'] = 100 * ((df['FGA_ROLL'] + 0.44 * df['FTA_ROLL'] + df['TOV_ROLL']) / df['MIN_ROLL'].replace(0, np.nan))
        
        # Consistency measure
        if 'PTS' in df.columns:
            df['PTS_CONSISTENCY'] = df['PTS'].rolling(window=window, min_periods=5).std()
        
        return df
    
    def assign_archetypes(self, df: pd.DataFrame) -> pd.DataFrame:
        """Assign all archetypes to each game"""
        
        archetype_df = df.copy()
        
        # Calculate rolling stats first (20-game window)
        archetype_df = self.calculate_rolling_stats(archetype_df)
        
        # Use rolling averages for archetype assignment
        roll_df = archetype_df.copy()
        for col in archetype_df.columns:
            if col.endswith('_ROLL'):
                base_col = col.replace('_ROLL', '')
                roll_df[base_col] = archetype_df[col]
        
        # Assign each archetype category
        for category, archetypes in self.archetypes.items():
            archetype_df[f'ARCHETYPE_{category.upper()}'] = 'Unknown'
            
            for archetype_name, archetype_def in archetypes.items():
                try:
                    mask = archetype_def['criteria'](roll_df)
                    archetype_df.loc[mask, f'ARCHETYPE_{category.upper()}'] = archetype_name
                except Exception as e:
                    # If criteria fails, skip this archetype
                    continue
        
        # Create composite archetype (combine multiple dimensions)
        archetype_df['COMPOSITE_ARCHETYPE'] = (
            archetype_df['ARCHETYPE_PRIMARY_ROLE'] + ' | ' +
            archetype_df['ARCHETYPE_SHOOTING_STYLE'] + ' | ' +
            archetype_df['ARCHETYPE_PLAYMAKING'] + ' | ' +
            archetype_df['ARCHETYPE_DEFENSE']
        )
        
        # Add numerical archetype strength scores
        archetype_df['SCORER_STRENGTH'] = roll_df.get('PTS', 0) / 30.0  # 0-1 scale
        archetype_df['PLAYMAKER_STRENGTH'] = roll_df.get('AST', 0) / 12.0
        archetype_df['REBOUNDER_STRENGTH'] = roll_df.get('REB', 0) / 15.0
        archetype_df['DEFENDER_STRENGTH'] = (roll_df.get('STL', 0) + roll_df.get('BLK', 0)) / 5.0
        archetype_df['SHOOTER_STRENGTH'] = roll_df.get('FG3M', 0) / 4.0
        
        return archetype_df
    
    def create_player_archetype(self, player_name: str) -> bool:
        """
        Create archetype CSV for a single player
        
        Args:
            player_name: Name of the player
            
        Returns:
            True if successful, False otherwise
        """
        player_dir = Path(self.data_dir) / player_name.replace(' ', '_')
        nba_file = player_dir / f"{player_name.replace(' ', '_')}_data.csv"
        archetype_file = player_dir / f"{player_name.replace(' ', '_')}_archetype.csv"
        
        # Check if already exists
        if archetype_file.exists():
            print(f"⏭️  {player_name}: Archetype already exists, skipping...")
            return True
        
        # Load NBA data
        if not nba_file.exists():
            return False
        
        try:
            df = pd.read_csv(nba_file)
            
            # Clean data
            df = df.replace('', np.nan)
            df['GAME_DATE'] = pd.to_datetime(df['GAME_DATE'])
            df = df.sort_values('GAME_DATE').reset_index(drop=True)
            
            # Ensure numeric columns
            numeric_cols = ['PTS', 'REB', 'AST', 'STL', 'BLK', 'FGM', 'FGA', 
                           'FG3M', 'FG3A', 'FTM', 'FTA', 'OREB', 'DREB', 'TOV', 'MIN']
            for col in numeric_cols:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Check for minimum games
            if len(df) < 20:
                print(f"❌ {player_name}: Insufficient games ({len(df)})")
                return False
            
            # Assign archetypes
            archetype_df = self.assign_archetypes(df)
            
            # Select archetype columns to save
            archetype_cols = ['GAME_DATE', 'SEASON', 'PLAYER_NAME'] + \
                           [col for col in archetype_df.columns if 'ARCHETYPE' in col or 'STRENGTH' in col]
            
            # Add some context columns
            context_cols = ['PTS', 'REB', 'AST', 'STL', 'BLK', 'FG3M', 'MIN', 'IS_HOME', 
                          'OPPONENT_TEAM', 'season_exp']
            for col in context_cols:
                if col in archetype_df.columns:
                    archetype_cols.append(col)
            
            # Save archetype data
            final_df = archetype_df[archetype_cols].copy()
            final_df.to_csv(archetype_file, index=False)
            
            print(f"✅ {player_name}: Archetype created ({len(final_df)} games)")
            return True
            
        except Exception as e:
            print(f"❌ {player_name}: {e}")
            return False
    
    def create_all_archetypes(self, limit: int = None):
        """
        Create archetypes for all players
        
        Args:
            limit: Limit number of players (for testing)
        """
        print("🏀 Creating Player Archetypes for All Players")
        print("=" * 60)
        
        data_path = Path(self.data_dir)
        player_dirs = [d for d in data_path.iterdir() 
                      if d.is_dir() and d.name != 'college']
        
        # Filter to only players with NBA data
        player_dirs = [d for d in player_dirs 
                      if (d / f"{d.name}_data.csv").exists()]
        
        if limit:
            player_dirs = player_dirs[:limit]
        
        print(f"📊 Processing {len(player_dirs)} players...")
        print(f"🎯 Creating {len(self.archetypes)} archetype dimensions")
        print()
        
        success_count = 0
        failed_count = 0
        skipped_count = 0
        
        for idx, player_dir in enumerate(player_dirs, 1):
            player_name = player_dir.name.replace('_', ' ')
            
            try:
                # Check if already exists
                archetype_file = player_dir / f"{player_dir.name}_archetype.csv"
                if archetype_file.exists():
                    skipped_count += 1
                    if idx % 100 == 0:
                        print(f"⏭️  Processed {idx}/{len(player_dirs)} players (skipped: {skipped_count})...")
                    continue
                
                if self.create_player_archetype(player_name):
                    success_count += 1
                else:
                    failed_count += 1
                
                if idx % 100 == 0:
                    print(f"✅ Processed {idx}/{len(player_dirs)} players...")
                    
            except Exception as e:
                print(f"❌ Error processing {player_name}: {e}")
                failed_count += 1
        
        print()
        print("=" * 60)
        print("🏁 Archetype Creation Complete!")
        print(f"✅ Successfully created: {success_count} players")
        print(f"⏭️  Skipped (already exist): {skipped_count} players")
        print(f"❌ Failed: {failed_count} players")
        print(f"📁 Archetypes saved to: data2/[Player_Name]/[Player_Name]_archetype.csv")
        
        return success_count


def main():
    """Main execution"""
    print("🏀 NBA Player Archetyping System")
    print("=" * 50)
    print("1. Test with single player")
    print("2. Test with limited players")
    print("3. Create archetypes for all players")
    print("4. Exit")
    print()
    
    choice = input("Select option (1-4): ").strip()
    
    archetyper = PlayerArchetyping()
    
    if choice == "1":
        player_name = input("Enter player name (e.g., Kevin Durant): ").strip()
        archetyper.create_player_archetype(player_name)
        
        # Show sample archetypes
        player_dir = Path('data2') / player_name.replace(' ', '_')
        archetype_file = player_dir / f"{player_name.replace(' ', '_')}_archetype.csv"
        if archetype_file.exists():
            df = pd.read_csv(archetype_file)
            print(f"\n📊 Sample Archetypes for {player_name}:")
            print(df[['GAME_DATE', 'SEASON', 'ARCHETYPE_PRIMARY_ROLE', 
                     'ARCHETYPE_SHOOTING_STYLE', 'ARCHETYPE_PLAYMAKING']].head(10))
    
    elif choice == "2":
        limit = int(input("How many players to process? ") or "10")
        archetyper.create_all_archetypes(limit=limit)
    
    elif choice == "3":
        confirm = input("Create archetypes for ALL players? (yes/no): ").strip().lower()
        if confirm == 'yes':
            archetyper.create_all_archetypes()
        else:
            print("❌ Cancelled")
    
    elif choice == "4":
        print("👋 Goodbye!")
    
    else:
        print("❌ Invalid choice")


if __name__ == "__main__":
    main()

