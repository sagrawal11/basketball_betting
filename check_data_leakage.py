#!/usr/bin/env python3
"""
Data Leakage Checker
===================
Verify that features don't contain future information
"""

import pandas as pd
import numpy as np
from pathlib import Path

def check_rolling_features(player_name: str = "Kevin Durant"):
    """Check if rolling averages properly exclude current game"""
    print(f"🔍 Checking Rolling Features for {player_name}")
    print("=" * 70)
    
    # Load features
    player_dir = Path('data2') / player_name.replace(' ', '_')
    features_file = player_dir / f"{player_name.replace(' ', '_')}_features.csv"
    
    if not features_file.exists():
        print(f"❌ Features file not found")
        return False
    
    df = pd.read_csv(features_file)
    
    # Check if rolling averages are calculated correctly
    # For game i, PTS_L3_AVG should be average of games i-3, i-2, i-1 (NOT including game i)
    
    print("\nTest 1: Rolling Average Calculation")
    print("-" * 70)
    
    # Manual calculation for games 10-15
    for i in range(10, 15):
        if i < len(df):
            current_pts = df.iloc[i]['PTS']
            l3_avg = df.iloc[i].get('PTS_L3_AVG', None)
            
            # Calculate what L3 should be (previous 3 games)
            if i >= 3:
                manual_l3 = df.iloc[i-3:i]['PTS'].mean()
                
                # Check if current game is included
                with_current = df.iloc[i-2:i+1]['PTS'].mean()
                
                print(f"Game {i}: PTS={current_pts:.0f}, L3_AVG={l3_avg:.2f}")
                print(f"  Expected (prev 3): {manual_l3:.2f}")
                print(f"  If includes current: {with_current:.2f}")
                
                if abs(l3_avg - manual_l3) < 0.01:
                    print(f"  ✅ CORRECT - Excludes current game")
                elif abs(l3_avg - with_current) < 0.01:
                    print(f"  ❌ LEAKAGE - Includes current game!")
                    return False
                else:
                    print(f"  ⚠️  Unexpected value")
                print()
    
    return True

def check_career_features(player_name: str = "Kevin Durant"):
    """Check if career averages properly exclude current game"""
    print(f"\n🔍 Checking Career Features for {player_name}")
    print("=" * 70)
    
    player_dir = Path('data2') / player_name.replace(' ', '_')
    features_file = player_dir / f"{player_name.replace(' ', '_')}_features.csv"
    
    df = pd.read_csv(features_file)
    
    print("\nTest 2: Career Average Calculation")
    print("-" * 70)
    
    # Check career average at game 100
    if len(df) > 100:
        i = 100
        current_pts = df.iloc[i]['PTS']
        career_avg = df.iloc[i].get('PTS_CAREER_AVG', None)
        
        # Manual calculation (should exclude current game)
        manual_career = df.iloc[:i]['PTS'].mean()
        with_current = df.iloc[:i+1]['PTS'].mean()
        
        print(f"Game {i}: PTS={current_pts:.0f}, CAREER_AVG={career_avg:.2f}")
        print(f"  Expected (games 0-{i-1}): {manual_career:.2f}")
        print(f"  If includes current: {with_current:.2f}")
        
        if abs(career_avg - manual_career) < 0.01:
            print(f"  ✅ CORRECT - Excludes current game")
            return True
        elif abs(career_avg - with_current) < 0.01:
            print(f"  ❌ LEAKAGE - Includes current game!")
            return False
    
    return True

def check_feature_contamination(player_name: str = "Kevin Durant"):
    """Check if features contain target variables"""
    print(f"\n🔍 Checking Feature Contamination for {player_name}")
    print("=" * 70)
    
    player_dir = Path('data2') / player_name.replace(' ', '_')
    features_file = player_dir / f"{player_name.replace(' ', '_')}_features.csv"
    
    df = pd.read_csv(features_file)
    
    targets = ['PTS', 'OREB', 'DREB', 'AST', 'FTM', 'FG3M', 'FGM', 'STL', 'BLK', 'TOV']
    
    print("\nTest 3: Target Variable Contamination")
    print("-" * 70)
    
    issues = []
    for target in targets:
        # Find columns that contain the target name
        target_cols = [c for c in df.columns if target in c]
        
        # Filter out legitimate uses (rolling, career, trends)
        suspicious = [c for c in target_cols 
                     if not any(x in c for x in ['_L3_', '_L5_', '_L10_', '_L20_', 
                                                   'CAREER', 'LAST_', 'TREND'])]
        
        if suspicious and target in suspicious:
            suspicious.remove(target)  # Target itself is OK (it's excluded in training)
        
        if suspicious:
            issues.append((target, suspicious))
    
    if issues:
        print("⚠️  Found potentially problematic features:")
        for target, feats in issues:
            print(f"\n  {target}:")
            for f in feats:
                print(f"    - {f}")
        return False
    else:
        print("✅ No suspicious target-derived features found")
        return True

def check_categorical_encoding():
    """Check if all categorical variables are encoded"""
    print(f"\n🔍 Checking Categorical Encoding")
    print("=" * 70)
    
    player_dir = Path('data2/Kevin_Durant')
    features_file = player_dir / "Kevin_Durant_features.csv"
    
    df = pd.read_csv(features_file)
    
    print("\nTest 4: Categorical Variable Encoding")
    print("-" * 70)
    
    # Find object dtype columns (categorical)
    categorical = df.select_dtypes(include=['object']).columns.tolist()
    
    # Exclude legitimate text columns
    exclude_text = ['PLAYER_NAME', 'GAME_DATE', 'SEASON', 'MATCHUP', 
                    'PLAYER_TEAM', 'OPPONENT_TEAM', 'OPPONENT']
    
    categorical = [c for c in categorical if c not in exclude_text]
    
    if categorical:
        print(f"⚠️  Found {len(categorical)} unencoded categorical columns:")
        for col in categorical[:10]:  # Show first 10
            print(f"  - {col}")
        return False
    else:
        print("✅ All categorical variables properly encoded")
        return True

def main():
    print("🏀 NBA Data Leakage Checker")
    print("=" * 70)
    print()
    
    all_passed = True
    
    # Run all checks
    if not check_rolling_features():
        all_passed = False
    
    if not check_career_features():
        all_passed = False
    
    if not check_feature_contamination():
        all_passed = False
    
    if not check_categorical_encoding():
        all_passed = False
    
    print()
    print("=" * 70)
    if all_passed:
        print("✅ ALL CHECKS PASSED - No data leakage detected!")
        print("🚀 Ready for feature engineering")
    else:
        print("❌ ISSUES FOUND - Fix before regenerating features")
        print("⚠️  Current features have data leakage!")
    print("=" * 70)

if __name__ == "__main__":
    main()

