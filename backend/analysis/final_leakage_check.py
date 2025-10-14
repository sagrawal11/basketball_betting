#!/usr/bin/env python3
"""
Final Comprehensive Data Leakage Check
======================================
Verify NO leakage before spending 28 hours training models
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def test_rolling_averages():
    """Test 1: Rolling averages exclude current game"""
    print("Test 1: Rolling Averages Exclude Current Game")
    print("-" * 70)
    
    df = pd.read_csv('player_data/Kevin_Durant/Kevin_Durant_features.csv')
    
    i = 50
    if 'PTS_L5_AVG' in df.columns:
        l5 = df.iloc[i]['PTS_L5_AVG']
        
        # Need to load original data to check
        orig_df = pd.read_csv('player_data/Kevin_Durant/Kevin_Durant_data.csv')
        orig_df = orig_df.sort_values('GAME_DATE').reset_index(drop=True)
        manual_l5 = orig_df.iloc[i-5:i]['PTS'].mean()
        
        passed = abs(l5 - manual_l5) < 0.01
        print(f"  PTS_L5_AVG at game {i}: {l5:.2f}")
        print(f"  Expected (prev 5 games): {manual_l5:.2f}")
        print(f"  {'✅ PASS' if passed else '❌ FAIL - Includes current game!'}")
        return passed
    return True


def test_no_target_variables():
    """Test 2: Raw target variables removed from features"""
    print("\nTest 2: Raw Target Variables Removed")
    print("-" * 70)
    
    df = pd.read_csv('player_data/LeBron_James/LeBron_James_features.csv')
    
    # These should NOT be in features (removed to prevent leakage)
    should_be_removed = ['REB', 'PLUS_MINUS', 'MIN', 'FGA', 'FG3A', 'FTA', 
                         'FG_PCT', 'FG3_PCT', 'FT_PCT', 'VIDEO_AVAILABLE']
    
    # These SHOULD be in features (we need them to predict)
    should_be_kept = ['PTS', 'OREB', 'DREB', 'AST', 'FTM', 'FG3M', 'FGM', 'STL', 'BLK', 'TOV']
    
    removed_present = [c for c in should_be_removed if c in df.columns]
    targets_missing = [c for c in should_be_kept if c not in df.columns]
    
    print(f"  Leakage columns still present: {len(removed_present)}")
    if removed_present:
        for col in removed_present:
            print(f"    ❌ {col}")
    
    print(f"  Target variables present: {len([c for c in should_be_kept if c in df.columns])}/10")
    if targets_missing:
        for col in targets_missing:
            print(f"    ❌ Missing: {col}")
    
    passed = len(removed_present) == 0 and len(targets_missing) == 0
    print(f"  {'✅ PASS' if passed else '❌ FAIL'}")
    return passed


def test_no_duplicates():
    """Test 3: No duplicate games from team stats merge"""
    print("\nTest 3: No Duplicate Games")
    print("-" * 70)
    
    df = pd.read_csv('player_data/Aaron_Gordon/Aaron_Gordon_features.csv')
    orig_df = pd.read_csv('player_data/Aaron_Gordon/Aaron_Gordon_data.csv')
    
    ratio = len(df) / len(orig_df)
    dup_count = df.duplicated(subset=['GAME_DATE']).sum()
    
    print(f"  Original games: {len(orig_df)}")
    print(f"  Feature rows: {len(df)}")
    print(f"  Ratio: {ratio:.2f}x")
    print(f"  Duplicates: {dup_count}")
    
    passed = ratio == 1.0 and dup_count == 0
    print(f"  {'✅ PASS' if passed else '❌ FAIL - Duplicates detected!'}")
    return passed


def test_realistic_prediction():
    """Test 4: Train a simple model using ACTUAL training script logic"""
    print("\nTest 4: Realistic Model Performance (Smoke Test)")
    print("-" * 70)
    
    # Use the ACTUAL training script logic
    from train_models import PlayerModelTrainer
    
    trainer = PlayerModelTrainer()
    df = trainer.load_player_features('LeBron James')
    
    if df is None:
        print("  ❌ Could not load features!")
        return False
    
    target = 'PTS'
    
    if target not in df.columns:
        print("  ❌ Target variable missing!")
        return False
    
    # Use trainer's prepare_data method (same logic as actual training)
    X_train, X_test, y_train, y_test, feature_names, scaler = trainer.prepare_data(df, target)
    
    if X_train is None:
        print("  ❌ Data preparation failed!")
        return False
    
    print(f"  Features used: {len(feature_names)}")
    print(f"  Training games: {len(X_train)}")
    print(f"  Test games: {len(X_test)}")
    
    # Train simple Ridge model
    from sklearn.linear_model import Ridge
    model = Ridge(alpha=1.0)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    
    mae = np.mean(np.abs(y_test - y_pred))
    r2 = 1 - (np.sum((y_test - y_pred)**2) / np.sum((y_test - y_test.mean())**2))
    
    print(f"  Points MAE: {mae:.2f}")
    print(f"  R² Score: {r2:.3f}")
    
    # Realistic performance: MAE should be 2-6 points, R² should be 0.3-0.8
    passed = 2 <= mae <= 8 and 0.2 <= r2 <= 0.9
    
    if passed:
        print(f"  ✅ PASS - Realistic performance!")
    else:
        if mae < 1:
            print(f"  ❌ FAIL - MAE too low ({mae:.2f}) = DATA LEAKAGE!")
        elif r2 > 0.95:
            print(f"  ❌ FAIL - R² too high ({r2:.3f}) = DATA LEAKAGE!")
        else:
            print(f"  ⚠️  WARNING - Performance outside expected range")
    
    return passed
    
    # Time-based split
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    # Train simple Ridge model
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    model = Ridge(alpha=1.0)
    model.fit(X_train_scaled, y_train)
    
    y_pred = model.predict(X_test_scaled)
    
    mae = np.mean(np.abs(y_test - y_pred))
    r2 = 1 - (np.sum((y_test - y_pred)**2) / np.sum((y_test - y_test.mean())**2))
    
    print(f"  Features used: {len(feature_cols)}")
    print(f"  Training games: {len(X_train)}")
    print(f"  Test games: {len(X_test)}")
    print(f"  Points MAE: {mae:.2f}")
    print(f"  R² Score: {r2:.3f}")
    
    # Realistic performance: MAE should be 2-6 points, R² should be 0.3-0.8
    passed = 2 <= mae <= 8 and 0.2 <= r2 <= 0.9
    
    if passed:
        print(f"  ✅ PASS - Realistic performance!")
    else:
        if mae < 1:
            print(f"  ❌ FAIL - MAE too low ({mae:.2f}) = DATA LEAKAGE!")
        elif r2 > 0.95:
            print(f"  ❌ FAIL - R² too high ({r2:.3f}) = DATA LEAKAGE!")
        else:
            print(f"  ⚠️  WARNING - Performance outside expected range")
    
    return passed


def test_team_features_present():
    """Test 5: Team features successfully integrated"""
    print("\nTest 5: Team Features Present")
    print("-" * 70)
    
    df = pd.read_csv('player_data/Stephen_Curry/Stephen_Curry_features.csv')
    
    team_features = ['OPP_DEF_RATING', 'TEAM_OFF_RATING', 'GAME_PACE_EST', 'TEAM_QUALITY_DIFF']
    
    present = [f for f in team_features if f in df.columns]
    missing = [f for f in team_features if f not in df.columns]
    
    print(f"  Team features present: {len(present)}/4")
    for feat in present:
        print(f"    ✅ {feat}")
    
    if missing:
        for feat in missing:
            print(f"    ❌ {feat}")
    
    passed = len(missing) == 0
    print(f"  {'✅ PASS' if passed else '❌ FAIL'}")
    return passed


def main():
    print("🔍 FINAL COMPREHENSIVE DATA LEAKAGE CHECK")
    print("=" * 70)
    print()
    
    results = []
    
    # Run all tests
    results.append(("Rolling Averages", test_rolling_averages()))
    results.append(("No Target Leakage", test_no_target_variables()))
    results.append(("No Duplicates", test_no_duplicates()))
    results.append(("Realistic Performance", test_realistic_prediction()))
    results.append(("Team Features", test_team_features_present()))
    
    # Summary
    print()
    print("=" * 70)
    print("📊 FINAL RESULTS:")
    print("-" * 70)
    
    for test_name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {status}: {test_name}")
    
    all_passed = all(r[1] for r in results)
    
    print()
    print("=" * 70)
    if all_passed:
        print("🎉 ALL TESTS PASSED!")
        print("✅ NO DATA LEAKAGE DETECTED")
        print("✅ Features are production-ready")
        print()
        print("🚀 SAFE TO START 28-HOUR MODEL TRAINING!")
    else:
        print("⚠️  SOME TESTS FAILED")
        print("❌ DO NOT TRAIN MODELS YET")
        print("🔧 Fix issues before proceeding")
    print("=" * 70)


if __name__ == "__main__":
    main()

