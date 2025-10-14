#!/usr/bin/env python3
"""
Rebounding Model Analysis
=========================
Analyze why rebounding predictions are poor and identify improvements
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")


def analyze_rebounding_features():
    """Analyze what features exist for rebounding and what's missing"""
    
    print("🏀 REBOUNDING MODEL ANALYSIS")
    print("=" * 80)
    
    # Load a sample player to see available features
    sample_players = ['LeBron_James', 'Stephen_Curry', 'Joel_Embiid', 'Giannis_Antetokounmpo']
    
    for player in sample_players:
        features_file = f'player_data/{player}/{player}_features.csv'
        
        if not Path(features_file).exists():
            continue
        
        df = pd.read_csv(features_file)
        
        print(f"\n{'=' * 80}")
        print(f"Player: {player.replace('_', ' ')}")
        print(f"{'=' * 80}")
        
        # Check OREB and DREB distributions
        if 'OREB' in df.columns and 'DREB' in df.columns:
            print(f"\nOREB stats:")
            print(f"  Mean: {df['OREB'].mean():.2f}")
            print(f"  Std: {df['OREB'].std():.2f}")
            print(f"  Min/Max: {df['OREB'].min():.0f} / {df['OREB'].max():.0f}")
            print(f"  % of games with 0 OREB: {(df['OREB'] == 0).sum() / len(df) * 100:.1f}%")
            
            print(f"\nDREB stats:")
            print(f"  Mean: {df['DREB'].mean():.2f}")
            print(f"  Std: {df['DREB'].std():.2f}")
            print(f"  Min/Max: {df['DREB'].min():.0f} / {df['DREB'].max():.0f}")
            print(f"  % of games with 0 DREB: {(df['DREB'] == 0).sum() / len(df) * 100:.1f}%")
        
        # Check what rebounding-related features exist
        rebound_features = [col for col in df.columns if 'REB' in col or 'REBOUND' in col]
        print(f"\nRebounding-related features ({len(rebound_features)}):")
        for feat in sorted(rebound_features)[:20]:  # Show first 20
            if feat in df.columns:
                print(f"  - {feat}: mean={df[feat].mean():.2f}, std={df[feat].std():.2f}")
        
        # Check position/height features
        if 'height_inches' in df.columns:
            print(f"\nHeight: {df['height_inches'].iloc[0]:.0f} inches")
        
        if 'position_encoded' in df.columns:
            print(f"Position encoded: {df['position_encoded'].iloc[0]}")
        
        # Check archetype rebounding features
        archetype_cols = [col for col in df.columns if 'ARCHETYPE' in col and 'REBOUND' in col]
        if archetype_cols:
            print(f"\nArchetype rebounding features:")
            for col in archetype_cols[:5]:
                print(f"  - {col}")
        
        break  # Just analyze first available player in detail


def identify_missing_features():
    """Identify what features are missing that could help rebounding"""
    
    print("\n" + "=" * 80)
    print("🔍 MISSING FEATURES FOR REBOUNDING")
    print("=" * 80)
    
    missing_features = [
        "Opponent's rebounding rate (how many rebounds do they allow?)",
        "Team total rebounding context",
        "Opponent's pace (faster = more possessions = more rebounds)",
        "Opponent's shot volume (more misses = more rebounds)",
        "Player's minutes rolling average (more minutes = more opportunities)",
        "Team's starting lineup context (playing with good rebounders?)",
        "Opponent's size/height features",
        "Shot location data (corner 3s vs long 2s affect rebound positioning)",
        "Home court rebounding advantage",
        "Back-to-back fatigue effect on rebounding",
    ]
    
    print("\nPotential features to add:")
    for i, feat in enumerate(missing_features, 1):
        print(f"  {i}. {feat}")


def analyze_model_errors():
    """Analyze where the rebounding models are making the biggest errors"""
    
    print("\n" + "=" * 80)
    print("📊 REBOUNDING MODEL ERROR ANALYSIS")
    print("=" * 80)
    
    # Collect rebounding metrics
    data = []
    
    for player_dir in Path('player_data').iterdir():
        if not player_dir.is_dir():
            continue
        
        for stat in ['OREB', 'DREB']:
            metrics_file = player_dir / 'models' / f'{stat}_metrics.json'
            
            if not metrics_file.exists():
                continue
            
            with open(metrics_file, 'r') as f:
                metrics = json.load(f)
            
            # Also load player features to get context
            features_file = player_dir / f"{player_dir.name}_features.csv"
            if features_file.exists():
                df = pd.read_csv(features_file)
                avg_value = df[stat].mean()
                position = df['position_encoded'].iloc[0] if 'position_encoded' in df.columns else -1
                height = df['height_inches'].iloc[0] if 'height_inches' in df.columns else np.nan
            else:
                avg_value = metrics['target_mean']
                position = -1
                height = np.nan
            
            data.append({
                'player': player_dir.name.replace('_', ' '),
                'stat': stat,
                'r2': metrics['metrics']['R2'],
                'mae': metrics['metrics']['MAE'],
                'target_mean': metrics['target_mean'],
                'avg_value': avg_value,
                'position': position,
                'height': height,
                'model': metrics['best_model']
            })
    
    df = pd.DataFrame(data)
    
    # Analyze by stat
    for stat in ['OREB', 'DREB']:
        stat_df = df[df['stat'] == stat]
        
        print(f"\n{stat} Analysis:")
        print(f"  Average R²: {stat_df['r2'].mean():.3f}")
        print(f"  Average MAE: {stat_df['mae'].mean():.2f}")
        print(f"  Average target value: {stat_df['target_mean'].mean():.2f}")
        
        # Correlations
        print(f"\n  R² correlation with:")
        print(f"    - Target mean: {stat_df[['r2', 'target_mean']].corr().iloc[0, 1]:.3f}")
        print(f"    - Height: {stat_df[['r2', 'height']].dropna().corr().iloc[0, 1]:.3f}")
        
        # Players with good vs bad predictions
        good_r2 = stat_df[stat_df['r2'] > 0.5].sort_values('r2', ascending=False).head(5)
        bad_r2 = stat_df[stat_df['r2'] < 0.0].sort_values('r2').head(5)
        
        print(f"\n  Top 5 players with good {stat} predictions (R² > 0.5):")
        for _, row in good_r2.iterrows():
            print(f"    {row['player']:<25} R²={row['r2']:.3f}, MAE={row['mae']:.2f}, Avg={row['avg_value']:.1f}")
        
        print(f"\n  Top 5 players with poor {stat} predictions (R² < 0):")
        for _, row in bad_r2.iterrows():
            print(f"    {row['player']:<25} R²={row['r2']:.3f}, MAE={row['mae']:.2f}, Avg={row['avg_value']:.1f}")
    
    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Rebounding Model Performance Analysis', fontsize=14, fontweight='bold')
    
    # OREB R² vs Target Mean
    oreb_df = df[df['stat'] == 'OREB']
    axes[0, 0].scatter(oreb_df['target_mean'], oreb_df['r2'], alpha=0.5)
    axes[0, 0].axhline(y=0, color='r', linestyle='--', label='Baseline (R²=0)')
    axes[0, 0].set_xlabel('Average OREB per game')
    axes[0, 0].set_ylabel('R² Score')
    axes[0, 0].set_title('OREB: R² vs Average Value')
    axes[0, 0].legend()
    axes[0, 0].grid(alpha=0.3)
    
    # DREB R² vs Target Mean
    dreb_df = df[df['stat'] == 'DREB']
    axes[0, 1].scatter(dreb_df['target_mean'], dreb_df['r2'], alpha=0.5, color='orange')
    axes[0, 1].axhline(y=0, color='r', linestyle='--', label='Baseline (R²=0)')
    axes[0, 1].set_xlabel('Average DREB per game')
    axes[0, 1].set_ylabel('R² Score')
    axes[0, 1].set_title('DREB: R² vs Average Value')
    axes[0, 1].legend()
    axes[0, 1].grid(alpha=0.3)
    
    # OREB distribution
    axes[1, 0].hist(oreb_df['r2'], bins=30, alpha=0.7, edgecolor='black')
    axes[1, 0].axvline(oreb_df['r2'].mean(), color='r', linestyle='--', label=f'Mean: {oreb_df["r2"].mean():.3f}')
    axes[1, 0].set_xlabel('R² Score')
    axes[1, 0].set_ylabel('Number of Players')
    axes[1, 0].set_title('OREB R² Distribution')
    axes[1, 0].legend()
    axes[1, 0].grid(alpha=0.3)
    
    # DREB distribution
    axes[1, 1].hist(dreb_df['r2'], bins=30, alpha=0.7, color='orange', edgecolor='black')
    axes[1, 1].axvline(dreb_df['r2'].mean(), color='r', linestyle='--', label=f'Mean: {dreb_df["r2"].mean():.3f}')
    axes[1, 1].set_xlabel('R² Score')
    axes[1, 1].set_ylabel('Number of Players')
    axes[1, 1].set_title('DREB R² Distribution')
    axes[1, 1].legend()
    axes[1, 1].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('rebounding_analysis.png', dpi=300, bbox_inches='tight')
    print(f"\n✅ Saved: rebounding_analysis.png")
    plt.close()


def suggest_improvements():
    """Suggest concrete improvements for rebounding models"""
    
    print("\n" + "=" * 80)
    print("💡 SUGGESTED IMPROVEMENTS FOR REBOUNDING MODELS")
    print("=" * 80)
    
    improvements = {
        "1. Add Opponent Rebounding Features": [
            "Merge opponent's defensive rebounding rate from team stats",
            "Add opponent's offensive rebounding rate",
            "Calculate opponent's total rebounds allowed per game",
        ],
        
        "2. Add Team Context Features": [
            "Team's total rebounds per game (playing with Jokic vs not)",
            "Team's rebounding rank in the league",
            "Percentage of team rebounds the player typically gets",
        ],
        
        "3. Add Opportunity-Based Features": [
            "Estimated rebounding opportunities (based on FGA from both teams)",
            "Team pace * opponent pace (more possessions = more opportunities)",
            "Opponent's FG% (lower % = more misses = more defensive rebounds)",
        ],
        
        "4. Improve Position/Size Features": [
            "Use height more prominently in rebounding predictions",
            "Create position-specific models (guards vs forwards vs centers)",
            "Add wingspan data if available",
        ],
        
        "5. Add Rolling Opportunity Features": [
            "Rolling average of minutes played (more minutes = more opportunities)",
            "Rolling average of team's total shot attempts",
            "Rolling average of opponent's total shot attempts",
        ],
        
        "6. Hyperparameter Tuning": [
            "Current models use default hyperparameters",
            "Could use GridSearchCV or Optuna for optimization",
            "Try different max_depth, learning_rate, n_estimators",
        ],
        
        "7. Try Different Approaches": [
            "Poisson regression (rebounds are count data)",
            "Quantile regression (predict distribution, not just mean)",
            "Ensemble of position-specific models",
        ],
    }
    
    for improvement, details in improvements.items():
        print(f"\n{improvement}")
        print("-" * 80)
        for detail in details:
            print(f"  • {detail}")
    
    print("\n" + "=" * 80)
    print("🎯 RECOMMENDED NEXT STEPS")
    print("=" * 80)
    print("""
1. START WITH LOW-HANGING FRUIT:
   - Add opponent rebounding rate features (we already have team stats!)
   - Add team pace and opponent pace interactions
   - Use height/position features more prominently

2. FEATURE ENGINEERING (Quick Win):
   - Calculate rebounding opportunities = (Team FGA + Opp FGA) * (1 - League Avg FG%)
   - Add player's share of team rebounds as a feature
   - Add minutes rolling average (already have MIN_L3_TOTAL, MIN_L5_TOTAL)

3. LONGER-TERM IMPROVEMENTS:
   - Position-specific models (train separate models for guards/forwards/centers)
   - Hyperparameter tuning with cross-validation
   - Try Poisson regression for count data

4. VALIDATION:
   - After changes, re-run final_leakage_check.py to ensure no leakage
   - Compare new R² scores to current baseline
   - Focus on improving players who average > 2 rebounds/game first
    """)


def main():
    """Main execution"""
    analyze_rebounding_features()
    identify_missing_features()
    analyze_model_errors()
    suggest_improvements()


if __name__ == "__main__":
    main()

