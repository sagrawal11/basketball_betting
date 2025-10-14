#!/usr/bin/env python3
"""
Model Performance Analysis
==========================
Analyze and visualize performance of all trained models
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict

sns.set_style("whitegrid")

class ModelPerformanceAnalyzer:
    """Analyze model performance across all players"""
    
    def __init__(self, data_dir: str = "player_data"):
        self.data_dir = Path(data_dir)
        self.target_variables = [
            'PTS', 'OREB', 'DREB', 'AST', 'FTM', 'FG3M', 'FGM', 'STL', 'BLK', 'TOV'
        ]
    
    def collect_all_metrics(self):
        """Collect metrics from all trained models"""
        all_metrics = []
        
        # Find all model directories
        for player_dir in self.data_dir.iterdir():
            if not player_dir.is_dir() or player_dir.name == 'college':
                continue
            
            models_dir = player_dir / 'models'
            summary_file = models_dir / 'model_summary.json'
            
            if not summary_file.exists():
                continue
            
            # Read summary
            with open(summary_file, 'r') as f:
                summary = json.load(f)
            
            player_name = summary['player_name']
            total_games = summary.get('total_games', 0)
            
            # Read detailed metrics for each stat
            for stat in self.target_variables:
                metrics_file = models_dir / f'{stat}_metrics.json'
                
                if not metrics_file.exists():
                    continue
                
                with open(metrics_file, 'r') as f:
                    metrics_data = json.load(f)
                
                # Extract key information
                all_metrics.append({
                    'player': player_name,
                    'stat': stat,
                    'best_model': metrics_data['best_model'],
                    'mae': metrics_data['metrics']['MAE'],
                    'rmse': metrics_data['metrics']['RMSE'],
                    'r2': metrics_data['metrics']['R2'],
                    'mape': metrics_data['metrics']['MAPE'],
                    'n_train': metrics_data['n_train'],
                    'n_test': metrics_data['n_test'],
                    'target_mean': metrics_data['target_mean'],
                    'target_std': metrics_data['target_std'],
                    'total_games': total_games
                })
        
        return pd.DataFrame(all_metrics)
    
    def print_summary_stats(self, df):
        """Print summary statistics"""
        print("\n" + "=" * 80)
        print("📊 MODEL PERFORMANCE SUMMARY")
        print("=" * 80)
        
        # Overall stats
        print(f"\n✅ Total Players with Trained Models: {df['player'].nunique()}")
        print(f"✅ Total Models Trained: {len(df)}")
        print(f"✅ Average Games per Player: {df.groupby('player')['total_games'].first().mean():.0f}")
        print(f"✅ Average Training Size: {df['n_train'].mean():.0f} games")
        print(f"✅ Average Test Size: {df['n_test'].mean():.0f} games")
        
        # Performance by stat
        print("\n" + "-" * 80)
        print("📈 PERFORMANCE BY STATISTIC")
        print("-" * 80)
        print(f"{'Stat':<8} {'Avg MAE':<10} {'Avg RMSE':<10} {'Avg R²':<10} {'Avg MAPE':<10} {'Avg Value':<12}")
        print("-" * 80)
        
        for stat in self.target_variables:
            stat_data = df[df['stat'] == stat]
            if len(stat_data) == 0:
                continue
            
            avg_mae = stat_data['mae'].mean()
            avg_rmse = stat_data['rmse'].mean()
            avg_r2 = stat_data['r2'].mean()
            avg_mape = stat_data['mape'].mean()
            avg_value = stat_data['target_mean'].mean()
            
            print(f"{stat:<8} {avg_mae:<10.2f} {avg_rmse:<10.2f} {avg_r2:<10.3f} {avg_mape:<10.1f}% {avg_value:<12.2f}")
        
        # Model type performance
        print("\n" + "-" * 80)
        print("🤖 MODEL TYPE USAGE")
        print("-" * 80)
        
        model_counts = df['best_model'].value_counts()
        for model, count in model_counts.items():
            pct = (count / len(df)) * 100
            print(f"  {model:<20} {count:>4} models ({pct:>5.1f}%)")
        
        # Data leakage check
        print("\n" + "-" * 80)
        print("🔍 DATA LEAKAGE CHECK")
        print("-" * 80)
        
        suspicious_models = df[(df['mae'] < 0.5) & (df['target_mean'] > 5)]
        if len(suspicious_models) > 0:
            print(f"⚠️  WARNING: {len(suspicious_models)} models with suspiciously low MAE!")
            for _, row in suspicious_models.head(10).iterrows():
                print(f"  - {row['player']} - {row['stat']}: MAE={row['mae']:.2f}, Mean={row['target_mean']:.2f}")
        else:
            print("✅ No suspicious models detected (all MAE values are realistic)")
        
        high_r2_models = df[(df['r2'] > 0.95)]
        if len(high_r2_models) > 0:
            print(f"\n⚠️  WARNING: {len(high_r2_models)} models with suspiciously high R²!")
            for _, row in high_r2_models.head(10).iterrows():
                print(f"  - {row['player']} - {row['stat']}: R²={row['r2']:.3f}")
        else:
            print("✅ No overly perfect models (no R² > 0.95)")
        
        # Top performers
        print("\n" + "-" * 80)
        print("🌟 TOP PERFORMING MODELS (by R²)")
        print("-" * 80)
        
        top_models = df.nlargest(10, 'r2')
        for idx, row in top_models.iterrows():
            print(f"  {row['player']:<25} {row['stat']:<6} R²={row['r2']:.3f}, MAE={row['mae']:.2f}, Model={row['best_model']}")
        
        print("\n" + "=" * 80)
    
    def create_visualizations(self, df):
        """Create performance visualizations"""
        print("\n📊 Creating visualizations...")
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Model Performance Analysis', fontsize=16, fontweight='bold')
        
        # 1. MAE by Statistic
        ax1 = axes[0, 0]
        stat_mae = df.groupby('stat')['mae'].agg(['mean', 'std']).reset_index()
        stat_mae = stat_mae.sort_values('mean')
        ax1.barh(stat_mae['stat'], stat_mae['mean'], xerr=stat_mae['std'], color='skyblue', edgecolor='black')
        ax1.set_xlabel('Mean Absolute Error (MAE)')
        ax1.set_title('Average MAE by Statistic')
        ax1.grid(axis='x', alpha=0.3)
        
        # 2. R² by Statistic
        ax2 = axes[0, 1]
        stat_r2 = df.groupby('stat')['r2'].agg(['mean', 'std']).reset_index()
        stat_r2 = stat_r2.sort_values('mean', ascending=False)
        ax2.barh(stat_r2['stat'], stat_r2['mean'], xerr=stat_r2['std'], color='lightgreen', edgecolor='black')
        ax2.set_xlabel('R² Score')
        ax2.set_title('Average R² by Statistic')
        ax2.set_xlim(0, 1)
        ax2.grid(axis='x', alpha=0.3)
        
        # 3. Model Type Distribution
        ax3 = axes[0, 2]
        model_counts = df['best_model'].value_counts()
        colors = plt.cm.Set3(np.linspace(0, 1, len(model_counts)))
        ax3.pie(model_counts.values, labels=model_counts.index, autopct='%1.1f%%', colors=colors)
        ax3.set_title('Best Model Distribution')
        
        # 4. MAE vs Target Mean (normalized error)
        ax4 = axes[1, 0]
        df['normalized_mae'] = df['mae'] / df['target_mean']
        for stat in self.target_variables:
            stat_data = df[df['stat'] == stat]
            ax4.scatter(stat_data['target_mean'], stat_data['mae'], label=stat, alpha=0.6, s=50)
        ax4.set_xlabel('Target Mean Value')
        ax4.set_ylabel('MAE')
        ax4.set_title('MAE vs Target Mean')
        ax4.legend(loc='upper left', fontsize=8)
        ax4.grid(alpha=0.3)
        
        # 5. R² Distribution
        ax5 = axes[1, 1]
        ax5.hist(df['r2'], bins=30, color='purple', alpha=0.7, edgecolor='black')
        ax5.axvline(df['r2'].mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {df["r2"].mean():.3f}')
        ax5.axvline(df['r2'].median(), color='green', linestyle='--', linewidth=2, label=f'Median: {df["r2"].median():.3f}')
        ax5.set_xlabel('R² Score')
        ax5.set_ylabel('Frequency')
        ax5.set_title('Distribution of R² Scores')
        ax5.legend()
        ax5.grid(alpha=0.3)
        
        # 6. MAPE by Statistic
        ax6 = axes[1, 2]
        stat_mape = df.groupby('stat')['mape'].agg(['mean', 'std']).reset_index()
        stat_mape = stat_mape.sort_values('mean')
        ax6.barh(stat_mape['stat'], stat_mape['mean'], xerr=stat_mape['std'], color='coral', edgecolor='black')
        ax6.set_xlabel('Mean Absolute Percentage Error (MAPE)')
        ax6.set_title('Average MAPE by Statistic')
        ax6.grid(axis='x', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('model_performance_analysis.png', dpi=300, bbox_inches='tight')
        print("✅ Saved: model_performance_analysis.png")
        
        # Additional plot: Performance by model type and statistic
        fig2, ax = plt.subplots(figsize=(14, 8))
        
        # Heatmap of average R² by (stat, model_type)
        pivot_data = df.pivot_table(values='r2', index='stat', columns='best_model', aggfunc='mean')
        sns.heatmap(pivot_data, annot=True, fmt='.3f', cmap='YlGnBu', ax=ax, cbar_kws={'label': 'R² Score'})
        ax.set_title('Average R² Score by Statistic and Model Type', fontsize=14, fontweight='bold')
        ax.set_xlabel('Model Type')
        ax.set_ylabel('Statistic')
        
        plt.tight_layout()
        plt.savefig('model_performance_heatmap.png', dpi=300, bbox_inches='tight')
        print("✅ Saved: model_performance_heatmap.png")
        
        plt.close('all')
    
    def analyze_star_players(self, df):
        """Analyze performance for star players"""
        print("\n" + "=" * 80)
        print("⭐ STAR PLAYER MODEL PERFORMANCE")
        print("=" * 80)
        
        star_players = [
            'LeBron James', 'Stephen Curry', 'Kevin Durant', 'James Harden',
            'Giannis Antetokounmpo', 'Luka Dončić', 'Joel Embiid', 'Nikola Jokic',
            'Kawhi Leonard', 'Damian Lillard', 'Jayson Tatum', 'Jimmy Butler'
        ]
        
        for player in star_players:
            player_data = df[df['player'] == player]
            
            if len(player_data) == 0:
                continue
            
            print(f"\n{player}")
            print("-" * 80)
            print(f"{'Stat':<8} {'Model':<18} {'MAE':<8} {'RMSE':<8} {'R²':<8} {'MAPE':<10}")
            print("-" * 80)
            
            for _, row in player_data.iterrows():
                print(f"{row['stat']:<8} {row['best_model']:<18} {row['mae']:<8.2f} {row['rmse']:<8.2f} {row['r2']:<8.3f} {row['mape']:<10.1f}%")
            
            avg_r2 = player_data['r2'].mean()
            avg_mae = player_data['mae'].mean()
            print("-" * 80)
            print(f"Average R²: {avg_r2:.3f}, Average MAE: {avg_mae:.2f}")
        
        print("\n" + "=" * 80)
    
    def export_to_csv(self, df):
        """Export metrics to CSV for further analysis"""
        output_file = 'all_model_metrics.csv'
        df.to_csv(output_file, index=False)
        print(f"\n💾 Saved all metrics to: {output_file}")


def main():
    """Main execution"""
    print("🏀 NBA Model Performance Analyzer")
    print("=" * 80)
    
    analyzer = ModelPerformanceAnalyzer()
    
    print("\n📥 Collecting metrics from all trained models...")
    df = analyzer.collect_all_metrics()
    
    if len(df) == 0:
        print("❌ No trained models found!")
        return
    
    # Print summary statistics
    analyzer.print_summary_stats(df)
    
    # Analyze star players
    analyzer.analyze_star_players(df)
    
    # Create visualizations
    analyzer.create_visualizations(df)
    
    # Export to CSV
    analyzer.export_to_csv(df)
    
    print("\n✅ Analysis complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()

