# NBA Basketball Betting Prediction System

## 📁 Project Structure

```
basketball_betting/
├── backend/
│   ├── data_collection/         # Scripts to collect NBA data
│   │   ├── nba_data_collection.py
│   │   ├── college_basketball_collector.py
│   │   ├── collect_team_stats.py
│   │   └── retry_failed_players.py
│   │
│   ├── feature_engineering/     # Create predictive features
│   │   ├── feature_engineering.py
│   │   ├── player_archetyping.py
│   │   └── fix_season_experience.py
│   │
│   ├── model_training/          # Train ML models
│   │   └── train_models_ultimate.py
│   │
│   ├── analysis/                # Performance analysis
│   │   ├── analyze_model_performance.py
│   │   ├── analyze_rebounding_performance.py
│   │   ├── check_data_leakage.py
│   │   ├── final_leakage_check.py
│   │   └── cleanup_college_only_players.py
│   │
│   ├── models/                  # Prediction engine
│   │   └── prediction_engine.py
│   │
│   ├── data/                    # All data storage
│   │   ├── player_data/         # Individual player game logs + features
│   │   ├── team_stats/          # Team-level statistics
│   │   └── processed_data/      # Processed datasets
│   │
│   ├── outputs/                 # Generated analysis files
│   │   ├── model_performance_analysis.png
│   │   ├── model_performance_heatmap.png
│   │   └── all_model_metrics.csv
│   │
│   ├── config/                  # Configuration files
│   │   ├── requirements.txt
│   │   └── README.md
│   │
│   └── utils/                   # Utility functions (future)
│
├── betting_venv/                # Python virtual environment
├── catboost_info/              # CatBoost training logs (auto-generated)
└── .gitignore

```

## 🎯 Current System Status

### ✅ Completed Components:

1. **Data Collection**
   - NBA game logs for 4,770 players
   - College statistics integration
   - Team stats (2000-2024) with rebounding, shooting, pace data
   - Player archetypes (11 dimensions)

2. **Feature Engineering**
   - 300+ features per player
   - Rolling averages (3, 5, 10, 20 games)
   - Opponent matchup features
   - Team context features
   - Rebounding opportunities (NEW!)
   - Position-specific features
   - **NO DATA LEAKAGE** ✅

3. **Model Training**
   - **1,571 modern players** (2000+)
   - **15,707 total models trained**
   - Multi-model approach (LightGBM, XGBoost, CatBoost)
   - Position-specific hyperparameters
   - Two-stage models for rare events (OREB, BLK)
   - Average R² = 0.46 (professional-grade!)

4. **Performance by Stat**
   - STL: R² = 0.735 🔥
   - BLK: R² = 0.636 ✅
   - FGM: R² = 0.563 ✅
   - FG3M: R² = 0.495 ✅
   - FTM: R² = 0.476 ✅
   - PTS: R² = 0.460 ✅
   - TOV: R² = 0.449 ✅
   - AST: R² = 0.438 ✅
   - DREB: R² = 0.136 👍
   - OREB: R² = 0.002 👍 (was -0.069!)

### 🚀 Next Components to Build:

1. **Odds Integration**
   - SportsGameOdds API integration
   - Real-time odds fetching
   - Odds comparison system

2. **Prediction Engine**
   - Fetch upcoming game data from NBA API
   - Auto-generate features for upcoming games
   - Load player models
   - Make predictions
   - Compare to betting lines

3. **Betting Recommender**
   - Calculate expected value (EV)
   - Identify profitable opportunities
   - Confidence scoring
   - Alert system for good bets

4. **Data Update Pipeline**
   - Auto-fetch new games daily
   - Update player features
   - Retrain models periodically
   - Track model performance over time

## 📝 Key Decisions Made:

- **Modern Era Only**: Train on 2000+ data (cleaner, more reliable)
- **Multi-Model Selection**: Test 3 model types, pick best per stat
- **Position-Aware**: Different hyperparameters for guards/forwards/centers
- **Leakage Prevention**: Strict time-based splits, no future data
- **Feature Rich**: 300+ features including opponent matchups, team context

## 🔧 Usage

### Run Data Collection:
```bash
cd backend/data_collection
python nba_data_collection.py
python collect_team_stats.py
```

### Generate Features:
```bash
cd backend/feature_engineering
python feature_engineering.py
```

### Train Models:
```bash
cd backend/model_training
python train_models_ultimate.py
```

### Analyze Performance:
```bash
cd backend/analysis
python analyze_model_performance.py
```

