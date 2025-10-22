# Prediction Fine-Tuning & Historical Tracking

This folder contains scripts for analyzing NBA game patterns and tracking prediction accuracy over time.

## 📁 Files

### Core Scripts

1. **`analyze_scoring_patterns.py`**
   - Analyzes 2024-25 season games to learn real NBA scoring distributions
   - Determines how scoring is split between starters, rotation, and bench
   - Outputs: `scoring_parameters.json` with learned formulas
   - Run once per season or when you want to update scoring models

2. **`daily_update.py`** ⭐ **Run Daily (or whenever)**
   - **Smart backfill:** Automatically catches up on all missed days since last run
   - Collects new game data from completed games
   - Updates all player CSV files with latest games
   - Regenerates features for updated players
   - Updates player archetypes
   - Retrains models with fresh data
   - Updates historical predictions with actual results
   - **Run daily, every 2 days, or whenever - it auto-catches up!**

3. **`prediction_storage.py`**
   - Helper module for saving/loading predictions
   - Handles JSON storage and CSV exports
   - Calculates accuracy metrics

### Data Files

- **`predictions/`** - Saved predictions and results
  - `*.json` - Individual game predictions with actual results
  - `all_predictions.csv` - All predictions in CSV format for analysis
  
- **`scoring_parameters.json`** - Learned NBA game patterns
  - Team scoring averages
  - Top 5/8 player percentages
  - Bench contribution estimates

## 🔄 Daily Workflow

### Every Game Day:

**Before Games (Automatic):**
1. Web app auto-generates predictions when you load the Games page
2. Predictions saved to `predictions/GAME_ID.json`

**After Games Finish:**
```bash
# Run this once - it handles EVERYTHING
python backend/prediction_fine_tuning/daily_update.py
```

### What It Does (Fully Automated):

**🎯 Smart Multi-Day Backfill:**
- Checks from last update date through today
- Catches up on ALL missed days automatically
- Skip a day or two? No problem - it handles it!

**📥 Data Collection:**
- Fetches actual scores & player stats from NBA API
- Appends new games to player CSV files

**🔧 Feature Engineering:**
- Regenerates all features for updated players
- Includes rolling averages, trends, opponent context

**🎭 Player Archetyping:**
- Updates player role classifications
- Adjusts for playing style changes

**🤖 Model Retraining:**
- Retrains models with latest game data
- Ensures predictions use most recent patterns

**📊 Historical Tracking:**
- Compares predictions vs actual results
- Calculates comprehensive accuracy metrics
- Updates History page data

### When to Run:

**Flexible Schedule:**
- **Daily** (recommended): Keep everything fresh
- **Every 2-3 days**: Script catches up automatically
- **After big game nights**: See your accuracy immediately

**Best Time:**
- **Morning after games** (9-10 AM): All NBA API data ready
- **Or anytime**: Script only processes finished games

## 📊 Accuracy Metrics Explained

### Overall Stat Accuracy
Comprehensive metric combining:
- **Team Score (30%)**: How close we predicted final scores
- **Player Points (40%)**: Individual scoring accuracy
- **Other Stats (30%)**: Rebounds, assists, steals, blocks

**Example:**
```
Overall: 60.2%
├─ Team Score: 93.1% (predicted 266 total, actual 249)
├─ Player Points: 54.5% (avg per-player accuracy)
└─ Other Stats: 34.8% (rebounds, assists, etc.)
```

### Avg Points Diff
Average error in points prediction per player
- `±5.8` means we're typically off by 5.8 points per player
- Lower is better

## 🎯 Historical Data

All predictions are stored in:
- **JSON files**: Full game details with player stats
- **CSV file**: Flattened for easy Excel/Python analysis
- **History page**: Beautiful UI showing predicted vs actual

## 💡 Tips

1. Run `update_history.py` once per day (after games finish)
2. Re-run `analyze_scoring_patterns.py` at end of season to update scoring formulas
3. Check `predictions/all_predictions.csv` for detailed analysis
4. History page auto-refreshes when you reload it

