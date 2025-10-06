# 🏀 NBA Basketball Betting Analytics Engine

A comprehensive sports betting analytics system that predicts over/under stats for NBA players using historical performance data and college basketball statistics for rookie archetyping.

## 📊 Project Overview

This system collects and analyzes:
- **NBA Game-by-Game Data**: Complete game logs for 4,954 NBA players (1990-2025)
- **College Basketball Data**: College statistics for 1,333+ NBA players (2005-2024)
- **Advanced Metrics**: PER, True Shooting %, Usage Rate, Game Score, and more
- **Contextual Features**: Home/away, rest days, opponent strength, time factors

## 🚀 Key Features

### NBA Data Collection
- ✅ **1.3M+ games** collected from NBA API
- ✅ **Complete player coverage** from 1990-2025
- ✅ **Advanced metrics** calculated for each game
- ✅ **Contextual features** (home/away, rest, opponent, etc.)
- ✅ **Checkpointing system** for reliable large-scale collection
- ✅ **Rate limiting** to respect API constraints

### College Basketball Data Collection
- ✅ **1,333+ players** with both college and NBA data
- ✅ **Sports Reference** web scraping for college stats
- ✅ **Comprehensive stats**: PPG, RPG, APG, shooting percentages, PER
- ✅ **Rookie archetyping** capability for predicting draft prospects

### Prediction Models (Coming Soon)
- 🔄 Player performance prediction models
- 🔄 Rookie NBA translation models based on college performance
- 🔄 Over/under betting recommendations

## 📁 Project Structure

```
basketball_betting/
├── nba_data_collection.py          # Main NBA data collector
├── college_basketball_collector.py # College data scraper
├── cleanup_college_only_players.py # Data cleanup utility
├── retry_failed_players.py         # Retry failed NBA collections
├── data2/                           # Player data storage
│   ├── Kevin_Durant/
│   │   ├── Kevin_Durant_data.csv
│   │   └── Kevin_Durant_college_data.csv
│   └── college/                     # Master college data
│       └── nba_rookies_college_data_2005_2024.csv
└── nba/
    ├── data/
    │   └── NBA-COMPLETE-playerlist.csv
    └── scripts/
        └── nba_data_collector.py
```

## 🛠️ Setup

### Prerequisites
- Python 3.8+
- Virtual environment (recommended)

### Installation

1. Clone the repository:
```bash
git clone <your-repo-url>
cd basketball_betting
```

2. Create and activate virtual environment:
```bash
python -m venv betting_venv
source betting_venv/bin/activate  # On Windows: betting_venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## 📊 Data Collection

### NBA Game Data

Collect NBA game-by-game data for all players:

```bash
python nba_data_collection.py
```

Features:
- Automatic checkpointing every 10 players
- Resume capability if interrupted
- Rate limiting (10 seconds between requests)
- Comprehensive error handling

Output: Individual CSV files for each player in `data2/[Player_Name]/`

### College Basketball Data

Collect college statistics for NBA players:

```bash
python college_basketball_collector.py
```

Options:
1. Test with single player (Kevin Durant)
2. Collect NBA rookies college data (2005-2024)
3. Collect current draft prospects
4. Create rookie archetypes

Output: 
- Individual files: `data2/[Player_Name]/[Player_Name]_college_data.csv`
- Master file: `data2/college/nba_rookies_college_data_2005_2024.csv`

## 📈 Data Schema

### NBA Game Data
- Basic stats: PTS, REB, AST, STL, BLK, FG%, 3P%, FT%
- Advanced metrics: TS%, PER, GAME_SCORE, PIE_EST, USG_RATE_EST
- Context: IS_HOME, OPPONENT, DAYS_SINCE_LAST_GAME, REST_FACTOR
- Temporal: GAME_DATE, GAME_DAY_OF_WEEK, GAME_MONTH, GAME_NUMBER_IN_SEASON

### College Data
- Basic stats: PPG, RPG, APG, FG%, 3P%, FT%
- Advanced: PER, Usage Rate, True Shooting %
- Context: College, Conference, Seasons played, Draft year

## 📊 Dataset Statistics

- **Total NBA Players**: 6,560
- **Players with NBA data**: 4,954
- **Players with college data**: 1,333
- **Players with both datasets**: 1,333
- **Total NBA games collected**: 1,300,000+
- **Data collection period**: 1990-2025

## 🎯 Use Cases

1. **Over/Under Betting**: Predict player performance for betting markets
2. **Rookie Projection**: Predict NBA success based on college performance
3. **Player Analysis**: Deep dive into player performance patterns
4. **Matchup Analysis**: Analyze player performance vs specific opponents
5. **Rest Impact**: Study how rest days affect performance

## 🔧 Utilities

### Cleanup College-Only Players
Remove college data for players who never played NBA games:

```bash
python cleanup_college_only_players.py
```

### Retry Failed Collections
Retry data collection for players who failed initially:

```bash
python retry_failed_players.py
```

## 🚧 Roadmap

- [ ] Build prediction models for player performance
- [ ] Create rookie archetype classification system
- [ ] Develop betting recommendation engine
- [ ] Add real-time game data integration
- [ ] Build web interface for predictions
- [ ] Add injury data integration

## 📝 Notes

- **Data Sources**: NBA API, Sports Reference
- **Rate Limiting**: Built-in delays to respect API limits
- **Checkpointing**: All collection scripts support resume functionality
- **Data Privacy**: No personal player information stored

## ⚠️ Disclaimer

This project is for educational and research purposes only. Always bet responsibly and within legal boundaries.

## 📄 License

MIT License - Feel free to use for your own projects!

## 🤝 Contributing

Contributions welcome! Feel free to open issues or submit pull requests.

---

**Built with ❤️ for basketball analytics and data science**

