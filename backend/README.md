# NBA prediction backend

Python pipeline: local **player CSVs** + **Kaggle team boxscores** → parquets → `feature_engine` → global LightGBM → Flask API.

**There is no `nba_api` dependency** — team data is built from files under `data/Kaggle NBA Team Data/`.

## Data layout

| Path | Role |
|------|------|
| `config/paths.py` | `DATA_DIR`, `TEAM_STATS_CSV`, `KAGGLE_TEAM_GAMES_PARQUET`, artifacts |
| `data/player_data/<Player>/<Player>_data.csv` | Raw game logs |
| `data/Kaggle NBA Team Data/processed/teams_boxscores.csv` | Kaggle game-level team box (source CSV) |
| `data/processed/kaggle_team_games.parquet` | Same data as parquet (canonical for `KG_PREV_*` features) |
| `data/team_stats/all_team_stats.{parquet,csv}` | Team × season pace & ratings |
| `data/auxiliary/` | DvP / injuries parquets (optional) |
| `models/artifacts/global/` | Trained models + `feature_columns.json` |

## Kaggle → parquets

```bash
cd backend && python data_collection/build_team_stats_from_kaggle.py
```

Writes game-level parquet, team×season parquet + CSV, and `team_stats_summary.json`.

Fallback without Kaggle: `python data_collection/build_team_stats_from_player_logs.py`

## Features & training

```bash
python model_training/train_global.py
```

`feature_engine.py` loads team stats from **`all_team_stats.parquet`** when present (else CSV), and adds **prior-game** Kaggle columns `KG_PREV_*` / `KG_OPP_PREV_*` from `kaggle_team_games.parquet`.

## API

```bash
python web/app.py
```

Local-data mode: no live NBA scoreboard. Use `POST /api/predict` or `GET /api/game/<id>/players?home=GSW&away=BOS`.

## Nightly job

```bash
python pipeline/nightly.py
```

Rebuilds Kaggle-derived team files when the source CSV exists; refreshes DvP and injuries.

## Frontend

See `../lovable/`; set `VITE_API_BASE` to your API URL.

## Full plan / checklist

See **[../PLANNING.md](../PLANNING.md)**.
