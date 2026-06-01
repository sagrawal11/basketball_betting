#!/usr/bin/env python3
"""
NBA Betting Assistant — Flask API for predictions.

No stats.nba.com / nba_api at runtime: schedules and rosters come from local `player_data`
and optional UI-provided matchups. Injuries are fetched live from ESPN.
"""

from __future__ import annotations

import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import pytz
import requests
import os
import redis
from flask import Flask, jsonify, request
from flask_cors import CORS

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.predictor import NBAPredictor
from data_collection.collect_injuries import fetch_team_injuries, _normalize_team_abbrev

app = Flask(__name__)
app.config["SECRET_KEY"] = "nba-betting-secret-key-change-in-production"

CORS(app)

redis_client = None
redis_url = os.environ.get("REDIS_URL") or os.environ.get("UPSTASH_REDIS_URL")
if redis_url:
    try:
        redis_client = redis.from_url(redis_url)
        print("Connected to Redis for baseline caching.")
    except Exception as e:
        print(f"Failed to connect to Redis: {e}")

backend_dir = Path(__file__).parent.parent
data_dir = backend_dir / "data"
models_dir = data_dir / "player_data"
predictions_dir = backend_dir / "data" / "predictions"

try:
    from utils.storage import download_api_bundle
    from config.paths import GLOBAL_MODEL_DIR
    bucket = os.environ.get("R2_BUCKET_NAME")
    if bucket:
        if not list(GLOBAL_MODEL_DIR.glob("*.joblib")):
            print("Models not found locally. Downloading API bundle from R2...")
            download_api_bundle(bucket)
            print("Download complete.")
except Exception as e:
    print(f"Failed to sync from R2: {e}")

predictor = NBAPredictor(data_dir=str(data_dir), models_dir=str(models_dir))

PLACEHOLDER_HEADSHOT = "https://via.placeholder.com/260x190?text=Player"

# ── Injury cache (avoid hammering ESPN on every request) ─────────────────────
_injury_cache: Dict[str, Any] = {}
_injury_cache_ts: float = 0.0
INJURY_CACHE_TTL = 120  # seconds


def _get_cached_injuries() -> Dict[str, List[Dict[str, Any]]]:
    """Return injuries grouped by team_abbrev, cached for INJURY_CACHE_TTL seconds."""
    global _injury_cache, _injury_cache_ts
    now = time.time()
    if _injury_cache and (now - _injury_cache_ts) < INJURY_CACHE_TTL:
        return _injury_cache

    try:
        from data_collection.collect_injuries import fetch_injuries_from_espn
        df = fetch_injuries_from_espn()
        if df.empty:
            _injury_cache = {}
        else:
            grouped: Dict[str, List[Dict[str, Any]]] = {}
            for _, row in df.iterrows():
                team = row.get("team_abbrev", "")
                if team not in grouped:
                    grouped[team] = []
                grouped[team].append(row.to_dict())
            _injury_cache = grouped
        _injury_cache_ts = now
    except Exception as e:
        print(f"Injury cache refresh failed: {e}")
        # Keep stale cache rather than returning nothing

    return _injury_cache


def _slug_to_display(slug: str) -> str:
    """Convert 'lebron_james' to 'LeBron James' (display-quality)."""
    return slug.replace("_", " ")


def _display_to_slug(display_name: str) -> str:
    """Convert 'LeBron James' to 'lebron_james'."""
    return display_name.lower().replace(" ", "_")


def _match_injury_to_roster(injury_name: str, roster_names: List[str]) -> Optional[str]:
    """Best-effort match ESPN injury name to our roster slug names.

    Handles common edge cases: Jr./Jr, III, accented chars.
    """
    injury_lower = injury_name.lower().strip()
    # Remove suffixes for matching
    clean = injury_lower.replace(".", "").replace("'", "").replace("'", "")

    for roster_name in roster_names:
        roster_lower = roster_name.lower().strip()
        roster_clean = roster_lower.replace(".", "").replace("'", "").replace("'", "")

        # Exact match
        if clean == roster_clean:
            return roster_name

        # Handle slug format (underscore vs space)
        if clean.replace(" ", "_") == roster_clean.replace(" ", "_"):
            return roster_name

    # Fuzzy: check if ESPN last name matches any roster last name
    injury_parts = injury_lower.split()
    if len(injury_parts) >= 2:
        injury_last = injury_parts[-1].replace(".", "")
        for roster_name in roster_names:
            roster_parts = roster_name.lower().split()
            if len(roster_parts) >= 2:
                roster_last = roster_parts[-1].replace(".", "")
                roster_first = roster_parts[0]
                injury_first = injury_parts[0]
                # Match last name + first initial
                if injury_last == roster_last and injury_first[0] == roster_first[0]:
                    return roster_name

    return None


def team_logo_url(abbrev: str) -> str:
    a = (abbrev or "nba").strip().lower()
    return f"https://a.espncdn.com/i/teamlogos/nba/500/{a}.png"


def _load_prediction_history() -> List[Dict[str, Any]]:
    """Load prediction history from JSON files in predictions_dir."""
    predictions_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for f in sorted(predictions_dir.glob("*.json")):
        try:
            results.append(json.loads(f.read_text(encoding="utf-8")))
        except Exception:
            pass
    return results


def _save_prediction(data: Dict[str, Any]) -> None:
    """Save a prediction to a JSON file."""
    predictions_dir.mkdir(parents=True, exist_ok=True)
    game_id = data.get("game_id", "unknown")
    game_date = data.get("game_date", "unknown")
    path = predictions_dir / f"{game_date}_{game_id}.json"
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def scan_players_on_teams(player_root: Path, home: str, away: str) -> List[Tuple[str, str, bool]]:
    """(display_name, team_abbrev, is_home) using latest PLAYER_TEAM in each player CSV."""
    out: List[Tuple[str, str, bool]] = []
    home_u, away_u = home.upper().strip(), away.upper().strip()
    for d in sorted(player_root.iterdir()):
        if not d.is_dir():
            continue
        slug = d.name
        csv_path = d / f"{slug}_data.csv"
        if not csv_path.exists():
            continue
        try:
            df = pd.read_csv(csv_path)
        except Exception:
            continue
        if "PLAYER_TEAM" not in df.columns or "GAME_DATE" not in df.columns or len(df) == 0:
            continue
            
        # Check if the player is actually active recently (last 45 days)
        # This naturally filters out players with season-ending injuries,
        # guys completely out of the rotation, and retired players.
        last_date_str = str(df["GAME_DATE"].iloc[-1])
        try:
            last_date = datetime.strptime(last_date_str, "%Y-%m-%d")
            days_since = (datetime.now() - last_date).days
            if days_since > 45:
                continue
        except Exception:
            continue
            
        team = str(df["PLAYER_TEAM"].iloc[-1] or "").strip().upper()
        name = slug.replace("_", " ")
        if team == home_u:
            out.append((name, home_u, True))
        elif team == away_u:
            out.append((name, away_u, False))
    return out


@app.route("/")
def index():
    return jsonify(
        {
            "service": "nba-betting-api",
            "docs": "Local-data mode — use /api/predict or /api/game/<id>/players?home=&away=",
        }
    )


ESPN_TO_NBA_MAP = {
    "GS": "GSW", "NO": "NOP", "NY": "NYK", 
    "SA": "SAS", "UTAH": "UTA", "WSH": "WAS"
}

@app.route("/api/games/today")
def get_todays_games():
    """Fetch live NBA schedule from ESPN API."""
    try:
        resp = requests.get("https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard", timeout=10)
        data = resp.json()
        
        games = []
        for ev in data.get("events", []):
            try:
                comp = ev["competitions"][0]
                competitors = comp["competitors"]
                home = next(c for c in competitors if c["homeAway"] == "home")["team"]
                away = next(c for c in competitors if c["homeAway"] == "away")["team"]
                
                home_abbrev = ESPN_TO_NBA_MAP.get(home["abbreviation"].upper(), home["abbreviation"].upper())
                away_abbrev = ESPN_TO_NBA_MAP.get(away["abbreviation"].upper(), away["abbreviation"].upper())
                
                games.append({
                    "id": ev["id"],
                    "time": ev["date"],
                    "date": datetime.now().strftime("%Y-%m-%d"),
                    "location": comp.get("venue", {}).get("fullName", "TBD"),
                    "homeTeam": {
                        "name": home.get("name", home_abbrev),
                        "abbrev": home_abbrev,
                        "logo": team_logo_url(home_abbrev)
                    },
                    "awayTeam": {
                        "name": away.get("name", away_abbrev),
                        "abbrev": away_abbrev,
                        "logo": team_logo_url(away_abbrev)
                    }
                })
            except Exception as e:
                print(f"Error parsing game: {e}")
                
        return jsonify({
            "success": True,
            "games": sorted(games, key=lambda x: x["time"]),
            "count": len(games)
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"success": False, "error": f"Failed to fetch standard ESPN scoreboard: {str(e)}", "games": []}), 500


@app.route("/api/injuries")
def get_injuries():
    """Get current injury report, optionally filtered by team."""
    team = request.args.get("team", "").upper().strip()
    injuries = _get_cached_injuries()

    if team:
        team_injuries = injuries.get(team, [])
        return jsonify({
            "success": True,
            "team": team,
            "injuries": team_injuries,
            "count": len(team_injuries),
        })

    # Return all injuries
    all_injuries = []
    for team_list in injuries.values():
        all_injuries.extend(team_list)
    return jsonify({
        "success": True,
        "injuries": all_injuries,
        "count": len(all_injuries),
    })


@app.route("/api/game/<game_id>/players")
def get_game_players(game_id):
    try:
        home_abbrev = request.args.get("home")
        away_abbrev = request.args.get("away")
        if not home_abbrev or not away_abbrev:
            return jsonify({"success": False, "error": "Missing home or away query param"}), 400

        # Parse optional manual exclusions from the frontend
        exclude_param = request.args.get("excludePlayers", "")
        manually_excluded = [p.strip() for p in exclude_param.split(",") if p.strip()] if exclude_param else []

        eastern = pytz.timezone("US/Eastern")
        today = datetime.now(eastern)
        game_date = today.strftime("%Y-%m-%d")
        
        # Check Redis Cache for baseline predictions
        cache_key = f"game:{game_id}:baseline:{game_date}"
        if not manually_excluded and redis_client:
            try:
                cached = redis_client.get(cache_key)
                if cached:
                    return jsonify(json.loads(cached))
            except Exception as e:
                print(f"Redis read error: {e}")

        season = predictor._get_current_season()

        # Step 1: Get full roster from local data
        roster = scan_players_on_teams(models_dir, home_abbrev, away_abbrev)
        roster_names = [name for name, _, _ in roster]

        # Step 2: Fetch live injuries for both teams
        injuries = _get_cached_injuries()
        home_injuries = injuries.get(home_abbrev.upper(), [])
        away_injuries = injuries.get(away_abbrev.upper(), [])

        # Build lookup: roster_name → injury info
        injury_map: Dict[str, Dict[str, Any]] = {}
        out_players: Dict[str, List[str]] = {home_abbrev.upper(): [], away_abbrev.upper(): []}

        for inj in home_injuries + away_injuries:
            espn_name = inj.get("player_name", "")
            matched = _match_injury_to_roster(espn_name, roster_names)
            if matched:
                injury_map[matched] = inj
                if inj.get("status", "").lower() == "out":
                    team = inj.get("team_abbrev", "")
                    if team in out_players:
                        out_players[team].append(matched)

        # Add manually excluded players to out lists
        for name in manually_excluded:
            for roster_name, team_abbrev, _ in roster:
                if roster_name.lower() == name.lower() or roster_name.replace(" ", "_").lower() == name.lower():
                    if team_abbrev in out_players and roster_name not in out_players[team_abbrev]:
                        out_players[team_abbrev].append(roster_name)
                    if roster_name not in injury_map:
                        injury_map[roster_name] = {
                            "status": "Out",
                            "injury_type": "Manual Exclusion",
                            "detail": "Excluded by user",
                        }

        # Step 3: Generate predictions for active players only
        players_with_predictions = []
        injury_report = []  # Sidelined players for the frontend

        for player_name, team_abbrev, is_home in roster:
            opponent_abbrev = away_abbrev if is_home else home_abbrev
            inj_info = injury_map.get(player_name)

            # Determine this player's injury status
            player_status = "Healthy"
            if inj_info:
                player_status = inj_info.get("status", "Healthy")

            # If player is Out, add to injury report and skip prediction
            if player_status.lower() == "out":
                injury_report.append({
                    "name": player_name,
                    "team": team_abbrev,
                    "is_home": is_home,
                    "status": "Out",
                    "injury_type": inj_info.get("injury_type", "") if inj_info else "",
                    "detail": inj_info.get("detail", "") if inj_info else "",
                    "side": inj_info.get("side", "") if inj_info else "",
                    "headshot_url": inj_info.get("headshot_url", "") if inj_info else "",
                })
                continue

            # Get teammates who are out (same team)
            same_team_out = out_players.get(team_abbrev, [])

            preds = predictor.predict_player_stats(
                player_name=player_name,
                opponent_team=opponent_abbrev,
                is_home=is_home,
                game_date=game_date,
                season=season,
                injury_status=player_status,
                teammates_out=same_team_out,
            )
            if not preds:
                continue

            slug = player_name.replace(" ", "_")
            csv_path = models_dir / slug / f"{slug}_data.csv"
            position = "G"
            try:
                position = str(pd.read_csv(csv_path).iloc[-1].get("position", "G"))
            except Exception:
                pass

            # Use ESPN headshot if available, otherwise placeholder
            headshot = PLACEHOLDER_HEADSHOT
            if inj_info and inj_info.get("headshot_url"):
                headshot = inj_info["headshot_url"]

            player_entry = {
                "name": player_name,
                "position": position,
                "image": headshot,
                "stats": {
                    "points": round(float(preds.get("PTS", 0)), 1),
                    "rebounds": round(
                        float(preds.get("DREB", 0)) + float(preds.get("OREB", 0)), 1
                    ),
                    "assists": round(float(preds.get("AST", 0)), 1),
                    "steals": round(float(preds.get("STL", 0)), 1),
                    "blocks": round(float(preds.get("BLK", 0)), 1),
                    "fg": "—",
                    "threePt": "—",
                    "ft": "—",
                },
                "team": team_abbrev,
                "is_home": is_home,
            }

            # Annotate injury status for Day-To-Day / Probable players
            if inj_info and player_status.lower() != "healthy":
                player_entry["injuryStatus"] = player_status
                player_entry["injuryDetail"] = f"{inj_info.get('injury_type', '')} ({inj_info.get('side', '')})".strip(" ()")

            players_with_predictions.append(player_entry)

        home_players = [p for p in players_with_predictions if p["is_home"]]
        away_players = [p for p in players_with_predictions if not p["is_home"]]

        home_score = sum(p["stats"]["points"] for p in home_players)
        away_score = sum(p["stats"]["points"] for p in away_players)
        if len(home_players) >= 8:
            home_final = home_score + 3
            away_final = away_score + 3
        elif len(home_players) >= 5:
            home_final = home_score / 0.828 if home_score else 104
            away_final = away_score / 0.828 if away_score else 104
        else:
            home_final = home_score or 104
            away_final = away_score or 104

        try:
            _save_prediction({
                "game_id": game_id,
                "game_date": game_date,
                "home_team": home_abbrev,
                "away_team": away_abbrev,
                "home_predicted_score": round(float(home_final), 1),
                "away_predicted_score": round(float(away_final), 1),
                "player_predictions": players_with_predictions,
                "injury_report": injury_report,
            })
        except Exception as save_error:
            print(f"Could not save prediction: {save_error}")

        response_data = {
            "success": True,
            "homeTeam": {
                "name": home_abbrev,
                "logo": team_logo_url(home_abbrev),
                "predictedScore": round(home_final),
            },
            "awayTeam": {
                "name": away_abbrev,
                "logo": team_logo_url(away_abbrev),
                "predictedScore": round(away_final),
            },
            "date": today.strftime("%B %d, %Y"),
            "time": "TBD",
            "location": "",
            "homePlayers": [
                {"name": p["name"], "position": p["position"], "image": p["image"], "stats": p["stats"],
                 **({"injuryStatus": p["injuryStatus"], "injuryDetail": p.get("injuryDetail", "")} if "injuryStatus" in p else {})}
                for p in home_players
            ],
            "awayPlayers": [
                {"name": p["name"], "position": p["position"], "image": p["image"], "stats": p["stats"],
                 **({"injuryStatus": p["injuryStatus"], "injuryDetail": p.get("injuryDetail", "")} if "injuryStatus" in p else {})}
                for p in away_players
            ],
            "injuryReport": injury_report,
        }

        if not manually_excluded and redis_client:
            try:
                # Cache for 12 hours
                redis_client.setex(cache_key, 12 * 3600, json.dumps(response_data))
            except Exception as e:
                print(f"Redis set error: {e}")

        return jsonify(response_data)

    except Exception as e:
        import traceback

        traceback.print_exc()
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/predict", methods=["POST"])
def get_prediction():
    try:
        data = request.json
        player_name = data.get("player_name")
        opponent = data.get("opponent")
        is_home = data.get("is_home", True)
        game_date = data.get("game_date", datetime.now().strftime("%Y-%m-%d"))
        season = data.get("season", predictor._get_current_season())
        injury_status = data.get("injury_status", "Healthy")
        teammates_out = data.get("teammates_out", [])

        predictions = predictor.predict_player_stats(
            player_name=player_name,
            opponent_team=opponent,
            is_home=is_home,
            game_date=game_date,
            season=season,
            injury_status=injury_status,
            teammates_out=teammates_out,
        )

        if predictions:
            return jsonify({"success": True, "player": player_name, "predictions": predictions})
        return jsonify({"success": False, "error": f"No trained model found for {player_name}"}), 404

    except Exception as e:
        import traceback

        traceback.print_exc()
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/generate-all-predictions", methods=["POST"])
def generate_all_predictions():
    return jsonify(
        {
            "success": False,
            "generated": 0,
            "message": "Bulk auto-generation from a live NBA feed is disabled. Call /api/game/<id>/players?home=&away= per matchup.",
        }
    )


@app.route("/api/history/games")
def get_historical_games():
    try:
        preds = _load_prediction_history()
        historical_games = []
        for p in preds:
            historical_games.append(
                {
                    "id": p.get("game_id"),
                    "homeTeam": {
                        "name": p.get("home_team", ""),
                        "abbrev": p.get("home_team", ""),
                        "logo": team_logo_url(str(p.get("home_team", ""))),
                        "predictedScore": p.get("home_predicted_score"),
                        "actualScore": p.get("home_actual_score") or 0,
                    },
                    "awayTeam": {
                        "name": p.get("away_team", ""),
                        "abbrev": p.get("away_team", ""),
                        "logo": team_logo_url(str(p.get("away_team", ""))),
                        "predictedScore": p.get("away_predicted_score"),
                        "actualScore": p.get("away_actual_score") or 0,
                    },
                    "time": "Saved",
                    "date": p.get("game_date", ""),
                    "location": "",
                    "accuracy": {
                        "scoreAccuracy": p.get("score_accuracy"),
                        "playerStatsAccuracy": p.get("player_stats_accuracy"),
                        "avgPointsDiff": p.get("avg_points_diff"),
                    },
                }
            )
        return jsonify({"success": True, "games": historical_games})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/analyze-bet", methods=["POST"])
def analyze_bet():
    try:
        data = request.json
        player_name = data.get("player")
        stat_type = data.get("stat")
        line = float(data.get("line", 0))
        odds = int(data.get("over_odds", -110))

        # Use the predictor to get the expected stat value
        prediction = predictor.predict_player_stats(
            player_name=player_name,
            opponent_team=data.get("opponent", ""),
            is_home=data.get("is_home", True),
            game_date=data.get("game_date", datetime.now().strftime("%Y-%m-%d")),
            season=data.get("season", predictor._get_current_season()),
        )

        if not prediction or stat_type not in prediction:
            return jsonify({"success": False, "error": f"Cannot predict {stat_type} for {player_name}"}), 404

        predicted_value = prediction[stat_type]
        edge = predicted_value - line

        if abs(edge) < 0.5:
            recommendation = "no_edge"
            confidence = "low"
        elif edge > 2.0:
            recommendation = "over"
            confidence = "high"
        elif edge > 0.5:
            recommendation = "over"
            confidence = "medium"
        elif edge < -2.0:
            recommendation = "under"
            confidence = "high"
        else:
            recommendation = "under"
            confidence = "medium"

        return jsonify({
            "success": True,
            "recommendation": {
                "player": player_name,
                "stat": stat_type,
                "line": line,
                "predicted": round(predicted_value, 1),
                "edge": round(edge, 1),
                "recommendation": recommendation,
                "confidence": confidence,
            },
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


if __name__ == "__main__":
    print("NBA Betting API (local-data mode)")
    print("http://localhost:5001")
    app.run(debug=True, port=5001, host="127.0.0.1")
