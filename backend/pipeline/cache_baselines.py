#!/usr/bin/env python3
"""
Pre-calculate baseline predictions for all games today/tomorrow and push them to Redis.
This script is meant to be run at the end of the nightly pipeline.
It uses the Flask test client to simulate requests, which automatically triggers
the Redis caching logic built into app.py.
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from web.app import app, redis_client

def cache_baselines():
    if not redis_client:
        print("WARNING: Redis client not configured. Caching will be skipped.")
        print("Set UPSTASH_REDIS_URL or REDIS_URL environment variable.")
        return

    print("Starting baseline caching...")
    with app.test_client() as client:
        print("Fetching today's schedule from ESPN...")
        resp = client.get("/api/games/today")
        if resp.status_code != 200:
            print(f"Failed to fetch games: {resp.status_code}")
            return
            
        data = resp.get_json()
        games = data.get("games", [])
        print(f"Found {len(games)} games to cache.")
        
        for game in games:
            game_id = game["id"]
            home = game["homeTeam"]["abbrev"]
            away = game["awayTeam"]["abbrev"]
            
            print(f"Generating baseline cache for {away} @ {home} (Game ID: {game_id})...")
            start_time = time.time()
            players_resp = client.get(f"/api/game/{game_id}/players?home={home}&away={away}")
            
            if players_resp.status_code == 200:
                elapsed = time.time() - start_time
                print(f"Successfully cached {away} @ {home} in {elapsed:.2f}s")
            else:
                print(f"Failed to cache {away} @ {home}: {players_resp.status_code}")

if __name__ == "__main__":
    cache_baselines()
