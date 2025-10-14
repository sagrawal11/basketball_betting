#!/usr/bin/env python3
"""
Retry Failed NBA Players
=======================
Retry data collection for players that failed during the initial collection
"""

import pandas as pd
import json
import os
from nba_data_collection import NBADataCollector

def retry_failed_players():
    """Retry data collection for failed players"""
    
    print("🔄 Retrying Failed NBA Players")
    print("=" * 40)
    
    # Load the collection report
    report_path = "player_data/nba_collection_report.json"
    if not os.path.exists(report_path):
        print(f"❌ Report not found: {report_path}")
        return
    
    with open(report_path, 'r') as f:
        report = json.load(f)
    
    failed_players = report.get('failed_players', [])
    no_data_players = report.get('no_data_players', [])
    
    print(f"📊 Failed players: {len(failed_players)}")
    print(f"📊 No data players: {len(no_data_players)}")
    print()
    
    if not failed_players and not no_data_players:
        print("✅ No failed players to retry!")
        return
    
    # Initialize collector with more aggressive retry settings
    collector = NBADataCollector()
    
    # Increase delays for failed players
    collector.request_delay = 15.0  # Even more respectful
    
    retry_count = 0
    success_count = 0
    
    # Retry failed players
    for player_name in failed_players[:50]:  # Limit to first 50 for testing
        print(f"🔄 Retrying {player_name}...")
        
        try:
            # Get NBA ID
            player_id = collector.get_player_nba_id(player_name)
            if player_id is None:
                print(f"❌ Could not find NBA ID for {player_name}")
                continue
            
            # Collect data
            result = collector.collect_player_data(player_name, player_id)
            
            if result['success']:
                print(f"✅ {player_name}: {result['games_collected']} games collected")
                success_count += 1
            else:
                print(f"❌ {player_name}: {result['error']}")
            
            retry_count += 1
            
            # Extra delay between retries
            import time
            time.sleep(20)  # 20 second delay between retries
            
        except Exception as e:
            print(f"❌ Error retrying {player_name}: {e}")
            retry_count += 1
    
    print(f"\n🏁 Retry complete!")
    print(f"🔄 Players retried: {retry_count}")
    print(f"✅ Successful: {success_count}")
    print(f"❌ Still failed: {retry_count - success_count}")

def analyze_failed_players():
    """Analyze why players failed"""
    
    print("🔍 Analyzing Failed Players")
    print("=" * 30)
    
    # Load the collection report
    report_path = "player_data/nba_collection_report.json"
    if not os.path.exists(report_path):
        print(f"❌ Report not found: {report_path}")
        return
    
    with open(report_path, 'r') as f:
        report = json.load(f)
    
    failed_players = report.get('failed_players', [])
    no_data_players = report.get('no_data_players', [])
    
    print(f"📊 Total failed: {len(failed_players)}")
    print(f"📊 No data: {len(no_data_players)}")
    
    # Show sample of failed players
    print(f"\n🔍 Sample failed players:")
    for player in failed_players[:10]:
        print(f"  - {player}")
    
    print(f"\n🔍 Sample no-data players:")
    for player in no_data_players[:10]:
        print(f"  - {player}")
    
    # Check if any failed players have data in player_data/
    print(f"\n🔍 Checking if any failed players actually have data:")
    player_data_dir = "player_data"
    if os.path.exists(player_data_dir):
        existing_players = [d for d in os.listdir(player_data_dir) if os.path.isdir(os.path.join(player_data_dir, d))]
        
        found_count = 0
        for player in failed_players[:20]:  # Check first 20
            player_dir = player.replace(' ', '_')
            if player_dir in existing_players:
                print(f"  ✅ {player} - Data exists!")
                found_count += 1
        
        print(f"📊 Found {found_count} failed players with existing data")

if __name__ == "__main__":
    print("🏀 NBA Failed Players Retry System")
    print("=" * 40)
    print("1. Analyze failed players")
    print("2. Retry failed players")
    print("3. Exit")
    print()
    
    choice = input("Select option (1-3): ").strip()
    
    if choice == "1":
        analyze_failed_players()
    elif choice == "2":
        retry_failed_players()
    elif choice == "3":
        print("👋 Goodbye!")
    else:
        print("❌ Invalid choice")
