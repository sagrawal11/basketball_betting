#!/usr/bin/env python3
"""
Nightly batch job: refresh auxiliary data, optionally rebuild features and retrain global models.

Typical schedule (after games finish, e.g. 1:30 AM ET):
  cd backend && python pipeline/nightly.py

Default (no extra flags):
  - Rebuild team stats + parquets from local Kaggle `processed/teams_boxscores.csv` when present
    (see data_collection/build_team_stats_from_kaggle.py); otherwise skip team stats.
  - DvP table, injuries — fast.

Optional heavy steps:
  --rebuild-features   Rebuild *_features.parquet for every player with raw CSV
  --retrain-global     Run model_training/train_global.py after features
  --fit-archetypes     Run archetype_engine.py before feature rebuild
  --dry-run            Print steps only

Player game-log appends from live box scores are not automated here yet; re-run your
bulk collector or a dedicated incremental script when that is wired.
"""
from __future__ import annotations

import argparse
import logging
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

BACKEND = Path(__file__).resolve().parent.parent
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

from config.paths import KAGGLE_TEAMS_BOXSCORES_CSV, PLAYER_DATA_DIR, PROCESSED_DIR, ADVANCED_DATA_DIR

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

LAST_RUN_FILE = PROCESSED_DIR / "nightly_last_run.txt"


def _run(cmd: list[str], dry_run: bool) -> None:
    logger.info("RUN: %s", " ".join(cmd))
    if dry_run:
        return
    subprocess.check_call(cmd, cwd=str(BACKEND))


def _python(*args: str) -> list[str]:
    return [sys.executable, *args]


def step_team_stats(dry_run: bool) -> None:
    if KAGGLE_TEAMS_BOXSCORES_CSV.exists():
        cmd = _python(
            str(BACKEND / "data_collection" / "build_team_stats_from_kaggle.py"),
            "--input",
            str(KAGGLE_TEAMS_BOXSCORES_CSV),
        )
        _run(cmd, dry_run)
        return
    logger.info(
        "Skipping team stats: no Kaggle boxscores at %s — add the dataset under data/Kaggle NBA Team Data/",
        KAGGLE_TEAMS_BOXSCORES_CSV,
    )


def step_update_kaggle(dry_run: bool) -> None:
    cmd = [
        "kaggle", "datasets", "download", "-d", "chrismunch/nba-game-team-statistics",
        "-f", "teams_boxscores.csv",
        "-p", str(KAGGLE_TEAMS_BOXSCORES_CSV.parent),
        "--unzip"
    ]
    if dry_run:
        logger.info("RUN: %s", " ".join(cmd))
        return
    logger.info("RUN: %s", " ".join(cmd))
    try:
        subprocess.check_call(cmd, cwd=str(BACKEND))
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        logger.warning("Kaggle download failed. Is ~/.kaggle/kaggle.json configured? Error: %s", e)

def step_update_advanced_sqlite(dry_run: bool) -> None:
    cmd = [
        "kaggle", "datasets", "download", "-d", "wyattowalsh/basketball",
        "-f", "nba.sqlite",
        "-p", str(ADVANCED_DATA_DIR),
        "--unzip"
    ]
    if dry_run:
        logger.info("RUN: %s", " ".join(cmd))
        return
    logger.info("RUN: %s", " ".join(cmd))
    try:
        subprocess.check_call(cmd, cwd=str(BACKEND))
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        logger.warning("WyattWalsh Advanced SQLite download failed. Is ~/.kaggle/kaggle.json configured? Error: %s", e)

def step_update_player_logs(dry_run: bool) -> None:
    _run(_python(str(BACKEND / "data_collection" / "update_player_logs.py")), dry_run)


def step_collect_dvp(dry_run: bool) -> None:
    _run(_python(str(BACKEND / "data_collection" / "collect_dvp.py")), dry_run)


def step_collect_injuries(dry_run: bool) -> None:
    _run(_python(str(BACKEND / "data_collection" / "collect_injuries.py")), dry_run)


def step_fit_archetypes(dry_run: bool, max_players: int | None) -> None:
    cmd = _python(str(BACKEND / "archetype_engine.py"))
    if max_players:
        cmd.extend(["--max-players", str(max_players)])
    _run(cmd, dry_run)


def step_rebuild_all_features(dry_run: bool, max_players: int | None) -> None:
    if dry_run:
        logger.info("[dry-run] rebuild features for player_data folders")
        return
    from feature_engine import default_engine
    import pandas as pd
    from datetime import datetime, timedelta

    fe = default_engine()
    dirs = sorted([p for p in PLAYER_DATA_DIR.iterdir() if p.is_dir()])
    if max_players:
        dirs = dirs[:max_players]
        
    cutoff = datetime.now() - timedelta(days=365)
    active_dirs = []
    for d in dirs:
        csv_path = d / f"{d.name}_data.csv"
        if csv_path.exists():
            try:
                df = pd.read_csv(csv_path, usecols=["GAME_DATE"])
                if pd.to_datetime(df["GAME_DATE"].iloc[-1]) >= cutoff:
                    active_dirs.append(d)
            except Exception:
                continue
                
    logger.info("Rebuilding features for %d active players (past 365 days)...", len(active_dirs))
    ok = 0
    for d in active_dirs:
        name = d.name.replace("_", " ")
        try:
            out = fe.build_training_features(name, save_parquet=True)
            if out is not None:
                ok += 1
        except Exception as e:
            logger.warning("Features failed for %s: %s", name, e)
    logger.info("Feature rebuild finished: %s / %s players", ok, len(active_dirs))


def step_train_global(dry_run: bool, max_players: int | None) -> None:
    cmd = _python(str(BACKEND / "model_training" / "train_hybrid.py"))
    if max_players:
        cmd.extend(["--max-players", str(max_players)])
    _run(cmd, dry_run)


def step_cache_baselines(dry_run: bool) -> None:
    _run(_python(str(BACKEND / "pipeline" / "cache_baselines.py")), dry_run)


def step_upload_artifacts(dry_run: bool) -> None:
    """Upload the cleaned ML dataset to R2 (~2.8 GB)."""
    if dry_run:
        logger.info("[dry-run] upload full training bundle to R2")
        return
        
    try:
        from utils.storage import upload_full_training_bundle
        import os
        
        bucket = os.environ.get("R2_BUCKET_NAME")
        if not bucket:
            logger.info("R2_BUCKET_NAME not set, skipping upload.")
            return
            
        logger.info("Uploading full training bundle (~2.8 GB)...")
        upload_full_training_bundle(bucket)
        
    except Exception as e:
        logger.warning("Failed to upload artifacts: %s", e)


def write_last_run() -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    LAST_RUN_FILE.write_text(datetime.now(timezone.utc).isoformat(), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description="NBA betting backend nightly pipeline")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--rebuild-features", action="store_true")
    ap.add_argument("--fit-archetypes", action="store_true")
    ap.add_argument("--retrain-global", action="store_true")
    ap.add_argument("--max-players", type=int, default=None, help="Cap for archetype/features/train")
    args = ap.parse_args()

    logger.info("Nightly pipeline start (backend=%s)", BACKEND)

    step_update_kaggle(args.dry_run)
    step_update_advanced_sqlite(args.dry_run)
    step_update_player_logs(args.dry_run)
    
    step_team_stats(args.dry_run)
    step_collect_dvp(args.dry_run)
    step_collect_injuries(args.dry_run)

    if args.fit_archetypes:
        step_fit_archetypes(args.dry_run, args.max_players)

    if args.rebuild_features:
        step_rebuild_all_features(args.dry_run, args.max_players)

    if args.retrain_global:
        step_train_global(args.dry_run, args.max_players)

    if not args.dry_run:
        write_last_run()
        
    step_cache_baselines(args.dry_run)
    step_upload_artifacts(args.dry_run)

    logger.info("Nightly pipeline finished.")


if __name__ == "__main__":
    main()
