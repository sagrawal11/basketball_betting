#!/usr/bin/env python3
"""
Nightly batch job: refresh auxiliary data, optionally rebuild features and retrain models.

Typical schedule (after games finish, e.g. 1:30 AM ET):
  cd backend && python pipeline/nightly.py

Default (no extra flags):
  - Refresh Kaggle team data + advanced SQLite, scrape new player logs.
  - Rebuild team stats parquets, DvP table, injuries.
  - Warm the Redis baseline cache and upload the training bundle to R2.

Optional heavy steps:
  --rebuild-features   Rebuild *_features.parquet for every active player
  --retrain-global     Run model_training/train_hybrid.py after features
  --fit-archetypes     Run archetype_engine.py before feature rebuild
  --dry-run            Print the step plan only

Resilience (Phase 11):
  - Every step runs in isolation: a failing step is logged and recorded but does NOT
    abort the rest of the run, so caching/upload still happen on a flaky scrape night.
  - Network steps retry with backoff. Steps can declare an intentional skip.
  - A lockfile prevents overlapping runs (stale locks are taken over).
  - A per-run status summary is written to `processed/nightly_last_run.json` (always),
    and the process exit code is non-zero only when a *critical* step fails.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
import time
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path

BACKEND = Path(__file__).resolve().parent.parent
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

from config.paths import (
    KAGGLE_TEAM_DATA_DIR,
    KAGGLE_TEAMS_BOXSCORES_CSV,
    PLAYER_DATA_DIR,
    PROCESSED_DIR,
    ADVANCED_DATA_DIR,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

LAST_RUN_FILE = PROCESSED_DIR / "nightly_last_run.json"
LOCK_FILE = PROCESSED_DIR / "nightly.lock"
LOCK_MAX_AGE_S = 6 * 3600  # a lock older than this is considered stale


class SkipStep(Exception):
    """Raised by a step to signal it intentionally did nothing (not a failure)."""


@dataclass
class StepResult:
    name: str
    status: str  # "ok" | "skipped" | "failed"
    duration_s: float
    attempts: int = 1
    critical: bool = False
    error: str | None = None


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _run(cmd: list[str], dry_run: bool) -> None:
    logger.info("RUN: %s", " ".join(cmd))
    if dry_run:
        return
    subprocess.check_call(cmd, cwd=str(BACKEND))


def _python(*args: str) -> list[str]:
    return [sys.executable, *args]


# ── Individual steps (bodies raise on failure; the runner handles isolation/retries) ──

def step_team_stats(dry_run: bool) -> None:
    if not KAGGLE_TEAMS_BOXSCORES_CSV.exists():
        raise SkipStep(
            f"no Kaggle boxscores at {KAGGLE_TEAMS_BOXSCORES_CSV} — add the dataset under data/Kaggle NBA Team Data/"
        )
    cmd = _python(
        str(BACKEND / "data_collection" / "build_team_stats_from_kaggle.py"),
        "--input",
        str(KAGGLE_TEAMS_BOXSCORES_CSV),
    )
    _run(cmd, dry_run)


def step_update_kaggle(dry_run: bool) -> None:
    # Download the FULL dataset, not a single file: Kaggle's single-file endpoint
    # (`-f <file>`) now returns 404. The zip unpacks into KAGGLE_TEAM_DATA_DIR as
    # `processed/teams_boxscores.csv` (+ cumulative_scraped/*), matching
    # KAGGLE_TEAMS_BOXSCORES_CSV. `--force` ensures the nightly pulls the latest.
    cmd = [
        "kaggle", "datasets", "download", "-d", "chrismunch/nba-game-team-statistics",
        "-p", str(KAGGLE_TEAM_DATA_DIR),
        "--unzip", "--force",
    ]
    _run(cmd, dry_run)


def step_update_advanced_sqlite(dry_run: bool) -> None:
    # NOTE: still uses Kaggle's single-file `-f` endpoint (which can 404 like the team
    # dataset did). Full-dataset download is not viable here — the wyattowalsh basketball
    # dataset is multi-GB. This step is non-critical and gated by "skip if local newer",
    # so an `-f` failure degrades gracefully (the existing nba.sqlite stays in place).
    cmd = [
        "kaggle", "datasets", "download", "-d", "wyattowalsh/basketball",
        "-f", "nba.sqlite",
        "-p", str(ADVANCED_DATA_DIR),
        "--unzip",
    ]
    _run(cmd, dry_run)


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

    from utils.storage import upload_full_training_bundle

    bucket = os.environ.get("R2_BUCKET_NAME")
    if not bucket:
        raise SkipStep("R2_BUCKET_NAME not set")

    logger.info("Uploading full training bundle (~2.8 GB)...")
    upload_full_training_bundle(bucket)


# ── Step runner: isolation + retries/backoff + skip handling ──

def run_step(name, fn, *, critical: bool, retries: int = 0, backoff: float = 5.0,
             dry_run: bool = False) -> StepResult:
    start = time.monotonic()
    attempt = 0
    while True:
        attempt += 1
        try:
            fn()
            dur = time.monotonic() - start
            logger.info("[OK] %s (%.1fs%s)", name, dur, f", x{attempt}" if attempt > 1 else "")
            return StepResult(name, "ok", dur, attempts=attempt, critical=critical)
        except SkipStep as e:
            dur = time.monotonic() - start
            logger.info("[SKIP] %s: %s", name, e)
            return StepResult(name, "skipped", dur, attempts=attempt, critical=critical)
        except Exception as e:
            if attempt <= retries and not dry_run:
                wait = backoff * attempt
                logger.warning("[RETRY] %s failed (attempt %d/%d): %s — retrying in %.0fs",
                               name, attempt, retries + 1, e, wait)
                time.sleep(wait)
                continue
            dur = time.monotonic() - start
            log = logger.error if critical else logger.warning
            log("[FAIL%s] %s after %d attempt(s): %s",
                "-CRITICAL" if critical else "", name, attempt, e)
            return StepResult(name, "failed", dur, attempts=attempt, critical=critical, error=str(e))


def _overall_status(results: list[StepResult]) -> str:
    if any(r.status == "failed" and r.critical for r in results):
        return "failed"
    if any(r.status == "failed" for r in results):
        return "partial"
    return "success"


def _log_summary(results: list[StepResult], status: str) -> None:
    logger.info("===== Nightly summary: %s =====", status.upper())
    for r in results:
        extra = f" (x{r.attempts})" if r.attempts > 1 else ""
        logger.info("  %-22s %-8s %6.1fs%s", r.name, r.status, r.duration_s, extra)
        if r.error:
            logger.info("       error: %s", r.error)


# ── Run lock (idempotency / overlap guard) ──

def _pid_alive(pid) -> bool:
    if not pid:
        return False
    try:
        os.kill(int(pid), 0)
    except ProcessLookupError:
        return False
    except (PermissionError, ValueError):
        return True
    return True


def acquire_lock() -> Path:
    """Create the run lock, taking over a stale one. Raises RuntimeError if a live run holds it."""
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    if LOCK_FILE.exists():
        try:
            data = json.loads(LOCK_FILE.read_text())
            pid = data.get("pid")
            started = datetime.fromisoformat(data.get("started_at"))
            age = (datetime.now(timezone.utc) - started).total_seconds()
            if _pid_alive(pid) and age < LOCK_MAX_AGE_S:
                raise RuntimeError(
                    f"Another nightly run is in progress (pid={pid}, started {data.get('started_at')}). Aborting."
                )
            logger.warning("Taking over stale lock (pid=%s, age=%.0fs).", pid, age)
        except RuntimeError:
            raise
        except Exception:
            logger.warning("Corrupt lock file at %s — removing.", LOCK_FILE)
        LOCK_FILE.unlink(missing_ok=True)
    LOCK_FILE.write_text(json.dumps({"pid": os.getpid(), "started_at": _now_iso()}))
    return LOCK_FILE


def release_lock() -> None:
    try:
        LOCK_FILE.unlink(missing_ok=True)
    except OSError:
        pass


def write_last_run(results: list[StepResult], status: str) -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "finished_at": _now_iso(),
        "status": status,
        "steps": [asdict(r) for r in results],
    }
    LAST_RUN_FILE.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _recent_success(hours: float) -> bool:
    if not LAST_RUN_FILE.exists():
        return False
    try:
        data = json.loads(LAST_RUN_FILE.read_text())
        if data.get("status") != "success":
            return False
        finished = datetime.fromisoformat(data["finished_at"])
        age_h = (datetime.now(timezone.utc) - finished).total_seconds() / 3600.0
        return age_h < hours
    except Exception:
        return False


def main() -> int:
    ap = argparse.ArgumentParser(description="NBA betting backend nightly pipeline")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--rebuild-features", action="store_true")
    ap.add_argument("--fit-archetypes", action="store_true")
    ap.add_argument("--retrain-global", action="store_true")
    ap.add_argument("--max-players", type=int, default=None, help="Cap for archetype/features/train")
    ap.add_argument("--skip-if-recent-hours", type=float, default=None,
                    help="Exit early if a successful run finished within this many hours")
    args = ap.parse_args()

    logger.info("Nightly pipeline start (backend=%s)", BACKEND)

    if not args.dry_run and args.skip_if_recent_hours and _recent_success(args.skip_if_recent_hours):
        logger.info("A successful run finished within %.1fh — skipping (per --skip-if-recent-hours).",
                    args.skip_if_recent_hours)
        return 0

    lock_held = False
    if not args.dry_run:
        try:
            acquire_lock()
            lock_held = True
        except RuntimeError as e:
            logger.error(str(e))
            return 1

    # (name, callable, critical, retries)
    plan: list[tuple] = [
        ("update_kaggle", lambda: step_update_kaggle(args.dry_run), False, 2),
        ("update_advanced_sqlite", lambda: step_update_advanced_sqlite(args.dry_run), False, 1),
        ("update_player_logs", lambda: step_update_player_logs(args.dry_run), False, 2),
        ("team_stats", lambda: step_team_stats(args.dry_run), False, 0),
        ("collect_dvp", lambda: step_collect_dvp(args.dry_run), False, 0),
        ("collect_injuries", lambda: step_collect_injuries(args.dry_run), False, 1),
    ]
    if args.fit_archetypes:
        plan.append(("fit_archetypes", lambda: step_fit_archetypes(args.dry_run, args.max_players), False, 0))
    if args.rebuild_features:
        plan.append(("rebuild_features", lambda: step_rebuild_all_features(args.dry_run, args.max_players), True, 0))
    if args.retrain_global:
        plan.append(("train", lambda: step_train_global(args.dry_run, args.max_players), True, 0))
    plan.append(("cache_baselines", lambda: step_cache_baselines(args.dry_run), True, 1))
    plan.append(("upload_artifacts", lambda: step_upload_artifacts(args.dry_run), False, 2))

    results: list[StepResult] = []
    try:
        for name, fn, critical, retries in plan:
            results.append(run_step(name, fn, critical=critical, retries=retries, dry_run=args.dry_run))
    finally:
        if lock_held:
            release_lock()

    status = _overall_status(results)
    _log_summary(results, status)
    if not args.dry_run:
        write_last_run(results, status)

    logger.info("Nightly pipeline finished (%s).", status)
    return 1 if status == "failed" else 0


if __name__ == "__main__":
    sys.exit(main())
