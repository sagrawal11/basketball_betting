#!/usr/bin/env python3
"""
Rebuild *_features.parquet for every player folder under player_data/.

Use after changing feature_engine (e.g. new Kaggle columns). Full corpus can take hours;
use --max-players for a smoke test.

  cd backend && python scripts/rebuild_all_features.py
  cd backend && python scripts/rebuild_all_features.py --max-players 100
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

BACKEND = Path(__file__).resolve().parent.parent
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

from config.paths import PLAYER_DATA_DIR
from feature_engine import default_engine

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    ap = argparse.ArgumentParser(description="Rebuild player feature parquets")
    ap.add_argument("--player-data", type=Path, default=PLAYER_DATA_DIR)
    ap.add_argument("--max-players", type=int, default=None)
    args = ap.parse_args()

    fe = default_engine()
    dirs = sorted([p for p in args.player_data.iterdir() if p.is_dir()])
    if args.max_players:
        dirs = dirs[: args.max_players]

    ok = 0
    for i, d in enumerate(dirs):
        name = d.name.replace("_", " ")
        try:
            out = fe.build_training_features(name, save_parquet=True)
            if out is not None and len(out) > 0:
                ok += 1
        except Exception as e:
            logger.warning("[%s/%s] %s: %s", i + 1, len(dirs), name, e)

    logger.info("Finished: %s / %s players with features", ok, len(dirs))


if __name__ == "__main__":
    main()
