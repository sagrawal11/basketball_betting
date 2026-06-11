"""Task 7.2 — the managed-column whitelist must drop raw same-game rate stats.

``_add_advanced_ratios`` computes TS_PCT, USG_PCT and *_PER100 from the CURRENT
game's makes/attempts/minutes, so the raw (non-rolling) versions are leakage:
at training they encode the outcome being predicted, and at inference they are
NaN/garbage because the current game hasn't happened. Their *rolling* versions
(TS_PCT_L5, USG_PCT_L10, ...) are shifted to prior games and are legitimate
features that must be kept.

``_clean_output_columns`` previously listed TS_PCT/USG_PCT in ``managed_exact``
(so they were kept) despite its own trailing comment saying they're leaky.
These tests pin the correct behavior: drop the raw rate stats, keep the rolling
ones.
"""
import numpy as np
import pandas as pd

import model_training.train_global as tg
from feature_engine import TARGET_STATS, default_engine

# Raw same-game rate stats that must NOT survive into the feature set.
LEAKY_RAW = {"TS_PCT", "USG_PCT", "PTS_PER100", "AST_PER100", "REB_PER100"}
# Rolling (prior-game) versions that MUST survive — regression guard against
# an over-broad fix that strips legitimate features.
MANAGED_ROLLING = {"TS_PCT_L5", "USG_PCT_L10", "PTS_PER100_L3", "AST_PER100_L5"}


def _frame_with_rate_stats(n: int = 24) -> pd.DataFrame:
    data: dict = {
        "GAME_DATE": pd.date_range("2022-10-01", periods=n, freq="D"),
        "SEASON": ["2022-23"] * n,
        "PLAYER_NAME": ["test player"] * n,
        "OPPONENT_TEAM": ["BOS"] * n,
        "PLAYER_TEAM": ["GSW"] * n,
        "POSITION_GROUP": ["G"] * n,
        "position_encoded": np.arange(n) % 5,
    }
    for t in TARGET_STATS:
        data[t] = np.linspace(0, 30, n)
    # a couple of clearly-managed features so the frame isn't degenerate
    data["PTS_L5"] = np.linspace(1, 100, n)
    data["KG_PREV_PTS"] = np.linspace(1, 100, n)
    # raw leaky rate stats
    for c in LEAKY_RAW:
        data[c] = np.linspace(0.4, 0.7, n)
    # legitimate rolling rate stats
    for c in MANAGED_ROLLING:
        data[c] = np.linspace(0.4, 0.7, n)
    return pd.DataFrame(data)


def test_clean_output_columns_drops_raw_rate_stats():
    engine = default_engine()
    cleaned = engine._clean_output_columns(_frame_with_rate_stats())
    leaked = LEAKY_RAW.intersection(cleaned.columns)
    assert not leaked, f"Raw same-game rate stats survived the whitelist: {sorted(leaked)}"


def test_clean_output_columns_keeps_rolling_rate_stats():
    engine = default_engine()
    cleaned = engine._clean_output_columns(_frame_with_rate_stats())
    dropped = MANAGED_ROLLING.difference(cleaned.columns)
    assert not dropped, f"Legitimate rolling rate stats were wrongly dropped: {sorted(dropped)}"


def test_feature_set_excludes_raw_rate_stats():
    """End-to-end through the unified selection used by both trainers."""
    _, feature_cols, _, _ = tg._feature_target_split(_frame_with_rate_stats())
    leaked = LEAKY_RAW.intersection(feature_cols)
    assert not leaked, f"Leaky raw rate stats present in model feature set: {sorted(leaked)}"
    # rolling versions must still be features
    assert MANAGED_ROLLING.issubset(set(feature_cols)), (
        f"Rolling rate features missing: {sorted(MANAGED_ROLLING.difference(feature_cols))}"
    )
