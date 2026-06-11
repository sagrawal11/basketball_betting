"""Task 7.1 — prove (and then guard) that train_global.py and train_hybrid.py
select the SAME feature columns from the same input frame.

Both trainers call ``save_feature_columns(feature_cols)`` and write to the same
``feature_columns.json``. If they disagree on which columns are features, then
whichever trainer ran last silently defines what inference expects — the core
train-serve skew bug. These tests pin the feature selection to a single source
of truth.
"""
import numpy as np
import pandas as pd

import model_training.train_global as tg
import model_training.train_hybrid as th
from feature_engine import TARGET_STATS


def _synthetic_feature_frame(n: int = 24) -> pd.DataFrame:
    """A frame that mixes three kinds of columns:

    * managed features that the FeatureEngine whitelist KEEPS,
    * leaky same-game raw columns the whitelist is designed to DROP,
    * targets + metadata.

    A correct, unified selection must yield identical feature sets regardless of
    which trainer produced them.
    """
    data: dict = {
        # --- metadata (non-numeric mostly; dropped from the numeric feature set) ---
        "GAME_DATE": pd.date_range("2022-10-01", periods=n, freq="D"),
        "SEASON": ["2022-23"] * n,
        "PLAYER_NAME": ["test player"] * n,
        "OPPONENT_TEAM": ["BOS"] * n,
        "PLAYER_TEAM": ["GSW"] * n,
        "POSITION_GROUP": ["G"] * n,
        "position_encoded": np.arange(n) % 5,
        "Game_ID": np.arange(n),  # ID_LIKE -> must be dropped by both
    }
    # --- targets ---
    for t in TARGET_STATS:
        data[t] = np.linspace(0, 30, n)
    # --- managed features that survive the whitelist ---
    for c in [
        "PTS_L5", "AST_L10", "TS_PCT_L5", "KG_PREV_PTS", "KG_OPP_PREV_PTS",
        "OPP_PRIOR_AVG_PTS", "ARCH_PROB_0", "ARCH_PROB_1", "DVP_PTS",
        "TEAM_PACE", "TEAM_OFF_RATING", "IS_HOME", "DAYS_REST",
    ]:
        data[c] = np.linspace(1, 100, n)
    # --- leaky same-game raw columns the whitelist DROPS but an unmanaged
    #     numeric split keeps (these are the divergence) ---
    for c in [
        "FGA", "FTA", "REB", "PLUS_MINUS", "FG_PCT", "FG3_PCT", "FT_PCT",
        "PF", "MIN", "TS_PCT", "USG_PCT", "PTS_PER100", "AST_PER100",
    ]:
        data[c] = np.linspace(1, 50, n)
    return pd.DataFrame(data)


def test_trainers_select_identical_feature_columns():
    """The two training scripts must agree on the feature column set, or
    feature_columns.json becomes order-of-execution dependent (train-serve skew).
    """
    df = _synthetic_feature_frame()

    _, global_cols, _, _ = tg._feature_target_split(df.copy())
    _, hybrid_cols, _, _ = th._feature_target_split(df.copy())

    only_global = sorted(set(global_cols) - set(hybrid_cols))
    only_hybrid = sorted(set(hybrid_cols) - set(global_cols))

    assert set(global_cols) == set(hybrid_cols), (
        "train_global.py and train_hybrid.py select different feature columns, "
        "so feature_columns.json depends on which trainer runs last.\n"
        f"  Only in train_global (leaky/unmanaged, expected to be dropped): {only_global}\n"
        f"  Only in train_hybrid: {only_hybrid}"
    )


def test_trainers_select_identical_targets_and_order():
    """Beyond set equality, the feature ORDER must match too — LightGBM consumes
    columns positionally, so an order mismatch is also train-serve skew.
    """
    df = _synthetic_feature_frame()
    _, global_cols, global_y, _ = tg._feature_target_split(df.copy())
    _, hybrid_cols, hybrid_y, _ = th._feature_target_split(df.copy())

    assert global_cols == hybrid_cols, "Feature column ORDER differs between trainers"
    assert global_y == hybrid_y, "Target column selection differs between trainers"
