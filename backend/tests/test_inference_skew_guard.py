"""Task 7.6 — make train-serve skew observable at inference.

predict_player_stats reindexes the inference row to the trained feature_columns
and fills anything missing/NaN with 0.0, logging only at DEBUG. For high-signal
managed features (ARCH_PROB_*, OPP_PRIOR_AVG_*, KG_PREV_*, DVP_*) a silent 0.0
is far from the training distribution — skew that hides instead of surfacing.

These tests pin a diagnostic (_feature_skew_report) and a guard
(_check_feature_skew) that count BOTH missing and present-but-NaN columns and
emit a structured WARNING when a non-trivial fraction were filled.
"""
import logging

import numpy as np
import pandas as pd
import pytest

from models.predictor import NBAPredictor

SENSITIVE = ["ARCH_PROB_0", "ARCH_PROB_1", "OPP_PRIOR_AVG_PTS", "KG_PREV_pts", "DVP_PTS"]
COLS = ["PTS_L5", "AST_L10", "TEAM_PACE", "position_encoded"] + SENSITIVE  # 9 columns


def _predictor(cols):
    # __new__ avoids loading models / FeatureEngine — the guard only needs
    # self.feature_columns and the class-level threshold/prefixes.
    p = NBAPredictor.__new__(NBAPredictor)
    p.feature_columns = list(cols)
    return p


def _full_row(cols):
    return pd.Series({c: 1.0 for c in cols})


def test_skew_report_clean_row_has_nothing_filled():
    rep = _predictor(COLS)._feature_skew_report(_full_row(COLS))
    assert rep["n_filled"] == 0
    assert rep["fraction_filled"] == 0.0
    assert rep["sensitive_filled"] == []


def test_skew_report_counts_missing_and_nan_including_sensitive():
    p = _predictor(COLS)
    row = _full_row(COLS)
    row["ARCH_PROB_0"] = np.nan          # present but NaN -> would be 0.0-filled
    row["OPP_PRIOR_AVG_PTS"] = np.nan      # present but NaN
    row = row.drop(["KG_PREV_pts", "PTS_L5"])  # missing entirely

    rep = p._feature_skew_report(row)

    assert rep["n_filled"] == 4
    assert rep["fraction_filled"] == pytest.approx(4 / len(COLS))
    # all three sensitive features that were missing/NaN must be reported
    assert set(rep["sensitive_filled"]) == {"ARCH_PROB_0", "OPP_PRIOR_AVG_PTS", "KG_PREV_pts"}


def test_check_feature_skew_warns_when_many_filled(caplog):
    p = _predictor(COLS)
    row = pd.Series({"PTS_L5": 1.0})  # 8/9 columns missing -> ~89%

    with caplog.at_level(logging.WARNING):
        rep = p._check_feature_skew("LeBron James", row)

    assert rep["fraction_filled"] >= p.SKEW_WARN_FRACTION
    warnings = [r for r in caplog.records if r.levelno >= logging.WARNING]
    assert any(
        "skew" in r.getMessage().lower() and "LeBron James" in r.getMessage()
        for r in warnings
    ), f"expected a structured skew warning naming the player; got {[r.getMessage() for r in warnings]}"


def test_check_feature_skew_silent_when_complete(caplog):
    p = _predictor(COLS)
    with caplog.at_level(logging.WARNING):
        p._check_feature_skew("LeBron James", _full_row(COLS))
    assert [r for r in caplog.records if r.levelno >= logging.WARNING] == []
