"""Task 7.4 — position_encoded must be a stable, frame-independent mapping.

The old code used ``pd.Categorical(df["position"]).codes``, whose integer codes
depend on which position strings happen to appear in the current frame. Because
training builds features one player at a time and inference builds a
single-player frame, the same position (e.g. a guard) could map to different
integers across players and between train and serve — categorical skew fed
straight into the model.

The fix derives position_encoded from the normalized POSITION_GROUP (G/F/C)
through a fixed dictionary, so the encoding never changes with frame content.
"""
import pandas as pd

from feature_engine import FeatureEngine


def _engine() -> FeatureEngine:
    # _prepare_raw only needs a bound instance (no heavy state) -> hermetic.
    return FeatureEngine.__new__(FeatureEngine)


def _frame(positions, start="2023-01-01") -> pd.DataFrame:
    n = len(positions)
    return pd.DataFrame(
        {
            "GAME_DATE": pd.date_range(start, periods=n, freq="D"),
            "position": positions,
        }
    )


def test_same_position_encodes_identically_across_frames():
    eng = _engine()
    # 'PG' co-occurs with different positions in each frame; per-frame Categorical
    # codes would assign it a different integer in each.
    a = eng._prepare_raw(_frame(["PG", "SF", "PG"]))
    b = eng._prepare_raw(_frame(["C", "PG", "C"]))

    code_a = a.loc[a["position"] == "PG", "position_encoded"].unique()
    code_b = b.loc[b["position"] == "PG", "position_encoded"].unique()

    assert len(code_a) == 1 and len(code_b) == 1
    assert code_a[0] == code_b[0], (
        f"'PG' encoded as {code_a[0]} in one frame but {code_b[0]} in another — "
        "per-frame categorical codes cause train-serve skew"
    )


def test_position_encoded_is_one_to_one_with_position_group():
    eng = _engine()
    # PG/SG/G-F all normalize to group G; SF/PF to F; C to C.
    df = eng._prepare_raw(_frame(["PG", "SG", "SF", "PF", "C", "G-F"]))

    codes_per_group = df.groupby("POSITION_GROUP")["position_encoded"].nunique()
    assert (codes_per_group == 1).all(), (
        "every POSITION_GROUP must have exactly one encoding; got "
        f"{codes_per_group.to_dict()}"
    )
    pairs = df[["POSITION_GROUP", "position_encoded"]].drop_duplicates()
    assert pairs["position_encoded"].nunique() == pairs["POSITION_GROUP"].nunique(), (
        "distinct position groups must map to distinct codes"
    )


def test_position_groups_map_to_fixed_codes():
    eng = _engine()
    df = eng._prepare_raw(_frame(["PG", "SF", "C"]))
    codes = dict(zip(df["POSITION_GROUP"], df["position_encoded"]))
    assert codes == {"G": 0, "F": 1, "C": 2}


def test_missing_position_defaults_to_guard_code():
    eng = _engine()
    df = eng._prepare_raw(
        pd.DataFrame({"GAME_DATE": pd.date_range("2023-01-01", periods=3, freq="D")})
    )
    assert (df["POSITION_GROUP"] == "G").all()
    # default must use the SAME code 'G' gets everywhere else (stability)
    g_ref = _engine()._prepare_raw(_frame(["PG", "SF", "C"]))
    g_code = g_ref.loc[g_ref["POSITION_GROUP"] == "G", "position_encoded"].iloc[0]
    assert (df["position_encoded"] == g_code).all()
