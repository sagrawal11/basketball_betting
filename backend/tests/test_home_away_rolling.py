"""Task 7.7 — home/away rolling features must not cross-contaminate.

HA_HOME_* should be a player's recent HOME-only form and HA_AWAY_* their recent
AWAY-only form. The old implementation rolled each venue subset, then
merge_asof-joined both back onto every row keyed on date only — so a home game
also received the most recent prior AWAY-rolled values (and vice-versa). The
venue split lost its meaning.

The fix computes each venue's rolling mean within that venue and attaches it
ONLY to that venue's rows; the other venue's columns stay NaN.

Data layout (chronological): 4 away games (PTS=30), then 4 home games (PTS=10),
then one away game. This guarantees the buggy code would borrow a *non-NaN*
opposite-venue value, making the contamination observable.
"""
import pandas as pd
import pytest

from feature_engine import FeatureEngine


def _eng() -> FeatureEngine:
    return FeatureEngine.__new__(FeatureEngine)  # method uses no instance state


def _frame() -> pd.DataFrame:
    rows = [
        ("2023-01-01", 0, 30),
        ("2023-01-02", 0, 30),
        ("2023-01-03", 0, 30),
        ("2023-01-04", 0, 30),
        ("2023-01-05", 1, 10),  # home game preceded by 4 away games
        ("2023-01-06", 1, 10),
        ("2023-01-07", 1, 10),
        ("2023-01-08", 1, 10),  # home game with >=3 prior home games
        ("2023-01-09", 0, 30),  # away game preceded by 4 home games
    ]
    df = pd.DataFrame(rows, columns=["GAME_DATE", "IS_HOME", "PTS"])
    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])
    df["MIN"] = 30.0
    df["FGM"] = 4.0
    return df


def _run():
    out = _eng()._add_home_away_rolling_features(_frame())
    return out.sort_values("GAME_DATE").reset_index(drop=True)


def _row(out, date):
    return out.loc[out["GAME_DATE"] == pd.Timestamp(date)].iloc[0]


def test_home_game_does_not_borrow_away_form():
    home = _row(_run(), "2023-01-05")
    assert pd.isna(home["HA_AWAY_PTS_L5"]), (
        f"home game carried away-form HA_AWAY_PTS_L5={home['HA_AWAY_PTS_L5']} "
        "(cross-contamination)"
    )


def test_away_game_does_not_borrow_home_form():
    away = _row(_run(), "2023-01-09")
    assert pd.isna(away["HA_HOME_PTS_L5"]), (
        f"away game carried home-form HA_HOME_PTS_L5={away['HA_HOME_PTS_L5']} "
        "(cross-contamination)"
    )


def test_home_form_uses_only_prior_home_games():
    out = _run()
    home = _row(out, "2023-01-08")  # prior home games are all PTS=10
    assert home["HA_HOME_PTS_L5"] == pytest.approx(10.0)
    assert pd.isna(home["HA_AWAY_PTS_L5"])


def test_away_form_uses_only_prior_away_games():
    out = _run()
    away = _row(out, "2023-01-09")  # prior away games are all PTS=30
    assert away["HA_AWAY_PTS_L5"] == pytest.approx(30.0)
    assert pd.isna(away["HA_HOME_PTS_L5"])


def test_ha_columns_are_always_present():
    out = _run()
    for side in ("HOME", "AWAY"):
        for col in ("PTS", "MIN", "FGM"):
            for w in (5, 10):
                assert f"HA_{side}_{col}_L{w}" in out.columns
