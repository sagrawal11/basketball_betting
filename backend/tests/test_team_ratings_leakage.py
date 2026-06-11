"""Task 7.3 — team ratings (TEAM_PACE / TEAM_OFF_RATING / TEAM_DEF_RATING) must
reflect only games that happened BEFORE the current game.

The old ``_merge_team_context`` merged ``all_team_stats`` on (team, SEASON),
i.e. a single FULL-SEASON aggregate. Every mid-season row therefore saw the
team's end-of-season rating — data from future games leaking into the features.

The fix computes a season-to-date rating from game-level data and attaches it
with ``merge_asof(direction="backward", allow_exact_matches=False)`` so a row
only ever sees strictly-prior games.
"""
import numpy as np
import pandas as pd
import pytest

import feature_engine as fe
from feature_engine import FeatureEngine

SEASON = "2023-24"
DATES = pd.to_datetime(
    ["2023-11-01", "2023-11-03", "2023-11-05", "2023-11-07", "2023-11-09"]
)
# Team AAA's per-game offensive rating. The team is weak early (90, 92, 94) and
# then spikes (130, 130). Full-season mean = 107.2 (the leaky value); the
# prior-only mean before the mid-season game on 11-05 is (90 + 92) / 2 = 91.
AAA_OFF = [90.0, 92.0, 94.0, 130.0, 130.0]
FULL_SEASON_AAA_OFF = float(np.mean(AAA_OFF))  # 107.2


def _season_aggregate() -> pd.DataFrame:
    """The full-season aggregate the OLD code merged (the leakage source)."""
    return pd.DataFrame(
        {
            "TEAM_ABBREVIATION": ["AAA", "BBB"],
            "SEASON": [SEASON, SEASON],
            "PACE": [100.0, 100.0],
            "OFF_RATING": [FULL_SEASON_AAA_OFF, 120.0],
            "DEF_RATING": [105.0, 108.0],
            "NET_RATING": [FULL_SEASON_AAA_OFF - 105.0, 12.0],
            "GP": [5, 5],
        }
    )


def _asof_timeline() -> pd.DataFrame:
    """Cumulative-through-game means — the schema the new builder produces.
    merge_asof(exclusive) over this yields the prior-games-only rating.
    """
    rows = []
    for team, vals in [("AAA", AAA_OFF), ("BBB", [120.0] * 5)]:
        cum = pd.Series(vals).expanding().mean().tolist()
        for d, c in zip(DATES, cum):
            rows.append(
                {
                    "team": team,
                    "game_date": d,
                    "season": SEASON,
                    "PACE": 100.0,
                    "OFF_RATING": c,
                    "DEF_RATING": 105.0,
                    "NET_RATING": c - 105.0,
                }
            )
    return pd.DataFrame(rows)


def _player_rows() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "GAME_DATE": DATES,
            "SEASON": [SEASON] * 5,
            "PLAYER_TEAM": ["AAA"] * 5,
            "OPPONENT_TEAM": ["BBB"] * 5,
        }
    )


def _engine() -> FeatureEngine:
    """Construct without __init__ so the test is hermetic (no disk / heavy load).
    _merge_team_context only depends on team_stats (old) / team_ratings (new).
    """
    eng = FeatureEngine.__new__(FeatureEngine)
    eng.team_stats = _season_aggregate()
    eng.team_ratings = _asof_timeline()
    return eng


def test_midseason_rating_uses_only_prior_games():
    out = _engine()._merge_team_context(_player_rows())
    mid = out.loc[out["GAME_DATE"] == pd.Timestamp("2023-11-05")].iloc[0]
    assert mid["TEAM_OFF_RATING"] == pytest.approx(91.0), (
        f"expected prior-only mean 91.0, got {mid['TEAM_OFF_RATING']}; "
        f"the full-season aggregate is {FULL_SEASON_AAA_OFF} (leakage)"
    )


def test_midseason_rating_is_not_the_full_season_aggregate():
    out = _engine()._merge_team_context(_player_rows())
    mid = out.loc[out["GAME_DATE"] == pd.Timestamp("2023-11-05")].iloc[0]
    assert mid["TEAM_OFF_RATING"] != pytest.approx(FULL_SEASON_AAA_OFF)
    # and it must not be pulled upward by the FUTURE 130-rated games
    assert mid["TEAM_OFF_RATING"] < 100.0


def test_first_game_of_season_has_no_prior_rating():
    out = _engine()._merge_team_context(_player_rows())
    first = out.loc[out["GAME_DATE"] == pd.Timestamp("2023-11-01")].iloc[0]
    assert pd.isna(first["TEAM_OFF_RATING"]), (
        "the first game has no prior games; a non-null rating means the current "
        "game or a season aggregate leaked in"
    )


def test_opponent_rating_also_prior_only():
    # Opponent BBB is constant 120, so any prior-only mean is 120; the point is
    # the OPP_* columns exist and come from the as-of timeline, not the aggregate.
    out = _engine()._merge_team_context(_player_rows())
    mid = out.loc[out["GAME_DATE"] == pd.Timestamp("2023-11-05")].iloc[0]
    assert mid["OPP_OFF_RATING"] == pytest.approx(120.0)


def test_builder_produces_cumulative_prior_inclusive_ratings(tmp_path):
    games = pd.DataFrame(
        {
            "team": ["AAA"] * 5,
            "season": [SEASON] * 5,
            "game_date": DATES,
            "off_rtg": AAA_OFF,
            "def_rtg": [105.0] * 5,
            "pace": [100.0] * 5,
            "net_rtg": [v - 105.0 for v in AAA_OFF],
            "game_id": list(range(5)),
            "team_id": [1] * 5,
        }
    )
    p = tmp_path / "kaggle_team_games.parquet"
    games.to_parquet(p, index=False)

    tl = fe._load_team_ratings_timeline(p)
    assert tl is not None
    tl = tl.sort_values("game_date").reset_index(drop=True)
    # cumulative mean THROUGH each game (so merge_asof-exclusive => prior only)
    assert tl["OFF_RATING"].tolist() == pytest.approx([90.0, 91.0, 92.0, 101.5, 107.2])
