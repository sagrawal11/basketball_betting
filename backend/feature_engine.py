"""
Unified feature engineering for training and inference (single code path).
Rolling stats use shift(1) before rolling to exclude the current game.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from config.paths import (
    DATA_DIR,
    DVP_PARQUET,
    PLAYER_DATA_DIR,
    TEAM_STATS_CSV,
    KAGGLE_TEAM_GAMES_PARQUET,
    ARCHETYPE_PIPELINE_PATH,
    FEATURE_COLUMNS_PATH,
    ARTIFACTS_DIR,
)
from archetype_engine import ArchetypeEngine

logger = logging.getLogger(__name__)

TARGET_STATS = ["PTS", "OREB", "DREB", "AST", "FTM", "FG3M", "FGM", "STL", "BLK", "TOV"]
ROLL_WINDOWS = [3, 5, 10, 20]

# Identifier-like numeric columns that must never be used as model features.
ID_LIKE = {"Player_ID", "Game_ID", "SEASON_ID", "VIDEO_AVAILABLE", "ARCH_ARGMAX"}
# Home / away–only rolling (separate from global rolls); keep small to limit width
HA_ROLL_WINDOWS = (5, 10)
HA_ROLL_COLS = ("PTS", "MIN", "FGM")


def _slug(name: str) -> str:
    return name.replace(" ", "_")


def _position_group(pos: Any) -> str:
    if pd.isna(pos):
        return "G"
    s = str(pos).upper()
    if "G" in s or "GUARD" in s:
        return "G"
    if "F" in s or "FORWARD" in s or "WING" in s:
        return "F"
    if "C" in s or "CENTER" in s:
        return "C"
    return "G"


# Fixed, frame-independent encoding for position groups. Order is arbitrary but
# must stay constant so the same position always maps to the same integer across
# players and between training and inference (no per-frame Categorical codes).
POSITION_GROUP_ENCODING = {"G": 0, "F": 1, "C": 2}


def _load_team_stats(path: Path) -> Optional[pd.DataFrame]:
    """Prefer `all_team_stats.parquet` next to CSV; fallback to CSV."""
    path = Path(path)
    pq = path.with_suffix(".parquet")
    if pq.exists():
        ts = pd.read_parquet(pq)
    elif path.exists():
        ts = pd.read_csv(path)
    else:
        logger.warning("Team stats not found at %s or %s", pq, path)
        return None
    if "TEAM_ABBREVIATION" not in ts.columns or "SEASON" not in ts.columns:
        return None
    ts = ts.drop_duplicates(subset=["TEAM_ABBREVIATION", "SEASON"], keep="last")
    return ts


# Do not shift these — IDs or non-meaningful numerics for prior-game context
_KG_SHIFT_EXCLUDE = frozenset({"game_id", "season", "team_id", "sub_season_id", "min"})


def _load_kaggle_prior_game_features(parquet_path: Path) -> Optional[pd.DataFrame]:
    """
    One row per (team, game_date): numeric columns are **previous game** values (groupby team shift 1).
    Avoids same-game leakage vs targets. Merge keys: team + game_date (normalized).
    """
    path = Path(parquet_path)
    if not path.exists():
        logger.warning("Kaggle team games parquet missing at %s — KG_PREV_* features off", path)
        return None
    kg = pd.read_parquet(path)
    if "team" not in kg.columns or "game_date" not in kg.columns:
        return None
    kg = kg.copy()
    kg["game_date"] = pd.to_datetime(kg["game_date"]).dt.normalize()
    kg = kg.sort_values(["team", "game_date"])
    num_cols = [
        c
        for c in kg.select_dtypes(include=[np.number]).columns
        if c not in _KG_SHIFT_EXCLUDE
    ]
    out = kg[["team", "game_date"]].copy()
    for c in num_cols:
        out[f"KG_PREV_{c}"] = kg.groupby("team", sort=False)[c].shift(1)
    return out


# Game-level rating columns (Kaggle) -> output rating names. Combined with an
# as-of (prior-only) merge these replace the leaky full-season aggregates.
_TEAM_RATING_SOURCES = {
    "pace": "PACE",
    "off_rtg": "OFF_RATING",
    "def_rtg": "DEF_RATING",
    "net_rtg": "NET_RATING",
}


def _load_team_ratings_timeline(parquet_path: Path) -> Optional[pd.DataFrame]:
    """Per (team, game_date) season-to-date team ratings.

    Each value is the cumulative mean THROUGH that game, computed within
    (team, season). Paired with ``merge_asof(direction="backward",
    allow_exact_matches=False)`` this yields a rating built only from games that
    happened strictly before the row's game — no same-game or future leakage,
    unlike the old full-season aggregate keyed on (team, SEASON).
    """
    path = Path(parquet_path)
    if not path.exists():
        logger.warning("Kaggle team games parquet missing at %s — team ratings off", path)
        return None
    try:
        kg = pd.read_parquet(path)
    except Exception:
        return None
    if "team" not in kg.columns or "game_date" not in kg.columns:
        return None
    present_src = [c for c in _TEAM_RATING_SOURCES if c in kg.columns]
    if not present_src:
        return None
    kg = kg.copy()
    kg["game_date"] = pd.to_datetime(kg["game_date"]).dt.normalize()
    group_keys = ["team", "season"] if "season" in kg.columns else ["team"]
    kg = kg.sort_values(group_keys + ["game_date"])
    out = kg[["team", "game_date"]].copy()
    if "season" in kg.columns:
        out["season"] = kg["season"]
    grp = kg.groupby(group_keys, sort=False)
    for src in present_src:
        out[_TEAM_RATING_SOURCES[src]] = grp[src].transform(lambda s: s.expanding().mean())
    return out


def _load_dvp() -> Optional[pd.DataFrame]:
    if not DVP_PARQUET.exists():
        return None
    try:
        return pd.read_parquet(DVP_PARQUET)
    except Exception:
        return None


def _build_kaggle_team_schedule_meta(parquet_path: Path) -> Optional[pd.DataFrame]:
    """Per team game: days since that team's previous game, B2B flag, playoff flag."""
    path = Path(parquet_path)
    if not path.exists():
        return None
    try:
        kg = pd.read_parquet(path, columns=["team", "game_date", "is_playoff"])
    except Exception:
        return None
    kg = kg.copy()
    kg["game_date"] = pd.to_datetime(kg["game_date"]).dt.normalize()
    kg = kg.sort_values(["team", "game_date"])
    kg["team_days_rest"] = kg.groupby("team", sort=False)["game_date"].diff().dt.days
    kg["team_is_b2b"] = (kg["team_days_rest"] <= 1).fillna(False).astype(int)
    kg.loc[kg["team_days_rest"].isna(), "team_is_b2b"] = 0
    if "is_playoff" in kg.columns:
        kg["is_playoff"] = kg["is_playoff"].fillna(0).astype(int)
    return kg


class FeatureEngine:
    """Builds training and inference features from raw player game logs."""

    def __init__(
        self,
        player_data_dir: Optional[Path] = None,
        team_stats_path: Optional[Path] = None,
        archetype_engine: Optional[ArchetypeEngine] = None,
    ):
        self.player_data_dir = Path(player_data_dir or PLAYER_DATA_DIR)
        self.team_stats_path = Path(team_stats_path or TEAM_STATS_CSV)
        self.team_stats = _load_team_stats(self.team_stats_path)
        # Prior-games-only team ratings (replaces the leaky full-season aggregate
        # that team_stats provided to _merge_team_context).
        self.team_ratings = _load_team_ratings_timeline(KAGGLE_TEAM_GAMES_PARQUET)
        self.kaggle_prior_games = _load_kaggle_prior_game_features(KAGGLE_TEAM_GAMES_PARQUET)
        self._kaggle_team_schedule_meta: Optional[pd.DataFrame] = None
        self.dvp = _load_dvp()
        self.archetype_engine = archetype_engine

    def _get_kaggle_team_schedule_meta(self) -> Optional[pd.DataFrame]:
        if self._kaggle_team_schedule_meta is None:
            self._kaggle_team_schedule_meta = _build_kaggle_team_schedule_meta(KAGGLE_TEAM_GAMES_PARQUET)
        return self._kaggle_team_schedule_meta

    def _merge_schedule_context_from_kaggle(self, df: pd.DataFrame) -> pd.DataFrame:
        """Opponent days rest / B2B and playoff flag from Kaggle team game rows."""
        meta = self._get_kaggle_team_schedule_meta()
        if meta is None or len(meta) == 0:
            df["OPP_DAYS_REST"] = np.nan
            df["OPP_IS_B2B"] = np.nan
            df["IS_PLAYOFF_GAME"] = np.nan
            return df
        x = df.copy()
        x["_dt"] = pd.to_datetime(x["GAME_DATE"]).dt.normalize()
        opp = meta.rename(
            columns={
                "team": "OPPONENT_TEAM",
                "game_date": "_gdo",
                "team_days_rest": "OPP_DAYS_REST",
                "team_is_b2b": "OPP_IS_B2B",
            }
        )
        x = x.merge(
            opp[["OPPONENT_TEAM", "_gdo", "OPP_DAYS_REST", "OPP_IS_B2B"]],
            left_on=["OPPONENT_TEAM", "_dt"],
            right_on=["OPPONENT_TEAM", "_gdo"],
            how="left",
        )
        x = x.drop(columns=["_gdo"], errors="ignore")
        ply = meta.rename(columns={"team": "PLAYER_TEAM", "game_date": "_gdp"})
        x = x.merge(
            ply[["PLAYER_TEAM", "_gdp", "is_playoff"]].rename(columns={"is_playoff": "IS_PLAYOFF_GAME"}),
            left_on=["PLAYER_TEAM", "_dt"],
            right_on=["PLAYER_TEAM", "_gdp"],
            how="left",
        )
        x = x.drop(columns=["_gdp"], errors="ignore")
        x = x.drop(columns=["_dt"], errors="ignore")
        return x

    def _add_home_away_rolling_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prior-only rolling means computed WITHIN each venue.

        ``HA_HOME_*`` is populated only on home-game rows (the mean of that
        player's prior HOME games) and ``HA_AWAY_*`` only on away-game rows; the
        opposite venue's columns stay NaN. This avoids the cross-contamination
        of the old merge_asof-on-date approach, where every row (home or away)
        received the most recent prior values from BOTH venues — so a home game
        carried away-form and vice-versa, and same-day rows were collapsed.
        """
        out = df.copy().reset_index(drop=True)
        if "IS_HOME" not in out.columns:
            return out
        cols_use = [c for c in HA_ROLL_COLS if c in out.columns]
        if not cols_use:
            return out

        # Pre-create the full HA schema so the columns are deterministically
        # present (other-venue rows simply remain NaN).
        for side_name in ("HOME", "AWAY"):
            for col in cols_use:
                for w in HA_ROLL_WINDOWS:
                    out[f"HA_{side_name}_{col}_L{w}"] = np.nan

        dt = pd.to_datetime(out["GAME_DATE"]).dt.normalize()
        chrono = dt.sort_values(kind="stable").index  # chronological row labels

        for side_name, home_val in (("HOME", 1), ("AWAY", 0)):
            mask = (out.loc[chrono, "IS_HOME"] == home_val).to_numpy()
            side_idx = chrono[mask]
            if len(side_idx) == 0:
                continue
            for col in cols_use:
                # shift(1) over this venue's chronological games -> prior only.
                shifted = out.loc[side_idx, col].shift(1)
                for w in HA_ROLL_WINDOWS:
                    rolled = shifted.rolling(w, min_periods=max(2, min(3, w))).mean()
                    out.loc[side_idx, f"HA_{side_name}_{col}_L{w}"] = rolled
        return out

    def _prepare_raw(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df = df.replace("", np.nan).replace(" ", np.nan)
        if "GAME_DATE" in df.columns:
            df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])
        df = df.sort_values("GAME_DATE").reset_index(drop=True)
        if "WL" in df.columns:
            df = df.drop(columns=["WL"])
        if "position" in df.columns:
            df["POSITION_GROUP"] = df["position"].apply(_position_group)
        elif "POSITION_GROUP" not in df.columns:
            df["POSITION_GROUP"] = "G"
        # Stable encoding derived from the normalized group via a fixed mapping —
        # never per-frame Categorical codes (which drift with frame content).
        df["position_encoded"] = (
            df["POSITION_GROUP"].map(POSITION_GROUP_ENCODING).fillna(0).astype(int)
        )
        if "IS_HOME" not in df.columns and "MATCHUP" in df.columns:
            df["IS_HOME"] = df["MATCHUP"].astype(str).str.contains("vs.", na=False).astype(int)
        if "height" in df.columns:
            df["height_inches"] = df["height"].map(self._height_to_inches)
        return df

    @staticmethod
    def _height_to_inches(h: Any) -> float:
        try:
            if isinstance(h, str) and "-" in h:
                a, b = h.split("-", 1)
                return int(a) * 12 + int(b)
        except Exception:
            pass
        return np.nan

    def _merge_team_context(self, df: pd.DataFrame) -> pd.DataFrame:
        """Attach season-to-date team & opponent ratings using ONLY games that
        happened before each row's game (no same-game or future leakage).

        Uses the prior-only ``team_ratings`` timeline + merge_asof(backward,
        exclusive), replacing the old full-season aggregate keyed on (team,
        SEASON) which leaked end-of-season data into every mid-season row.
        """
        rating_out = ["PACE", "OFF_RATING", "DEF_RATING", "NET_RATING"]
        timeline = getattr(self, "team_ratings", None)
        if timeline is None or len(timeline) == 0:
            for c in rating_out:
                df[f"TEAM_{c}"] = np.nan
                df[f"OPP_{c}"] = np.nan
            return df

        present = [c for c in rating_out if c in timeline.columns]
        x = df.copy()
        x["_gd"] = pd.to_datetime(x["GAME_DATE"]).dt.normalize()
        x["_idx"] = np.arange(len(x), dtype=np.int64)

        # Player team: most recent STRICTLY-prior team game.
        tl_team = timeline.rename(
            columns={"team": "PLAYER_TEAM", "game_date": "_gdt",
                     **{c: f"TEAM_{c}" for c in present}}
        )
        team_cols = [f"TEAM_{c}" for c in present]
        m = pd.merge_asof(
            x.sort_values("_gd"),
            tl_team[["PLAYER_TEAM", "_gdt"] + team_cols].sort_values("_gdt"),
            left_on="_gd",
            right_on="_gdt",
            by="PLAYER_TEAM",
            direction="backward",
            allow_exact_matches=False,
        )
        m = m.drop(columns=["_gdt"], errors="ignore")

        # Opponent team: same prior-only as-of logic.
        tl_opp = timeline.rename(
            columns={"team": "OPPONENT_TEAM", "game_date": "_gdo",
                     **{c: f"OPP_{c}" for c in present}}
        )
        opp_cols = [f"OPP_{c}" for c in present]
        m = pd.merge_asof(
            m.sort_values("_gd"),
            tl_opp[["OPPONENT_TEAM", "_gdo"] + opp_cols].sort_values("_gdo"),
            left_on="_gd",
            right_on="_gdo",
            by="OPPONENT_TEAM",
            direction="backward",
            allow_exact_matches=False,
        )
        m = (
            m.sort_values("_idx")
            .drop(columns=["_idx", "_gd", "_gdo"], errors="ignore")
            .reset_index(drop=True)
        )

        # Guarantee a stable column schema even if a source rating was absent.
        for c in rating_out:
            if f"TEAM_{c}" not in m.columns:
                m[f"TEAM_{c}"] = np.nan
            if f"OPP_{c}" not in m.columns:
                m[f"OPP_{c}"] = np.nan
        return m

    def _merge_kaggle_prior_game_context(self, df: pd.DataFrame) -> pd.DataFrame:
        """Attach prior-game team box stats from Kaggle (shifted); opponent side with KG_OPP_PREV_*."""
        prior = self.kaggle_prior_games
        if prior is None or len(prior) == 0:
            return df
        
        # We must use merge_asof to handle future games where the exact date isn't in 'prior' yet
        x = df.copy()
        x["_gd"] = pd.to_datetime(x["GAME_DATE"]).dt.normalize()
        x["_idx"] = np.arange(len(x), dtype=np.int64)
        
        # 1. Merge player team context
        pl = prior.rename(columns={"team": "PLAYER_TEAM", "game_date": "_gd_pl"})
        kgp = [c for c in pl.columns if c.startswith("KG_PREV_")]
        
        m = pd.merge_asof(
            x.sort_values("_gd"),
            pl[["PLAYER_TEAM", "_gd_pl"] + kgp].sort_values("_gd_pl"),
            left_on="_gd",
            right_on="_gd_pl",
            by="PLAYER_TEAM",
            direction="backward",
            allow_exact_matches=False
        )
        m = m.drop(columns=["_gd_pl"], errors="ignore")
        
        # 2. Merge opponent team context
        op = prior.rename(columns={"team": "OPPONENT_TEAM", "game_date": "_gd_op"})
        ren = {c: c.replace("KG_PREV_", "KG_OPP_PREV_", 1) for c in kgp}
        op = op.rename(columns=ren)
        kgo = list(ren.values())
        
        m = pd.merge_asof(
            m.sort_values("_gd"),
            op[["OPPONENT_TEAM", "_gd_op"] + kgo].sort_values("_gd_op"),
            left_on="_gd",
            right_on="_gd_op",
            by="OPPONENT_TEAM",
            direction="backward",
            allow_exact_matches=False
        )
        
        m = m.sort_values("_idx").drop(columns=["_idx", "_gd", "_gd_op"], errors="ignore")
        return m

    def _merge_dvp(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.dvp is None or len(self.dvp) == 0:
            df["DVP_PTS"] = np.nan
            df["DVP_REB"] = np.nan
            df["DVP_AST"] = np.nan
            return df
        dvp = self.dvp.copy()
        
        if "GAME_DATE" not in dvp.columns:
            # Fallback to old format if Parquet isn't rebuilt yet
            keys = [k for k in ["OPPONENT_TEAM", "SEASON", "POSITION_GROUP"] if k in dvp.columns]
            sub = dvp[keys + ["PTS_ALLOWED_PER36", "REB_ALLOWED_PER36", "AST_ALLOWED_PER36"]].drop_duplicates(subset=keys, keep="last")
            return df.merge(sub.rename(columns={"PTS_ALLOWED_PER36": "DVP_PTS", "REB_ALLOWED_PER36": "DVP_REB", "AST_ALLOWED_PER36": "DVP_AST"}), on=keys, how="left")
            
        dvp["_dt"] = pd.to_datetime(dvp["GAME_DATE"]).dt.normalize()
        x = df.copy()
        # Protect against empty or missing matches
        if "GAME_DATE" not in x.columns:
            x["DVP_PTS"] = np.nan
            x["DVP_REB"] = np.nan
            x["DVP_AST"] = np.nan
            return x

        x["_dt"] = pd.to_datetime(x["GAME_DATE"]).dt.normalize()
        x["_idx"] = np.arange(len(x), dtype=np.int64)
        
        sub = dvp[["OPPONENT_TEAM", "POSITION_GROUP", "_dt", "PTS_ALLOWED_PER36", "REB_ALLOWED_PER36", "AST_ALLOWED_PER36"]].dropna(subset=["_dt"])
        
        m = pd.merge_asof(
            x.sort_values("_dt"),
            sub.sort_values("_dt"),
            on="_dt",
            by=["OPPONENT_TEAM", "POSITION_GROUP"],
            direction="backward",
            allow_exact_matches=False
        )
        
        m = m.sort_values("_idx").drop(columns=["_idx", "_dt"], errors="ignore")
        m = m.rename(columns={"PTS_ALLOWED_PER36": "DVP_PTS", "REB_ALLOWED_PER36": "DVP_REB", "AST_ALLOWED_PER36": "DVP_AST"})
        return m

    def _merge_clutch_stats(self, df: pd.DataFrame) -> pd.DataFrame:
        """Merge clutch time logic built from the SQLite Advanced dataset."""
        clutch_path = DATA_DIR / "auxiliary" / "clutch_stats.parquet"
        if not clutch_path.exists():
            df["CLUTCH_SHOTS_ATT"] = np.nan
            df["CLUTCH_SHOTS_MADE"] = np.nan
            return df
            
        try:
            clutch = pd.read_parquet(clutch_path)
            # Safe forward-fill merge_asof based on game date just like DvP
            if "GAME_DATE" in clutch.columns and "PLAYER_NAME" in clutch.columns:
                clutch["_dt"] = pd.to_datetime(clutch["GAME_DATE"]).dt.normalize()
                x = df.copy()
                x["_dt"] = pd.to_datetime(x["GAME_DATE"]).dt.normalize()
                x["_idx"] = np.arange(len(x), dtype=np.int64)
                
                m = pd.merge_asof(
                    x.sort_values("_dt"),
                    clutch.dropna(subset=["_dt"]).sort_values("_dt"),
                    on="_dt",
                    by="PLAYER_NAME",
                    direction="backward",
                    allow_exact_matches=False
                )
                return m.sort_values("_idx").drop(columns=["_idx", "_dt"], errors="ignore")
        except Exception:
            pass
            
        df["CLUTCH_SHOTS_ATT"] = np.nan
        df["CLUTCH_SHOTS_MADE"] = np.nan
        return df

    def _add_advanced_ratios(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        
        # True Shooting % -> TS = PTS / (2 * (FGA + 0.44 * FTA))
        if "FGA" in out.columns and "FTA" in out.columns and "PTS" in out.columns:
            denom_ts = 2 * (out["FGA"] + 0.44 * out["FTA"])
            # avoid div by zero
            out["TS_PCT"] = np.where(denom_ts > 0, out["PTS"] / denom_ts, 0.0)
        else:
            out["TS_PCT"] = np.nan
            
        # Pace/Possession adjustment
        pace = out.get("TEAM_PACE", out.get("OPP_PACE", 100.0))
        pace = pace.fillna(100.0).clip(lower=1.0)
        
        if "MIN" in out.columns:
            mins = out["MIN"].fillna(0.0).clip(lower=1.0)
            # Factor = 4800 / (pace * mins) => scale stat up to Per-100-Poss
            factor = 4800.0 / (pace * mins)
            
            for s in ["PTS", "AST", "REB", "OREB", "DREB", "STL", "BLK", "TOV", "FGA", "FG3A"]:
                if s in out.columns:
                    out[f"{s}_PER100"] = out[s] * factor
                    
            if "FGA" in out.columns and "FTA" in out.columns and "TOV" in out.columns:
                # Approximate Usage %
                out["USG_PCT"] = 100.0 * ((out["FGA"] + 0.44 * out["FTA"] + out["TOV"]) * 48.0) / (mins * pace)
            else:
                out["USG_PCT"] = np.nan
        else:
            out["USG_PCT"] = np.nan
            
        return out

    def _rolling_features(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        roll_cols = [
            "PTS",
            "OREB",
            "DREB",
            "AST",
            "STL",
            "BLK",
            "TOV",
            "FGM",
            "FG3M",
            "FTM",
            "MIN",
            "TS_PCT",
            "USG_PCT",
            "PTS_PER100",
            "AST_PER100",
            "REB_PER100",
        ]
        for c in roll_cols:
            if c not in out.columns:
                out[c] = np.nan
        new_cols = {}
        for w in ROLL_WINDOWS:
            for c in roll_cols:
                shifted = out[c].shift(1)
                new_cols[f"{c}_L{w}"] = shifted.rolling(window=w, min_periods=max(2, min(3, w))).mean()
                new_cols[f"{c}_L{w}_STD"] = shifted.rolling(window=w, min_periods=max(2, min(3, w))).std()
        new_cols["DAYS_REST"] = out["GAME_DATE"].diff().dt.days.clip(lower=0).fillna(3)
        new_cols["IS_B2B"] = (new_cols["DAYS_REST"] <= 1).astype(int)

        # Vectorized GAMES_LAST_7: count games within trailing 7-day window
        dates = out["GAME_DATE"].values
        g7 = np.zeros(len(out), dtype=np.int64)
        for i in range(1, len(out)):
            cutoff = dates[i] - np.timedelta64(7, "D")
            g7[i] = int(np.sum((dates[:i] >= cutoff)))
        new_cols["GAMES_LAST_7"] = g7

        out = pd.concat([out, pd.DataFrame(new_cols, index=out.index)], axis=1)

        if "MIN_L10" in out.columns and "MIN_L10_STD" in out.columns:
            out["MIN_CV_L10"] = out["MIN_L10_STD"] / (out["MIN_L10"].clip(lower=0) + 1e-6)
        return out

    def _expanding_opponent_means(self, df: pd.DataFrame, stat: str) -> pd.Series:
        """Expanding mean of target vs opponent from strictly prior games (no leakage)."""
        if "OPPONENT_TEAM" not in df.columns or stat not in df.columns:
            return pd.Series(np.nan, index=df.index)
        out: List[float] = []
        cum: Dict[Any, Tuple[float, int]] = {}
        for i in range(len(df)):
            row = df.iloc[i]
            opp = row.get("OPPONENT_TEAM")
            prior = cum.get(opp, (0.0, 0))
            out.append(prior[0] / prior[1] if prior[1] > 0 else np.nan)
            v = row[stat]
            if pd.notna(opp) and pd.notna(v):
                s, c = cum.get(opp, (0.0, 0))
                cum[opp] = (s + float(v), c + 1)
        return pd.Series(out, index=df.index)

    def build_training_features(
        self,
        player_name: str,
        save_parquet: bool = True,
    ) -> Optional[pd.DataFrame]:
        slug = _slug(player_name)
        player_dir = self.player_data_dir / slug
        raw_path = player_dir / f"{slug}_data.csv"
        if not raw_path.exists():
            return None
        df = pd.read_csv(raw_path)
        return self._build_from_dataframe(df, player_name, slug, save_parquet)

    def _build_from_dataframe(
        self,
        df: pd.DataFrame,
        player_name: str,
        slug: str,
        save_parquet: bool,
    ) -> pd.DataFrame:
        df = self._prepare_raw(df)
        df = self._merge_team_context(df)
        df = self._merge_kaggle_prior_game_context(df)
        df = self._merge_dvp(df)
        df = self._merge_clutch_stats(df)
        df = self._merge_schedule_context_from_kaggle(df)
        df = self._add_advanced_ratios(df)
        df = self._rolling_features(df)
        df = self._add_home_away_rolling_features(df)
        prior_cols = {f"OPP_PRIOR_AVG_{t}": self._expanding_opponent_means(df, t) for t in TARGET_STATS}
        df = pd.concat([df, pd.DataFrame(prior_cols)], axis=1)
        if self.archetype_engine is not None:
            # transform_frame guarantees a fixed ARCH_PROB_* schema (prior-filled
            # on failure); no longer swallowed, so a genuine bug fails loudly
            # instead of silently dropping the archetype columns.
            df = self.archetype_engine.transform_frame(df)
        df["PLAYER_NAME"] = player_name
        df = self._clean_output_columns(df)
        if save_parquet:
            out_dir = self.player_data_dir / slug
            out_dir.mkdir(parents=True, exist_ok=True)
            pq = out_dir / f"{slug}_features.parquet"
            try:
                df.to_parquet(pq, index=False)
            except Exception as e:
                logger.warning("Parquet save failed for %s: %s", slug, e)
        return df

    def _clean_output_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Strictly whitelist only managed features and required metadata."""
        # Identification / Metadata
        keep = [
            "GAME_DATE", "PLAYER_NAME", "OPPONENT_TEAM", "PLAYER_TEAM", "SEASON", 
            "MATCHUP", "Game_ID", "Player_ID", "POSITION_GROUP", "position_encoded"
        ]

        # Targets (needed for training parquets)
        keep.extend(TARGET_STATS)
        
        # Managed Features
        managed_prefixes = [
            "KG_PREV_", "KG_OPP_PREV_", "DVP_", "CLUTCH_", "OPP_", 
            "HA_HOME_", "HA_AWAY_", "ARCH_PROB_", "OPP_PRIOR_AVG_"
        ]
        
        # Exact Managed Stat Names.
        # NOTE: raw same-game TS_PCT / USG_PCT are intentionally absent here —
        # they are leaky (computed from the current game) and are dropped by the
        # explicit exclusion below. Only their rolling (_L*) versions are kept.
        managed_exact = [
            "IS_HOME", "height_inches", "TEAM_PACE", "TEAM_OFF_RATING",
            "TEAM_DEF_RATING", "TEAM_NET_RATING", "REST_FACTOR",
            "DAYS_REST", "IS_B2B", "GAMES_LAST_7",
            "MIN_CV_L10", "IS_PLAYOFF_GAME", "OPPONENT_STRENGTH"
        ]
        
        # Per-100 and Rolling patterns
        for c in df.columns:
            # Check prefixes
            if any(c.startswith(p) for p in managed_prefixes):
                keep.append(c)
                continue
            # Check exacts
            if c in managed_exact:
                keep.append(c)
                continue
            # Check patterns like PTS_PER100 or PTS_L3
            # IMPORTANT: We only want ROLLING per-100 or rolling ratios. 
            # Non-rolling ratios (TS_PCT, USG_PCT, PTS_PER100) are LEAKY.
            if "_L3" in c or "_L5" in c or "_L10" in c or "_L20" in c:
                # This catches PTS_PER100_L3 AND TS_PCT_L3, which is what we want.
                keep.append(c)
                continue
            
            # If we reach here, and it's TS_PCT or USG_PCT or raw PER100, we EXCLUDE it.
            if c in ("TS_PCT", "USG_PCT") or "_PER100" in c:
                continue
        
        # Unique set and filter
        keep = list(dict.fromkeys([c for c in keep if c in df.columns]))
        return df[keep]

    def build_inference_row(
        self,
        player_name: str,
        opponent_team: str,
        is_home: bool,
        game_date: str,
        season: str,
    ) -> Optional[pd.Series]:
        """Single feature vector for an upcoming game (uses only past games)."""
        slug = _slug(player_name)
        raw_path = self.player_data_dir / slug / f"{slug}_data.csv"
        if not raw_path.exists():
            return None
        hist = pd.read_csv(raw_path)
        hist = self._prepare_raw(hist)
        next_date = pd.to_datetime(game_date)
        last = hist.iloc[-1]
        placeholder = {c: last.get(c) for c in hist.columns}
        for c in hist.columns:
            if c in (
                "PTS", "OREB", "DREB", "AST", "STL", "BLK", "TOV", "FGM", "FGA", "FG3M", "FG3A", "FTM", "FTA",
                "MIN", "PF", "REB", "PLUS_MINUS", "FG_PCT", "FG3_PCT", "FT_PCT",
            ):
                placeholder[c] = np.nan
        placeholder["GAME_DATE"] = next_date
        placeholder["OPPONENT_TEAM"] = opponent_team
        placeholder["PLAYER_TEAM"] = last.get("PLAYER_TEAM")
        placeholder["SEASON"] = season
        placeholder["IS_HOME"] = int(is_home)
        pt = placeholder.get("PLAYER_TEAM", "")
        placeholder["MATCHUP"] = f"{pt} vs. {opponent_team}" if is_home else f"{pt} @ {opponent_team}"
        extended = pd.concat([hist, pd.DataFrame([placeholder])], ignore_index=True)
        built = self._build_from_dataframe(extended, player_name, slug, save_parquet=False)
        return built.iloc[-1]


def load_feature_columns() -> Optional[List[str]]:
    if not FEATURE_COLUMNS_PATH.exists():
        return None
    with open(FEATURE_COLUMNS_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("columns")


def save_feature_columns(columns: List[str]) -> None:
    # Guard against destructive overwrites: if a previous trainer already wrote a
    # different feature set, surface it loudly. With a unified selection
    # (split_features_targets) every trainer writes the same columns, so a
    # mismatch here means the skew bug has reappeared.
    existing = load_feature_columns()
    if existing is not None and existing != columns:
        added = [c for c in columns if c not in existing]
        removed = [c for c in existing if c not in columns]
        logger.warning(
            "Overwriting feature_columns.json with a DIFFERENT feature set "
            "(added=%s, removed=%s). All trainers should agree via "
            "split_features_targets().",
            added,
            removed,
        )
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(FEATURE_COLUMNS_PATH, "w", encoding="utf-8") as f:
        json.dump({"columns": columns}, f, indent=2)


def default_engine() -> FeatureEngine:
    # If an archetype model is present it MUST load — a corrupt/unreadable model
    # silently becoming None would drop ARCH_PROB_* columns and cause train-serve
    # skew. Absence of the file is allowed (pre-archetype bootstrap phase).
    arch = None
    if ARCHETYPE_PIPELINE_PATH.exists():
        arch = ArchetypeEngine.load(ARCHETYPE_PIPELINE_PATH)
    return FeatureEngine(archetype_engine=arch)


def split_features_targets(
    df: pd.DataFrame,
    engine: Optional[FeatureEngine] = None,
) -> Tuple[pd.DataFrame, List[str], List[str], pd.Series]:
    """Single source of truth for feature/target column selection.

    Every trainer (train_global.py, train_hybrid.py) MUST call this so that the
    feature set written to feature_columns.json is identical no matter which
    trainer ran last. Diverging selection here is the root cause of train-serve
    skew.

    Returns ``(X_numeric, feature_cols, target_cols, seasons)`` where:
      * the frame is chronologically sorted by GAME_DATE (when present),
      * columns are restricted to the managed whitelist via
        ``FeatureEngine._clean_output_columns``,
      * targets and identifier-like columns are removed from the feature matrix.
    """
    if engine is None:
        engine = default_engine()

    df = df.copy()
    if "GAME_DATE" in df.columns:
        df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])
        df = df.sort_values("GAME_DATE").reset_index(drop=True)

    seasons = df["SEASON"] if "SEASON" in df.columns else pd.Series(["Unknown"] * len(df))

    # Restrict to the strictly-managed feature whitelist (the same cleaning the
    # FeatureEngine applies when it writes training parquets and inference rows).
    df = engine._clean_output_columns(df)

    num = df.select_dtypes(include=[np.number])
    y_cols = [c for c in TARGET_STATS if c in num.columns]
    drop_cols = [c for c in num.columns if c in TARGET_STATS or c in ID_LIKE]
    X_num = num.drop(columns=drop_cols, errors="ignore")
    X_cols = list(X_num.columns)
    return X_num, X_cols, y_cols, seasons
