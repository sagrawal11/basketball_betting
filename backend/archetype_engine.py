"""
Data-driven player archetypes: PCA + Gaussian Mixture on shifted rolling per-36 profiles.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

TARGET_FEATURES = [
    "PTS_PER100_L10", "AST_PER100_L10", "REB_PER100_L10",
    "STL_L10", "BLK_L10", "TOV_L10",
    "USG_PCT_L10", "TS_PCT_L10", "MIN_L10"
]

class ArchetypeEngine:
    """PCA + GMM on shifted rolling per-36 features; soft cluster probabilities per game."""

    def __init__(
        self,
        n_pca: int = 8,
        n_components: int = 10,
        random_state: int = 42,
        max_iter: int = 200,
    ):
        self.n_pca = n_pca
        self.n_components = n_components
        self.random_state = random_state
        self.max_iter = max_iter
        self.scaler: Optional[StandardScaler] = None
        self.pca: Optional[PCA] = None
        self.gmm: Optional[GaussianMixture] = None
        self.feature_columns_: List[str] = []

    def fit(self, feature_matrix: pd.DataFrame, sample_rows: Optional[int] = 200_000) -> "ArchetypeEngine":
        Xdf = feature_matrix.select_dtypes(include=[np.number]).copy()
        Xdf = Xdf.replace([np.inf, -np.inf], np.nan).dropna()
        if len(Xdf) == 0:
            raise ValueError("No rows to fit archetypes")
        if sample_rows and len(Xdf) > sample_rows:
            Xdf = Xdf.sample(n=sample_rows, random_state=self.random_state)
        self.feature_columns_ = list(Xdf.columns)
        X = Xdf.values.astype(np.float64)

        n_pca = min(self.n_pca, X.shape[1], max(1, X.shape[0] - 1))
        n_comp = min(self.n_components, max(2, len(X) // 20))

        self.scaler = StandardScaler()
        Xs = self.scaler.fit_transform(X)
        self.pca = PCA(n_components=n_pca, random_state=self.random_state)
        Xp = self.pca.fit_transform(Xs)
        self.gmm = GaussianMixture(
            n_components=n_comp,
            covariance_type="full",
            random_state=self.random_state,
            max_iter=self.max_iter,
        )
        self.gmm.fit(Xp)
        logger.info(
            "ArchetypeEngine fit: samples=%s features=%s pca=%s gmm_k=%s",
            len(X),
            len(self.feature_columns_),
            n_pca,
            n_comp,
        )
        return self

    @property
    def n_archetypes(self) -> int:
        """Number of ARCH_PROB_* columns this engine produces (GMM components)."""
        return int(self.gmm.n_components) if self.gmm is not None else 0

    @property
    def prior_(self) -> Optional[np.ndarray]:
        """GMM mixture weights — the marginal archetype distribution used as the
        deterministic fill when per-row probabilities can't be computed."""
        return self.gmm.weights_ if self.gmm is not None else None

    def _transform(self, X: np.ndarray) -> np.ndarray:
        assert self.scaler is not None and self.pca is not None and self.gmm is not None
        Xs = self.scaler.transform(X)
        return self.pca.transform(Xs)

    def predict_proba(self, feature_matrix: pd.DataFrame) -> np.ndarray:
        assert self.gmm is not None
        cols = self.feature_columns_
        X = feature_matrix.reindex(columns=cols, fill_value=0.0).values.astype(np.float64)
        Z = self._transform(X)
        return self.gmm.predict_proba(Z)

    def transform_frame(self, df: pd.DataFrame) -> pd.DataFrame:
        """Attach a FIXED ARCH_PROB_0..N-1 schema (+ ARCH_ARGMAX).

        The column set is deterministic for a fitted/loaded engine. If per-row
        probabilities can't be computed for any reason, every row is filled with
        the GMM training prior (mixture weights) instead of dropping the columns
        — guaranteeing identical features at training and inference time.
        """
        out = df.copy()
        n = self.n_archetypes
        if n == 0:
            raise RuntimeError("ArchetypeEngine is not fitted/loaded; cannot transform")

        proba = None
        try:
            feat = df.copy()
            for c in TARGET_FEATURES:
                if c not in feat.columns:
                    feat[c] = 0.0
            feat = feat[TARGET_FEATURES].fillna(0.0)
            proba = self.predict_proba(feat)
        except Exception as e:
            logger.warning("Archetype predict failed; filling training prior: %s", e)
            proba = None

        if proba is None or getattr(proba, "shape", (0,))[0] != len(out):
            proba = np.tile(self.prior_, (len(out), 1))

        for j in range(n):
            out[f"ARCH_PROB_{j}"] = proba[:, j]
        out["ARCH_ARGMAX"] = np.argmax(proba, axis=1)
        return out

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "scaler": self.scaler,
                "pca": self.pca,
                "gmm": self.gmm,
                "feature_columns_": self.feature_columns_,
                "n_pca": self.n_pca,
                "n_components": self.n_components,
            },
            path,
        )

    @classmethod
    def load(cls, path: Path) -> "ArchetypeEngine":
        path = Path(path)
        blob = joblib.load(path)
        eng = cls(
            n_pca=blob.get("n_pca", 8),
            n_components=blob.get("n_components", 10),
        )
        eng.scaler = blob["scaler"]
        eng.pca = blob["pca"]
        eng.gmm = blob["gmm"]
        eng.feature_columns_ = blob.get("feature_columns_", [])
        return eng


def fit_archetypes_from_player_dirs(
    player_data_dir: Path,
    output_path: Path,
    max_players: Optional[int] = None,
) -> ArchetypeEngine:
    player_data_dir = Path(player_data_dir)
    frames: List[pd.DataFrame] = []
    dirs = sorted([p for p in player_data_dir.iterdir() if p.is_dir()])
    if max_players:
        dirs = dirs[:max_players]
    for d in dirs:
        slug = d.name
        pq_path = d / f"{slug}_features.parquet"
        if not pq_path.exists():
            continue
        try:
            df = pd.read_parquet(pq_path)
            if len(df) < 25:
                continue
            
            missing = [c for c in TARGET_FEATURES if c not in df.columns]
            if missing:
                continue

            am = df[TARGET_FEATURES].copy()
            frames.append(am)
        except Exception as e:
            logger.warning("Skip %s: %s", slug, e)
    if not frames:
        raise RuntimeError("No player data found to fit archetypes")
    stacked = pd.concat(frames, ignore_index=True)
    engine = ArchetypeEngine()
    engine.fit(stacked)
    engine.save(output_path)
    return engine


if __name__ == "__main__":
    import argparse
    import sys

    BACKEND = Path(__file__).resolve().parent
    if str(BACKEND) not in sys.path:
        sys.path.insert(0, str(BACKEND))
    from config.paths import ARCHETYPE_PIPELINE_PATH, PLAYER_DATA_DIR

    ap = argparse.ArgumentParser(description="Fit PCA+GMM archetype model on player logs")
    ap.add_argument("--player-data", type=Path, default=PLAYER_DATA_DIR)
    ap.add_argument("--out", type=Path, default=ARCHETYPE_PIPELINE_PATH)
    ap.add_argument("--max-players", type=int, default=None)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO)
    fit_archetypes_from_player_dirs(args.player_data, args.out, max_players=args.max_players)
    print(f"Saved archetype model to {args.out}")
