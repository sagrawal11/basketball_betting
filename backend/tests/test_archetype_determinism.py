"""Task 7.5 — ARCH_PROB_* columns must be deterministically present.

The archetype transform and the archetype-engine load were both wrapped in
silent ``except`` blocks, so the ARCH_PROB_* columns could appear in one code
path but vanish in another (training vs inference, or player-to-player). That
is train-serve skew: feature_columns.json may expect ARCH_PROB_* while a given
build silently omits them.

The fix:
  * ``transform_frame`` always emits a fixed ARCH_PROB_0..N-1 schema; rows whose
    probabilities can't be computed fall back to the GMM training prior
    (gmm.weights_), not 0.0.
  * ``default_engine`` fails loudly when a present archetype model can't load,
    instead of silently returning an engine with no archetype columns.
"""
import numpy as np
import pandas as pd
import pytest

import feature_engine as fe
from archetype_engine import ArchetypeEngine, TARGET_FEATURES


def _feature_df(rows: int, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame({c: rng.normal(10.0, 3.0, rows) for c in TARGET_FEATURES})


def _fitted_engine(rows: int = 200, seed: int = 1) -> ArchetypeEngine:
    eng = ArchetypeEngine(n_components=4)
    eng.fit(_feature_df(rows, seed), sample_rows=None)
    return eng


def _raise(*_a, **_k):
    raise RuntimeError("simulated GMM/predict failure")


def test_transform_frame_emits_fixed_schema_for_normal_frame():
    eng = _fitted_engine()
    out = eng.transform_frame(_feature_df(7, seed=3))
    arch_cols = [c for c in out.columns if c.startswith("ARCH_PROB_")]
    assert len(arch_cols) == eng.n_archetypes
    assert "ARCH_ARGMAX" in out.columns


def test_transform_frame_fills_prior_when_predict_fails(monkeypatch):
    eng = _fitted_engine()
    # Simulate any internal failure (schema drift, corrupt component, etc.).
    monkeypatch.setattr(eng, "predict_proba", _raise)

    out = eng.transform_frame(_feature_df(5, seed=2))

    arch_cols = sorted(
        (c for c in out.columns if c.startswith("ARCH_PROB_")),
        key=lambda c: int(c.rsplit("_", 1)[1]),
    )
    # columns are still present (no skew) ...
    assert len(arch_cols) == eng.n_archetypes
    # ... and filled with the training prior, not 0.0
    row0 = out.loc[0, arch_cols].to_numpy(dtype=float)
    np.testing.assert_allclose(row0, eng.prior_)
    assert not np.allclose(row0, 0.0)


def test_default_engine_raises_on_corrupt_archetype_model(tmp_path, monkeypatch):
    bad = tmp_path / "archetype_gmm.joblib"
    bad.write_bytes(b"not a real joblib payload")
    monkeypatch.setattr(fe, "ARCHETYPE_PIPELINE_PATH", bad)

    with pytest.raises(Exception):
        fe.default_engine()
