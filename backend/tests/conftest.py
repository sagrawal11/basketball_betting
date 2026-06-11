"""Pytest configuration for backend tests.

Ensures the ``backend/`` directory is importable so tests can ``import
feature_engine`` and ``import model_training.train_*`` exactly the way the
training scripts do at runtime.
"""
import sys
from pathlib import Path

BACKEND = Path(__file__).resolve().parent.parent
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))
