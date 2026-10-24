"""Shared paths and imports.

The dashboard reuses the feature code from ../pm so the live dashboard computes exactly the
same features as train.py does on recorded sessions. Nothing in pm/ is copied or edited.
"""
import sys
from pathlib import Path

PLATFORM = Path(__file__).resolve().parent
PM_DIR = PLATFORM.parent / "pm"
ROOT = PLATFORM.parent
MODELS = PLATFORM / "models"
DATA = PLATFORM / "data"
DUMMY_MODEL = MODELS / "anomaly_dummy.joblib"

if str(PM_DIR) not in sys.path:
    sys.path.insert(0, str(PM_DIR))

import features  # noqa: E402,F401  (pm/features.py)
import synthetic  # noqa: E402,F401  (pm/synthetic.py)
