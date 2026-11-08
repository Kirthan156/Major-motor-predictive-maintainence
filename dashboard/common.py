"""Shared paths and imports for the dashboard.

The dashboard reuses the feature code in ../pm, so the live dashboard computes exactly the same
features as pm/dataset.py and pm/train.py compute on recorded sessions. Nothing in pm/ is copied.
"""
import sys
from pathlib import Path

DASHBOARD = Path(__file__).resolve().parent
ROOT = DASHBOARD.parent
PM_DIR = ROOT / "pm"
MODELS = ROOT / "models"              # trained models from pm/train.py, train_baseline.py, or plugins
PLUGINS = DASHBOARD / "model_plugins"  # example plugin template lives here
DATA = DASHBOARD / "data"             # live score logs (CSV) written by server.py
RAW = ROOT / "data" / "raw"           # recorded sessions (.jsonl) from pm/recorder.py
SAMPLE = ROOT / "data" / "sample"     # small real recordings shipped with the repo for the demo
DUMMY_MODEL = DASHBOARD / "models" / "anomaly_synthetic.joblib"

# Kept for older imports
PLATFORM = DASHBOARD

if str(PM_DIR) not in sys.path:
    sys.path.insert(0, str(PM_DIR))

import dataset  # noqa: E402,F401  (pm/dataset.py)
import features  # noqa: E402,F401  (pm/features.py)
import synthetic  # noqa: E402,F401  (pm/synthetic.py)
