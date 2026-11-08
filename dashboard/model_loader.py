"""Pluggable model loading. See docs/MODEL_INTERFACE.md for the full contract.

The dashboard accepts two kinds of model file, for the anomaly model and for the optional fault
classifier alike:

1. A .joblib dict (what pm/train.py and dashboard/train_baseline.py save)
     anomaly:    {"model", "features": [...], "threshold": float}
                 score = -model.score_samples(X)  (higher = more abnormal)
                 optional: "score_sign" (+1 if score_samples already means higher = worse),
                           "name", "source", "ref_median"/"ref_scale" (dicts, for 'why' text)
     classifier: {"model", "features": [...], "classes": [...]}  with predict_proba
   "kind": "anomaly" | "classifier" is optional; a dict with "classes" is taken as a classifier.

2. A Python plugin file (models/<anything>.py) for models that a joblib dict cannot describe:
   Keras/PyTorch autoencoders, 1D-CNN or LSTM on raw bursts, ensembles, etc. It must define
       def load(model_dir: pathlib.Path) -> object
   and the object must have
       predict(feats: dict, msg: dict, history: list[dict]) -> dict
   returning any of: anomaly_score (float, higher = worse), threshold (float),
   predicted_fault (str), fault_confidence (float).
   Optional attributes: name (str), source (str), threshold (float), features (list).
   Template: dashboard/model_plugins/example_plugin.py
"""
import importlib.util
import math
from pathlib import Path

import numpy as np
import pandas as pd

from common import DUMMY_MODEL, MODELS


class ModelError(RuntimeError):
    pass


def _frame(feats, cols):
    x = pd.DataFrame([feats]).reindex(columns=cols)
    return x.apply(pd.to_numeric, errors="coerce").astype(float)


class JoblibAnomaly:
    kind = "anomaly"

    def __init__(self, bundle, path):
        try:
            self.model, self.features, self.threshold = bundle["model"], list(bundle["features"]), float(bundle["threshold"])
        except KeyError as e:
            raise ModelError(f"{path}: anomaly .joblib needs keys model, features, threshold (missing {e})")
        self.sign = float(bundle.get("score_sign", -1.0))
        self.name = bundle.get("name") or Path(path).stem
        self.source = bundle.get("source") or bundle.get("trained_on") or "recorded data"
        self.path = str(path)
        self.ref_median = bundle.get("ref_median")
        self.ref_scale = bundle.get("ref_scale")
        if self.ref_median is None and hasattr(self.model, "named_steps"):
            sc = self.model.named_steps.get("standardscaler")
            if sc is not None:
                self.ref_median = dict(zip(self.features, sc.mean_))
                self.ref_scale = dict(zip(self.features, sc.scale_))
        self.missing_warned = False

    def predict(self, feats, msg=None, history=None):
        x = _frame(feats, self.features)
        n_missing = int(x.iloc[0].isna().sum())
        if n_missing > 0.2 * len(self.features) and not self.missing_warned:
            print(f"WARNING: {n_missing}/{len(self.features)} model features are missing in the live data. "
                  "The model was trained on a different firmware or feature version.")
            self.missing_warned = True
        if hasattr(self.model, "score_samples"):
            raw = float(self.model.score_samples(x)[0])
        elif hasattr(self.model, "decision_function"):
            raw = float(self.model.decision_function(x)[0])
        else:
            raise ModelError(f"{self.path}: model has neither score_samples nor decision_function")
        return {"anomaly_score": self.sign * raw, "threshold": self.threshold}


class JoblibClassifier:
    kind = "classifier"

    def __init__(self, bundle, path):
        try:
            self.model, self.features, self.classes = bundle["model"], list(bundle["features"]), list(bundle["classes"])
        except KeyError as e:
            raise ModelError(f"{path}: classifier .joblib needs keys model, features, classes (missing {e})")
        self.name = bundle.get("name") or Path(path).stem
        self.source = bundle.get("source") or "recorded data"
        self.path = str(path)
        self.threshold = None

    def predict(self, feats, msg=None, history=None):
        proba = self.model.predict_proba(_frame(feats, self.features))[0]
        i = int(np.argmax(proba))
        return {"predicted_fault": str(self.classes[i]), "fault_confidence": float(proba[i])}


class Plugin:
    def __init__(self, path):
        spec = importlib.util.spec_from_file_location(f"model_plugin_{Path(path).stem}", path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        if not hasattr(mod, "load"):
            raise ModelError(f"{path}: plugin must define load(model_dir)")
        self.obj = mod.load(Path(path).parent)
        if not hasattr(self.obj, "predict"):
            raise ModelError(f"{path}: load() must return an object with predict(feats, msg, history)")
        self.name = getattr(self.obj, "name", Path(path).stem)
        self.source = getattr(self.obj, "source", "plugin")
        self.threshold = getattr(self.obj, "threshold", None)
        self.features = getattr(self.obj, "features", None)
        self.ref_median = getattr(self.obj, "ref_median", None)
        self.ref_scale = getattr(self.obj, "ref_scale", None)
        self.kind = getattr(self.obj, "kind", "anomaly")
        self.path = str(path)

    def predict(self, feats, msg=None, history=None):
        out = dict(self.obj.predict(feats, msg, history) or {})
        if "threshold" not in out and self.threshold is not None:
            out["threshold"] = float(self.threshold)
        return out


def load_model(path):
    path = Path(path)
    if not path.exists():
        raise ModelError(f"No model file at {path}")
    if path.suffix == ".py":
        return Plugin(path)
    import joblib
    bundle = joblib.load(path)
    if not isinstance(bundle, dict):
        raise ModelError(f"{path}: expected a dict with model/features/... (see docs/MODEL_INTERFACE.md)")
    kind = bundle.get("kind") or ("classifier" if "classes" in bundle else "anomaly")
    return JoblibClassifier(bundle, path) if kind == "classifier" else JoblibAnomaly(bundle, path)


def find_anomaly_model():
    """'auto': the best anomaly model that exists, in this order:
    1. models/anomaly_iforest.joblib   (pm/train.py, trained on your labelled sessions)
    2. any other models/anomaly_*.joblib or models/anomaly_*.py (e.g. from the Phase II model code), newest first
    3. models/anomaly_baseline.joblib  (dashboard/train_baseline.py on real healthy sessions)
    4. dashboard/models/anomaly_synthetic.joblib (synthetic data, pipeline demo only)
    """
    first = MODELS / "anomaly_iforest.joblib"
    if first.exists():
        return first
    others = [p for p in MODELS.glob("anomaly_*") if p.suffix in (".joblib", ".py")
              and p.name not in ("anomaly_baseline.joblib",)]
    if others:
        return max(others, key=lambda p: p.stat().st_mtime)
    base = MODELS / "anomaly_baseline.joblib"
    if base.exists():
        return base
    return DUMMY_MODEL if DUMMY_MODEL.exists() else None


def finite(v):
    try:
        return v is not None and math.isfinite(float(v))
    except (TypeError, ValueError):
        return False
