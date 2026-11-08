# Model interface: how a trained model plugs into the live dashboard

Any model you train later (Random Forest, XGBoost, Isolation Forest, autoencoder, 1D-CNN, LSTM...)
plugs into `dashboard/server.py` without changing the dashboard code, as long as it follows this page.

## 1. The features the model sees

Live scoring builds each row exactly the way `pm/dataset.py` builds a training row:

```python
feats = features.extract(msg)                       # pm/features.py, ~420 columns per 2 s message
history.append(feats)                               # last 15 messages (30 s) of the same device
row = {**feats, **features.temporal(history)}       # + tm_* trend columns  -> ~454 columns
```

- Train on `data/dataset.csv` from `python pm/dataset.py`, or call `dataset.load_sessions()`.
- Learn only from `dataset.feature_columns(df)`. It drops metadata, diagnostics and columns that
  describe the day instead of the motor (voltage, ambient temperature, humidity, Wi-Fi...).
- **Never change `pm/features.py` after training without retraining.** If you must, bump
  `CACHE_VERSION` in `pm/dataset.py` so the cached feature CSVs are rebuilt.
- Messages are firmware v3 (`fw: "v3"`). Raw signals are base64 little-endian int16. Decode with
  `features._decode()`. Accelerometer in g = counts / `vib["lsb_per_g"]` (4096 at ±8 g). Gyro in
  deg/s = counts / `vib["lsb_per_dps"]`. Audio: `msg["audio"]["data"]`, 4096 samples at 16 kHz.

## 2. Option A: a `.joblib` dict (scikit-learn style models)

Save the model with `joblib.dump(bundle, "models/<name>.joblib")`.

**Anomaly model** (trained on healthy data only):

```python
bundle = {
    "kind": "anomaly",                 # optional
    "name": "Isolation Forest v2",     # shown on the dashboard
    "source": "14 healthy sessions",   # shown on the dashboard
    "model": pipeline,                 # has score_samples() (or decision_function())
    "features": cols,                  # list of column names, in training order
    "threshold": thr,                  # alarm when score > thr
    "score_sign": -1.0,                # score = score_sign * model.score_samples(X)
                                       # -1 for sklearn (higher score_samples = more normal)
}
```

**Fault classifier**:

```python
bundle = {"kind": "classifier", "model": clf, "features": cols, "classes": ["healthy", "imbalance", ...]}
# clf must have predict_proba(); classes[i] is the label of column i
```

This is the format `pm/train.py` already writes (`anomaly_iforest.joblib`, `fault_rf.joblib`, `fault_xgb.joblib`).

**Threshold.** Set it from sessions the model did *not* train on (leave-one-session-out), for example the
99th percentile of held-out healthy scores. A threshold from training rows is too low and gives false alarms.

## 3. Option B: a Python plugin (deep learning, raw-signal models, ensembles)

Save `models/anomaly_<name>.py` (anomaly) or any name for a classifier. Template:
`dashboard/model_plugins/example_plugin.py`.

```python
def load(model_dir):               # model_dir = the folder the .py file is in (models/)
    return MyModel(model_dir)

class MyModel:
    name = "Conv autoencoder"      # optional, shown on the dashboard
    source = "14 healthy sessions" # optional
    threshold = 0.012              # optional if predict() returns it
    kind = "anomaly"               # or "classifier"
    def predict(self, feats, msg, history):
        ...
        return {"anomaly_score": score, "threshold": self.threshold}
        # classifier: {"predicted_fault": "imbalance", "fault_confidence": 0.83}
```

- `feats`: the ~454-column feature dict above. `msg`: the raw message. `history`: last ≤15 feature dicts.
- Import TensorFlow/PyTorch **inside** the plugin, so the dashboard still starts without them.
- `predict()` runs every 2 s, so keep it under ~200 ms. Load weights once, in `load()`.
- Higher `anomaly_score` must mean more abnormal.

## 4. Choosing which model the dashboard uses

| How | Example |
|---|---|
| Automatic (default) | `DASHBOARD_MODEL=auto`. The dashboard picks `models/anomaly_iforest.joblib`, then the newest other `models/anomaly_*.joblib` or `.py`, then `models/anomaly_baseline.joblib`. |
| In `.env` | `DASHBOARD_MODEL=../models/anomaly_autoencoder.py` |
| On the command line | `python server.py --source mqtt --model ../models/anomaly_autoencoder.py --classifier ../models/fault_xgb.joblib` |

The model note under **Condition** on the page shows which file is loaded.

## 5. Check before a demo

```
cd dashboard
python selftest.py --model ../models/<your model> [--classifier ../models/<classifier>]
```

It checks that every feature the model needs exists live, that healthy recordings stay normal, and that
a demo fault raises the alarm.

## 6. The second detector: change since baseline

Next to the model score, the dashboard runs a **change detector** (`dashboard/scoring.py`, `ChangeDetector`).
It learns the first 30 readings (1 minute) of each run as that run's normal, then scores how far 75 key
features have moved (robust z-scores, median of the last 5 readings, mean of the 3 largest). Alarm
threshold 6.0 gave about 0.5 % of healthy readings over the threshold on the 7 Nov sessions.

Why it exists: on the 7 Nov data the differences *between* healthy sessions (speed, mounting, the
motor rocking) are larger than many single-fault changes, so a model trained on all healthy sessions
cannot alarm on a small fault. The change detector compares the motor with itself a minute ago, so it
catches a fault that starts during a run. It cannot see a fault that was already there when the baseline
was learned. Press **Re-learn baseline** after changing the speed regulator. Turn it off with `--no-change`.
