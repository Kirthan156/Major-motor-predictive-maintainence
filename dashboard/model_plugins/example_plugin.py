"""Template for a model plugin. Copy it to ../../models/anomaly_<name>.py and edit.

The dashboard calls load(model_dir) once at start, then predict(feats, msg, history) for every
2-second reading. Use a plugin when your model is not a plain scikit-learn .joblib: a Keras or
PyTorch autoencoder, a 1D-CNN / LSTM on the raw vibration burst, an ensemble, and so on.

    python server.py --model ../models/anomaly_example.py
    (or DASHBOARD_MODEL=../models/anomaly_example.py in .env)

As written, this runs with no extra installs: it scores each reading by how far a handful of key
features are from the healthy medians in models/healthy_reference.json (made by
dashboard/train_baseline.py). Replace the body of predict() with your own model.

Arguments of predict()
  feats    dict of ~450 features for this reading: pm/features.extract(msg) + pm/features.temporal(history)
           (exactly the columns pm/dataset.py writes for training)
  msg      the raw firmware v3 message (dict). Raw signals: decode with pm/features._decode:
             vib = msg["vib"]; ax_g = _decode(vib["x"]) / vib["lsb_per_g"]   (1024 samples at vib["fs"] Hz)
             gyro: vib["gx"] / vib["lsb_per_dps"];  audio: _decode(msg["audio"]["data"]) at 16 kHz
  history  list of the last <=15 feature dicts of this device (oldest first, current last)

Return a dict with any of
  anomaly_score     float, higher = more abnormal                       (needed for an anomaly model)
  threshold         float, alarm when anomaly_score > threshold
  predicted_fault   str, e.g. "imbalance"                                (for a classifier)
  fault_confidence  float 0..1
"""
import json
from pathlib import Path

import numpy as np

KEY_FEATURES = ["ax_1x", "ay_1x", "ax_2x", "amag_vel_rms_mms", "gmag_rms", "tilt_pitch_deg",
                "current", "aud_rms_dbfs", "ax_kurtosis"]


class ExampleModel:
    name = "Example plugin (robust z-score on key features)"
    source = "healthy_reference.json"
    kind = "anomaly"            # or "classifier" if you only return predicted_fault
    threshold = 4.0

    def __init__(self, model_dir):
        ref = json.loads((Path(model_dir) / "healthy_reference.json").read_text())
        self.med, self.scale = ref["median"], ref["scale"]
        # Expose these so the dashboard can explain an alarm ("most unusual readings"):
        self.ref_median = {k: self.med[k] for k in KEY_FEATURES if k in self.med}
        self.ref_scale = {k: self.scale[k] for k in KEY_FEATURES if k in self.scale}
        # --- Keras example (uncomment, and pip install tensorflow):
        # import tensorflow as tf
        # self.net = tf.keras.models.load_model(Path(model_dir) / "autoencoder.keras")
        # self.threshold = float(json.loads((Path(model_dir) / "autoencoder.json").read_text())["threshold"])

    def predict(self, feats, msg, history):
        z = []
        for k in self.ref_median:
            v = feats.get(k)
            if v is not None and np.isfinite(v) and self.ref_scale[k] > 0:
                z.append(abs((v - self.ref_median[k]) / self.ref_scale[k]))
        score = float(np.mean(sorted(z)[-3:])) if z else float("nan")
        # --- 1D-CNN / autoencoder on the raw burst, for example:
        # from features import _decode
        # vib = msg["vib"]
        # x = np.stack([_decode(vib[a]) / vib["lsb_per_g"] for a in "xyz"], axis=-1)   # (1024, 3)
        # x = (x - x.mean(0)) / (x.std(0) + 1e-9)
        # recon = self.net.predict(x[None], verbose=0)[0]
        # score = float(np.mean((recon - x) ** 2))
        return {"anomaly_score": score, "threshold": self.threshold}


def load(model_dir):
    return ExampleModel(model_dir)
