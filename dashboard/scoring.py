"""Turn one raw message into a dashboard point: features, anomaly score, alarm state."""
from collections import deque

import joblib
import numpy as np
import pandas as pd

from common import features

# Plain-language names so the dashboard can say why a window looks abnormal.
SLOW_NAMES = {"current": "current", "power": "power", "voltage": "voltage", "pf": "power factor",
              "rpm": "speed", "rot_hz": "running speed", "sound_dbfs": "sound level",
              "temp_rise_c": "temperature rise"}
VIB_NAMES = {"rms_g": "vibration RMS", "peak_g": "vibration peak", "p2p_g": "vibration peak-to-peak",
             "crest": "crest factor (spikiness)", "kurtosis": "impacts (kurtosis)", "skew": "vibration skew",
             "dom_hz": "dominant vibration frequency", "dom_amp_g": "dominant vibration peak",
             "centroid_hz": "vibration frequency centre", "1x_g": "vibration at running speed",
             "2x_g": "vibration at 2x speed", "3x_g": "vibration at 3x speed", "bpf_g": "blade-pass vibration"}


def nice_name(feature):
    if feature in SLOW_NAMES:
        return SLOW_NAMES[feature]
    axis, _, rest = feature.partition("_")
    axis = "overall" if axis == "mag" else axis
    if rest.startswith("band_"):
        lo, hi = rest.split("_")[1:3]
        return f"{axis} vibration energy {lo}-{hi} Hz"
    return f"{axis} {VIB_NAMES[rest]}" if rest in VIB_NAMES else feature


class Scorer:
    """Wraps an anomaly .joblib from train_dummy.py or pm/train.py (same format).

    A single window over the threshold is not an alarm: at a 99th-percentile threshold a healthy
    motor still crosses it about once every 100 windows (~3 min at 2 s per window). The alarm
    needs `alarm_k` of the last `alarm_n` windows over the threshold.
    """

    def __init__(self, path, alarm_k=3, alarm_n=5, classifier_path=None):
        b = joblib.load(path)
        self.model, self.cols, self.threshold = b["model"], b["features"], float(b["threshold"])
        self.source = b.get("source", "recorded data")
        self.path = str(path)
        scaler = self.model.named_steps.get("standardscaler")
        self.mean = pd.Series(scaler.mean_, index=self.cols) if scaler is not None else None
        self.scale = pd.Series(scaler.scale_, index=self.cols) if scaler is not None else None
        self.recent = deque(maxlen=alarm_n)
        self.alarm_k = alarm_k
        self.clf = joblib.load(classifier_path) if classifier_path else None

    def __call__(self, feats):
        x = pd.DataFrame([feats]).reindex(columns=self.cols)
        score = float(-self.model.score_samples(x)[0])      # higher = more abnormal
        over = score > self.threshold
        self.recent.append(over)
        out = {"anomaly_score": score, "threshold": self.threshold, "over_threshold": over,
               "alarm": sum(self.recent) >= self.alarm_k, "top_deviations": []}
        if self.mean is not None:
            z = ((x.iloc[0] - self.mean) / self.scale).dropna()
            for name in z.abs().sort_values(ascending=False).index[:4]:
                out["top_deviations"].append({"feature": name, "label": nice_name(name),
                                              "z": round(float(z[name]), 1)})
        if self.clf:
            proba = self.clf["model"].predict_proba(pd.DataFrame([feats])[self.clf["features"]])[0]
            i = int(proba.argmax())
            out["predicted_fault"], out["fault_confidence"] = self.clf["classes"][i], float(proba[i])
        return out


def spectrum(msg, max_hz=200.0, bins=160):
    """Vibration amplitude spectrum (mg) summed over the 3 axes, cut to 0..max_hz, for the chart."""
    vib = msg["vib"]
    fs, lsb = float(vib["fs"]), float(vib.get("lsb_per_g", 8192))
    total = None
    for a in ("x", "y", "z"):
        sig = np.asarray(vib[a], dtype=float) / lsb
        sig = (sig - sig.mean()) * np.hanning(len(sig))
        amp = 2.0 * np.abs(np.fft.rfft(sig)) / np.hanning(len(sig)).sum()
        total = amp if total is None else np.sqrt(total ** 2 + amp ** 2)
    freqs = np.fft.rfftfreq(len(vib["x"]), 1.0 / fs)
    keep = freqs <= max_hz
    freqs, total = freqs[keep], total[keep] * 1000
    step = max(1, len(freqs) // bins)
    return {"hz": [round(float(f), 1) for f in freqs[::step]],
            "mg": [round(float(v), 3) for v in total.reshape(-1)[: len(freqs) // step * step]
                   .reshape(-1, step).max(axis=1)]}


def to_point(msg, feats, result, ts):
    """Compact record sent to the browser and written to the log."""
    def r(v, d=3):
        try:
            return None if v is None or np.isnan(v) else round(float(v), d)
        except (TypeError, ValueError):
            return None
    return {
        "t": ts, "seq": msg.get("seq"), "device": msg.get("device_id"),
        "rpm": r(msg.get("rpm"), 0), "voltage": r(msg.get("voltage"), 1),
        "current": r(msg.get("current"), 4), "power": r(msg.get("power"), 1),
        "pf": r(msg.get("pf")), "sound_dbfs": r(msg.get("sound_dbfs"), 1),
        "motor_temp_c": r(msg.get("ds18b20_temp_c"), 2), "ambient_c": r(msg.get("dht22_temp_c"), 2),
        "temp_rise_c": r(feats.get("temp_rise_c"), 2), "humidity": r(msg.get("dht22_humidity"), 1),
        "vib_rms_mg": r(feats["mag_rms_g"] * 1000, 2), "vib_1x_mg": r(feats["x_1x_g"] * 1000, 2),
        "kurtosis": r(feats["x_kurtosis"], 2),
        "anomaly_score": r(result["anomaly_score"], 4), "threshold": r(result["threshold"], 4),
        "over_threshold": result["over_threshold"], "alarm": result["alarm"],
        "top_deviations": result["top_deviations"],
        "predicted_fault": result.get("predicted_fault"),
    }
