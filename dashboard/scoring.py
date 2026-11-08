"""Feature row -> anomaly score, alarm state, 'why' text and optional fault guess.

Also builds the compact point the browser shows, and the vibration spectrum chart.
"""
import math
from collections import deque

import numpy as np

from common import features
from model_loader import finite, load_model

# ---------------------------------------------------------------- plain-language feature names

GROUP = {"ax": "X vibration", "ay": "Y vibration", "az": "Z vibration", "amag": "overall vibration",
         "gx": "X rocking (gyro)", "gy": "Y rocking (gyro)", "gz": "Z rocking (gyro)", "gmag": "overall rocking (gyro)",
         "aud": "sound", "el": "electrical", "th": "temperature", "tilt": "mounting tilt", "xc": "axis relation",
         "rpm": "speed"}
STAT = {"rms": "RMS", "std": "spread", "peak": "peak", "p2p": "peak-to-peak", "crest": "crest factor (spikiness)",
        "kurtosis": "impacts (kurtosis)", "skew": "skew", "impulse": "impulse factor", "shape": "shape factor",
        "clearance": "clearance factor", "1x": "at running speed (1x)", "2x": "at 2x speed", "3x": "at 3x speed",
        "4x": "at 4x speed", "5x": "at 5x speed", "half_x": "at half speed (looseness)", "bpf": "at blade-pass",
        "2bpf": "at 2x blade-pass", "2line": "at 100 Hz (electrical)", "vel_rms_mms": "velocity RMS (mm/s)",
        "vel_1x_mms": "velocity at 1x", "dom_hz": "dominant frequency", "dom_amp": "dominant peak",
        "centroid_hz": "frequency centre", "spread_hz": "frequency spread", "entropy": "spectral entropy",
        "flatness": "spectral flatness", "rolloff85_hz": "roll-off frequency", "harmonic_ratio": "share in harmonics",
        "rms_dbfs": "level (dBFS)", "100hz_db": "mains hum", "zcr": "zero-crossing rate", "diff_rms": "jerk",
        "mean_dps": "mean rotation rate", "frame_db_std": "loudness variation", "pitch_deg": "pitch", "roll_deg": "roll"}
SINGLE = {"current": "current", "power": "power", "pf": "power factor", "rpm": "speed", "rot_hz": "running speed",
          "rot_hz_vib": "running speed (from vibration)", "th_rise_c": "temperature rise above room",
          "el_apparent_va": "apparent power", "el_reactive_var": "reactive power", "el_phase_deg": "phase angle",
          "el_impedance_ohm": "impedance", "el_power_per_rpm": "power per RPM", "rpm_std": "speed ripple",
          "rpm_range": "speed range", "rpm_slip": "slip", "aud_dbfs_device": "sound level"}


def nice_name(feature):
    if feature in SINGLE:
        return SINGLE[feature]
    if feature.startswith("tm_"):
        base, _, kind = feature[3:].rpartition("_")
        what = {"delta": "change since last reading", "mean5": "10 s average", "std5": "10 s variation"}.get(kind, kind)
        return f"{nice_name(base)}, {what}" if base else feature
    grp, _, rest = feature.partition("_")
    if grp not in GROUP:
        return feature
    if rest.startswith("band_"):
        lo, hi = rest.split("_")[1:3]
        return f"{GROUP[grp]} energy {lo}-{hi} Hz"
    if rest.startswith("mfcc"):
        return f"sound timbre ({rest.replace('_', ' ')})"
    if rest.startswith(("bpf1", "bpf2", "bpf3")):
        return f"sound at blade-pass x{rest[3]}"
    return f"{GROUP[grp]} {STAT.get(rest, rest.replace('_', ' '))}"


# ---------------------------------------------------------------- change-since-baseline detector

# Physically meaningful, glitch-tolerant features. Gyro kurtosis/peaks and RPM min/max/range are left
# out on purpose: in the 7 Nov healthy data they spike on single windows (I2C and TCRT5000 glitches).
CHANGE_FEATURES = (
    [f"a{a}_{s}" for a in "xyz" for s in ("rms", "1x", "2x", "3x", "half_x", "bpf", "vel_rms_mms", "crest",
                                           "kurtosis", "band_10_20", "band_20_30", "band_30_50", "band_50_80",
                                           "band_80_120", "band_120_200")]
    + [f"g{a}_{s}" for a in "xyz" for s in ("rms", "1x", "2x")]
    + ["amag_rms", "amag_vel_rms_mms", "gmag_rms", "tilt_pitch_deg", "tilt_roll_deg", "current", "power", "pf",
       "el_power_per_rpm", "rpm", "aud_rms_dbfs", "aud_centroid_hz", "aud_bpf1_db", "aud_100hz_db", "aud_frame_db_std"]
    + [f"aud_band_{lo}_{hi}" for lo, hi in ((125, 250), (250, 500), (500, 1000), (1000, 2000), (2000, 4000), (4000, 8000))]
)


class ChangeDetector:
    """How far the motor has moved from how it looked at the start of this run.

    The first `learn` windows (default 30 = 1 minute) after start, or after 'Re-learn baseline' on
    the page, become this run's reference. Every later window is smoothed (median of the last 5),
    compared feature by feature with the reference (robust z-score, capped at 10), and the score is
    the mean of the 3 largest |z|. Per-feature scales have a floor of half the spread seen across all
    healthy sessions, so tiny within-run noise cannot cause alarms.

    Tuned on the 7 Nov healthy sessions (10 runs): threshold 6.0 gave ~0.5 % of healthy windows
    over the threshold. It catches faults that START during a run. It cannot see a fault that was
    already there when the baseline was learned, which is what the global model is for.
    Learn the baseline only while the motor is known to be healthy and at the speed you will run.
    """

    def __init__(self, global_scale=None, learn=30, threshold=6.0, smooth=5, top_k=3, floor=0.5):
        self.global_scale = global_scale or {}
        self.learn_n, self.threshold, self.smooth, self.top_k, self.floor = learn, threshold, smooth, top_k, floor
        self.reset()

    def reset(self):
        self.buf, self.recent, self.med, self.scale, self.base_rpm = [], [], None, None, None

    @property
    def state(self):
        return "ready" if self.med is not None else f"learning {len(self.buf)}/{self.learn_n}"

    def __call__(self, feats):
        row = {k: float(feats[k]) for k in CHANGE_FEATURES if finite(feats.get(k))}
        if self.med is None:
            self.buf.append(row)
            if len(self.buf) >= self.learn_n:
                import pandas as pd
                b = pd.DataFrame(self.buf)
                med = b.median()
                mad = (b - med).abs().median() * 1.4826
                self.med, self.scale = {}, {}
                self.base_rpm = float(med["rpm"]) if "rpm" in med and finite(med["rpm"]) else None
                for k in med.index:
                    g = self.global_scale.get(k)
                    s = max(float(mad[k]), self.floor * g) if finite(g) else float(mad[k]) * 3
                    if finite(med[k]) and finite(s) and s > 0:
                        self.med[k], self.scale[k] = float(med[k]), s
            return {"change_score": None, "change_state": self.state, "change_deviations": []}
        self.recent = (self.recent + [row])[-self.smooth:]
        z = {}
        for k, m in self.med.items():
            vals = sorted(r[k] for r in self.recent if k in r)
            if vals:
                z[k] = max(min((vals[len(vals) // 2] - m) / self.scale[k], 10.0), -10.0)
        top = sorted(z, key=lambda k: -abs(z[k]))
        score = float(np.mean([abs(z[k]) for k in top[:self.top_k]])) if top else None
        rpm = row.get("rpm")
        speed_changed = bool(self.base_rpm and rpm and abs(rpm / self.base_rpm - 1) > 0.02)
        return {"change_score": score, "change_threshold": self.threshold, "change_state": "ready",
                "speed_changed": speed_changed, "baseline_rpm": self.base_rpm,
                "change_deviations": [{"feature": k, "label": nice_name(k), "z": round(z[k], 1)} for k in top[:4]]}


# ---------------------------------------------------------------- scorer

class Scorer:
    """Anomaly model (required) + change detector + fault classifier (optional) + alarm rule.

    One window over a threshold is not an alarm: at a 99th-percentile threshold a healthy motor
    still crosses it about once every 100 windows (~3 min at 2 s each). An alarm needs alarm_k of
    the last alarm_n windows over the threshold, for the model score OR the change score.
    """

    def __init__(self, model_path, alarm_k=3, alarm_n=5, classifier_path=None, threshold=None,
                 change=True, change_threshold=6.0, learn=30):
        self.anom = load_model(model_path)
        self.clf = load_model(classifier_path) if classifier_path else None
        self.override = threshold
        self.threshold = threshold if threshold is not None else self.anom.threshold
        self.alarm_k, self.alarm_n = alarm_k, alarm_n
        self.recent, self.recent_change = deque(maxlen=alarm_n), deque(maxlen=alarm_n)
        self.source, self.path, self.name = self.anom.source, self.anom.path, self.anom.name
        self.ref_median, self.ref_scale = load_reference(self.anom)
        self.change = ChangeDetector(self.ref_scale, learn=learn, threshold=change_threshold) if change else None

    def reset(self):
        """Forget the alarm history and re-learn the change baseline from the next windows."""
        self.recent.clear()
        self.recent_change.clear()
        if self.change:
            self.change.reset()

    def __call__(self, feats, msg=None, history=None):
        out = {"top_deviations": []}
        out.update(self.anom.predict(feats, msg, history))
        if self.override is not None:
            out["threshold"] = self.override
        score, thr = out.get("anomaly_score"), out.get("threshold", self.threshold)
        self.threshold = thr
        over = bool(finite(score) and finite(thr) and score > thr)
        self.recent.append(over)
        model_alarm = sum(self.recent) >= self.alarm_k

        change_alarm, change_over = False, False
        if self.change:
            ch = self.change(feats)
            out.update(ch)
            change_over = bool(finite(ch["change_score"]) and ch["change_score"] > self.change.threshold)
            if ch["change_score"] is not None:
                self.recent_change.append(change_over)
            change_alarm = sum(self.recent_change) >= self.alarm_k
        out.update(over_threshold=over, change_over=change_over, model_alarm=model_alarm,
                   change_alarm=change_alarm, alarm=model_alarm or change_alarm)

        if self.ref_median:
            z = {}
            for k, m in self.ref_median.items():
                if k.startswith("tm_"):
                    continue
                v, s = feats.get(k), self.ref_scale.get(k) if self.ref_scale else None
                if finite(v) and finite(m) and finite(s) and s > 0:
                    z[k] = (float(v) - m) / s
            for k in sorted(z, key=lambda k: -abs(z[k]))[:4]:
                out["top_deviations"].append({"feature": k, "label": nice_name(k), "z": round(z[k], 1)})
        if self.clf:
            out.update(self.clf.predict(feats, msg, history))
        return out


def load_reference(model):
    """Healthy median and robust spread per feature, for the 'why' text and the change detector.
    From models/healthy_reference.json (written by train_baseline.py) when it exists, because it is
    robust to the single-window sensor glitches in the data; else from the model itself."""
    import json
    from common import MODELS
    ref = MODELS / "healthy_reference.json"
    if ref.exists():
        d = json.loads(ref.read_text())
        if d.get("median") and d.get("scale"):
            return d["median"], d["scale"]
    return getattr(model, "ref_median", None) or {}, getattr(model, "ref_scale", None) or {}


# ---------------------------------------------------------------- browser payloads

def spectrum(msg, max_hz=200.0, bins=160):
    """Vibration amplitude spectrum (mg), combined over the 3 axes, 0..max_hz, for the chart."""
    vib = msg.get("vib")
    if not vib:
        return None
    fs, lsb = float(vib["fs"]), float(vib.get("lsb_per_g", 8192))
    total, n = None, None
    for a in ("x", "y", "z"):
        sig = features._decode(vib.get(a))
        if sig is None:
            continue
        sig = sig / lsb
        n = len(sig)
        w = np.hanning(n)
        amp = 2.0 * np.abs(np.fft.rfft((sig - sig.mean()) * w)) / w.sum()
        total = amp if total is None else np.sqrt(total ** 2 + amp ** 2)
    if total is None:
        return None
    freqs = np.fft.rfftfreq(n, 1.0 / fs)
    keep = freqs <= max_hz
    freqs, total = freqs[keep], total[keep] * 1000
    step = max(1, len(freqs) // bins)
    m = len(freqs) // step * step
    return {"hz": [round(float(f), 1) for f in freqs[:m:step]],
            "mg": [round(float(v), 3) for v in total[:m].reshape(-1, step).max(axis=1)]}


def to_point(msg, feats, result, ts):
    """Compact record sent to the browser and written to the CSV log."""
    def r(v, d=3):
        try:
            f = float(v)
        except (TypeError, ValueError):
            return None
        return None if math.isnan(f) or math.isinf(f) else round(f, d)
    amag = feats.get("amag_rms")
    return {
        "t": ts, "seq": msg.get("seq"), "device": msg.get("device_id"), "fw": msg.get("fw"),
        "rpm": r(msg.get("rpm"), 0), "voltage": r(msg.get("voltage"), 1),
        "current": r(msg.get("current"), 4), "power": r(msg.get("power"), 1),
        "pf": r(msg.get("pf")), "sound_dbfs": r(msg.get("sound_dbfs"), 1),
        "motor_temp_c": r(msg.get("ds18b20_temp_c"), 2), "ambient_c": r(msg.get("dht22_temp_c"), 2),
        "temp_rise_c": r(feats.get("th_rise_c"), 2), "humidity": r(msg.get("dht22_humidity"), 1),
        "vib_rms_mg": r(amag * 1000 if finite(amag) else None, 1),
        "vel_rms_mms": r(feats.get("amag_vel_rms_mms"), 2),
        "vib_1x_mg": r(feats.get("ax_1x", math.nan) * 1000, 1),
        "kurtosis": r(feats.get("ax_kurtosis"), 2),
        "rot_hz": r(feats.get("rot_hz"), 2),
        "wifi_rssi": r(msg.get("wifi_rssi"), 0),
        "anomaly_score": r(result.get("anomaly_score"), 4), "threshold": r(result.get("threshold"), 4),
        "over_threshold": bool(result.get("over_threshold")), "alarm": bool(result.get("alarm")),
        "model_alarm": bool(result.get("model_alarm")), "change_alarm": bool(result.get("change_alarm")),
        "change_score": r(result.get("change_score"), 3), "change_threshold": r(result.get("change_threshold"), 2),
        "change_over": bool(result.get("change_over")), "change_state": result.get("change_state"),
        "change_deviations": result.get("change_deviations", []),
        "speed_changed": bool(result.get("speed_changed")), "baseline_rpm": r(result.get("baseline_rpm"), 0),
        "top_deviations": result.get("top_deviations", []),
        "predicted_fault": result.get("predicted_fault"),
        "fault_confidence": r(result.get("fault_confidence"), 2),
    }
