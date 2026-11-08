"""Demo motor for the dashboard: real recorded healthy messages, with faults injected on demand.

It plays the firmware v3 messages recorded on the rig (data/sample/*.jsonl, or data/raw/ if that
is empty) in a loop. When you pick a fault on the page, every following message is changed the
way that fault would change it: extra 1x vibration for imbalance, impacts and half-speed content
for a loose blade, a tilt for a foot wedge, more rocking for a one-sided mount, lower current and
higher sound for air blockage.

The injected faults are an approximation for demonstrating the dashboard. They are NOT data from
real faults, so never report detection numbers from them.

If no recordings exist at all it falls back to pm/synthetic.py messages.
"""
import base64
import copy
import json
import time
from pathlib import Path

import numpy as np

from common import RAW, SAMPLE, features, synthetic

LABELS = ["healthy", "imbalance", "loose_blade", "foot_wedge", "one_sided_mount", "air_blockage"]


def _enc(sig_counts):
    return base64.b64encode(np.clip(np.round(sig_counts), -32768, 32767).astype("<i2").tobytes()).decode()


def load_messages(paths):
    msgs = []
    for p in paths:
        with open(p) as fh:
            for line in fh:
                line = line.strip()
                if line:
                    m = json.loads(line)
                    if m.get("vib"):
                        msgs.append(m)
    return msgs


def default_recordings():
    for d in (SAMPLE, RAW):
        files = sorted(Path(d).glob("*.jsonl"))
        if files:
            return files
    return []


class MotorSimulator:
    def __init__(self, seed=None, files=None):
        self.rng = np.random.default_rng(seed)
        files = files if files else default_recordings()
        self.msgs = load_messages(files) if files else []
        self.files = [str(f) for f in files] if files else []
        self.kind = "recorded" if self.msgs else "synthetic"
        if not self.msgs:
            self.sess = {"rot_hz": 21.7, "mount_gain": 1.0, "noise_g": 0.01, "i0": 0.27, "amb": 28, "snd": -52}
            self.synth = synthetic.session(self.rng, "demo", "healthy", 0, 400)
        self.label, self.severity = "healthy", 0.0
        self.i = 0
        self.seq = 0
        self.t0 = time.time()

    def set_fault(self, label, severity):
        if label not in LABELS:
            raise ValueError(f"unknown label {label!r}; use one of {LABELS}")
        self.label = label
        self.severity = 0.0 if label == "healthy" else float(min(max(severity, 0.0), 3.0))

    def next_message(self):
        if self.msgs:
            base = self.msgs[self.i % len(self.msgs)]
        else:
            base = self.synth[self.i % len(self.synth)]
        self.i += 1
        msg = copy.deepcopy(base)
        for k in ("session", "label", "severity", "notes", "received_at"):
            msg.pop(k, None)
        msg["device_id"] = "demo_motor"
        msg["seq"] = self.seq
        msg["uptime_ms"] = int((time.time() - self.t0) * 1000) + 1
        self.seq += 1
        if self.label != "healthy" and self.severity > 0:
            inject(msg, self.label, self.severity, self.rng)
        return msg


def _gain_at(sig, fs, freq_gain, tol_hz=1.5):
    """Multiply the spectrum of sig around each frequency by a gain (keeps the rotor's own phase)."""
    spec = np.fft.rfft(sig)
    f = np.fft.rfftfreq(len(sig), 1.0 / fs)
    for f0, g in freq_gain:
        m = np.abs(f - f0) <= tol_hz
        spec[m] *= g
    return np.fft.irfft(spec, n=len(sig))


def inject(msg, label, sev, rng):
    """Change one real message in place the way the fault would (approximation, demo only).

    Imbalance and mounting faults scale the rotor's existing 1x / 2x vibration (so the change
    stays locked to the real rotation), a loose blade adds half-speed content and impacts, and the
    mounting faults also tilt the sensor (static g offset) and increase the gyro rocking.
    """
    vib = msg["vib"]
    fs = float(vib["fs"])
    lsb_g = float(vib.get("lsb_per_g", 8192))
    acc = {a: features._decode(vib[a]) / lsb_g for a in "xyz"}            # g
    gyr = {a: features._decode(vib.get("g" + a)) for a in "xyz"}          # raw counts
    n = len(acc["x"])
    t = np.arange(n) / fs
    rpm = msg.get("rpm") or 0
    rot = rpm / 60.0 if rpm and rpm > 0 else features.estimate_rotation_hz(list(acc.values()), fs)
    gscale = 1.0

    if label == "imbalance":
        for a in ("x", "y"):
            acc[a] = _gain_at(acc[a], fs, [(rot, 1 + 0.35 * sev)])
        msg["current"] = _scale(msg.get("current"), 1 + 0.005 * sev)
    elif label == "loose_blade":
        sx, sz = acc["x"].std(), acc["z"].std()
        acc["x"] = acc["x"] + 0.12 * sev * sx * np.sin(np.pi * rot * t + rng.uniform(0, 6.28))   # 0.5x
        period = int(fs / max(rot, 1))
        for start in range(int(rng.integers(0, period)), n, period):
            if rng.random() < 0.7:
                k = np.arange(min(25, n - start))
                hit = 0.8 * sev * sz * np.exp(-k / 5) * np.sin(2 * np.pi * 180 * k / fs)
                acc["z"][start:start + len(k)] += hit
                acc["x"][start:start + len(k)] += 0.5 * hit
        for key, f in (("rpm_std", 1 + 0.5 * sev), ("rpm_max", 1 + 0.01 * sev)):
            msg[key] = _scale(msg.get(key), f)
        _audio(msg, gain=1 + 0.1 * sev, clicks=sev, rng=rng)
    elif label == "foot_wedge":
        acc["x"] = acc["x"] + 0.03 * sev                                   # tilt
        acc["z"] = acc["z"] - 0.01 * sev
        for a in ("x", "y", "z"):
            acc[a] = _gain_at(acc[a], fs, [(2 * rot, 1 + 0.4 * sev)])
        gscale = 1 + 0.15 * sev
    elif label == "one_sided_mount":
        acc["y"] = _gain_at(acc["y"] + 0.02 * sev, fs, [(rot, 1 + 0.3 * sev), (2 * rot, 1 + 0.3 * sev)])
        gscale = 1 + 0.3 * sev
    elif label == "air_blockage":
        msg["current"] = _scale(msg.get("current"), 1 - 0.03 * sev)
        msg["power"] = _scale(msg.get("power"), 1 - 0.035 * sev)
        for key in ("rpm", "rpm_min", "rpm_max"):
            msg[key] = _scale(msg.get(key), 1 + 0.01 * sev)
        if msg.get("sound_dbfs") is not None:
            msg["sound_dbfs"] = msg["sound_dbfs"] + 1.5 * sev
        _audio(msg, gain=10 ** (1.5 * sev / 20), clicks=0, rng=rng)

    for a in "xyz":
        vib[a] = _enc(acc[a] * lsb_g)
        if gyr[a] is not None and gscale != 1.0:
            m = gyr[a].mean()
            vib["g" + a] = _enc(m + (gyr[a] - m) * gscale)


def _scale(v, f):
    return None if v is None else float(v) * f


def _audio(msg, gain, clicks, rng):
    au = msg.get("audio")
    if not au or not au.get("data"):
        return
    x = features._decode(au["data"])
    m = x.mean()
    y = m + (x - m) * gain
    if clicks:
        s = float((x - m).std())
        for _ in range(int(3 * clicks)):
            i = int(rng.integers(0, len(y) - 40))
            y[i:i + 40] += 3 * s * np.exp(-np.arange(40) / 8) * rng.choice([-1, 1])
    au["data"] = _enc(y)
