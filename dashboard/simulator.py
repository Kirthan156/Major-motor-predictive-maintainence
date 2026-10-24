"""Dummy motor: one firmware-v2 style message every call, from pm/synthetic.py.

Unlike synthetic.session(), which writes a whole recording at once, this keeps one motor
"installation" (rotation speed, mount stiffness, noise, ambient temperature) fixed and lets the
dashboard switch the fault and its severity while it runs.
"""
import time

import numpy as np

from common import synthetic

LABELS = ["healthy"] + synthetic.FAULTS


def new_installation(rng):
    # Same distributions as synthetic.session(), so a healthy simulator looks like training data.
    return {"rot_hz": rng.normal(21.7, 0.4), "mount_gain": rng.uniform(0.8, 1.25),
            "noise_g": rng.uniform(0.008, 0.014), "i0": rng.normal(0.27, 0.006),
            "amb": rng.uniform(24, 32), "snd": rng.normal(-52, 1.5)}


class MotorSimulator:
    def __init__(self, seed=None, warmup_msgs=150):
        self.rng = np.random.default_rng(seed)
        self.sess = new_installation(self.rng)
        self.label, self.severity = "healthy", 0.0
        self.seq = 0
        self.warmup_msgs = warmup_msgs
        self.t0 = time.time()

    def set_fault(self, label, severity):
        if label not in LABELS:
            raise ValueError(f"unknown label {label!r}; use one of {LABELS}")
        self.label = label
        self.severity = 0.0 if label == "healthy" else float(min(max(severity, 0.0), 3.0))

    def next_message(self):
        rng, sess, sev = self.rng, self.sess, self.severity
        label = self.label if sev > 0 else "healthy"
        vib, rot = synthetic.burst(rng, label, sev, sess)
        cur = sess["i0"] * (1 - (0.03 * sev if label == "air_blockage" else 0)
                            + (0.01 * sev if label in ("foot_wedge", "one_sided_mount") else 0))
        cur += rng.normal(0, 0.002)
        warm = min(self.seq / self.warmup_msgs, 1) * 4
        msg = {
            "device_id": "simulator", "fw": "v2", "seq": self.seq,
            "uptime_ms": int((time.time() - self.t0) * 1000),
            "ds18b20_temp_c": sess["amb"] + 6 + warm + rng.normal(0, 0.1),
            "dht22_temp_c": sess["amb"] + rng.normal(0, 0.1), "dht22_humidity": rng.uniform(40, 70),
            "rpm": rot * 60 + rng.normal(0, 5), "voltage": rng.normal(236, 2), "current": cur,
            "power": cur * 236 * 0.94, "pf": 0.94 + rng.normal(0, 0.005), "mains_hz": 50.0,
            "energy": 0.0,
            "sound_dbfs": sess["snd"] + (1.5 * sev if label == "air_blockage" else 0)
                          + (0.8 * sev if label == "loose_blade" else 0) + rng.normal(0, 0.5),
            "vib": vib,
        }
        self.seq += 1
        return msg
