"""Generate fake sessions in the exact firmware v2 message format.

Purpose: test the whole recorder -> features -> training pipeline before real fault data
exists. The numbers it produces say NOTHING about the real rig; never put them in the report
as results.

    python synthetic.py --out ../data/synthetic
    python train.py --data ../data/synthetic
"""
import argparse
import json
from pathlib import Path

import numpy as np

FS, N, LSB = 998.5, 1024, 8192
FAULTS = ["imbalance", "loose_blade", "foot_wedge", "one_sided_mount", "air_blockage"]


def burst(rng, label, sev, sess):
    t = np.arange(N) / FS
    rot = sess["rot_hz"] * (1 + (0.01 * sev if label == "air_blockage" else 0)) + rng.normal(0, 0.05)
    g = sess["mount_gain"]
    noise = sess["noise_g"]
    ph = rng.uniform(0, 2 * np.pi)
    x = 0.010 * g * np.sin(2 * np.pi * rot * t + ph) + 0.004 * np.sin(2 * np.pi * 3 * rot * t)
    y = 0.010 * g * np.cos(2 * np.pi * rot * t + ph) + 0.004 * np.cos(2 * np.pi * 3 * rot * t)
    z = 0.004 * g * np.sin(2 * np.pi * 2 * rot * t + ph)
    z += 0.003 * np.sin(2 * np.pi * 100 * t)                      # 2 x mains hum

    if label == "imbalance":
        a = 0.012 * sev * g
        x += a * np.sin(2 * np.pi * rot * t + ph)
        y += a * np.cos(2 * np.pi * rot * t + ph)
    elif label == "loose_blade":
        for k in range(2, 6):
            x += 0.003 * sev / k * np.sin(2 * np.pi * k * rot * t + rng.uniform(0, 6))
        hits = np.zeros(N)
        period = int(FS / rot)
        for start in range(rng.integers(0, period), N, period):
            if rng.random() < 0.3 + 0.2 * sev:
                hits[start:start + 6] += 0.04 * sev * np.exp(-np.arange(min(6, N - start)))
        x += hits
        z += 0.6 * hits
    elif label == "foot_wedge":
        z += 0.010 * sev * g * np.sin(2 * np.pi * 2 * rot * t + ph)
        z += 0.008 * sev * np.sin(2 * np.pi * rot * t)
    elif label == "one_sided_mount":
        x += 0.010 * sev * g * np.sin(2 * np.pi * rot * t + ph)
        x += 0.006 * sev * np.sin(2 * np.pi * 2 * rot * t)
        x += 0.004 * sev * np.sin(2 * np.pi * 34 * t)               # shifted structural resonance
    elif label == "air_blockage":
        y += 0.004 * sev * np.sin(2 * np.pi * 3 * rot * t + rng.uniform(0, 6))

    x += rng.normal(0, noise, N)
    y += rng.normal(0, noise, N)
    z += 1.0 + rng.normal(0, noise, N)                             # gravity on z

    raw = lambda s: np.clip(np.round(s * LSB), -32768, 32767).astype(int).tolist()  # noqa: E731
    return {"fs": FS, "lsb_per_g": LSB, "n": N, "x": raw(x), "y": raw(y), "z": raw(z)}, rot


def session(rng, sid, label, sev, n_msgs):
    sess = {"rot_hz": rng.normal(21.7, 0.4), "mount_gain": rng.uniform(0.8, 1.25),
            "noise_g": rng.uniform(0.008, 0.014), "i0": rng.normal(0.27, 0.006),
            "amb": rng.uniform(24, 32), "snd": rng.normal(-52, 1.5)}
    out = []
    for i in range(n_msgs):
        vib, rot = burst(rng, label, sev, sess)
        cur = sess["i0"] * (1 - (0.03 * sev if label == "air_blockage" else 0)
                            + (0.01 * sev if label in ("foot_wedge", "one_sided_mount") else 0))
        cur += rng.normal(0, 0.002)
        warm = min(i / n_msgs, 1) * 4
        out.append({
            "device_id": "synthetic", "fw": "v2", "seq": i, "uptime_ms": i * 2000,
            "ds18b20_temp_c": sess["amb"] + 6 + warm + rng.normal(0, 0.1),
            "dht22_temp_c": sess["amb"] + rng.normal(0, 0.1), "dht22_humidity": rng.uniform(40, 70),
            "rpm": rot * 60 + rng.normal(0, 5), "voltage": rng.normal(236, 2), "current": cur,
            "power": cur * 236 * 0.94, "pf": 0.94 + rng.normal(0, 0.005), "mains_hz": 50.0,
            "energy": 0.0,
            "sound_dbfs": sess["snd"] + (1.5 * sev if label == "air_blockage" else 0)
                          + (0.8 * sev if label == "loose_blade" else 0) + rng.normal(0, 0.5),
            "vib": vib,
            "session": sid, "label": label, "severity": sev, "notes": "SYNTHETIC",
            "received_at": f"synthetic-{i}",
        })
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default=str(Path(__file__).resolve().parent.parent / "data" / "synthetic"))
    p.add_argument("--msgs", type=int, default=60, help="messages per session (2 s each)")
    p.add_argument("--repeats", type=int, default=2, help="sessions per label and severity")
    p.add_argument("--seed", type=int, default=7)
    a = p.parse_args()
    rng = np.random.default_rng(a.seed)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    plan = [("healthy", 0)] * (a.repeats * 3) + [(f, s) for f in FAULTS for s in (1, 2, 3)] * a.repeats
    counts = {}
    for label, sev in plan:
        counts[(label, sev)] = counts.get((label, sev), 0) + 1
        sid = f"syn_{label}_s{sev}_r{counts[(label, sev)]}"
        with (out / f"{sid}.jsonl").open("w") as fh:
            for rec in session(rng, sid, label, sev, a.msgs):
                fh.write(json.dumps(rec, separators=(",", ":")) + "\n")
    print(f"Wrote {len(plan)} synthetic sessions to {out}")


if __name__ == "__main__":
    main()
