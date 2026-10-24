"""Feature extraction for one motor message (one ~1 s vibration burst + slow sensors).

The same function is used by the live recorder and by training, so the features a model
sees at training time are exactly the ones it sees live.
"""
import math

import numpy as np

BLADES = 3              # fan blades; blade-pass frequency = BLADES x rotation
MIN_ROT_HZ = 8.0        # search range for the 1x rotation peak when RPM is missing
MAX_ROT_HZ = 30.0
BANDS_HZ = [(2, 10), (10, 40), (40, 100), (100, 200), (200, 450)]

SLOW_FIELDS = ["voltage", "current", "power", "pf", "rpm", "sound_dbfs",
               "ds18b20_temp_c", "dht22_temp_c", "dht22_humidity"]


def _num(v):
    try:
        f = float(v)
    except (TypeError, ValueError):
        return math.nan
    return f


def _time_features(sig, prefix):
    rms = float(np.sqrt(np.mean(sig ** 2)))
    peak = float(np.max(np.abs(sig)))
    std = float(np.std(sig))
    if std > 0:
        z = sig / std
        kurt = float(np.mean(z ** 4))
        skew = float(np.mean(z ** 3))
    else:
        kurt = skew = 0.0
    return {
        f"{prefix}_rms_g": rms,
        f"{prefix}_peak_g": peak,
        f"{prefix}_p2p_g": float(np.ptp(sig)),
        f"{prefix}_crest": peak / rms if rms > 0 else 0.0,
        f"{prefix}_kurtosis": kurt,
        f"{prefix}_skew": skew,
    }


def _amp_at(freqs, amp, f0, tol_hz):
    if not f0 or f0 <= 0:
        return 0.0
    m = (freqs >= f0 - tol_hz) & (freqs <= f0 + tol_hz)
    return float(amp[m].max()) if m.any() else 0.0


def _spectrum_features(sig, fs, rot_hz, prefix):
    n = len(sig)
    win = np.hanning(n)
    spec = np.fft.rfft(sig * win)
    amp = 2.0 * np.abs(spec) / win.sum()          # single-sided amplitude in g
    freqs = np.fft.rfftfreq(n, 1.0 / fs)
    tol = max(1.5 * fs / n, 1.0)

    valid = freqs >= 2.0
    dom_idx = np.argmax(amp * valid)
    power = amp ** 2
    total = power[valid].sum() or 1e-12

    out = {
        f"{prefix}_dom_hz": float(freqs[dom_idx]),
        f"{prefix}_dom_amp_g": float(amp[dom_idx]),
        f"{prefix}_centroid_hz": float((freqs[valid] * power[valid]).sum() / total),
        f"{prefix}_1x_g": _amp_at(freqs, amp, rot_hz, tol),
        f"{prefix}_2x_g": _amp_at(freqs, amp, 2 * rot_hz, tol),
        f"{prefix}_3x_g": _amp_at(freqs, amp, 3 * rot_hz, tol),
        f"{prefix}_bpf_g": _amp_at(freqs, amp, BLADES * rot_hz, tol),
    }
    for lo, hi in BANDS_HZ:
        m = (freqs >= lo) & (freqs < hi)
        out[f"{prefix}_band_{lo}_{hi}"] = float(power[m].sum() / total)
    return out


def estimate_rotation_hz(vib_axes, fs):
    """1x rotation frequency from the vibration spectrum, used when the RPM sensor reads 0."""
    total = None
    for sig in vib_axes:
        n = len(sig)
        amp = np.abs(np.fft.rfft(sig * np.hanning(n)))
        total = amp if total is None else total + amp
    freqs = np.fft.rfftfreq(len(vib_axes[0]), 1.0 / fs)
    m = (freqs >= MIN_ROT_HZ) & (freqs <= MAX_ROT_HZ)
    return float(freqs[m][np.argmax(total[m])]) if m.any() else 0.0


def extract(msg):
    """Return a flat dict of features for one message, or None if it has no vibration burst."""
    vib = msg.get("vib")
    if not vib:
        return None
    fs = float(vib["fs"])
    scale = float(vib.get("lsb_per_g", 8192))
    axes = {}
    for a in ("x", "y", "z"):
        sig = np.asarray(vib[a], dtype=float) / scale
        axes[a] = sig - sig.mean()            # remove gravity and sensor offset

    rpm = _num(msg.get("rpm"))
    if rpm and rpm > 0 and not math.isnan(rpm):
        rot_hz, rot_src = rpm / 60.0, 1.0
    else:
        rot_hz, rot_src = estimate_rotation_hz(list(axes.values()), fs), 0.0

    feats = {"fs_hz": fs, "rot_hz": rot_hz, "rot_from_rpm": rot_src}
    for a, sig in axes.items():
        feats.update(_time_features(sig, a))
        feats.update(_spectrum_features(sig, fs, rot_hz, a))

    mag = np.sqrt(axes["x"] ** 2 + axes["y"] ** 2 + axes["z"] ** 2)
    feats["mag_rms_g"] = float(np.sqrt(np.mean(mag ** 2)))
    feats["mag_kurtosis"] = float(np.mean(((mag - mag.mean()) / (mag.std() or 1)) ** 4))

    for f in SLOW_FIELDS:
        feats[f] = _num(msg.get(f))
    feats["temp_rise_c"] = feats["ds18b20_temp_c"] - feats["dht22_temp_c"]
    return feats


# Feature groups, used for the ablation study (which sensors actually matter).
def feature_groups(columns):
    vib = [c for c in columns if c[:2] in ("x_", "y_", "z_") or c.startswith("mag_") or c == "rot_hz"]
    elec = [c for c in ("current", "power", "pf", "voltage") if c in columns]
    other = [c for c in ("rpm", "sound_dbfs", "temp_rise_c") if c in columns]
    return {
        "vibration": vib,
        "vibration+electrical": vib + elec,
        "all": vib + elec + other,
    }
