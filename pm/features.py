"""Feature extraction for one motor message (one ~1 s vibration burst + audio + slow sensors).

The same code is used by the live recorder and by training, so the features a model sees at
training time are exactly the ones it sees live.

Works with firmware v2 (accelerometer only, JSON arrays) and v3 (accelerometer + gyro + raw
audio, base64). Columns a message cannot provide come out as NaN, so every row has the same
columns.

Feature families (prefix):
  ax_ ay_ az_  accelerometer axes (g)        gx_ gy_ gz_  gyroscope axes (deg/s)
  amag_ gmag_  vector magnitudes             xc_          cross-axis relations
  aud_         audio                         el_          electrical (derived)
  rpm_         speed and speed ripple        th_          thermal / environment
  tilt_        static orientation (mean of the accelerometer = how the motor sits)
"""
import base64
import math

import numpy as np

BLADES = 3              # fan blades; blade-pass frequency = BLADES x rotation
POLES = 4               # 1479 RPM on 50 Hz mains -> 4-pole motor (sync 1500 RPM)
MIN_ROT_HZ = 8.0        # search range for the 1x rotation peak when RPM is missing
MAX_ROT_HZ = 30.0
G = 9.80665

VIB_BANDS = [(2, 10), (10, 20), (20, 30), (30, 50), (50, 80), (80, 120), (120, 200), (200, 300), (300, 450)]
AUD_BANDS = [(20, 125), (125, 250), (250, 500), (500, 1000), (1000, 2000), (2000, 4000), (4000, 8000)]
N_MFCC = 13

# Columns that describe the recording, not the motor. Kept in the dataset, never used for training.
NON_FEATURES = {
    "fs_hz", "rot_from_rpm", "vib_err", "vib_max_gap_us", "wifi_rssi", "free_heap", "cycle_ms",
    "uptime_ms", "energy", "mains_hz", "voltage", "th_ambient_c", "th_humidity", "th_dew_point_c",
    "th_motor_c", "th_mpu_c", "rpm_pulses", "aud_n",
}


# ---------------------------------------------------------------- helpers

def _num(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return math.nan


def _decode(v):
    """JSON list (v2) or base64 little-endian int16 (v3) -> float array."""
    if v is None:
        return None
    if isinstance(v, str):
        return np.frombuffer(base64.b64decode(v), dtype="<i2").astype(float)
    return np.asarray(v, dtype=float)


def _safe_div(a, b):
    return a / b if b else 0.0


def _time_features(sig, p):
    """Classic condition-monitoring statistics of one zero-mean signal."""
    absx = np.abs(sig)
    rms = float(np.sqrt(np.mean(sig ** 2)))
    peak = float(absx.max())
    mean_abs = float(absx.mean())
    std = float(sig.std())
    z = sig / std if std > 0 else sig * 0
    sqrt_mean = float(np.mean(np.sqrt(absx)) ** 2)
    diff = np.diff(sig)
    return {
        f"{p}_rms": rms,
        f"{p}_std": std,
        f"{p}_peak": peak,
        f"{p}_p2p": float(np.ptp(sig)),
        f"{p}_mean_abs": mean_abs,
        f"{p}_crest": _safe_div(peak, rms),
        f"{p}_impulse": _safe_div(peak, mean_abs),
        f"{p}_shape": _safe_div(rms, mean_abs),
        f"{p}_clearance": _safe_div(peak, sqrt_mean),
        f"{p}_kurtosis": float(np.mean(z ** 4)),
        f"{p}_skew": float(np.mean(z ** 3)),
        f"{p}_p01": float(np.percentile(sig, 1)),
        f"{p}_p99": float(np.percentile(sig, 99)),
        f"{p}_iqr": float(np.subtract(*np.percentile(sig, [75, 25]))),
        f"{p}_zcr": float(np.mean(np.signbit(sig[1:]) != np.signbit(sig[:-1]))),
        f"{p}_diff_rms": float(np.sqrt(np.mean(diff ** 2))),   # jerk-like, sensitive to impacts
    }


def _spectrum(sig, fs):
    n = len(sig)
    win = np.hanning(n)
    amp = 2.0 * np.abs(np.fft.rfft(sig * win)) / win.sum()     # single-sided peak amplitude
    freqs = np.fft.rfftfreq(n, 1.0 / fs)
    return freqs, amp


def _amp_at(freqs, amp, f0, tol):
    if not f0 or f0 <= 0 or f0 > freqs[-1]:
        return 0.0
    m = (freqs >= f0 - tol) & (freqs <= f0 + tol)
    return float(amp[m].max()) if m.any() else 0.0


def _spectral_shape(freqs, power, p, lo=2.0):
    valid = freqs >= lo
    f, pw = freqs[valid], power[valid]
    total = pw.sum() or 1e-30
    pn = pw / total
    centroid = float((f * pn).sum())
    spread = float(np.sqrt(((f - centroid) ** 2 * pn).sum()))
    cum = np.cumsum(pn)
    rolloff = float(f[min(np.searchsorted(cum, 0.85), len(f) - 1)])
    flatness = float(np.exp(np.mean(np.log(pw + 1e-30))) / (pw.mean() + 1e-30))
    entropy = float(-(pn * np.log2(pn + 1e-30)).sum() / np.log2(len(pn)))
    dom = int(np.argmax(pw))
    return {
        f"{p}_dom_hz": float(f[dom]),
        f"{p}_dom_amp": float(np.sqrt(pw[dom])),
        f"{p}_centroid_hz": centroid,
        f"{p}_spread_hz": spread,
        f"{p}_rolloff85_hz": rolloff,
        f"{p}_flatness": flatness,
        f"{p}_entropy": entropy,
    }


def _vib_axis_features(sig, fs, rot_hz, mains_hz, p, to_velocity):
    out = _time_features(sig, p)
    freqs, amp = _spectrum(sig, fs)
    power = amp ** 2
    out.update(_spectral_shape(freqs, power, p))
    tol = max(1.5 * fs / len(sig), 1.0)
    harm = 0.0
    for k in (1, 2, 3, 4, 5):
        a = _amp_at(freqs, amp, k * rot_hz, tol)
        out[f"{p}_{k}x"] = a
        harm += a ** 2
    out[f"{p}_half_x"] = _amp_at(freqs, amp, 0.5 * rot_hz, tol)      # looseness / rub
    out[f"{p}_bpf"] = _amp_at(freqs, amp, BLADES * rot_hz, tol)
    out[f"{p}_2bpf"] = _amp_at(freqs, amp, 2 * BLADES * rot_hz, tol)
    out[f"{p}_2line"] = _amp_at(freqs, amp, 2 * (mains_hz or 50.0), tol)  # electrical (magnetic) vibration
    total = power[freqs >= 2].sum() or 1e-30
    out[f"{p}_harmonic_ratio"] = float(harm / total)
    out[f"{p}_2x_1x"] = _safe_div(out[f"{p}_2x"], out[f"{p}_1x"])
    out[f"{p}_3x_1x"] = _safe_div(out[f"{p}_3x"], out[f"{p}_1x"])
    for lo, hi in VIB_BANDS:
        m = (freqs >= lo) & (freqs < hi)
        out[f"{p}_band_{lo}_{hi}"] = float(power[m].sum() / total)

    if to_velocity:
        # ISO 10816-style velocity RMS in mm/s, 10-450 Hz, from the acceleration spectrum.
        n = len(sig)
        spec = np.fft.rfft(sig * G)                      # m/s^2
        f = np.fft.rfftfreq(n, 1.0 / fs)
        band = (f >= 10) & (f <= 450)
        a_amp = 2.0 * np.abs(spec[band]) / n
        v_amp = a_amp / (2 * np.pi * f[band]) * 1000.0   # mm/s
        out[f"{p}_vel_rms_mms"] = float(np.sqrt(np.sum(v_amp ** 2) / 2.0))
        out[f"{p}_vel_1x_mms"] = float(_safe_div(out[f"{p}_1x"] * G * 1000.0, 2 * np.pi * rot_hz)) if rot_hz else 0.0
    return out


def estimate_rotation_hz(axes, fs):
    """1x rotation frequency from the vibration spectrum, used when the RPM sensor reads 0."""
    total = None
    for sig in axes:
        amp = np.abs(np.fft.rfft(sig * np.hanning(len(sig))))
        total = amp if total is None else total + amp
    freqs = np.fft.rfftfreq(len(axes[0]), 1.0 / fs)
    m = (freqs >= MIN_ROT_HZ) & (freqs <= MAX_ROT_HZ)
    if not m.any():
        return 0.0
    i = np.flatnonzero(m)[np.argmax(total[m])]
    # parabolic interpolation for sub-bin accuracy
    if 0 < i < len(total) - 1:
        a, b, c = np.log(total[i - 1] + 1e-12), np.log(total[i] + 1e-12), np.log(total[i + 1] + 1e-12)
        d = 0.5 * (a - c) / (a - 2 * b + c) if (a - 2 * b + c) != 0 else 0.0
        return float(freqs[i] + d * (freqs[1] - freqs[0]))
    return float(freqs[i])


def _phase_at(sig, fs, f0):
    n = len(sig)
    spec = np.fft.rfft(sig * np.hanning(n))
    freqs = np.fft.rfftfreq(n, 1.0 / fs)
    return float(np.angle(spec[np.argmin(np.abs(freqs - f0))]))


def _cross_features(a, fs, rot_hz):
    out = {}
    pairs = [("x", "y"), ("x", "z"), ("y", "z")]
    for u, v in pairs:
        cu, cv = a[u], a[v]
        su, sv = cu.std(), cv.std()
        out[f"xc_corr_{u}{v}"] = float(np.mean(cu * cv) / (su * sv)) if su > 0 and sv > 0 else 0.0
        out[f"xc_rms_ratio_{u}{v}"] = _safe_div(float(np.sqrt(np.mean(cu ** 2))), float(np.sqrt(np.mean(cv ** 2))))
        if rot_hz:
            d = _phase_at(cu, fs, rot_hz) - _phase_at(cv, fs, rot_hz)
            d = (d + np.pi) % (2 * np.pi) - np.pi
            out[f"xc_phase1x_{u}{v}_deg"] = float(np.degrees(abs(d)))   # ~90 deg for pure imbalance
        else:
            out[f"xc_phase1x_{u}{v}_deg"] = math.nan
    return out


def _mel_filterbank(n_filters, n_fft, fs, fmin=20.0, fmax=None):
    fmax = fmax or fs / 2
    mel = lambda f: 2595 * np.log10(1 + f / 700.0)          # noqa: E731
    inv = lambda m: 700 * (10 ** (m / 2595.0) - 1)           # noqa: E731
    pts = inv(np.linspace(mel(fmin), mel(fmax), n_filters + 2))
    bins = np.floor((n_fft + 1) * pts / fs).astype(int)
    fb = np.zeros((n_filters, n_fft // 2 + 1))
    for i in range(1, n_filters + 1):
        l, c, r = bins[i - 1], bins[i], bins[i + 1]
        for k in range(l, c):
            fb[i - 1, k] = (k - l) / max(c - l, 1)
        for k in range(c, r):
            fb[i - 1, k] = (r - k) / max(r - c, 1)
    return fb


_FB_CACHE = {}


def _audio_features(x, fs, rot_hz):
    out = {"aud_n": float(len(x))}
    x = x - x.mean()
    full = x / 32768.0
    rms = float(np.sqrt(np.mean(full ** 2)))
    out["aud_rms_dbfs"] = float(20 * np.log10(rms + 1e-12))
    out.update({k.replace("aud_t", "aud"): v for k, v in _time_features(full, "aud_t").items()
                if not k.endswith(("_p01", "_p99", "_iqr", "_mean_abs"))})

    freqs, amp = _spectrum(full, fs)
    power = amp ** 2
    out.update(_spectral_shape(freqs, power, "aud", lo=20.0))
    total = power[freqs >= 20].sum() or 1e-30
    for lo, hi in AUD_BANDS:
        m = (freqs >= lo) & (freqs < hi)
        out[f"aud_band_{lo}_{hi}"] = float(power[m].sum() / total)
    tol = max(1.5 * fs / len(full), 2.0)
    if rot_hz:
        for k in (1, 2, 3):
            out[f"aud_bpf{k}_db"] = float(20 * np.log10(_amp_at(freqs, amp, k * BLADES * rot_hz, tol) + 1e-12))
    else:
        for k in (1, 2, 3):
            out[f"aud_bpf{k}_db"] = math.nan
    out["aud_100hz_db"] = float(20 * np.log10(_amp_at(freqs, amp, 100.0, tol) + 1e-12))   # mains hum

    # MFCCs: 32 ms frames, 16 ms hop, 26 mel filters; mean and std over frames
    n_fft, hop = 512, 256
    if len(full) >= n_fft:
        key = (n_fft, fs)
        if key not in _FB_CACHE:
            fb = _mel_filterbank(26, n_fft, fs)
            k = np.arange(26)
            dct = np.cos(np.pi / 26 * (k + 0.5)[None, :] * np.arange(N_MFCC)[:, None])
            _FB_CACHE[key] = (fb, dct)
        fb, dct = _FB_CACHE[key]
        frames = np.lib.stride_tricks.sliding_window_view(full, n_fft)[::hop] * np.hanning(n_fft)
        pspec = np.abs(np.fft.rfft(frames, axis=1)) ** 2
        logmel = np.log(pspec @ fb.T + 1e-12)
        mfcc = logmel @ dct.T
        for i in range(N_MFCC):
            out[f"aud_mfcc{i}_mean"] = float(mfcc[:, i].mean())
            out[f"aud_mfcc{i}_std"] = float(mfcc[:, i].std())
        # frame-to-frame loudness variation (rattles, intermittent rubbing)
        frame_db = 10 * np.log10(pspec.sum(axis=1) + 1e-12)
        out["aud_frame_db_std"] = float(frame_db.std())
        out["aud_frame_db_range"] = float(np.ptp(frame_db))
    return out


def _dew_point(t, rh):
    if math.isnan(t) or math.isnan(rh) or rh <= 0:
        return math.nan
    a, b = 17.62, 243.12
    g = math.log(rh / 100.0) + a * t / (b + t)
    return b * g / (a - g)


# ---------------------------------------------------------------- main entry

def extract(msg):
    """Return a flat dict of features for one message, or None if it has no vibration burst."""
    vib = msg.get("vib")
    if not vib:
        return None
    fs = float(vib["fs"])
    lsb_g = float(vib.get("lsb_per_g", 8192))
    lsb_dps = float(vib.get("lsb_per_dps", 65.5))
    mains_hz = _num(msg.get("mains_hz"))
    mains_hz = 50.0 if math.isnan(mains_hz) else mains_hz

    acc_raw = {a: _decode(vib.get(a)) / lsb_g for a in ("x", "y", "z")}
    acc = {a: s - s.mean() for a, s in acc_raw.items()}
    gyro = {}
    for a in ("x", "y", "z"):
        g = _decode(vib.get("g" + a))
        if g is not None:
            g = g / lsb_dps
            gyro[a] = g

    rpm = _num(msg.get("rpm"))
    if rpm and rpm > 0 and not math.isnan(rpm):
        rot_hz, rot_src = rpm / 60.0, 1.0
    else:
        rot_hz, rot_src = estimate_rotation_hz(list(acc.values()), fs), 0.0

    f = {"fs_hz": fs, "rot_hz": rot_hz, "rot_from_rpm": rot_src,
         "rot_hz_vib": estimate_rotation_hz(list(acc.values()), fs),
         "vib_err": _num(vib.get("err")), "vib_max_gap_us": _num(vib.get("max_gap_us"))}

    # static orientation: how the motor sits (foot wedge / one-sided mount tilt it)
    mx, my, mz = (float(acc_raw[a].mean()) for a in ("x", "y", "z"))
    f["tilt_ax_mean_g"], f["tilt_ay_mean_g"], f["tilt_az_mean_g"] = mx, my, mz
    f["tilt_pitch_deg"] = float(np.degrees(np.arctan2(-mx, np.hypot(my, mz))))
    f["tilt_roll_deg"] = float(np.degrees(np.arctan2(my, mz)))

    for a, sig in acc.items():
        f.update(_vib_axis_features(sig, fs, rot_hz, mains_hz, f"a{a}", to_velocity=True))
    mag = np.sqrt(sum(s ** 2 for s in acc.values()))
    f.update(_time_features(mag - mag.mean(), "amag"))
    f["amag_vel_rms_mms"] = float(np.sqrt(sum(f[f"a{a}_vel_rms_mms"] ** 2 for a in "xyz")))
    f.update(_cross_features(acc, fs, rot_hz))

    if gyro:
        for a, sig in gyro.items():
            f[f"g{a}_mean_dps"] = float(sig.mean())
            f.update(_vib_axis_features(sig - sig.mean(), fs, rot_hz, mains_hz, f"g{a}", to_velocity=False))
        gm = np.sqrt(sum((s - s.mean()) ** 2 for s in gyro.values()))
        f.update(_time_features(gm - gm.mean(), "gmag"))

    audio = msg.get("audio")
    if audio and audio.get("data"):
        f.update(_audio_features(_decode(audio["data"]), float(audio["fs"]), rot_hz))
    f["aud_dbfs_device"] = _num(msg.get("sound_dbfs"))

    # electrical
    v, i, p, pf = (_num(msg.get(k)) for k in ("voltage", "current", "power", "pf"))
    for k in ("voltage", "current", "power", "energy", "pf", "mains_hz"):
        f[k] = _num(msg.get(k))
    s = v * i
    f["el_apparent_va"] = s
    f["el_reactive_var"] = math.sqrt(max(s * s - p * p, 0.0)) if not (math.isnan(s) or math.isnan(p)) else math.nan
    f["el_phase_deg"] = math.degrees(math.acos(min(max(pf, -1), 1))) if not math.isnan(pf) else math.nan
    f["el_impedance_ohm"] = v / i if i and i > 0.005 else math.nan
    f["el_power_per_rpm"] = p / rpm if rpm and rpm > 0 else math.nan
    f["el_current_per_vib"] = i / f["amag_rms"] if f["amag_rms"] > 0 else math.nan

    # speed
    sync_rpm = 120.0 * mains_hz / POLES
    f["rpm"] = rpm
    for k in ("rpm_min", "rpm_max", "rpm_std", "rpm_pulses"):
        f[k] = _num(msg.get(k))
    f["rpm_range"] = f["rpm_max"] - f["rpm_min"]
    f["rpm_slip"] = (sync_rpm - rot_hz * 60.0) / sync_rpm if rot_hz else math.nan

    # thermal / environment
    tm, ta, rh, tmpu = (_num(msg.get(k)) for k in ("ds18b20_temp_c", "dht22_temp_c", "dht22_humidity", "mpu_temp_c"))
    f["th_motor_c"], f["th_ambient_c"], f["th_humidity"], f["th_mpu_c"] = tm, ta, rh, tmpu
    f["th_rise_c"] = tm - ta
    f["th_mpu_rise_c"] = tmpu - ta
    f["th_dew_point_c"] = _dew_point(ta, rh)

    # diagnostics
    for k in ("wifi_rssi", "free_heap", "cycle_ms", "uptime_ms"):
        f[k] = _num(msg.get(k))
    return f


# ---------------------------------------------------------------- temporal features

TEMPORAL_BASE = ["amag_rms", "amag_vel_rms_mms", "ax_1x", "ay_1x", "current", "power", "pf", "rpm",
                 "aud_rms_dbfs", "th_motor_c", "th_rise_c", "gmag_rms"]


def temporal(history):
    """Features over the last few messages of the same run. history = list of feature dicts,
    oldest first, ending with the current one. Used identically in training and live."""
    cur = history[-1]
    out = {}
    for k in TEMPORAL_BASE:
        vals = np.array([h.get(k, math.nan) for h in history], dtype=float)
        x = cur.get(k, math.nan)
        prev = vals[-2] if len(vals) > 1 else math.nan
        w5 = vals[-5:]
        out[f"tm_{k}_delta"] = x - prev
        out[f"tm_{k}_mean5"] = float(np.nanmean(w5)) if np.isfinite(w5).any() else math.nan
        out[f"tm_{k}_std5"] = float(np.nanstd(w5)) if np.isfinite(w5).sum() > 1 else math.nan
    # motor temperature slope over the window (C per minute), 2 s per message
    t = np.array([h.get("th_motor_c", math.nan) for h in history], dtype=float)
    ok = np.isfinite(t)
    out["tm_motor_slope_c_per_min"] = (float(np.polyfit(np.arange(len(t))[ok] * 2.0, t[ok], 1)[0] * 60.0)
                                       if ok.sum() >= 5 else math.nan)
    return out


TEMPORAL_WINDOW = 15   # messages (30 s)


# Feature groups, used for the ablation study (which sensors actually matter).
def feature_groups(columns):
    pick = lambda *pre: [c for c in columns if c.startswith(pre)]   # noqa: E731
    accel = pick("ax_", "ay_", "az_", "amag_", "xc_") + [c for c in ("rot_hz", "rot_hz_vib") if c in columns]
    gyro = pick("gx_", "gy_", "gz_", "gmag_")
    elec = [c for c in columns if c in ("current", "power", "pf") or c.startswith("el_")]
    audio = pick("aud_")
    other = pick("rpm", "th_", "tilt_", "tm_")
    groups = {"accelerometer": accel}
    if gyro:
        groups["accel+gyro"] = accel + gyro
    groups["accel+gyro+electrical"] = accel + gyro + elec
    if audio:
        groups["accel+gyro+electrical+audio"] = accel + gyro + elec + audio
    groups["all"] = accel + gyro + elec + audio + other
    return groups
