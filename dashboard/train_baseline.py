"""Train the baseline anomaly model the dashboard uses: Isolation Forest on HEALTHY sessions only.

    python train_baseline.py                          # real sessions in ../data/raw  -> ../models/anomaly_baseline.joblib
    python train_baseline.py --data D:\\my_sessions    # another folder of .jsonl sessions
    python train_baseline.py --synthetic              # synthetic sessions -> models/anomaly_synthetic.joblib (demo only)

What it does
  1. Builds the feature table with pm/dataset.py (same features the live dashboard computes).
  2. Keeps rows labelled 'healthy'. Fault sessions, if any, are only used to report detection.
  3. Sets the alarm threshold from sessions the model did NOT see (leave-one-session-out):
     threshold = 99th percentile of held-out healthy scores, so ~1 % of healthy windows cross it.
  4. Fits the final model on all healthy sessions and saves it in the format the dashboard and
     pm/recorder.py both read: {"model", "features", "threshold", ...}.
  5. Writes reports/baseline_anomaly.md with the false-alarm rate per held-out session.

This needs at least 3 healthy sessions. More sessions, on different days, give a better model.
"""
import argparse
import json
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from common import DUMMY_MODEL, MODELS, RAW, ROOT, dataset, features, synthetic


def fit(X, seed=0):
    return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                         IsolationForest(n_estimators=300, max_samples=min(256, len(X)),
                                         contamination="auto", random_state=seed, n_jobs=-1)).fit(X)


def alarm_rate(over, k=3, n=5):
    """Fraction of windows where the dashboard's k-of-n alarm would be on."""
    over = np.asarray(over, dtype=int)
    if len(over) < n:
        return 0.0
    run = np.convolve(over, np.ones(n, dtype=int), mode="valid")
    return float((run >= k).mean())


def synthetic_table(seed=11, sessions=8, msgs=90):
    rng = np.random.default_rng(seed)
    rows = []
    for k in range(sessions):
        sid = f"synthetic_healthy_r{k}"
        hist = []
        for rec in synthetic.session(rng, sid, "healthy", 0, msgs):
            f = features.extract(rec)
            hist = (hist + [f])[-features.TEMPORAL_WINDOW:]
            rows.append({**f, **features.temporal(hist), "session": sid, "label": "healthy", "severity": 0})
    return pd.DataFrame(rows)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", default=str(RAW), help="folder with recorded .jsonl sessions")
    p.add_argument("--out", help="output .joblib (default ../models/anomaly_baseline.joblib)")
    p.add_argument("--percentile", type=float, default=99.0, help="threshold percentile of held-out healthy scores")
    p.add_argument("--synthetic", action="store_true", help="train on synthetic data (pipeline demo only)")
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args(argv)

    if a.synthetic:
        df = synthetic_table()
        out = Path(a.out) if a.out else DUMMY_MODEL
        source = "synthetic data (demo only)"
    else:
        df = dataset.load_sessions(a.data)
        out = Path(a.out) if a.out else MODELS / "anomaly_baseline.joblib"
        source = f"{df.loc[df['label'] == 'healthy', 'session'].nunique()} real healthy sessions"

    healthy = df[df["label"] == "healthy"].reset_index(drop=True)
    sessions = list(healthy["session"].unique())
    if len(sessions) < 3:
        raise SystemExit(f"Need at least 3 healthy sessions, found {len(sessions)} in {a.data}")
    cols = dataset.feature_columns(healthy)
    print(f"{len(healthy)} healthy windows from {len(sessions)} sessions, {len(cols)} features")

    # Threshold from sessions the model never saw, so the false-alarm rate is honest.
    held = {}
    for s in sessions:
        m = fit(healthy.loc[healthy["session"] != s, cols], a.seed)
        held[s] = -m.score_samples(healthy.loc[healthy["session"] == s, cols])
    all_held = np.concatenate(list(held.values()))
    threshold = float(np.percentile(all_held, a.percentile))

    lines = [f"# Baseline anomaly model ({datetime.now():%d %b %Y %H:%M})", "",
             f"Isolation Forest on {len(healthy)} healthy windows, {len(sessions)} sessions, {len(cols)} features.",
             f"Threshold = {a.percentile:g}th percentile of leave-one-session-out healthy scores = **{threshold:.4f}**.",
             "", "| Held-out session | Windows | Over threshold | Alarm on (3 of 5) |", "|---|---|---|---|"]
    for s in sessions:
        sc = held[s]
        lines.append(f"| {s} | {len(sc)} | {(sc > threshold).mean():.1%} | {alarm_rate(sc > threshold):.1%} |")
    lines.append(f"| **all** | {len(all_held)} | {(all_held > threshold).mean():.1%} | "
                 f"{np.mean([alarm_rate(held[s] > threshold) for s in sessions]):.1%} |")

    final = fit(healthy[cols], a.seed)
    X = healthy[cols].apply(pd.to_numeric, errors="coerce")
    med = X.median()
    iqr = (X.quantile(0.75) - X.quantile(0.25)) / 1.349
    scale = iqr.where(iqr > 1e-12, X.std()).replace(0, np.nan)

    faults = df[df["label"] != "healthy"]
    if len(faults):
        sc = -final.score_samples(faults[cols])
        lines += ["", "Detection on fault sessions (model never saw them):", "",
                  "| Label | Severity | Windows | Over threshold |", "|---|---|---|---|"]
        for (lab, sev), idx in faults.reset_index(drop=True).groupby(["label", "severity"]).groups.items():
            lines.append(f"| {lab} | {sev} | {len(idx)} | {(sc[np.asarray(idx)] > threshold).mean():.0%} |")

    out.parent.mkdir(parents=True, exist_ok=True)
    MODELS.mkdir(parents=True, exist_ok=True)
    joblib.dump({"kind": "anomaly", "name": "Isolation Forest baseline", "model": final, "features": cols,
                 "threshold": threshold, "source": source, "score_sign": -1.0,
                 "ref_median": {k: float(v) for k, v in med.items() if np.isfinite(v)},
                 "ref_scale": {k: float(v) for k, v in scale.items() if np.isfinite(v)},
                 "sessions": sessions, "created": datetime.now().isoformat(timespec="seconds"),
                 "feature_version": dataset.CACHE_VERSION}, out)
    if not a.synthetic:   # healthy reference for the dashboard's 'why' text and change detector
        (MODELS / "healthy_reference.json").write_text(json.dumps({
            "about": "Median and robust spread of each feature over all healthy windows (train_baseline.py).",
            "sessions": sessions, "median": {k: float(v) for k, v in med.items() if np.isfinite(v)},
            "scale": {k: float(v) for k, v in scale.items() if np.isfinite(v)}}, indent=0))
    report = "\n".join(lines)
    print(report)
    if not a.synthetic:
        rep = ROOT / "reports" / "baseline_anomaly.md"
        rep.parent.mkdir(exist_ok=True)
        rep.write_text(report + "\n")
        print(f"\nReport: {rep}")
    print(f"Saved {out}")
    return out


if __name__ == "__main__":
    main()
