"""Train the anomaly detector the dashboard uses until real data exists.

    python train_dummy.py

Fits the same model as pm/train.py (median imputer + scaler + isolation forest on healthy
windows only) on synthetic healthy sessions, and saves it in the same .joblib format, so the
server loads either file the same way. Once you have real healthy recordings, run
pm/train.py and start the server with --model ../models/anomaly_iforest.joblib.

The detection numbers printed here are for synthetic faults. They only show the pipeline works;
they say nothing about the real rig.
"""
import argparse

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from common import DUMMY_MODEL, MODELS, features, synthetic

# Same exclusions as pm/dataset.feature_columns: absolute temperatures and humidity
# depend on the weather, not the motor; temp_rise_c is kept.
SKIP = {"fs_hz", "rot_from_rpm", "ds18b20_temp_c", "dht22_temp_c", "dht22_humidity"}


def windows(rng, label, sev, n_sessions, msgs, start=0):
    rows = []
    for k in range(n_sessions):
        sid = f"dummy_{label}_s{sev}_r{start + k}"
        for rec in synthetic.session(rng, sid, label, sev, msgs):
            f = features.extract(rec)
            f.update(session=sid, label=label, severity=sev)
            rows.append(f)
    return pd.DataFrame(rows)


def fit(X):
    return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                         IsolationForest(n_estimators=300, contamination="auto", random_state=0)).fit(X)


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--sessions", type=int, default=8, help="healthy dummy sessions")
    p.add_argument("--msgs", type=int, default=90, help="windows per session (2 s each)")
    p.add_argument("--seed", type=int, default=11)
    a = p.parse_args(argv)
    rng = np.random.default_rng(a.seed)

    healthy = windows(rng, "healthy", 0, a.sessions, a.msgs)
    cols = [c for c in healthy.columns if c not in SKIP | {"session", "label", "severity"}]

    # Threshold from sessions the model never saw, so the false-alarm rate is honest.
    held_out = []
    for s in healthy["session"].unique():
        m = fit(healthy.loc[healthy["session"] != s, cols])
        held_out.append(-m.score_samples(healthy.loc[healthy["session"] == s, cols]))
    held_out = np.concatenate(held_out)
    threshold = float(np.percentile(held_out, 99))
    model = fit(healthy[cols])

    print(f"Trained on {len(healthy)} healthy dummy windows, {len(cols)} features")
    print(f"Threshold (99th percentile of unseen healthy windows): {threshold:.4f}")
    faults = pd.concat([windows(rng, f, s, 1, 30) for f in synthetic.FAULTS for s in (1, 2, 3)])
    fault_scores = -model.score_samples(faults[cols])
    print("Detection rate on dummy faults (pipeline check only, not a result):")
    for (lab, sev), idx in faults.reset_index(drop=True).groupby(["label", "severity"]).groups.items():
        print(f"  {lab:16s} severity {sev}: {(fault_scores[np.asarray(idx)] > threshold).mean():.0%}")

    MODELS.mkdir(exist_ok=True)
    joblib.dump({"model": model, "features": cols, "threshold": threshold, "source": "synthetic"},
                DUMMY_MODEL)
    print(f"Saved {DUMMY_MODEL}")


if __name__ == "__main__":
    main()
