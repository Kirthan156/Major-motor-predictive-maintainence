"""Load recorded sessions (data/raw/*.jsonl) into one feature table.

    python dataset.py                 # writes ../data/dataset.csv (one row per 2 s message)
    python dataset.py --data ../data/synthetic --out ../data/synthetic_dataset.csv
"""
import argparse
import json
from collections import deque
from pathlib import Path

import pandas as pd

from features import NON_FEATURES, TEMPORAL_WINDOW, extract, temporal

META = ["session", "label", "severity", "received_at", "seq", "fw"]
CACHE_VERSION = "v3"


def session_rows(path):
    rows, hist = [], deque(maxlen=TEMPORAL_WINDOW)
    with path.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            feats = extract(rec)
            if feats is None:
                continue
            hist.append(feats)
            feats = {**feats, **temporal(list(hist))}
            feats.update({k: rec.get(k) for k in META})
            rows.append(feats)
    return rows


def load_sessions(raw_dir, cache=True):
    raw_dir = Path(raw_dir)
    files = sorted(raw_dir.glob("*.jsonl"))
    if not files:
        raise SystemExit(f"No .jsonl sessions in {raw_dir}. Record some with recorder.py first.")
    frames = []
    for f in files:
        cache_file = f.with_suffix(f".features_{CACHE_VERSION}.csv")
        if cache and cache_file.exists() and cache_file.stat().st_mtime >= f.stat().st_mtime:
            frames.append(pd.read_csv(cache_file))
            continue
        df = pd.DataFrame(session_rows(f))
        if cache and not df.empty:
            df.to_csv(cache_file, index=False)
        frames.append(df)
    data = pd.concat(frames, ignore_index=True)
    data["severity"] = data["severity"].fillna(0).astype(int)
    return data


def feature_columns(df):
    """Columns a model may learn from: everything except metadata, diagnostics and columns that
    identify the day or the session rather than the motor's condition."""
    skip = set(META) | NON_FEATURES | {"notes"}
    cols = [c for c in df.columns if c not in skip and pd.api.types.is_numeric_dtype(df[c])]
    return [c for c in cols if df[c].notna().any()]


if __name__ == "__main__":
    root = Path(__file__).resolve().parent.parent
    p = argparse.ArgumentParser()
    p.add_argument("--data", default=str(root / "data" / "raw"))
    p.add_argument("--out", default=str(root / "data" / "dataset.csv"))
    a = p.parse_args()
    df = load_sessions(a.data)
    df.to_csv(a.out, index=False)
    print(f"{len(df)} rows x {df.shape[1]} columns ({len(feature_columns(df))} usable as model features) -> {a.out}")
