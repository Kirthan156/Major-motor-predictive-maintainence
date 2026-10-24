"""Load recorded sessions (data/raw/*.jsonl) into one feature table."""
import json
from pathlib import Path

import pandas as pd

from features import extract

META = ["session", "label", "severity", "received_at", "seq"]


def load_sessions(raw_dir, cache=True):
    raw_dir = Path(raw_dir)
    files = sorted(raw_dir.glob("*.jsonl"))
    if not files:
        raise SystemExit(f"No .jsonl sessions in {raw_dir}. Record some with recorder.py first.")
    frames = []
    for f in files:
        cache_file = f.with_suffix(".features.csv")
        if cache and cache_file.exists() and cache_file.stat().st_mtime >= f.stat().st_mtime:
            frames.append(pd.read_csv(cache_file))
            continue
        rows = []
        with f.open() as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                feats = extract(rec)
                if feats is None:
                    continue
                feats.update({k: rec.get(k) for k in META})
                rows.append(feats)
        df = pd.DataFrame(rows)
        if cache and not df.empty:
            df.to_csv(cache_file, index=False)
        frames.append(df)
    data = pd.concat(frames, ignore_index=True)
    data["severity"] = data["severity"].fillna(0).astype(int)
    return data


def feature_columns(df):
    skip = set(META) | {"fs_hz", "rot_from_rpm", "ds18b20_temp_c", "dht22_temp_c", "dht22_humidity"}
    return [c for c in df.columns if c not in skip]
