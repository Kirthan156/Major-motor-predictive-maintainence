"""Offline check of the whole PC-side chain, no ESP32 and no broker needed.

    python selftest.py                                  # model 'auto', recordings from data/sample or data/raw
    python selftest.py --model ../models/anomaly_myautoencoder.py
    python selftest.py --classifier ../models/fault_rf.joblib

It checks, and prints PASS / FAIL for each:
  1. a recorded message turns into features (pm/features.py)
  2. the model loads and every feature it needs exists in the live feature row
  3. healthy recorded readings stay below the alarm (few or no alarms)
  4. a demo fault injected after the baseline is learned raises the alarm
Run it after you add or retrain a model, before a demo.
"""
import argparse
import sys

import numpy as np

from model_loader import find_anomaly_model, load_model
from pipeline import LivePipeline
from scoring import Scorer
from simulator import MotorSimulator, default_recordings


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="auto")
    p.add_argument("--classifier")
    p.add_argument("--fault", default="imbalance")
    p.add_argument("--severity", type=float, default=2.0)
    a = p.parse_args(argv)
    fails = 0

    def report(ok, name, detail):
        nonlocal fails
        fails += 0 if ok else 1
        print(f"{'PASS' if ok else 'FAIL'}  {name}: {detail}")

    files = default_recordings()
    report(bool(files), "recordings", f"{len(files)} file(s), first {files[0]}" if files else
           "no .jsonl in data/sample or data/raw (the demo will fall back to synthetic data)")
    model = str(find_anomaly_model()) if a.model == "auto" else a.model
    if model in (None, "None"):
        report(False, "model", "no model found; run python train_baseline.py first")
        return 1
    m = load_model(model)
    report(True, "model loads", f"{m.name} from {model}")

    sim = MotorSimulator(seed=0, files=files[:1] if files else None)
    pipe = LivePipeline()
    feats, hist = pipe(sim.next_message())
    report(feats is not None, "features", f"{len(feats)} columns from one message" if feats else "extract() returned None")
    cols = getattr(m, "features", None)
    if cols:
        missing = [c for c in cols if c not in feats]
        report(len(missing) <= 0.05 * len(cols), "model columns present",
               f"{len(cols) - len(missing)}/{len(cols)}" + (f"; missing e.g. {missing[:5]}" if missing else ""))

    scorer = Scorer(model, classifier_path=a.classifier, learn=30)
    pipe.reset()
    sim = MotorSimulator(seed=0, files=files[:1] if files else None)
    healthy = [scorer(*_step(pipe, sim)) for _ in range(60)]
    after = healthy[35:]
    alarms = sum(r["alarm"] for r in after)
    scores = [r["anomaly_score"] for r in after]
    report(alarms <= 2, "healthy stays normal", f"{alarms}/{len(after)} alarm readings; model score "
           f"{np.nanmin(scores):.3f}..{np.nanmax(scores):.3f} vs threshold {scorer.threshold:.3f}")
    if a.classifier:
        print(f"      classifier says: {healthy[-1].get('predicted_fault')} ({healthy[-1].get('fault_confidence', 0):.2f})")

    sim.set_fault(a.fault, a.severity)
    faulty = [scorer(*_step(pipe, sim)) for _ in range(15)]
    hit = sum(r["alarm"] for r in faulty)
    model_hit = sum(r["over_threshold"] for r in faulty)
    report(hit > 0, f"demo fault '{a.fault}' severity {a.severity:g} raises the alarm",
           f"alarm on {hit}/15 readings (model score over threshold on {model_hit}/15, "
           f"change score up to {max((r.get('change_score') or 0) for r in faulty):.1f})")
    if a.classifier:
        print(f"      classifier says: {faulty[-1].get('predicted_fault')} ({faulty[-1].get('fault_confidence', 0):.2f})")
    print("\nAll checks passed." if not fails else f"\n{fails} check(s) failed.")
    return 1 if fails else 0


def _step(pipe, sim):
    msg = sim.next_message()
    feats, hist = pipe(msg)
    return feats, msg, hist


if __name__ == "__main__":
    sys.exit(main())
