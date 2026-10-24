"""Train and evaluate the fault classifier and the healthy-only anomaly detector.

    python train.py --data ../data/raw

Evaluation is leave-one-session-out: every window of a session is tested by a model that
never saw any window of that session. A random row split is also printed, only to show how
much it overstates accuracy (adjacent 2 s windows of one run are near-duplicates).

Outputs
  ../models/fault_rf.joblib, fault_xgb.joblib (if xgboost is installed), anomaly_iforest.joblib
  ../reports/results.json, results.md, confusion_<model>.csv, feature_importance.csv
"""
import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (average_precision_score, classification_report, confusion_matrix,
                             f1_score, roc_auc_score)
from sklearn.model_selection import LeaveOneGroupOut, cross_val_predict, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler

from dataset import feature_columns, load_sessions
from features import feature_groups

ROOT = Path(__file__).resolve().parent.parent


def make_rf():
    return make_pipeline(SimpleImputer(strategy="median"),
                         RandomForestClassifier(n_estimators=300, min_samples_leaf=2,
                                                class_weight="balanced", n_jobs=-1, random_state=0))


def make_xgb(n_classes):
    try:
        from xgboost import XGBClassifier
    except ImportError:
        return None
    return make_pipeline(SimpleImputer(strategy="median"),
                         XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.08, subsample=0.8,
                                       colsample_bytree=0.8, objective="multi:softprob",
                                       num_class=n_classes, n_jobs=-1, random_state=0))


def evaluate_classifier(name, model, X, y_enc, groups, classes, df, reports):
    logo = LeaveOneGroupOut()
    pred = cross_val_predict(model, X, y_enc, groups=groups, cv=logo, n_jobs=1)
    f1 = f1_score(y_enc, pred, average="macro")
    acc = float((pred == y_enc).mean())

    Xtr, Xte, ytr, yte = train_test_split(X, y_enc, test_size=0.25, random_state=0, stratify=y_enc)
    leaky = f1_score(yte, model.fit(Xtr, ytr).predict(Xte), average="macro")

    cm = pd.DataFrame(confusion_matrix(y_enc, pred, labels=range(len(classes))),
                      index=[f"true_{c}" for c in classes], columns=[f"pred_{c}" for c in classes])
    cm.to_csv(reports / f"confusion_{name}.csv")

    sev_recall = {}
    for (lab, sev), idx in df.groupby(["label", "severity"]).groups.items():
        idx = np.asarray(list(idx))
        sev_recall[f"{lab}/{sev}"] = float((pred[idx] == y_enc[idx]).mean())

    report = classification_report(y_enc, pred, target_names=classes, output_dict=True, zero_division=0)
    print(f"\n[{name}] leave-one-session-out macro-F1 {f1:.3f}  accuracy {acc:.3f}"
          f"   (random row split would claim macro-F1 {leaky:.3f})")
    print(cm.to_string())
    return {"macro_f1_session_cv": f1, "accuracy_session_cv": acc, "macro_f1_random_split_LEAKY": leaky,
            "per_class": {c: {"recall": report[c]["recall"], "precision": report[c]["precision"],
                              "f1": report[c]["f1-score"]} for c in classes},
            "recall_by_label_severity": sev_recall}


def ablation(X_all, y_enc, groups, cols):
    out = {}
    for gname, gcols in feature_groups(cols).items():
        gcols = [c for c in gcols if c in X_all.columns]
        pred = cross_val_predict(make_rf(), X_all[gcols], y_enc, groups=groups, cv=LeaveOneGroupOut())
        out[gname] = {"n_features": len(gcols), "macro_f1_session_cv": f1_score(y_enc, pred, average="macro")}
        print(f"  ablation {gname:24s} {len(gcols):3d} features  macro-F1 {out[gname]['macro_f1_session_cv']:.3f}")
    return out


def anomaly(df, X, models, reports):
    healthy = df["label"] == "healthy"
    h_sessions = df.loc[healthy, "session"].unique()
    if len(h_sessions) < 2:
        print("\nAnomaly detector skipped: needs at least 2 healthy sessions.")
        return None

    def fit(Xh):
        return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                             IsolationForest(n_estimators=300, contamination="auto", random_state=0)).fit(Xh)

    scores = pd.Series(np.nan, index=df.index)
    for s in h_sessions:                      # each healthy session scored by a model that never saw it
        tr = healthy & (df["session"] != s)
        m = fit(X[tr])
        te = df["session"] == s
        scores[te] = -m.score_samples(X[te])
    final = fit(X[healthy])
    train_scores = -final.score_samples(X[healthy])
    threshold = float(np.percentile(train_scores, 99))
    scores[~healthy] = -final.score_samples(X[~healthy])

    y = (~healthy).astype(int)
    res = {"roc_auc": roc_auc_score(y, scores), "pr_auc": average_precision_score(y, scores),
           "threshold_p99_healthy": threshold,
           "false_alarm_rate_heldout_healthy": float((scores[healthy] > threshold).mean()),
           "detection_rate_by_label_severity": {}}
    for (lab, sev), idx in df[~healthy].groupby(["label", "severity"]).groups.items():
        res["detection_rate_by_label_severity"][f"{lab}/{sev}"] = float((scores[idx] > threshold).mean())
    pd.DataFrame({"session": df["session"], "label": df["label"], "severity": df["severity"],
                  "anomaly_score": scores}).to_csv(reports / "anomaly_scores.csv", index=False)
    joblib.dump({"model": final, "features": list(X.columns), "threshold": threshold},
                models / "anomaly_iforest.joblib")
    print(f"\n[isolation forest, healthy-only] ROC-AUC {res['roc_auc']:.3f}  PR-AUC {res['pr_auc']:.3f}"
          f"  false alarms on unseen healthy {res['false_alarm_rate_heldout_healthy']:.1%}")
    for k, v in res["detection_rate_by_label_severity"].items():
        print(f"  detected {k:24s} {v:.1%}")
    return res


def write_markdown(results, path):
    lines = [f"# Results ({results['data_dir']})", "",
             f"{results['n_windows']} windows from {results['n_sessions']} sessions. "
             "All scores are leave-one-session-out.", ""]
    if results.get("synthetic"):
        lines += ["**SYNTHETIC DATA: these numbers only prove the pipeline runs. Do not report them.**", ""]
    lines += ["| Model | Macro-F1 | Accuracy | Random-split F1 (leaky) |", "|---|---|---|---|"]
    for name, r in results["classifiers"].items():
        lines.append(f"| {name} | {r['macro_f1_session_cv']:.3f} | {r['accuracy_session_cv']:.3f} "
                     f"| {r['macro_f1_random_split_LEAKY']:.3f} |")
    lines += ["", "## Sensor ablation (random forest)", "", "| Features | Count | Macro-F1 |", "|---|---|---|"]
    for g, r in results["ablation"].items():
        lines.append(f"| {g} | {r['n_features']} | {r['macro_f1_session_cv']:.3f} |")
    a = results.get("anomaly")
    if a:
        lines += ["", "## Anomaly detection (trained on healthy only)", "",
                  f"ROC-AUC {a['roc_auc']:.3f}, PR-AUC {a['pr_auc']:.3f}, "
                  f"false alarms on unseen healthy sessions {a['false_alarm_rate_heldout_healthy']:.1%}.", "",
                  "| Fault / severity | Detected |", "|---|---|"]
        lines += [f"| {k} | {v:.1%} |" for k, v in a["detection_rate_by_label_severity"].items()]
    path.write_text("\n".join(lines) + "\n")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default=str(ROOT / "data" / "raw"))
    p.add_argument("--models", default=str(ROOT / "models"))
    p.add_argument("--reports", default=str(ROOT / "reports"))
    a = p.parse_args()
    models, reports = Path(a.models), Path(a.reports)
    models.mkdir(parents=True, exist_ok=True)
    reports.mkdir(parents=True, exist_ok=True)

    df = load_sessions(a.data)
    cols = feature_columns(df)
    X = df[cols].astype(float)
    groups = df["session"].values
    print(f"{len(df)} windows, {df['session'].nunique()} sessions, {len(cols)} features")
    print(df.groupby(["label", "severity"])["session"].nunique().rename("sessions").to_string())

    results = {"data_dir": a.data, "n_windows": len(df), "n_sessions": int(df["session"].nunique()),
               "synthetic": bool((df.get("session").astype(str).str.startswith("syn_")).all()),
               "classifiers": {}}

    le = LabelEncoder().fit(df["label"])
    y_enc = le.transform(df["label"])
    classes = list(le.classes_)
    if len(classes) >= 2 and df["session"].nunique() >= 3:
        cands = {"random_forest": make_rf(), "xgboost": make_xgb(len(classes))}
        for name, model in cands.items():
            if model is None:
                continue
            results["classifiers"][name] = evaluate_classifier(name, model, X, y_enc, groups, classes, df, reports)
            model.fit(X, y_enc)
            short = "rf" if name == "random_forest" else "xgb"
            joblib.dump({"model": model, "features": cols, "classes": classes}, models / f"fault_{short}.joblib")
        rf = joblib.load(models / "fault_rf.joblib")["model"][-1]
        imp = pd.Series(rf.feature_importances_, index=cols).sort_values(ascending=False)
        imp.to_csv(reports / "feature_importance.csv", header=["importance"])
        print("\nTop features:\n" + imp.head(12).to_string())
        print("\nSensor ablation:")
        results["ablation"] = ablation(X, y_enc, groups, cols)
    else:
        print("Classifier skipped: needs at least 2 labels and 3 sessions.")
        results["ablation"] = {}

    results["anomaly"] = anomaly(df, X, models, reports)
    (reports / "results.json").write_text(json.dumps(results, indent=2))
    write_markdown(results, reports / "results.md")
    print(f"\nSaved models to {models} and reports to {reports}")


if __name__ == "__main__":
    main()
