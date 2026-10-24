"""MQTT -> labelled session file + InfluxDB (+ optional live prediction).

Replaces mqtt_to_influx.py. Every message from the ESP32 is:
  1. saved, raw vibration burst included, to data/raw/<session>.jsonl with its label,
  2. turned into features (features.py) and written to InfluxDB, tagged with the label,
  3. optionally scored by a trained model, with the prediction written to InfluxDB too.

Examples
  Record 5 minutes of healthy baseline:
    python recorder.py --session s01_healthy --label healthy --minutes 5
  Record blade imbalance, severity 2 of 3:
    python recorder.py --session s07_imb2 --label imbalance --severity 2 --minutes 5
  Live monitoring with the trained model (no recording):
    python recorder.py --live --model ../models/fault_rf.joblib --anomaly ../models/anomaly_iforest.joblib
"""
import argparse
import json
import math
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import paho.mqtt.client as mqtt
from dotenv import load_dotenv

from features import extract

LABELS = ["healthy", "imbalance", "loose_blade", "foot_wedge", "one_sided_mount", "air_blockage"]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--session", help="unique id for this run, e.g. s07_imb2 (one file per session)")
    p.add_argument("--label", choices=LABELS)
    p.add_argument("--severity", type=int, default=0, help="0 for healthy, 1-3 for faults")
    p.add_argument("--notes", default="", help="free text saved with every record (what you did to the rig)")
    p.add_argument("--minutes", type=float, default=0, help="stop after this long (0 = until Ctrl+C)")
    p.add_argument("--live", action="store_true", help="do not save a session file, just stream to InfluxDB")
    p.add_argument("--no-influx", action="store_true", help="skip InfluxDB (record to file only)")
    p.add_argument("--model", help="fault classifier .joblib from train.py, for live prediction")
    p.add_argument("--anomaly", help="anomaly detector .joblib from train.py, for live scoring")
    p.add_argument("--out", default=str(Path(__file__).resolve().parent.parent / "data" / "raw"))
    a = p.parse_args()
    if not a.live and not (a.session and a.label):
        p.error("--session and --label are required when recording (or use --live)")
    if a.label == "healthy":
        a.severity = 0
    return a


class Influx:
    def __init__(self):
        from influxdb_client import InfluxDBClient
        from influxdb_client.client.write_api import SYNCHRONOUS
        self.bucket = os.getenv("INFLUX_BUCKET")
        self.org = os.getenv("INFLUX_ORG")
        self.client = InfluxDBClient(url=os.getenv("INFLUX_URL"), token=os.getenv("INFLUX_TOKEN"), org=self.org)
        self.write_api = self.client.write_api(write_options=SYNCHRONOUS)

    def write(self, msg, feats, tags, pred):
        from influxdb_client import Point
        p = Point("motor_readings")
        for k, v in tags.items():
            if v not in (None, ""):
                p = p.tag(k, str(v))
        values = dict(feats or {})
        for k in ("voltage", "current", "power", "energy", "pf", "mains_hz", "rpm", "sound_dbfs",
                  "ds18b20_temp_c", "dht22_temp_c", "dht22_humidity"):
            values.setdefault(k, msg.get(k))
        for k, v in values.items():
            try:
                f = float(v)
            except (TypeError, ValueError):
                continue
            if not math.isnan(f) and not math.isinf(f):
                p = p.field(k, f)
        if pred:
            p = p.tag("predicted", pred["label"])
            for k in ("confidence", "anomaly_score", "is_anomaly"):
                if k in pred:
                    p = p.field(k, float(pred[k]))
        self.write_api.write(bucket=self.bucket, org=self.org, record=p)


class Predictor:
    def __init__(self, model_path, anomaly_path):
        import joblib
        self.clf = joblib.load(model_path) if model_path else None
        self.anom = joblib.load(anomaly_path) if anomaly_path else None

    def __call__(self, feats):
        import pandas as pd
        out = {}
        if self.clf:
            x = pd.DataFrame([feats])[self.clf["features"]]
            proba = self.clf["model"].predict_proba(x)[0]
            i = int(proba.argmax())
            out["label"] = self.clf["classes"][i]
            out["confidence"] = float(proba[i])
        if self.anom:
            x = pd.DataFrame([feats])[self.anom["features"]]
            score = float(-self.anom["model"].score_samples(x)[0])   # higher = more abnormal
            out["anomaly_score"] = score
            out["is_anomaly"] = float(score > self.anom["threshold"])
            out.setdefault("label", "anomaly" if score > self.anom["threshold"] else "normal")
        return out


def main():
    args = parse_args()
    load_dotenv()
    broker, port, topic = os.getenv("MQTT_BROKER", "localhost"), int(os.getenv("MQTT_PORT", 1883)), os.getenv("MQTT_TOPIC", "motor/sensors")

    influx = None if args.no_influx else Influx()
    predictor = Predictor(args.model, args.anomaly) if (args.model or args.anomaly) else None

    out_file = None
    if not args.live:
        out_dir = Path(args.out)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{args.session}.jsonl"
        if out_path.exists():
            print(f"Appending to existing session file {out_path}")
        out_file = out_path.open("a")
        print(f"Recording session '{args.session}' label={args.label} severity={args.severity} -> {out_path}")

    tags = {"session": args.session, "label": args.label,
            "severity": args.severity if args.label else None}
    stats = {"n": 0, "no_vib": 0, "start": time.time()}

    def on_connect(client, userdata, flags, reason_code, properties=None):
        if reason_code == 0:
            client.subscribe(topic)
            print(f"Connected to {broker}:{port}, subscribed to {topic}")
        else:
            print(f"MQTT connect failed: {reason_code}")

    def on_message(client, userdata, m):
        try:
            msg = json.loads(m.payload.decode())
        except Exception as e:
            print(f"Bad message ({len(m.payload)} bytes): {e}")
            return
        stats["n"] += 1
        msg["received_at"] = datetime.now(timezone.utc).isoformat()
        if out_file:
            msg.update({"session": args.session, "label": args.label, "severity": args.severity, "notes": args.notes})
            out_file.write(json.dumps(msg, separators=(",", ":")) + "\n")
            out_file.flush()

        feats = extract(msg)
        if feats is None:
            stats["no_vib"] += 1
            if stats["no_vib"] in (1, 10, 100):
                print("Message has no vibration burst. Is the ESP32 running firmware v2?")
        pred = predictor(feats) if (predictor and feats) else None
        if influx:
            try:
                influx.write(msg, feats, {**tags, "device_id": msg.get("device_id")}, pred)
            except Exception as e:
                print(f"InfluxDB write failed: {e}")

        line = f"#{stats['n']:4d}"
        if feats:
            line += (f"  vib {feats['mag_rms_g']*1000:6.1f} mg  fs {feats['fs_hz']:6.1f} Hz"
                     f"  1x {feats['rot_hz']:5.1f} Hz{'' if feats['rot_from_rpm'] else '(est)'}")
        fmt = lambda v, d: "--" if v is None else f"{v:.{d}f}"  # noqa: E731
        line += (f"  I {fmt(msg.get('current'), 3)} A  {fmt(msg.get('sound_dbfs'), 1)} dBFS"
                 f"  {fmt(msg.get('rpm'), 0)} rpm")
        if pred:
            line += f"  -> {pred.get('label')}"
            if "confidence" in pred:
                line += f" ({pred['confidence']:.2f})"
            if "anomaly_score" in pred:
                line += f" anomaly {pred['anomaly_score']:.3f}"
        print(line)

    try:
        client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
    except AttributeError:   # paho-mqtt 1.x
        client = mqtt.Client()
        _oc = on_connect
        on_connect = lambda c, u, f, rc: _oc(c, u, f, rc)   # noqa: E731
    client.on_connect = on_connect
    client.on_message = on_message
    client.connect(broker, port, 60)
    client.loop_start()
    try:
        while True:
            time.sleep(0.5)
            if args.minutes and time.time() - stats["start"] >= args.minutes * 60:
                print(f"Reached {args.minutes} min, stopping.")
                break
    except KeyboardInterrupt:
        print("Stopping.")
    finally:
        client.loop_stop()
        client.disconnect()
        if out_file:
            out_file.close()
        print(f"{stats['n']} messages received, {stats['no_vib']} without vibration data.")


if __name__ == "__main__":
    main()
