"""Live motor dashboard: reads sensor messages, scores them, streams them to the browser.

    python server.py                                   # demo motor: real recorded data + injectable faults
    python server.py --source mqtt                     # real ESP32 via the MQTT broker in ../.env
    python server.py --source replay --file ../data/raw/s01_sp1.jsonl     # play back one recording
    python server.py --source replay --file ../data/raw --interval 0.5     # play back a whole folder

Data path (mqtt):
    ESP32-S3 (firmware v3) --Wi-Fi/MQTT, topic motor/sensors--> Mosquitto on this PC
      --> this server: pm/features.py extract + temporal (same as training)
      --> anomaly model (+ change detector, + optional fault classifier)
      --> WebSocket --> browser on the PC or a phone
    pm/recorder.py can run at the same time on the same broker to save sessions and fill InfluxDB.

Model: --model auto picks the best model in ../models (see model_loader.find_anomaly_model).
Open the page from a phone on the same Wi-Fi at http://<this PC's IPv4>:8000.
"""
import argparse
import asyncio
import csv
import json
import os
import threading
import time
from collections import deque
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path

import uvicorn
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

try:
    from dotenv import load_dotenv
except ImportError:   # optional; settings can also come from flags
    load_dotenv = None

from common import DASHBOARD, DATA, RAW, ROOT
from model_loader import find_anomaly_model
from pipeline import LivePipeline
from scoring import Scorer, spectrum, to_point
from simulator import LABELS, MotorSimulator

LOG_FIELDS = ["t", "seq", "device", "fw", "rpm", "voltage", "current", "power", "pf", "sound_dbfs",
              "motor_temp_c", "ambient_c", "temp_rise_c", "humidity", "vib_rms_mg", "vel_rms_mms", "vib_1x_mg",
              "kurtosis", "rot_hz", "anomaly_score", "threshold", "over_threshold", "change_score",
              "change_threshold", "change_over", "model_alarm", "change_alarm", "alarm", "predicted_fault",
              "fault_confidence", "sim_label", "sim_severity"]


class Hub:
    def __init__(self, args):
        self.args = args
        self.scorer = Scorer(args.model, args.alarm_k, args.alarm_n, args.classifier, args.threshold,
                             change=not args.no_change, change_threshold=args.change_threshold, learn=args.learn)
        self.pipeline = LivePipeline()
        self.sim = MotorSimulator(seed=args.seed) if args.source == "sim" else None
        self.history = deque(maxlen=args.history)
        self.events = deque(maxlen=200)
        self.last_spectrum = None
        self.clients = set()
        self.was_alarm = False
        self.received = 0
        self.skipped = 0
        self.last_rx = None
        self.mqtt_state = "not used" if args.source != "mqtt" else "connecting"
        DATA.mkdir(exist_ok=True)
        self.log_path = DATA / f"live_{datetime.now():%Y%m%d_%H%M%S}.csv"
        self.log = self.log_path.open("w", newline="")
        self.writer = csv.DictWriter(self.log, LOG_FIELDS, extrasaction="ignore")
        self.writer.writeheader()
        self.raw = None
        if args.save_raw:
            RAW.mkdir(parents=True, exist_ok=True)
            p = RAW / f"live_{datetime.now():%Y%m%d_%H%M%S}.jsonl"
            self.raw = p.open("a")
            print(f"Saving raw messages to {p} (label them healthy only if you know they are)")

    def status(self):
        s = self.scorer
        return {"source": self.args.source, "replay_file": self.args.file,
                "model_name": s.name, "model_source": s.source, "model_path": s.path, "threshold": s.threshold,
                "classifier": s.clf.path if s.clf else None,
                "change_enabled": s.change is not None,
                "change_threshold": s.change.threshold if s.change else None,
                "change_state": s.change.state if s.change else None,
                "alarm_rule": f"{s.alarm_k} of the last {s.alarm_n} readings",
                "received": self.received, "skipped": self.skipped,
                "last_rx_age_s": None if self.last_rx is None else round(time.time() - self.last_rx, 1),
                "mqtt": self.mqtt_state, "broker": f"{self.args.broker}:{self.args.port_mqtt}" if self.args.source == "mqtt" else None,
                "log": str(self.log_path),
                "sim": {"label": self.sim.label, "severity": self.sim.severity, "labels": LABELS,
                        "kind": self.sim.kind} if self.sim else None}

    async def handle(self, msg):
        self.last_rx = time.time()
        if self.raw:
            self.raw.write(json.dumps({**msg, "received_at": datetime.now(timezone.utc).isoformat()},
                                      separators=(",", ":")) + "\n")
            self.raw.flush()
        try:
            feats, hist = self.pipeline(msg)
        except Exception as e:   # noqa: BLE001  (a corrupt message must not stop the dashboard)
            self.skipped += 1
            print(f"Skipped message seq={msg.get('seq')}: feature extraction failed ({type(e).__name__}: {e})")
            return
        if feats is None:            # no vibration burst (MPU read dropped on the ESP32, or v1 firmware)
            self.skipped += 1
            if self.skipped in (1, 10, 100):
                print(f"Message without a vibration burst skipped ({self.skipped} so far)")
            return
        self.received += 1
        result = self.scorer(feats, msg, hist)
        ts = datetime.now(timezone.utc).isoformat(timespec="milliseconds")
        point = to_point(msg, feats, result, ts)
        if self.sim:
            point["sim_label"], point["sim_severity"] = self.sim.label, self.sim.severity
        self.history.append(point)
        self.last_spectrum = spectrum(msg)
        self.writer.writerow(point)
        self.log.flush()
        event = None
        if point["alarm"] and not self.was_alarm:
            devs = point["change_deviations"] if point["change_alarm"] else point["top_deviations"]
            why = ", ".join(d["label"] for d in devs[:2])
            which = "change since baseline" if point["change_alarm"] and not point["model_alarm"] else \
                "model" if point["model_alarm"] and not point["change_alarm"] else "model and change since baseline"
            event = {"t": ts, "kind": "alarm", "text": f"Anomaly alarm ({which}). Most unusual: {why}"}
        elif not point["alarm"] and self.was_alarm:
            event = {"t": ts, "kind": "clear", "text": "Back to normal"}
        self.was_alarm = point["alarm"]
        if event:
            self.events.append(event)
            print(f"{ts} {event['text']}")
        if self.args.verbose:
            print(f"#{point['seq']} score {point['anomaly_score']} change {point['change_score']} alarm {point['alarm']}")
        await self.broadcast({"type": "point", "point": point, "spectrum": self.last_spectrum,
                              "event": event, "status": self.status()})

    async def broadcast(self, payload):
        text = json.dumps(payload)
        for ws in list(self.clients):
            try:
                await ws.send_text(text)
            except Exception:   # noqa: BLE001
                self.clients.discard(ws)

    def add_event(self, kind, text):
        e = {"t": datetime.now(timezone.utc).isoformat(timespec="seconds"), "kind": kind, "text": text}
        self.events.append(e)
        return e

    async def run_sim(self):
        print(f"Demo motor: {self.sim.kind} messages" + (f" from {len(self.sim.files)} recording(s)" if self.sim.files else ""))
        while True:
            await self.handle(self.sim.next_message())
            await asyncio.sleep(self.args.interval)

    def replay_files(self):
        p = Path(self.args.file)
        return sorted(p.glob("*.jsonl")) if p.is_dir() else [p]

    async def run_replay(self):
        """Play recorded sessions (one JSON message per line) as if they were arriving live."""
        while True:
            for f in self.replay_files():
                print(f"Replaying {f}")
                with open(f) as fh:
                    for line in fh:
                        if line.strip():
                            await self.handle(json.loads(line))
                            await asyncio.sleep(self.args.interval)
            if not self.args.loop:
                print("Replay finished (use --loop to repeat). The page stays up; Ctrl+C to stop.")
                return

    async def run_mqtt(self):
        import paho.mqtt.client as mqtt
        loop, queue = asyncio.get_running_loop(), asyncio.Queue(maxsize=100)
        broker, port, topic = self.args.broker, self.args.port_mqtt, self.args.topic

        def on_connect(client, userdata, flags, reason_code, properties=None):
            if getattr(reason_code, "is_failure", reason_code != 0):
                self.mqtt_state = f"refused ({reason_code})"
                print(f"MQTT connect refused: {reason_code}")
                return
            client.subscribe(topic)
            self.mqtt_state = "connected"
            print(f"MQTT connected to {broker}:{port}, subscribed to {topic}. Waiting for the ESP32...")

        def on_disconnect(client, *args):
            self.mqtt_state = "disconnected, retrying"
            print(f"MQTT disconnected from {broker}:{port}, retrying")

        def on_message(client, userdata, m):
            try:
                msg = json.loads(m.payload)
            except ValueError:
                print(f"Skipped a message that is not JSON ({len(m.payload)} bytes; cut off?)")
                return
            if not m.payload or not isinstance(msg, dict):
                return
            loop.call_soon_threadsafe(lambda: queue.full() or queue.put_nowait(msg))

        try:
            client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id=f"dashboard-{os.getpid()}")
        except AttributeError:   # paho-mqtt 1.x
            client = mqtt.Client(client_id=f"dashboard-{os.getpid()}")
        if self.args.mqtt_user:
            client.username_pw_set(self.args.mqtt_user, self.args.mqtt_password)
        client.on_connect, client.on_message, client.on_disconnect = on_connect, on_message, on_disconnect
        client.reconnect_delay_set(1, 10)
        print(f"Connecting to MQTT broker {broker}:{port} ...")
        client.connect_async(broker, port)
        threading.Thread(target=client.loop_forever, kwargs={"retry_first_connection": True},
                         daemon=True).start()
        while True:
            await self.handle(await queue.get())


class FaultRequest(BaseModel):
    label: str
    severity: float = 2.0


def build_app(hub):
    @asynccontextmanager
    async def lifespan(app):
        runner = {"sim": hub.run_sim, "mqtt": hub.run_mqtt, "replay": hub.run_replay}[hub.args.source]
        task = asyncio.create_task(runner())
        yield
        task.cancel()

    app = FastAPI(title="Motor live dashboard", lifespan=lifespan)

    @app.get("/")
    def index():
        return FileResponse(DASHBOARD / "static" / "index.html")

    @app.get("/api/status")
    def status():
        return hub.status()

    @app.get("/api/history")
    def history():
        return {"points": list(hub.history), "spectrum": hub.last_spectrum,
                "events": list(hub.events), "status": hub.status()}

    @app.post("/api/baseline/reset")
    async def reset_baseline():
        hub.scorer.reset()
        n = hub.scorer.change.learn_n if hub.scorer.change else 0
        event = hub.add_event("info", f"Re-learning the baseline from the next {n} readings. Keep the motor healthy meanwhile.")
        await hub.broadcast({"type": "status", "status": hub.status(), "event": event})
        return hub.status()

    @app.post("/api/sim/fault")
    async def set_fault(req: FaultRequest):
        if not hub.sim:
            raise HTTPException(400, "Fault injection only works with the demo motor (--source sim).")
        try:
            hub.sim.set_fault(req.label, req.severity)
        except ValueError as e:
            raise HTTPException(400, str(e))
        text = ("Demo motor set to healthy" if hub.sim.label == "healthy"
                else f"Demo fault injected: {hub.sim.label.replace('_', ' ')}, severity {hub.sim.severity:g}")
        event = hub.add_event("sim", text)
        await hub.broadcast({"type": "status", "status": hub.status(), "event": event})
        return hub.status()

    @app.websocket("/ws")
    async def ws(sock: WebSocket):
        await sock.accept()
        hub.clients.add(sock)
        try:
            while True:
                await sock.receive_text()
        except WebSocketDisconnect:
            hub.clients.discard(sock)

    app.mount("/static", StaticFiles(directory=DASHBOARD / "static"), name="static")
    return app


def pick_model(arg):
    if arg and arg != "auto":
        return arg
    found = find_anomaly_model()
    if found:
        return str(found)
    import train_baseline
    if any(RAW.glob("*.jsonl")):
        print("No model yet: training the baseline on the sessions in data/raw (about 30 s)...")
        try:
            return str(train_baseline.main([]))
        except SystemExit as e:
            print(f"Could not train on real data ({e}); using synthetic data instead.")
    print("No model and no recordings: training a synthetic demo model (about 10 s)...")
    return str(train_baseline.main(["--synthetic"]))


def parse_args(argv=None):
    if load_dotenv:
        load_dotenv(ROOT / ".env")
    env = os.getenv
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--source", choices=["sim", "mqtt", "replay"], default=env("DASHBOARD_SOURCE", "sim"))
    p.add_argument("--file", default=env("DASHBOARD_REPLAY") or None, help="recorded .jsonl (or a folder of them) for --source replay")
    p.add_argument("--loop", action="store_true", help="replay forever")
    p.add_argument("--model", default=env("DASHBOARD_MODEL", "auto"),
                   help="anomaly model (.joblib or plugin .py); 'auto' = best one in ../models")
    p.add_argument("--classifier", default=env("DASHBOARD_CLASSIFIER") or None,
                   help="optional fault classifier (.joblib or plugin .py), e.g. ../models/fault_rf.joblib")
    p.add_argument("--threshold", type=float, default=float(env("DASHBOARD_THRESHOLD")) if env("DASHBOARD_THRESHOLD") else None,
                   help="override the model's alarm threshold")
    p.add_argument("--no-change", action="store_true", help="turn off the change-since-baseline detector")
    p.add_argument("--change-threshold", type=float, default=float(env("DASHBOARD_CHANGE_THRESHOLD", 6.0)))
    p.add_argument("--learn", type=int, default=int(env("DASHBOARD_LEARN", 30)),
                   help="readings used to learn the change baseline (30 = 1 minute)")
    p.add_argument("--interval", type=float, default=2.0, help="seconds between demo or replayed readings")
    p.add_argument("--seed", type=int)
    p.add_argument("--alarm-k", type=int, default=3)
    p.add_argument("--alarm-n", type=int, default=5)
    p.add_argument("--history", type=int, default=1800, help="points kept in memory (1800 = 1 h at 2 s)")
    p.add_argument("--save-raw", action="store_true", help="also save every received message to data/raw/live_*.jsonl")
    p.add_argument("--broker", default=env("MQTT_BROKER", "localhost"))
    p.add_argument("--port-mqtt", type=int, default=int(env("MQTT_PORT", 1883)))
    p.add_argument("--topic", default=env("MQTT_TOPIC", "motor/sensors"))
    p.add_argument("--mqtt-user", default=env("MQTT_USER") or None)
    p.add_argument("--mqtt-password", default=env("MQTT_PASSWORD") or None)
    p.add_argument("--host", default=env("DASHBOARD_HOST", "0.0.0.0"))
    p.add_argument("--port", type=int, default=int(env("DASHBOARD_PORT", 8000)))
    p.add_argument("--verbose", action="store_true", help="print every scored reading")
    a = p.parse_args(argv)
    if a.source == "replay" and not (a.file and os.path.exists(a.file)):
        raise SystemExit("--source replay needs --file pointing to a recorded .jsonl session or a folder of them")
    a.model = pick_model(a.model)
    if not os.path.exists(a.model):
        raise SystemExit(f"No model at {a.model}")
    return a


def main(argv=None):
    a = parse_args(argv)
    hub = Hub(a)
    print(f"Source: {a.source}   model: {a.model} ({hub.scorer.source})"
          + (f"   classifier: {a.classifier}" if a.classifier else ""))
    print(f"Dashboard: http://localhost:{a.port}  (phone on the same Wi-Fi: http://<this PC's IPv4>:{a.port})")
    print(f"Scores are logged to {hub.log_path}")
    uvicorn.run(build_app(hub), host=a.host, port=a.port, log_level="warning")


if __name__ == "__main__":
    main()
