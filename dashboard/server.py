"""Live motor dashboard: reads messages, scores them, streams them to the browser.

    python server.py                                   # dummy motor (default), http://localhost:8000
    python server.py --source mqtt                     # real ESP32 via the MQTT broker in ../.env
    python server.py --source replay --file ../data/raw/s01_healthy.jsonl   # play back a recording

Model: by default the real model from ../models/anomaly_iforest.joblib (made by pm/train.py) if it
exists, otherwise the dummy model in models/anomaly_dummy.joblib (trained on first start).
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

import uvicorn
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

try:
    from dotenv import load_dotenv
except ImportError:   # optional; settings can also come from flags
    load_dotenv = None

from common import DATA, DUMMY_MODEL, PLATFORM, ROOT, features
from scoring import Scorer, spectrum, to_point
from simulator import LABELS, MotorSimulator

LOG_FIELDS = ["t", "seq", "device", "rpm", "voltage", "current", "power", "pf", "sound_dbfs",
              "motor_temp_c", "ambient_c", "temp_rise_c", "humidity", "vib_rms_mg", "vib_1x_mg",
              "kurtosis", "anomaly_score", "threshold", "over_threshold", "alarm", "predicted_fault",
              "sim_label", "sim_severity"]


class Hub:
    def __init__(self, args):
        self.args = args
        self.scorer = Scorer(args.model, args.alarm_k, args.alarm_n, args.classifier)
        self.sim = MotorSimulator(seed=args.seed) if args.source == "sim" else None
        self.history = deque(maxlen=args.history)
        self.events = deque(maxlen=200)
        self.last_spectrum = None
        self.clients = set()
        self.was_alarm = False
        self.received = 0
        DATA.mkdir(exist_ok=True)
        self.log_path = DATA / f"live_{datetime.now():%Y%m%d_%H%M%S}.csv"
        self.log = self.log_path.open("w", newline="")
        self.writer = csv.DictWriter(self.log, LOG_FIELDS, extrasaction="ignore")
        self.writer.writeheader()

    def status(self):
        return {"source": self.args.source, "replay_file": self.args.file, "model_source": self.scorer.source,
                "model_path": self.scorer.path, "threshold": self.scorer.threshold,
                "alarm_rule": f"{self.scorer.alarm_k} of last {self.scorer.recent.maxlen} windows",
                "received": self.received, "log": str(self.log_path),
                "sim": {"label": self.sim.label, "severity": self.sim.severity, "labels": LABELS}
                if self.sim else None}

    async def handle(self, msg):
        feats = features.extract(msg)
        if feats is None:            # v1 firmware message without a vibration burst
            return
        self.received += 1
        result = self.scorer(feats)
        ts = datetime.now(timezone.utc).isoformat(timespec="seconds")
        point = to_point(msg, feats, result, ts)
        if self.sim:
            point["sim_label"], point["sim_severity"] = self.sim.label, self.sim.severity
        self.history.append(point)
        self.last_spectrum = spectrum(msg)
        self.writer.writerow(point)
        self.log.flush()
        event = None
        if point["alarm"] and not self.was_alarm:
            why = ", ".join(d["label"] for d in point["top_deviations"][:2])
            event = {"t": ts, "kind": "alarm", "text": f"Anomaly alarm. Most unusual: {why}"}
        elif not point["alarm"] and self.was_alarm:
            event = {"t": ts, "kind": "clear", "text": "Back to normal"}
        self.was_alarm = point["alarm"]
        if event:
            self.events.append(event)
        await self.broadcast({"type": "point", "point": point, "spectrum": self.last_spectrum,
                              "event": event})

    async def broadcast(self, payload):
        text = json.dumps(payload)
        for ws in list(self.clients):
            try:
                await ws.send_text(text)
            except Exception:
                self.clients.discard(ws)

    async def run_sim(self):
        while True:
            await self.handle(self.sim.next_message())
            await asyncio.sleep(self.args.interval)

    async def run_replay(self):
        """Play a recorded session (one JSON message per line) as if it were arriving live."""
        while True:
            with open(self.args.file) as fh:
                for line in fh:
                    if line.strip():
                        await self.handle(json.loads(line))
                        await asyncio.sleep(self.args.interval)
            if not self.args.loop:
                print(f"Replay of {self.args.file} finished")
                return

    async def run_mqtt(self):
        import paho.mqtt.client as mqtt
        loop, queue = asyncio.get_running_loop(), asyncio.Queue(maxsize=100)
        broker, port, topic = self.args.broker, self.args.port_mqtt, self.args.topic

        def on_connect(client, userdata, flags, reason_code, properties=None):
            if reason_code != 0:
                print(f"MQTT connect refused: {reason_code}")
                return
            client.subscribe(topic)
            print(f"MQTT connected to {broker}:{port}, subscribed to {topic}. Waiting for the ESP32...")

        def on_disconnect(client, *args):
            print(f"MQTT disconnected from {broker}:{port}, retrying")

        def on_message(client, userdata, m):
            try:
                msg = json.loads(m.payload)
            except ValueError:
                print("Skipped a message that is not JSON")
                return
            loop.call_soon_threadsafe(lambda: queue.full() or queue.put_nowait(msg))

        try:
            client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
        except AttributeError:   # paho-mqtt 1.x
            client = mqtt.Client()
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
        return FileResponse(PLATFORM / "static" / "index.html")

    @app.get("/api/status")
    def status():
        return hub.status()

    @app.get("/api/history")
    def history():
        return {"points": list(hub.history), "spectrum": hub.last_spectrum,
                "events": list(hub.events), "status": hub.status()}

    @app.post("/api/sim/fault")
    async def set_fault(req: FaultRequest):
        if not hub.sim:
            raise HTTPException(400, "Fault injection only works with the dummy motor (--source sim).")
        try:
            hub.sim.set_fault(req.label, req.severity)
        except ValueError as e:
            raise HTTPException(400, str(e))
        text = ("Dummy motor set to healthy" if hub.sim.label == "healthy"
                else f"Dummy fault injected: {hub.sim.label.replace('_', ' ')}, severity {hub.sim.severity:g}")
        event = {"t": datetime.now(timezone.utc).isoformat(timespec="seconds"), "kind": "sim", "text": text}
        hub.events.append(event)
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

    app.mount("/static", StaticFiles(directory=PLATFORM / "static"), name="static")
    return app


def pick_model(arg):
    if arg != "auto":
        return arg
    real = ROOT / "models" / "anomaly_iforest.joblib"
    if real.exists():
        return str(real)
    if not DUMMY_MODEL.exists():
        print("No model yet: training the dummy model first (about 10 s)...")
        import train_dummy
        train_dummy.main([])
    return str(DUMMY_MODEL)


def main():
    if load_dotenv:
        load_dotenv(ROOT / ".env")
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--source", choices=["sim", "mqtt", "replay"], default=os.getenv("DASHBOARD_SOURCE", "sim"))
    p.add_argument("--file", help="recorded session .jsonl for --source replay")
    p.add_argument("--loop", action="store_true", help="replay the file forever")
    p.add_argument("--model", default=os.getenv("DASHBOARD_MODEL", "auto"),
                   help="anomaly .joblib; 'auto' = ../models/anomaly_iforest.joblib if present, else the dummy model")
    p.add_argument("--classifier", default=os.getenv("DASHBOARD_CLASSIFIER") or None,
                   help="optional fault_rf.joblib from pm/train.py, to also show which fault it looks like")
    p.add_argument("--interval", type=float, default=2.0, help="seconds between dummy or replayed readings")
    p.add_argument("--seed", type=int)
    p.add_argument("--alarm-k", type=int, default=3)
    p.add_argument("--alarm-n", type=int, default=5)
    p.add_argument("--history", type=int, default=1800, help="points kept in memory (1800 = 1 h at 2 s)")
    p.add_argument("--broker", default=os.getenv("MQTT_BROKER", "localhost"))
    p.add_argument("--port-mqtt", type=int, default=int(os.getenv("MQTT_PORT", 1883)))
    p.add_argument("--topic", default=os.getenv("MQTT_TOPIC", "motor/sensors"))
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=int(os.getenv("DASHBOARD_PORT", 8000)))
    a = p.parse_args()
    if a.source == "replay" and not (a.file and os.path.exists(a.file)):
        raise SystemExit("--source replay needs --file pointing to a recorded .jsonl session")
    a.model = pick_model(a.model)
    if not os.path.exists(a.model):
        raise SystemExit(f"No model at {a.model}")
    print(f"Source: {a.source}   model: {a.model}")
    print(f"Dashboard: http://localhost:{a.port}  (phone on the same Wi-Fi: http://<this PC's IPv4>:{a.port})")
    uvicorn.run(build_app(Hub(a)), host=a.host, port=a.port, log_level="warning")


if __name__ == "__main__":
    main()
