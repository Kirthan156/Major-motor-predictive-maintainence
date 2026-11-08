"""Pretend to be the ESP32: publish a recorded session to the MQTT broker, one message every 2 s.

Use it to test the whole PC side (Mosquitto -> server.py / recorder.py / InfluxDB) without the rig:

    python publish_recording.py ../data/sample/s01_sp1.jsonl
    python publish_recording.py ../data/raw --interval 0.5 --loop

It strips the labels that recorder.py added, so the messages look exactly like the firmware's.
"""
import argparse
import json
import os
import time
from pathlib import Path

from common import ROOT

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None


def main():
    if load_dotenv:
        load_dotenv(ROOT / ".env")
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("path", help="a .jsonl session or a folder of them")
    p.add_argument("--broker", default=os.getenv("MQTT_BROKER", "localhost"))
    p.add_argument("--port", type=int, default=int(os.getenv("MQTT_PORT", 1883)))
    p.add_argument("--topic", default=os.getenv("MQTT_TOPIC", "motor/sensors"))
    p.add_argument("--interval", type=float, default=2.0)
    p.add_argument("--loop", action="store_true")
    p.add_argument("--limit", type=int, default=0, help="stop after this many messages (0 = all)")
    a = p.parse_args()

    import paho.mqtt.client as mqtt
    try:
        client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id=f"fake-esp32-{os.getpid()}")
    except AttributeError:
        client = mqtt.Client(client_id=f"fake-esp32-{os.getpid()}")
    client.connect(a.broker, a.port)
    client.loop_start()
    src = Path(a.path)
    files = sorted(src.glob("*.jsonl")) if src.is_dir() else [src]
    sent = 0
    try:
        while True:
            for f in files:
                print(f"Publishing {f} to {a.broker}:{a.port} topic {a.topic}")
                with open(f) as fh:
                    for line in fh:
                        if not line.strip():
                            continue
                        msg = json.loads(line)
                        for k in ("session", "label", "severity", "notes", "received_at"):
                            msg.pop(k, None)
                        info = client.publish(a.topic, json.dumps(msg, separators=(",", ":")), qos=0)
                        info.wait_for_publish(5)
                        sent += 1
                        print(f"  sent #{msg.get('seq')} ({len(line)} bytes)")
                        if a.limit and sent >= a.limit:
                            return
                        time.sleep(a.interval)
            if not a.loop:
                return
    finally:
        client.loop_stop()
        client.disconnect()
        print(f"{sent} messages published")


if __name__ == "__main__":
    main()
