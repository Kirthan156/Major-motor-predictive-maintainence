"""Check that real ESP32 readings reach this PC and look sane, before trusting the dashboard.

    python check_connection.py              # listens for 30 s on the broker/topic from ../.env
    python check_connection.py --seconds 60 --broker 192.168.1.50

Prints PASS / WARN / FAIL for each check, and what to fix. Exit code 0 if nothing failed.
"""
import argparse
import json
import math
import os
import threading
import time

from common import ROOT, features

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None

REQUIRED = ["device_id", "seq", "rpm", "voltage", "current", "power", "sound_dbfs",
            "ds18b20_temp_c", "dht22_temp_c", "vib"]


def num(v):
    try:
        f = float(v)
        return None if math.isnan(f) else f
    except (TypeError, ValueError):
        return None


def check(msgs, seconds, bad_json):
    results = []

    def add(level, name, detail):
        results.append((level, name, detail))

    if not msgs:
        add("FAIL", "messages received", f"none in {seconds} s. Is the ESP32 powered and on the same Wi-Fi? "
            "Is MQTT_BROKER in secrets.h this PC's current IPv4 (ipconfig)? Is Mosquitto running with "
            "'listener 1883 0.0.0.0' and 'allow_anonymous true'? Is port 1883 allowed in Windows Firewall?")
        return results
    rate = len(msgs) / seconds
    add("PASS" if rate > 0.3 else "WARN", "messages received",
        f"{len(msgs)} in {seconds} s ({1 / rate:.1f} s apart; firmware v2 sends one every ~2 s)")
    if bad_json:
        add("WARN", "valid JSON", f"{bad_json} messages were not valid JSON (cut off? check the payload size)")

    last = msgs[-1]
    missing = [k for k in REQUIRED if k not in last]
    add("FAIL" if missing else "PASS", "message fields",
        f"missing {missing}. Flash firmware v2 (firmware/motor_node_v2)" if missing else "all expected fields present")

    vib = [m.get("vib") for m in msgs if m.get("vib")]
    if not vib:
        add("FAIL", "vibration burst", "every message has vib=null. The MPU6050 is not read: check its wiring "
            "(SDA/SCL) and the Serial Monitor. Without vibration the model cannot run.")
    else:
        fs = num(vib[-1].get("fs")) or 0
        n = len(vib[-1].get("x", []))
        add("PASS" if 900 <= fs <= 1100 and n >= 512 else "WARN", "vibration burst",
            f"{n} samples at {fs:.0f} Hz (expect 1024 at ~1000 Hz); {len(vib)}/{len(msgs)} messages have it")
        try:
            f = features.extract(next(m for m in msgs if m.get("vib")))
            add("PASS", "feature extraction", f"{len(f)} features, vibration {f['mag_rms_g'] * 1000:.1f} mg RMS, "
                f"running speed {f['rot_hz']:.1f} Hz ({'from RPM sensor' if f['rot_from_rpm'] else 'estimated from vibration'})")
        except Exception as e:   # noqa: BLE001
            add("FAIL", "feature extraction", f"{type(e).__name__}: {e}")

    def series(key):
        return [v for v in (num(m.get(key)) for m in msgs) if v is not None]

    cur = series("current")
    if not cur:
        add("WARN", "current (PZEM)", "no readings: PZEM not answering (check RX/TX and its 230 V supply)")
    elif max(cur) < 0.1:
        add("WARN", "current (PZEM)", f"max {max(cur):.3f} A; a 60 W fan should draw ~0.26 A. See 'PZEM current' "
            "in docs/recording-and-training.md. The model still runs, but current features are useless.")
    else:
        add("PASS", "current (PZEM)", f"{sum(cur) / len(cur):.3f} A average")

    rpm = series("rpm")
    if not rpm or max(rpm) <= 0:
        add("WARN", "speed (TCRT5000)", "0 RPM: the IR sensor is not seeing the reflective mark. Features fall "
            "back to estimating speed from vibration, which works but is less exact.")
    else:
        add("PASS" if 600 < sum(rpm) / len(rpm) < 2000 else "WARN", "speed (TCRT5000)",
            f"{sum(rpm) / len(rpm):.0f} RPM average (a table fan runs ~1000-1400)")

    snd = series("sound_dbfs")
    if not snd:
        add("WARN", "sound (INMP441)", "no readings")
    elif max(snd) > 0:
        add("WARN", "sound (INMP441)", f"{max(snd):.0f} is not a dBFS value (must be below 0). "
            "Old v1 firmware sends the raw level; flash v2")
    elif max(snd) - min(snd) < 0.05 and len(snd) > 3:
        add("WARN", "sound (INMP441)", f"stuck at {snd[0]:.1f} dBFS: check the I2S wiring")
    else:
        add("PASS", "sound (INMP441)", f"{min(snd):.1f} to {max(snd):.1f} dBFS")

    for key, name in (("ds18b20_temp_c", "motor temperature (DS18B20)"), ("dht22_temp_c", "room temperature (DHT22)")):
        t = series(key)
        add("PASS" if t and -10 < t[-1] < 120 else "WARN", name,
            f"{t[-1]:.1f} °C" if t else "no readings (sensor disconnected?)")

    seqs = [m.get("seq") for m in msgs if isinstance(m.get("seq"), int)]
    gaps = sum(1 for a, b in zip(seqs, seqs[1:]) if b != a + 1)
    if gaps:
        add("WARN", "dropped messages", f"{gaps} gaps in the sequence numbers (weak Wi-Fi or ESP32 restarts)")
    return results


def main():
    if load_dotenv:
        load_dotenv(ROOT / ".env")
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--broker", default=os.getenv("MQTT_BROKER", "localhost"))
    p.add_argument("--port", type=int, default=int(os.getenv("MQTT_PORT", 1883)))
    p.add_argument("--topic", default=os.getenv("MQTT_TOPIC", "motor/sensors"))
    p.add_argument("--seconds", type=int, default=30)
    a = p.parse_args()

    import paho.mqtt.client as mqtt
    msgs, bad, connected = [], [0], threading.Event()

    def on_connect(client, userdata, flags, reason_code, properties=None):
        if reason_code == 0:
            client.subscribe(a.topic)
            connected.set()

    def on_message(client, userdata, m):
        try:
            msgs.append(json.loads(m.payload))
            print(f"  message {len(msgs)}: {len(m.payload)} bytes")
        except ValueError:
            bad[0] += 1

    try:
        client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
    except AttributeError:   # paho-mqtt 1.x
        client = mqtt.Client()
    client.on_connect, client.on_message = on_connect, on_message
    try:
        client.connect(a.broker, a.port)
    except OSError as e:
        print(f"FAIL  broker: cannot reach {a.broker}:{a.port} ({e}). Start Mosquitto, or fix MQTT_BROKER in .env.")
        raise SystemExit(1)
    client.loop_start()
    if not connected.wait(5):
        print(f"FAIL  broker: {a.broker}:{a.port} did not accept the connection.")
        raise SystemExit(1)
    print(f"PASS  broker: connected to {a.broker}:{a.port}, listening on '{a.topic}' for {a.seconds} s ...")
    time.sleep(a.seconds)
    client.loop_stop()

    results = check(msgs, a.seconds, bad[0])
    print()
    for level, name, detail in results:
        print(f"{level:5s} {name}: {detail}")
    fails = sum(1 for r in results if r[0] == "FAIL")
    print("\nReady: run  python server.py --source mqtt" if not fails else "\nFix the FAIL lines first.")
    raise SystemExit(1 if fails else 0)


if __name__ == "__main__":
    main()
