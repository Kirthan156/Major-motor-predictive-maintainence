# Data path: from the sensors to the anomaly dashboard

Every stage, what it does, which file does it, and what to check when it breaks.

```
[1] Sensors ──▶ [2] ESP32-S3 firmware v3 ──Wi-Fi/MQTT──▶ [3] Mosquitto broker (PC, port 1883)
                                                                │  topic motor/sensors
                       ┌────────────────────────────────────────┼─────────────────────────────┐
                       ▼                                        ▼                             ▼
          [4] dashboard/server.py                   [5] pm/recorder.py              check_connection.py
              pipeline.py  features (pm/features.py)    data/raw/<session>.jsonl         (diagnostics)
              scoring.py   model + change detector      InfluxDB (optional)
              WebSocket ──▶ [6] browser (PC/phone)            │
                                                              ▼
                                               [7] pm/dataset.py ─▶ pm/train.py / dashboard/train_baseline.py
                                                                         │
                                                                         ▼
                                                               models/*.joblib or *.py ─▶ back into [4]
```

## [1] Sensors (what is measured)

| Sensor | Where on the rig | Sent as |
|---|---|---|
| MPU6050 accelerometer + gyroscope | on top of the motor core | 1024-sample burst per axis at 1 kHz (±8 g, ±1000 °/s) + chip temperature |
| INMP441 microphone | behind the motor, under the capacitor | 4096 raw samples at 16 kHz (256 ms) + level in dBFS |
| TCRT5000 IR | below/in front of the motor, reading a taped stick on the shaft | RPM, plus min / max / std of the rotation period in the cycle |
| PZEM-004T 100 A (18 turns) | back panel, apart from the rest | voltage, current, power, energy, power factor, mains frequency |
| DS18B20 | on the motor core, with the MPU6050 | motor temperature |
| DHT22 | beside the TCRT5000 | board-area temperature and humidity |

## [2] Firmware v3 (`firmware/motor_node_v3/motor_node_v3.ino`)

- One cycle every ~2 s (`CYCLE_MS 2000`; real cycle ~1.3-1.5 s plus sending). Each cycle: read the
  slow sensors, capture the 1 s vibration burst and the 256 ms audio clip, build one JSON message,
  stream it with `beginPublish` (no PubSubClient buffer limit), update the OLED.
- Raw samples are base64 little-endian int16 to keep the message at ~28 KB.
- Settings at the top of the sketch: `WIFI_SSID`, `WIFI_PASSWORD`, `MQTT_BROKER` (the PC's IPv4),
  `MQTT_TOPIC "motor/sensors"`, `CT_TURNS 18.0`, pins.

### Message format (one per cycle)

```json
{
  "device_id": "esp32s3_motor_node1", "fw": "v3", "seq": 312, "uptime_ms": 633238,
  "ds18b20_temp_c": 30.44, "dht22_temp_c": 31.2, "dht22_humidity": 60.1, "mpu_temp_c": 47.65,
  "rpm": 1313.3, "rpm_min": 1253.2, "rpm_max": 1378.7, "rpm_std": 29.16, "rpm_pulses": 44,
  "voltage": 238.6, "current": 0.1824, "power": 40.04, "energy": 0.0207, "pf": 0.92, "mains_hz": 50.0,
  "sound_dbfs": -19.94, "wifi_rssi": -55, "free_heap": 150996, "cycle_ms": 1326,
  "audio": {"fs": 16000, "n": 4096, "enc": "b64le16", "data": "<base64>"},
  "vib": {"fs": 1000.0, "n": 1024, "enc": "b64le16", "lsb_per_g": 4096, "lsb_per_dps": 32.8,
          "err": 0, "max_gap_us": 1001,
          "x": "<base64>", "y": "<base64>", "z": "<base64>", "gx": "<base64>", "gy": "<base64>", "gz": "<base64>"}
}
```

`vib` is `null` when the burst failed (I2C errors); such messages are skipped by the dashboard and the
dataset. Missing sensor values are `null`. `recorder.py` adds `session`, `label`, `severity`, `notes`,
`received_at` when it saves a message.

Decode a signal in Python:

```python
from features import _decode            # pm/features.py
ax_g = _decode(msg["vib"]["x"]) / msg["vib"]["lsb_per_g"]       # numpy array, g
gx_dps = _decode(msg["vib"]["gx"]) / msg["vib"]["lsb_per_dps"]   # deg/s
audio = _decode(msg["audio"]["data"]) / 32768.0                  # -1..1 full scale
```

## [3] Broker (Mosquitto)

Plain MQTT on TCP 1883, anonymous, listening on all interfaces (`listener 1883 0.0.0.0`). Any number
of programs can subscribe to `motor/sensors` at once; each gets every message. QoS 0: a message lost on
Wi-Fi is not resent (the `seq` number shows gaps).

Test it without the rig: `python dashboard/publish_recording.py data/sample/s01_sp1.jsonl` publishes a
recording exactly as the ESP32 would.

## [4] Dashboard service (`dashboard/`)

| File | Job |
|---|---|
| `server.py` | gets messages from one source (`mqtt`, `replay`, or the `sim` demo motor), runs the steps below, keeps the last hour in memory, serves the page and pushes each new point over a WebSocket. Logs every scored reading to `dashboard/data/live_<time>.csv`. |
| `pipeline.py` | `features.extract(msg)` + `features.temporal(last 15)` per device, the same as `pm/dataset.py`. Resets the 30 s history if the ESP32 restarts (uptime goes back) or after a 30 s gap. |
| `model_loader.py` | loads the anomaly model and the optional classifier: `.joblib` dicts or Python plugins (see `MODEL_INTERFACE.md`). |
| `scoring.py` | model score + change-since-baseline score + 3-of-5 alarm rule + "most unusual readings"; builds the compact point for the browser and the vibration spectrum. |
| `simulator.py` | the demo motor: real recorded messages in a loop, with faults injected on request. |

HTTP endpoints: `/` page, `/api/history` (last hour + events + status), `/api/status`,
`POST /api/baseline/reset`, `POST /api/sim/fault {"label","severity"}` (demo only), `/ws` WebSocket.

Per reading, the browser gets: time, seq, device, rpm, voltage, current, power, pf, sound_dbfs,
motor_temp_c, ambient_c, temp_rise_c, humidity, vib_rms_mg, vel_rms_mms, vib_1x_mg, kurtosis, rot_hz,
anomaly_score, threshold, change_score, change_threshold, over_threshold, change_over, model_alarm,
change_alarm, alarm, top_deviations, change_deviations, predicted_fault, fault_confidence. The same
fields go to the CSV log.

Speed: extraction + scoring take ~70 ms per message on a laptop, well inside the 2 s cycle.

## [5] Recorder (`pm/recorder.py`) and InfluxDB

- `python recorder.py --session s15_sp1 --label healthy --minutes 5` saves every message, raw bursts
  included, to `data/raw/s15_sp1.jsonl` and (without `--no-influx`) writes the features to InfluxDB
  measurement `motor_readings`, tagged with session / label / severity / device.
- `python recorder.py --live --anomaly ../models/anomaly_baseline.joblib --no-influx` scores live in the
  terminal instead of recording; with InfluxDB it also writes `anomaly_score`, `is_anomaly`, `predicted`.
- InfluxDB settings come from `.env`: `INFLUX_URL`, `INFLUX_TOKEN`, `INFLUX_ORG`, `INFLUX_BUCKET`.
- Do not run the old `mqtt_to_influx.py` bridge at the same time: it would write every reading twice.

## [6] Browser

`dashboard/static/index.html`, plain HTML + canvas, no internet needed. Shows the condition (Normal /
Watch / Anomaly), the six key readings, model score, change score, vibration, current, temperature rise
and the latest vibration spectrum with 1x/2x/3x markers, an event list, and a table of the last 20 readings.

## [7] Training

| Command | Uses | Writes |
|---|---|---|
| `python pm/dataset.py` | `data/raw/*.jsonl` | `data/dataset.csv` (~1 row per message, ~460 columns) |
| `python dashboard/train_baseline.py` | healthy sessions | `models/anomaly_baseline.joblib`, `models/healthy_reference.json`, `reports/baseline_anomaly.md` |
| `python pm/train.py` | all sessions | `models/anomaly_iforest.joblib`, `models/fault_rf.joblib`, `models/fault_xgb.joblib` (needs fault sessions), `reports/` |
| Phase II model code | `data/dataset.csv` or the raw sessions | any model following `docs/MODEL_INTERFACE.md` |

Always evaluate by holding out whole sessions, never random rows: neighbouring 2 s windows of one
run are almost identical, so a random split gives inflated scores.

## Start-up order (every session at the rig)

1. Mosquitto window (`mosquitto.exe -c mosquitto.conf -v`).
2. `influxd` (only if you use InfluxDB).
3. Power the ESP32; OLED shows `Sending ok`.
4. `cd dashboard` → `python check_connection.py` (first time that day).
5. `python server.py --source mqtt` → open http://localhost:8000 (phone: `http://<PC IPv4>:8000`).
6. Optional, second terminal: `cd pm` → `python recorder.py ...` to record while you watch.

Stop in reverse order: Ctrl+C the Python windows, power off the ESP32, close Mosquitto.
