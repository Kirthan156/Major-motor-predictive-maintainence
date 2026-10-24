// Motor predictive maintenance node, firmware v2 (ESP32-S3).
//
// Changes from v1 (handoff PDF):
//  - Vibration: every cycle takes a 1024-sample burst of raw accelerometer data at ~1 kHz
//    and publishes the raw samples, so RMS / kurtosis / FFT features can be computed.
//    v1 published one instantaneous sample every 2 s, which cannot show vibration at all.
//  - RPM: measured from the time between pulses instead of counting pulses in 1 s
//    (counting gave +/-60 RPM steps at one pulse per rev).
//  - Sound: uses the real 24-bit sample, removes the DC offset and reports dBFS.
//    This is the fix for the "sound level stuck near 32000" problem.
//  - PZEM: also publishes power factor and mains frequency.
//  - Wi-Fi and broker settings live in secrets.h (not committed).
//
// Board: ESP32S3 Dev Module, USB CDC On Boot Enabled, PSRAM Disabled.
// Libraries: MPU6050_light, OneWire, DallasTemperature, DHT sensor library,
// Adafruit SSD1306 + GFX, PZEM004Tv30, PubSubClient.  (ArduinoJson is no longer needed.)

#include <WiFi.h>
#include <PubSubClient.h>
#include <Wire.h>
#include <MPU6050_light.h>
#include <OneWire.h>
#include <DallasTemperature.h>
#include <DHT.h>
#include <Adafruit_GFX.h>
#include <Adafruit_SSD1306.h>
#include <PZEM004Tv30.h>
#include <driver/i2s.h>
#include <math.h>

#include "secrets.h"   // copy secrets.h.example to secrets.h and fill it in

const int           MQTT_PORT      = 1883;
const char*         MQTT_CLIENT_ID = "esp32s3_motor_node1";
const char*         MQTT_TOPIC     = "motor/sensors";
const unsigned long CYCLE_MS       = 2000;   // one message per cycle

#define SDA_PIN     8
#define SCL_PIN     9
#define DS18B20_PIN 4
#define DHT22_PIN   6
#define RPM_PIN     5
#define PZEM_RX_PIN 17
#define PZEM_TX_PIN 18
#define I2S_WS_PIN  13
#define I2S_SCK_PIN 14
#define I2S_SD_PIN  15
#define PULSES_PER_REV 1

// PZEM current, power and energy are divided by this. 1.0 = raw ring values.
// Calibrate with a known bulb: CT_TURNS = PZEM current reading / true current.
const float CT_TURNS = 1.0;

// Vibration burst
#define MPU_ADDR        0x68
#define VIB_N           1024     // samples per axis per burst (~1.02 s at 1 kHz)
#define VIB_PERIOD_US   1000     // 1 kHz target
#define VIB_LSB_PER_G   8192     // +/-4 g range, set by mpu.begin(1, 1)

// RPM: ignore pulses closer than this (debounce; 5 ms = 12000 RPM)
#define RPM_MIN_GAP_US  5000
#define RPM_TIMEOUT_US  1500000  // no pulse for 1.5 s -> RPM 0

#define SCREEN_WIDTH   128
#define SCREEN_HEIGHT  64
#define SCREEN_ADDRESS 0x3C
#define OLED_RESET     -1

#define I2S_PORT       I2S_NUM_0
#define SOUND_N        1024

WiFiClient espClient;
PubSubClient mqttClient(espClient);
Adafruit_SSD1306 display(SCREEN_WIDTH, SCREEN_HEIGHT, &Wire, OLED_RESET);
MPU6050 mpu(Wire);
OneWire oneWire(DS18B20_PIN);
DallasTemperature ds18b20(&oneWire);
DHT dht22(DHT22_PIN, DHT22);
PZEM004Tv30 pzem(Serial1, PZEM_RX_PIN, PZEM_TX_PIN);

static int16_t vibX[VIB_N], vibY[VIB_N], vibZ[VIB_N];
static float   vibFs = 0;
static bool    vibOk = false;
static float   vibRmsG = 0;          // for the OLED only; real features are computed in Python

static int32_t soundBuf[SOUND_N];
static char    payload[24576];

float ds18b20Temp = NAN, dht22Temp = NAN, dht22Hum = NAN;
float pzemVoltage = NAN, pzemCurrent = NAN, pzemPower = NAN, pzemEnergy = NAN;
float pzemPf = NAN, pzemFreq = NAN;
float soundDbfs = NAN;
float currentRpm = 0;
bool  dhtOk = false, pzemOk = false;
unsigned long lastCycle = 0, lastDisplay = 0;
uint32_t seq = 0;

volatile uint32_t lastPulseUs = 0, sumPeriodUs = 0, periodCount = 0;

void IRAM_ATTR onRpmPulse() {
  uint32_t t = micros();
  uint32_t dt = t - lastPulseUs;
  if (lastPulseUs != 0 && dt < RPM_MIN_GAP_US) return;
  if (lastPulseUs != 0) { sumPeriodUs += dt; periodCount++; }
  lastPulseUs = t;
}

void updateRpm() {
  noInterrupts();
  uint32_t sum = sumPeriodUs, cnt = periodCount, last = lastPulseUs;
  sumPeriodUs = 0; periodCount = 0;
  interrupts();
  if (cnt > 0) {
    currentRpm = 60.0e6f * cnt / ((float)sum * PULSES_PER_REV);
  } else if (last == 0 || micros() - last > RPM_TIMEOUT_US) {
    currentRpm = 0;
  }
}

// ---------- sound ----------

void setupI2SMic() {
  i2s_config_t cfg = {
    .mode = (i2s_mode_t)(I2S_MODE_MASTER | I2S_MODE_RX),
    .sample_rate = 16000,
    .bits_per_sample = I2S_BITS_PER_SAMPLE_32BIT,
    .channel_format = I2S_CHANNEL_FMT_ONLY_LEFT,
    .communication_format = I2S_COMM_FORMAT_STAND_I2S,
    .intr_alloc_flags = 0,
    .dma_buf_count = 4,
    .dma_buf_len = 256,
    .use_apll = false
  };
  i2s_pin_config_t pins = {
    .bck_io_num = I2S_SCK_PIN,
    .ws_io_num = I2S_WS_PIN,
    .data_out_num = I2S_PIN_NO_CHANGE,
    .data_in_num = I2S_SD_PIN
  };
  i2s_driver_install(I2S_PORT, &cfg, 0, NULL);
  i2s_set_pin(I2S_PORT, &pins);
}

// Returns sound level in dBFS (0 = full scale, quieter is more negative), or NAN.
float readSoundDbfs() {
  size_t bytesRead = 0;
  // The DMA buffers hold stale audio from before the vibration burst; drain them first.
  i2s_read(I2S_PORT, soundBuf, sizeof(soundBuf), &bytesRead, 0);
  i2s_read(I2S_PORT, soundBuf, sizeof(soundBuf), &bytesRead, 200 / portTICK_PERIOD_MS);
  int n = bytesRead / sizeof(int32_t);
  if (n < 64) return NAN;
  // INMP441 sends 24-bit samples left-aligned in a 32-bit word.
  double mean = 0;
  for (int i = 0; i < n; i++) mean += (soundBuf[i] >> 8);
  mean /= n;
  double sumSq = 0;
  for (int i = 0; i < n; i++) {
    double s = (soundBuf[i] >> 8) - mean;   // remove the DC offset
    sumSq += s * s;
  }
  double rms = sqrt(sumSq / n);
  return (float)(20.0 * log10(rms / 8388608.0 + 1e-12));
}

// ---------- vibration ----------

bool readAccelRaw(int16_t &x, int16_t &y, int16_t &z) {
  Wire.beginTransmission(MPU_ADDR);
  Wire.write(0x3B);                       // ACCEL_XOUT_H
  if (Wire.endTransmission(false) != 0) return false;
  if (Wire.requestFrom((uint8_t)MPU_ADDR, (uint8_t)6) != 6) return false;
  uint8_t b[6];
  for (int i = 0; i < 6; i++) b[i] = Wire.read();   // read in order; never combine reads in one expression
  x = (int16_t)((b[0] << 8) | b[1]);
  y = (int16_t)((b[2] << 8) | b[3]);
  z = (int16_t)((b[4] << 8) | b[5]);
  return true;
}

void writeMpuReg(uint8_t reg, uint8_t val) {
  Wire.beginTransmission(MPU_ADDR);
  Wire.write(reg);
  Wire.write(val);
  Wire.endTransmission();
}

// Blocks for ~1 s. The OLED and MQTT are paused while it runs.
void captureVibrationBurst() {
  int errors = 0;
  uint32_t t0 = micros(), tFirst = 0, tLast = 0;
  for (int i = 0; i < VIB_N; i++) {
    uint32_t target = t0 + (uint32_t)i * VIB_PERIOD_US;
    while ((int32_t)(micros() - target) < 0) { }
    uint32_t t = micros();
    if (i == 0) tFirst = t;
    tLast = t;
    if (!readAccelRaw(vibX[i], vibY[i], vibZ[i])) {
      errors++;
      if (i > 0) { vibX[i] = vibX[i - 1]; vibY[i] = vibY[i - 1]; vibZ[i] = vibZ[i - 1]; }
      else { vibX[i] = vibY[i] = vibZ[i] = 0; }
    }
  }
  vibFs = (VIB_N - 1) * 1.0e6f / (float)(tLast - tFirst);
  vibOk = errors < VIB_N / 100;   // allow under 1% failed reads

  // Rough magnitude RMS (gravity removed by mean) for the OLED.
  double mx = 0, my = 0, mz = 0;
  for (int i = 0; i < VIB_N; i++) { mx += vibX[i]; my += vibY[i]; mz += vibZ[i]; }
  mx /= VIB_N; my /= VIB_N; mz /= VIB_N;
  double ss = 0;
  for (int i = 0; i < VIB_N; i++) {
    double dx = vibX[i] - mx, dy = vibY[i] - my, dz = vibZ[i] - mz;
    ss += dx * dx + dy * dy + dz * dz;
  }
  vibRmsG = sqrt(ss / VIB_N) / VIB_LSB_PER_G;
}

// ---------- slow sensors ----------

void readSlowSensors() {
  // DS18B20: read the conversion started last cycle (2 s ago), then start the next one.
  float dsT = ds18b20.getTempCByIndex(0);
  ds18b20Temp = (dsT == DEVICE_DISCONNECTED_C) ? NAN : dsT;
  ds18b20.requestTemperatures();

  float h = dht22.readHumidity();
  float t = dht22.readTemperature();
  dhtOk = !isnan(h) && !isnan(t);
  if (dhtOk) { dht22Hum = h; dht22Temp = t; }

  float v = pzem.voltage(), c = pzem.current(), p = pzem.power(), e = pzem.energy();
  float pf = pzem.pf(), f = pzem.frequency();
  pzemOk = !isnan(v) && !isnan(c) && !isnan(p);
  pzemVoltage = v;
  pzemCurrent = isnan(c) ? NAN : c / CT_TURNS;
  pzemPower   = isnan(p) ? NAN : p / CT_TURNS;
  pzemEnergy  = isnan(e) ? NAN : e / CT_TURNS;
  pzemPf = pf;
  pzemFreq = f;
}

// ---------- JSON (built by hand: ArduinoJson needs ~30 KB to hold 3072 samples) ----------

size_t payloadLen = 0;

void put(const char* s) {
  size_t n = strlen(s);
  if (payloadLen + n < sizeof(payload)) { memcpy(payload + payloadLen, s, n); payloadLen += n; payload[payloadLen] = 0; }
}

void putNum(const char* key, float v, int decimals) {
  char tmp[48];
  if (isnan(v) || isinf(v)) snprintf(tmp, sizeof(tmp), "\"%s\":null,", key);
  else snprintf(tmp, sizeof(tmp), "\"%s\":%.*f,", key, decimals, v);
  put(tmp);
}

void putArray(const char* key, const int16_t* a, int n) {
  char tmp[16];
  snprintf(tmp, sizeof(tmp), "\"%s\":[", key);
  put(tmp);
  for (int i = 0; i < n; i++) {
    snprintf(tmp, sizeof(tmp), i ? ",%d" : "%d", a[i]);
    put(tmp);
  }
  put("]");
}

size_t buildPayload(unsigned long now) {
  payloadLen = 0;
  payload[0] = 0;
  char tmp[96];
  snprintf(tmp, sizeof(tmp), "{\"device_id\":\"%s\",\"fw\":\"v2\",\"seq\":%lu,\"uptime_ms\":%lu,",
           MQTT_CLIENT_ID, (unsigned long)seq, now);
  put(tmp);
  putNum("ds18b20_temp_c", ds18b20Temp, 2);
  putNum("dht22_temp_c", dhtOk ? dht22Temp : NAN, 2);
  putNum("dht22_humidity", dhtOk ? dht22Hum : NAN, 2);
  putNum("rpm", currentRpm, 1);
  putNum("voltage", pzemVoltage, 1);
  putNum("current", pzemCurrent, 4);
  putNum("power", pzemPower, 2);
  putNum("energy", pzemEnergy, 4);
  putNum("pf", pzemPf, 2);
  putNum("mains_hz", pzemFreq, 1);
  putNum("sound_dbfs", soundDbfs, 2);
  if (vibOk) {
    snprintf(tmp, sizeof(tmp), "\"vib\":{\"fs\":%.2f,\"lsb_per_g\":%d,\"n\":%d,", vibFs, VIB_LSB_PER_G, VIB_N);
    put(tmp);
    putArray("x", vibX, VIB_N); put(",");
    putArray("y", vibY, VIB_N); put(",");
    putArray("z", vibZ, VIB_N);
    put("}}");
  } else {
    put("\"vib\":null}");
  }
  return payloadLen;
}

// ---------- network ----------

void connectWiFi() {
  Serial.print("Connecting to WiFi");
  WiFi.begin(WIFI_SSID, WIFI_PASSWORD);
  while (WiFi.status() != WL_CONNECTED) { delay(400); Serial.print("."); }
  Serial.println();
  Serial.print("WiFi connected, IP: ");
  Serial.println(WiFi.localIP());
}

void connectMQTT() {
  while (!mqttClient.connected()) {
    Serial.print("Connecting to MQTT broker...");
    if (mqttClient.connect(MQTT_CLIENT_ID)) {
      Serial.println("connected");
    } else {
      Serial.print("failed, rc=");
      Serial.print(mqttClient.state());
      Serial.println(" retrying in 2s");
      delay(2000);
    }
  }
}

bool publishPayload(size_t len) {
  // Stream the message so PubSubClient does not need a 20 KB internal buffer.
  if (!mqttClient.beginPublish(MQTT_TOPIC, len, false)) return false;
  size_t written = mqttClient.write((const uint8_t*)payload, len);
  return mqttClient.endPublish() && written == len;
}

// ---------- display ----------

void drawDisplay(unsigned long now) {
  display.clearDisplay();
  display.setCursor(0, 0);
  if ((now / 3000) % 2 == 0) {
    display.println("Motor Monitor (1/2)");
    display.println("-------------------");
    display.print("DHT T:"); display.print(dhtOk ? dht22Temp : 0, 1);
    display.print(" H:"); display.println(dhtOk ? dht22Hum : 0, 1);
    display.print("Motor T: "); display.print(ds18b20Temp, 1); display.println(" C");
    display.print("RPM: "); display.println(currentRpm, 0);
    display.print("Sound: "); display.print(soundDbfs, 1); display.println(" dBFS");
  } else {
    display.println("Motor Monitor (2/2)");
    display.println("-------------------");
    if (pzemOk) {
      display.print("V:"); display.print(pzemVoltage, 0);
      display.print(" PF:"); display.println(pzemPf, 2);
      display.print("I:"); display.print(pzemCurrent, 3); display.println(" A");
      display.print("P:"); display.print(pzemPower, 1); display.println(" W");
    } else {
      display.println("PZEM: NO DATA");
      display.println("check TX/RX wires");
      display.println("");
    }
    display.print("Vib: "); display.print(vibRmsG * 1000, 0);
    display.print(" mg @"); display.print(vibFs, 0); display.println("Hz");
  }
  display.display();
}

// ---------- main ----------

void setup() {
  Serial.begin(115200);
  unsigned long t0 = millis();
  while (!Serial && millis() - t0 < 5000) delay(10);
  Wire.begin(SDA_PIN, SCL_PIN);
  Wire.setClock(400000);

  if (!display.begin(SSD1306_SWITCHCAPVCC, SCREEN_ADDRESS)) {
    Serial.println("OLED not found!");
    while (1) delay(1000);
  }
  display.clearDisplay();
  display.setTextSize(1);
  display.setTextColor(SSD1306_WHITE);
  display.setCursor(0, 0);
  display.println("Booting up (v2)...");
  display.display();

  // Gyro +/-500 dps, accel +/-4 g (fan vibration with a loose blade can pass 2 g).
  if (mpu.begin(1, 1) != 0) {
    Serial.println("MPU6050 init failed!");
    while (1) delay(1000);
  }
  writeMpuReg(0x1A, 0x01);   // CONFIG: DLPF 184 Hz, anti-aliasing for 1 kHz sampling
  writeMpuReg(0x19, 0x00);   // SMPLRT_DIV: 1 kHz output rate

  ds18b20.begin();
  ds18b20.setWaitForConversion(false);
  ds18b20.requestTemperatures();
  dht22.begin();

  pinMode(RPM_PIN, INPUT);
  attachInterrupt(digitalPinToInterrupt(RPM_PIN), onRpmPulse, FALLING);

  // Uncomment once to reset the PZEM energy counter, upload, then comment it out again.
  // pzem.resetEnergy();

  setupI2SMic();
  connectWiFi();
  mqttClient.setServer(MQTT_BROKER, MQTT_PORT);
  mqttClient.setKeepAlive(30);
}

void loop() {
  if (WiFi.status() != WL_CONNECTED) connectWiFi();
  if (!mqttClient.connected()) connectMQTT();
  mqttClient.loop();

  unsigned long now = millis();

  if (now - lastCycle >= CYCLE_MS) {
    lastCycle = now;
    seq++;
    readSlowSensors();
    updateRpm();
    soundDbfs = readSoundDbfs();
    captureVibrationBurst();
    mqttClient.loop();

    size_t len = buildPayload(now);
    if (len >= sizeof(payload) - 1) {
      Serial.println("Payload buffer full, message dropped.");
    } else if (publishPayload(len)) {
      Serial.printf("Published #%lu: %u bytes, vib fs=%.1f Hz rms=%.0f mg, I=%.4f A, %.1f dBFS, %.0f RPM\n",
                    (unsigned long)seq, (unsigned)len, vibFs, vibRmsG * 1000, pzemCurrent, soundDbfs, currentRpm);
    } else {
      Serial.println("MQTT publish failed.");
    }
  }

  if (now - lastDisplay >= 250) {
    lastDisplay = now;
    drawDisplay(now);
  }
}
