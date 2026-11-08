// Motor predictive maintenance node, firmware v3 (ESP32-S3).
//
// v3 sends everything the sensors can give, so the PC can compute as many features as possible:
//  - MPU6050: 1024-sample burst at 1 kHz of accelerometer X/Y/Z AND gyroscope X/Y/Z, plus the
//    chip's own temperature. (v2 sent accelerometer only.)
//  - INMP441: 4096 raw audio samples at 16 kHz (256 ms), so the PC can compute a full sound
//    spectrum, MFCCs etc. sound_dbfs is still sent too. (v2 sent one number.)
//  - TCRT5000: RPM plus min / max / std of the rotation period (speed ripple).
//  - PZEM: voltage, current, power, energy, power factor, mains frequency.
//  - DS18B20 + DHT22: motor temperature, ambient temperature and humidity.
//  - Diagnostics: Wi-Fi signal, free memory, burst read errors, cycle time.
// Raw samples are sent as base64 int16 (little-endian) to keep the message ~30 KB.
//
// The OLED shows one page with only the important values.
//
// Board: ESP32S3 Dev Module, USB CDC On Boot Enabled, PSRAM Disabled.
// Libraries: MPU6050_light, OneWire, DallasTemperature, DHT sensor library,
// Adafruit SSD1306 + GFX, PZEM004Tv30, PubSubClient.

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

// ---------------- WIFI + MQTT (fill these in; blank them before sharing the code) ----------------
#define WIFI_SSID     "your_wifi_name"
#define WIFI_PASSWORD "your_wifi_password"
#define MQTT_BROKER   "192.168.x.x"   // your PC's IPv4 from ipconfig

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

// PZEM current, power and energy are divided by this (18 turns of the live wire through the ring).
const float CT_TURNS = 18.0;

// Vibration burst
#define MPU_ADDR         0x68
#define VIB_N            1024     // samples per axis per burst (~1.02 s at 1 kHz)
#define VIB_PERIOD_US    1000     // 1 kHz
#define VIB_LSB_PER_G    4096     // +/-8 g, set by mpu.begin(2, 2) (the fan rig clipped at +/-4 g)
#define GYRO_LSB_PER_DPS 32.8     // +/-1000 deg/s, set by mpu.begin(2, 2)

// Audio
#define I2S_PORT     I2S_NUM_0
#define AUDIO_FS     16000
#define AUDIO_N      4096         // 256 ms per message

// RPM: ignore pulses closer than this (debounce; 5 ms = 12000 RPM)
#define RPM_MIN_GAP_US  5000
#define RPM_TIMEOUT_US  1500000   // no pulse for 1.5 s -> RPM 0

#define SCREEN_WIDTH   128
#define SCREEN_HEIGHT  64
#define SCREEN_ADDRESS 0x3C
#define OLED_RESET     -1

WiFiClient espClient;
PubSubClient mqttClient(espClient);
// The last two numbers keep the I2C bus at 400 kHz after every screen update. The library's
// default drops it back to 100 kHz, which made a 14-byte MPU read take ~1.6 ms (only ~609 Hz).
Adafruit_SSD1306 display(SCREEN_WIDTH, SCREEN_HEIGHT, &Wire, OLED_RESET, 400000UL, 400000UL);
MPU6050 mpu(Wire);
OneWire oneWire(DS18B20_PIN);
DallasTemperature ds18b20(&oneWire);
DHT dht22(DHT22_PIN, DHT22);
PZEM004Tv30 pzem(Serial1, PZEM_RX_PIN, PZEM_TX_PIN);

static int16_t ax[VIB_N], ay[VIB_N], az[VIB_N], gx[VIB_N], gy[VIB_N], gz[VIB_N];
static float   vibFs = 0, vibRmsG = 0, mpuTempC = NAN;
static bool    vibOk = false;
static int     vibErrors = 0;
static uint32_t vibMaxGapUs = 0;

static int32_t audioRaw[AUDIO_N];
static int16_t audio16[AUDIO_N];
static int     audioN = 0;

static char    payload[40960];

float ds18b20Temp = NAN, dht22Temp = NAN, dht22Hum = NAN;
float pzemVoltage = NAN, pzemCurrent = NAN, pzemPower = NAN, pzemEnergy = NAN;
float pzemPf = NAN, pzemFreq = NAN;
float soundDbfs = NAN;
float currentRpm = 0, rpmMin = NAN, rpmMax = NAN, rpmStd = NAN;
uint32_t rpmPulses = 0;
bool  dhtOk = false, pzemOk = false, lastPublishOk = false;
unsigned long lastCycle = 0, lastDisplay = 0, lastCycleMs = 0;
uint32_t seq = 0;

// ---------- RPM (interrupt) ----------

volatile uint32_t lastPulseUs = 0, periodCount = 0, minPeriodUs = 0xFFFFFFFF, maxPeriodUs = 0;
volatile uint64_t sumPeriodUs = 0, sumSqPeriodUs = 0;

void IRAM_ATTR onRpmPulse() {
  uint32_t t = micros();
  uint32_t dt = t - lastPulseUs;
  if (lastPulseUs != 0 && dt < RPM_MIN_GAP_US) return;
  if (lastPulseUs != 0) {
    sumPeriodUs += dt;
    sumSqPeriodUs += (uint64_t)dt * dt;
    periodCount++;
    if (dt < minPeriodUs) minPeriodUs = dt;
    if (dt > maxPeriodUs) maxPeriodUs = dt;
  }
  lastPulseUs = t;
}

void updateRpm() {
  noInterrupts();
  uint64_t sum = sumPeriodUs, sumSq = sumSqPeriodUs;
  uint32_t cnt = periodCount, last = lastPulseUs, pmin = minPeriodUs, pmax = maxPeriodUs;
  sumPeriodUs = 0; sumSqPeriodUs = 0; periodCount = 0; minPeriodUs = 0xFFFFFFFF; maxPeriodUs = 0;
  interrupts();
  rpmPulses = cnt;
  if (cnt > 0) {
    double mean = (double)sum / cnt;                       // us per pulse
    double var = (double)sumSq / cnt - mean * mean;
    currentRpm = 60.0e6 / (mean * PULSES_PER_REV);
    rpmMax = 60.0e6 / ((double)pmin * PULSES_PER_REV);
    rpmMin = 60.0e6 / ((double)pmax * PULSES_PER_REV);
    // std of RPM, propagated from the std of the period
    rpmStd = (var > 0) ? currentRpm * sqrt(var) / mean : 0;
  } else {
    if (last == 0 || micros() - last > RPM_TIMEOUT_US) currentRpm = 0;
    rpmMin = rpmMax = rpmStd = NAN;
  }
}

// ---------- sound ----------

void setupI2SMic() {
  i2s_config_t cfg = {
    .mode = (i2s_mode_t)(I2S_MODE_MASTER | I2S_MODE_RX),
    .sample_rate = AUDIO_FS,
    .bits_per_sample = I2S_BITS_PER_SAMPLE_32BIT,
    .channel_format = I2S_CHANNEL_FMT_ONLY_LEFT,
    .communication_format = I2S_COMM_FORMAT_STAND_I2S,
    .intr_alloc_flags = 0,
    .dma_buf_count = 8,
    .dma_buf_len = 512,
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

// Records AUDIO_N fresh samples. Sets soundDbfs and fills audio16 (DC removed, 16-bit).
void captureAudio() {
  size_t bytesRead = 0;
  // The DMA buffers hold stale audio from before the vibration burst; drain them first.
  i2s_read(I2S_PORT, audioRaw, sizeof(audioRaw), &bytesRead, 0);
  size_t got = 0;
  unsigned long t0 = millis();
  while (got < sizeof(audioRaw) && millis() - t0 < 500) {
    i2s_read(I2S_PORT, (uint8_t*)audioRaw + got, sizeof(audioRaw) - got, &bytesRead, 100 / portTICK_PERIOD_MS);
    got += bytesRead;
  }
  int n = got / sizeof(int32_t);
  audioN = n;
  if (n < 256) { soundDbfs = NAN; audioN = 0; return; }
  // INMP441 sends 24-bit samples left-aligned in a 32-bit word.
  double mean = 0;
  for (int i = 0; i < n; i++) mean += (audioRaw[i] >> 8);
  mean /= n;
  double sumSq = 0;
  for (int i = 0; i < n; i++) {
    double s = (audioRaw[i] >> 8) - mean;               // remove the DC offset
    sumSq += s * s;
    double s16 = s / 256.0;                             // 24-bit -> 16-bit
    if (s16 > 32767) s16 = 32767;
    if (s16 < -32768) s16 = -32768;
    audio16[i] = (int16_t)lrint(s16);
  }
  soundDbfs = (float)(20.0 * log10(sqrt(sumSq / n) / 8388608.0 + 1e-12));
}

// ---------- vibration ----------

// Reads accel, temperature and gyro in one 14-byte transfer.
bool readMpuRaw(int i, int16_t &tempRaw) {
  Wire.beginTransmission(MPU_ADDR);
  Wire.write(0x3B);                       // ACCEL_XOUT_H
  if (Wire.endTransmission(false) != 0) return false;
  if (Wire.requestFrom((uint8_t)MPU_ADDR, (uint8_t)14) != 14) return false;
  uint8_t b[14];
  for (int k = 0; k < 14; k++) b[k] = Wire.read();   // read in order; never combine reads in one expression
  ax[i] = (int16_t)((b[0] << 8) | b[1]);
  ay[i] = (int16_t)((b[2] << 8) | b[3]);
  az[i] = (int16_t)((b[4] << 8) | b[5]);
  tempRaw = (int16_t)((b[6] << 8) | b[7]);
  gx[i] = (int16_t)((b[8] << 8) | b[9]);
  gy[i] = (int16_t)((b[10] << 8) | b[11]);
  gz[i] = (int16_t)((b[12] << 8) | b[13]);
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
  Wire.setClock(400000);   // make sure nothing has slowed the bus down
  int errors = 0;
  double tempSum = 0;
  int tempCount = 0;
  uint32_t t0 = micros(), tFirst = 0, tLast = 0, maxGap = 0;
  for (int i = 0; i < VIB_N; i++) {
    uint32_t target = t0 + (uint32_t)i * VIB_PERIOD_US;
    while ((int32_t)(micros() - target) < 0) { }
    uint32_t t = micros();
    if (i == 0) tFirst = t;
    else if (t - tLast > maxGap) maxGap = t - tLast;
    tLast = t;
    int16_t tempRaw = 0;
    bool ok = readMpuRaw(i, tempRaw);
    if (!ok) ok = readMpuRaw(i, tempRaw);   // one quick retry
    if (ok) {
      tempSum += tempRaw; tempCount++;
    } else {
      errors++;
      if (i > 0) {
        ax[i] = ax[i - 1]; ay[i] = ay[i - 1]; az[i] = az[i - 1];
        gx[i] = gx[i - 1]; gy[i] = gy[i - 1]; gz[i] = gz[i - 1];
      } else {
        ax[i] = ay[i] = az[i] = gx[i] = gy[i] = gz[i] = 0;
      }
    }
  }
  vibFs = (VIB_N - 1) * 1.0e6f / (float)(tLast - tFirst);
  vibErrors = errors;
  vibMaxGapUs = maxGap;
  mpuTempC = tempCount ? (float)(tempSum / tempCount / 340.0 + 36.53) : NAN;
  // Reject the burst if more than 2% of reads failed, or if sampling stalled anywhere
  // (a stall makes the samples unevenly spaced, which corrupts the FFT).
  vibOk = errors <= VIB_N / 50 && maxGap <= 3 * VIB_PERIOD_US;

  // Accelerometer magnitude RMS (gravity removed by mean), for the OLED and serial log.
  double mx = 0, my = 0, mz = 0;
  for (int i = 0; i < VIB_N; i++) { mx += ax[i]; my += ay[i]; mz += az[i]; }
  mx /= VIB_N; my /= VIB_N; mz /= VIB_N;
  double ss = 0;
  for (int i = 0; i < VIB_N; i++) {
    double dx = ax[i] - mx, dy = ay[i] - my, dz = az[i] - mz;
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
  // Drop corrupted frames (seen once: 184820 A). Raw limits: 100 A ring, mains 80-300 V.
  if (isnan(v) || v < 80 || v > 300 || isnan(c) || c < 0 || c > 100 || isnan(p) || p < 0 || p > 23000) {
    v = c = p = e = pf = f = NAN;
  }
  pzemOk = !isnan(v) && !isnan(c) && !isnan(p);
  pzemVoltage = v;
  pzemCurrent = isnan(c) ? NAN : c / CT_TURNS;
  pzemPower   = isnan(p) ? NAN : p / CT_TURNS;
  pzemEnergy  = isnan(e) ? NAN : e / CT_TURNS;
  pzemPf = pf;
  pzemFreq = f;
}

// ---------- JSON (built by hand; raw samples as base64) ----------

size_t payloadLen = 0;

void put(const char* s) {
  size_t n = strlen(s);
  if (payloadLen + n < sizeof(payload)) { memcpy(payload + payloadLen, s, n); payloadLen += n; payload[payloadLen] = 0; }
  else payloadLen = sizeof(payload);   // mark as overflowed
}

void putNum(const char* key, float v, int decimals) {
  char tmp[48];
  if (isnan(v) || isinf(v)) snprintf(tmp, sizeof(tmp), "\"%s\":null,", key);
  else snprintf(tmp, sizeof(tmp), "\"%s\":%.*f,", key, decimals, v);
  put(tmp);
}

void putInt(const char* key, long v) {
  char tmp[48];
  snprintf(tmp, sizeof(tmp), "\"%s\":%ld,", key, v);
  put(tmp);
}

static const char B64[] = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

// "key":"<base64 of n int16 values, little-endian>"
void putB64(const char* key, const int16_t* a, int n) {
  char tmp[24];
  snprintf(tmp, sizeof(tmp), "\"%s\":\"", key);
  put(tmp);
  const uint8_t* d = (const uint8_t*)a;   // ESP32 is little-endian
  size_t len = (size_t)n * 2;
  size_t need = ((len + 2) / 3) * 4 + 2;
  if (payloadLen + need >= sizeof(payload)) { payloadLen = sizeof(payload); return; }
  char* o = payload + payloadLen;
  size_t i = 0;
  for (; i + 2 < len; i += 3) {
    uint32_t v = (d[i] << 16) | (d[i + 1] << 8) | d[i + 2];
    *o++ = B64[(v >> 18) & 63]; *o++ = B64[(v >> 12) & 63]; *o++ = B64[(v >> 6) & 63]; *o++ = B64[v & 63];
  }
  if (i < len) {
    uint32_t v = d[i] << 16;
    if (i + 1 < len) v |= d[i + 1] << 8;
    *o++ = B64[(v >> 18) & 63]; *o++ = B64[(v >> 12) & 63];
    *o++ = (i + 1 < len) ? B64[(v >> 6) & 63] : '=';
    *o++ = '=';
  }
  *o++ = '"';
  *o = 0;
  payloadLen = o - payload;
}

size_t buildPayload(unsigned long now) {
  payloadLen = 0;
  payload[0] = 0;
  char tmp[160];
  snprintf(tmp, sizeof(tmp), "{\"device_id\":\"%s\",\"fw\":\"v3\",\"seq\":%lu,\"uptime_ms\":%lu,",
           MQTT_CLIENT_ID, (unsigned long)seq, now);
  put(tmp);
  putNum("ds18b20_temp_c", ds18b20Temp, 3);
  putNum("dht22_temp_c", dhtOk ? dht22Temp : NAN, 2);
  putNum("dht22_humidity", dhtOk ? dht22Hum : NAN, 2);
  putNum("mpu_temp_c", mpuTempC, 2);
  putNum("rpm", currentRpm, 1);
  putNum("rpm_min", rpmMin, 1);
  putNum("rpm_max", rpmMax, 1);
  putNum("rpm_std", rpmStd, 2);
  putInt("rpm_pulses", rpmPulses);
  putNum("voltage", pzemVoltage, 1);
  putNum("current", pzemCurrent, 4);
  putNum("power", pzemPower, 2);
  putNum("energy", pzemEnergy, 4);
  putNum("pf", pzemPf, 2);
  putNum("mains_hz", pzemFreq, 1);
  putNum("sound_dbfs", soundDbfs, 2);
  putInt("wifi_rssi", WiFi.RSSI());
  putInt("free_heap", (long)ESP.getFreeHeap());
  putInt("cycle_ms", (long)lastCycleMs);
  if (audioN > 0) {
    snprintf(tmp, sizeof(tmp), "\"audio\":{\"fs\":%d,\"n\":%d,\"enc\":\"b64le16\",", AUDIO_FS, audioN);
    put(tmp);
    putB64("data", audio16, audioN);
    put("},");
  } else {
    put("\"audio\":null,");
  }
  if (vibOk) {
    snprintf(tmp, sizeof(tmp),
             "\"vib\":{\"fs\":%.2f,\"n\":%d,\"enc\":\"b64le16\",\"lsb_per_g\":%d,\"lsb_per_dps\":%.1f,\"err\":%d,\"max_gap_us\":%lu,",
             vibFs, VIB_N, VIB_LSB_PER_G, GYRO_LSB_PER_DPS, vibErrors, (unsigned long)vibMaxGapUs);
    put(tmp);
    putB64("x", ax, VIB_N); put(",");
    putB64("y", ay, VIB_N); put(",");
    putB64("z", az, VIB_N); put(",");
    putB64("gx", gx, VIB_N); put(",");
    putB64("gy", gy, VIB_N); put(",");
    putB64("gz", gz, VIB_N);
    put("}}");
  } else {
    snprintf(tmp, sizeof(tmp), "\"vib\":null,\"vib_err\":%d,\"vib_max_gap_us\":%lu}", vibErrors, (unsigned long)vibMaxGapUs);
    put(tmp);
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
  // Stream the message so PubSubClient does not need a 30 KB internal buffer.
  if (!mqttClient.beginPublish(MQTT_TOPIC, len, false)) return false;
  size_t written = mqttClient.write((const uint8_t*)payload, len);
  return mqttClient.endPublish() && written == len;
}

// ---------- display: one page, important values only ----------

void drawDisplay() {
  bool running = (!isnan(pzemCurrent) && pzemCurrent > 0.02) || vibRmsG > 0.05;
  display.clearDisplay();
  display.setCursor(0, 0);

  display.setTextSize(2);
  if (running) { display.print(currentRpm, 0); display.println(" rpm"); }
  else display.println("STOPPED");
  display.setTextSize(1);

  display.print("Vib   "); display.print(vibRmsG * 1000, 0); display.println(" mg");
  if (pzemOk) {
    display.print("Power "); display.print(pzemPower, 1); display.print("W ");
    display.print(pzemCurrent, 3); display.println("A");
  } else {
    display.println("Power --  (PZEM?)");
  }
  display.print("Motor "); display.print(ds18b20Temp, 1); display.print("C");
  if (dhtOk && !isnan(ds18b20Temp)) { display.print(" +"); display.print(ds18b20Temp - dht22Temp, 1); }
  display.println();
  display.print("Sound "); display.print(soundDbfs, 0); display.println(" dB");
  display.print(lastPublishOk ? "Sending ok  #" : "NOT SENDING #"); display.println(seq);
  display.display();
}

// ---------- main ----------

void setup() {
  Serial.begin(115200);
  unsigned long t0 = millis();
  while (!Serial && millis() - t0 < 5000) delay(10);
  Wire.begin(SDA_PIN, SCL_PIN);
  Wire.setClock(400000);
  Wire.setTimeOut(3);   // ms; the default 50 ms stall wrecks 1 kHz sampling when a read fails

  if (!display.begin(SSD1306_SWITCHCAPVCC, SCREEN_ADDRESS)) {
    Serial.println("OLED not found!");
    while (1) delay(1000);
  }
  display.clearDisplay();
  display.setTextSize(1);
  display.setTextColor(SSD1306_WHITE);
  display.setCursor(0, 0);
  display.println("Booting up (v3)...");
  display.display();

  // Gyro +/-1000 dps, accel +/-8 g: the test with the blade fitted clipped at +/-4 g and +/-500 dps.
  if (mpu.begin(2, 2) != 0) {
    Serial.println("MPU6050 init failed!");
    while (1) delay(1000);
  }
  writeMpuReg(0x1A, 0x01);   // CONFIG: DLPF 184 Hz, anti-aliasing for 1 kHz sampling
  writeMpuReg(0x19, 0x00);   // SMPLRT_DIV: 1 kHz output rate

  ds18b20.begin();
  ds18b20.setResolution(12);           // 0.0625 C steps
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
    captureAudio();
    mqttClient.loop();
    captureVibrationBurst();
    mqttClient.loop();
    lastCycleMs = millis() - now;

    size_t len = buildPayload(now);
    if (len >= sizeof(payload) - 1) {
      lastPublishOk = false;
      Serial.println("Payload buffer full, message dropped.");
    } else if ((lastPublishOk = publishPayload(len))) {
      Serial.printf("Published #%lu: %u bytes, vib %s fs=%.1f Hz rms=%.0f mg err=%d gap=%lu us, I=%.4f A, %.1f dBFS (%d samples), %.0f RPM (+/-%.1f), cycle %lu ms\n",
                    (unsigned long)seq, (unsigned)len, vibOk ? "ok" : "DROPPED", vibFs, vibRmsG * 1000,
                    vibErrors, (unsigned long)vibMaxGapUs, pzemCurrent, soundDbfs, audioN, currentRpm,
                    isnan(rpmStd) ? 0.0f : rpmStd, lastCycleMs);
    } else {
      Serial.println("MQTT publish failed.");
    }
  }

  if (now - lastDisplay >= 250) {
    lastDisplay = now;
    drawDisplay();
  }
}
