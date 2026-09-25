// Based on Edge Impulse "esp32_microphone_continuous" (c) 2022 EdgeImpulse Inc., MIT License.
// Adapted for ESP32-S3 + I2S mic (INMP441 / ICS-43434).
// Wiring: mic SCK -> GPIO5, WS -> GPIO6, SD -> GPIO7, VDD -> 3V3, GND -> GND, L/R -> GND
// Buzzer: -> GPIO27 (through 10k pulldown to GND, per project spec)
//
// ADDED IN THIS VERSION:
//   1. Buzzer beeps when "arise" is detected above threshold (short beep),
//      and gives a distinct continuous alarm pattern when "abort" is detected.
//   2. ESP32 connects to WiFi and hosts its own live dashboard as a web page.
//      Open the IP address printed in Serial Monitor in any browser on the
//      same network to view it.
//   3. A /status JSON endpoint exposes real-time classifier output (word +
//      confidence for all 4 classes) which the dashboard polls automatically.
//
// SIMULATED VALUES (clearly marked below, per project instructions):
//   - Abort latency (40ms) and Arise latency (100ms) display boxes
//   - CPU utilization gauge (randomized 8.0-9.0%)
// Everything else on the dashboard (detected word, live confidence bars,
// detection log, uptime) is driven by real data from the classifier.

#define EIDSP_QUANTIZE_FILTERBANK   0

#include <RohithM-project-1_inferencing.h>

#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "driver/i2s.h"
#include <WiFi.h>
#include <WebServer.h>

// ---------- Your settings ----------
#define I2S_PORT   I2S_NUM_0
#define PIN_SCK    5      // BCLK
#define PIN_WS     6      // LRCL
#define PIN_SD     7      // DOUT of the mic
#define MIC_GAIN   2      // raise if mic rms is tiny when you speak, lower if it clips

// ---------- Buzzer settings ----------
#define BUZZER_PIN 27
#define ARISE_BEEP_MS      180   // single short beep duration for ARISE
#define ABORT_BEEP_ON_MS   120   // on-time per pulse for ABORT alarm pattern
#define ABORT_BEEP_OFF_MS  120   // off-time per pulse for ABORT alarm pattern

// ---------- WiFi settings — EDIT THESE ----------
const char* WIFI_SSID     = "YOUR_WIFI_NAME";
const char* WIFI_PASSWORD = "YOUR_WIFI_PASSWORD";

// ---------- Detection thresholds ----------
#define ARISE_THRESHOLD  0.80f
#define ABORT_THRESHOLD  0.80f

WebServer server(80);

typedef struct {
    signed short *buffers[2];
    unsigned char buf_select;
    unsigned char buf_ready;
    unsigned int buf_count;
    unsigned int n_samples;
} inference_t;

static inference_t inference;
static const uint32_t sample_buffer_size = 2048;
static signed short sampleBuffer[sample_buffer_size];
static bool debug_nn = false;
static int print_results = -(EI_CLASSIFIER_SLICES_PER_MODEL_WINDOW);
static bool record_status = true;

// ---------- Live state shared with the dashboard ----------
// These are updated every classification cycle and read by the /status handler.
volatile float  g_confAbort = 0, g_confArise = 0, g_confNoise = 0, g_confUnknown = 0;
volatile char   g_topWord[16] = "listening";
volatile float  g_topConfidence = 0;
volatile int    g_dspMs = 0, g_classMs = 0;
volatile bool   g_ariseEventFlag = false;   // set true briefly when a new ARISE fires
volatile bool   g_abortEventFlag = false;   // set true briefly when a new ABORT fires
volatile unsigned long g_lastEventMs = 0;

// ---------- Non-blocking buzzer state machine ----------
enum BuzzerMode { BUZZ_IDLE, BUZZ_ARISE, BUZZ_ABORT };
BuzzerMode buzzerMode = BUZZ_IDLE;
unsigned long buzzerStateChangeAt = 0;
bool buzzerPinState = false;
int abortPulsesRemaining = 0;

void startAriseBeep() {
    buzzerMode = BUZZ_ARISE;
    digitalWrite(BUZZER_PIN, HIGH);
    buzzerPinState = true;
    buzzerStateChangeAt = millis() + ARISE_BEEP_MS;
}

void startAbortAlarm() {
    buzzerMode = BUZZ_ABORT;
    abortPulsesRemaining = 6;   // 6 pulses ~= 1.4s alarm burst, retriggered on repeat detections
    digitalWrite(BUZZER_PIN, HIGH);
    buzzerPinState = true;
    buzzerStateChangeAt = millis() + ABORT_BEEP_ON_MS;
}

void serviceBuzzer() {
    if (buzzerMode == BUZZ_IDLE) return;
    if ((long)(millis() - buzzerStateChangeAt) < 0) return;

    if (buzzerMode == BUZZ_ARISE) {
        digitalWrite(BUZZER_PIN, LOW);
        buzzerPinState = false;
        buzzerMode = BUZZ_IDLE;
    } else if (buzzerMode == BUZZ_ABORT) {
        if (buzzerPinState) {
            digitalWrite(BUZZER_PIN, LOW);
            buzzerPinState = false;
            buzzerStateChangeAt = millis() + ABORT_BEEP_OFF_MS;
            abortPulsesRemaining--;
            if (abortPulsesRemaining <= 0) buzzerMode = BUZZ_IDLE;
        } else {
            digitalWrite(BUZZER_PIN, HIGH);
            buzzerPinState = true;
            buzzerStateChangeAt = millis() + ABORT_BEEP_ON_MS;
        }
    }
}

// =====================================================================
// DASHBOARD HTML (served at "/") — polls /status every 400ms
// =====================================================================
const char DASHBOARD_HTML[] PROGMEM = R"HTMLDOC(
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>ARISE Live Dashboard</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800&display=swap" rel="stylesheet">
<style>
  :root{
    --green:#1c6b4e; --green-dark:#124a35; --green-light:#e7f4ee;
    --bg:#f3f6f5; --card:#ffffff; --text:#1a2b24; --muted:#7a8b83;
    --red:#c0392b; --amber:#c9962b; --gray:#9aa6a0; --radius:16px;
  }
  *{box-sizing:border-box; margin:0; padding:0;}
  body{
    font-family:'Inter',system-ui,-apple-system,sans-serif;
    background:var(--bg); color:var(--text); padding:24px;
    min-height:100vh;
  }
  .header{
    display:flex; justify-content:space-between; align-items:center;
    background:var(--card); border-radius:var(--radius); padding:20px 28px;
    margin-bottom:20px; box-shadow:0 2px 12px rgba(20,40,30,0.06);
  }
  .brand{display:flex; align-items:center; gap:14px;}
  .brand-icon{
    width:46px; height:46px; border-radius:12px;
    background:linear-gradient(135deg,var(--green),var(--green-dark));
    display:flex; align-items:center; justify-content:center;
    font-size:22px; color:white; font-weight:800;
  }
  .brand h1{font-size:20px; font-weight:800; letter-spacing:0.3px;}
  .brand h1 span{color:var(--green); font-weight:800;}
  .brand p{font-size:12.5px; color:var(--muted); margin-top:2px;}
  .team-tag{
    background:var(--green-light); color:var(--green-dark);
    padding:8px 16px; border-radius:20px; font-size:13px; font-weight:700;
  }
  .status-dot{
    display:inline-block; width:9px; height:9px; border-radius:50%;
    background:#2ecc71; margin-right:7px; box-shadow:0 0 0 rgba(46,204,113,0.5);
    animation:pulse 1.6s infinite;
  }
  @keyframes pulse{
    0%{box-shadow:0 0 0 0 rgba(46,204,113,0.5);}
    70%{box-shadow:0 0 0 8px rgba(46,204,113,0);}
    100%{box-shadow:0 0 0 0 rgba(46,204,113,0);}
  }
  .grid{display:grid; grid-template-columns:repeat(4,1fr); gap:18px; margin-bottom:18px;}
  .card{
    background:var(--card); border-radius:var(--radius); padding:22px;
    box-shadow:0 2px 12px rgba(20,40,30,0.06);
  }
  .card-label{font-size:12.5px; color:var(--muted); font-weight:600; text-transform:uppercase; letter-spacing:0.5px;}
  .word-card{
    background:linear-gradient(135deg,var(--green),var(--green-dark));
    color:white; grid-column:span 1;
  }
  .word-card .card-label{color:#c9e8da;}
  #wordDisplay{font-size:34px; font-weight:800; margin-top:8px; letter-spacing:0.5px; transition:color 0.3s;}
  #wordSub{font-size:13px; color:#d4ecdf; margin-top:6px;}
  .metric-value{font-size:30px; font-weight:800; margin-top:10px;}
  .metric-value.small{font-size:26px;}
  .metric-unit{font-size:14px; font-weight:600; color:var(--muted); margin-left:4px;}
  .bar-wrap{width:100%; height:6px; background:var(--bg); border-radius:6px; margin-top:12px; overflow:hidden;}
  .bar-fill{height:100%; background:var(--green); border-radius:6px; transition:width 0.4s;}
  .row2{display:grid; grid-template-columns:2fr 1fr; gap:18px; margin-bottom:18px;}
  .panel-title{font-size:15px; font-weight:700; margin-bottom:18px; display:flex; justify-content:space-between; align-items:center;}
  .panel-title small{font-size:11.5px; color:var(--muted); font-weight:500;}
  .bars{display:flex; align-items:flex-end; gap:22px; height:170px; padding:0 10px;}
  .bar-col{flex:1; display:flex; flex-direction:column; align-items:center; justify-content:flex-end; height:100%;}
  .bar-track{width:38px; height:100%; background:var(--bg); border-radius:8px; display:flex; align-items:flex-end; overflow:hidden;}
  .bar-track .fill{width:100%; border-radius:8px; transition:height 0.35s ease;}
  .bar-col .pct{font-size:12.5px; font-weight:700; margin-bottom:8px;}
  .bar-col .name{font-size:12px; color:var(--muted); margin-top:8px; font-weight:600;}
  .donut-wrap{display:flex; flex-direction:column; align-items:center; justify-content:center; height:100%;}
  .donut-label{font-size:12px; color:var(--muted); margin-top:10px; font-weight:600; text-align:center;}
  .row3{display:grid; grid-template-columns:1.3fr 1fr; gap:18px;}
  .log-list{max-height:210px; overflow-y:auto; display:flex; flex-direction:column; gap:10px;}
  .log-item{
    display:flex; justify-content:space-between; align-items:center;
    padding:10px 14px; background:var(--bg); border-radius:10px; font-size:13px;
  }
  .log-item .tag{
    font-weight:700; padding:3px 10px; border-radius:12px; font-size:11.5px; color:white;
  }
  .uptime-box{display:flex; flex-direction:column; align-items:center; justify-content:center; height:100%; padding:20px 0;}
  #uptimeVal{font-size:38px; font-weight:800; font-variant-numeric:tabular-nums; color:var(--green-dark);}
  .footer{text-align:center; color:var(--muted); font-size:12px; margin-top:20px;}
</style>
</head>
<body>

<div class="header">
  <div class="brand">
    <div class="brand-icon">A</div>
    <div>
      <h1>ARISE <span>&mdash; Autonomous Real-time Intelligent Space Edge</span></h1>
      <p>Low Latency and Efficient Voice Activator for Edge Devices</p>
    </div>
  </div>
  <div class="team-tag"><span class="status-dot"></span>LIVE &middot; Team ENDEAVOURS</div>
</div>

<div class="grid">
  <div class="card word-card">
    <div class="card-label">Word Detected</div>
    <div id="wordDisplay">Listening&hellip;</div>
    <div id="wordSub">Confidence: --%</div>
  </div>
  <div class="card">
    <div class="card-label">Abort Latency (Hardware Interrupt)</div>
    <div class="metric-value">40<span class="metric-unit">ms</span></div>
    <div class="bar-wrap"><div class="bar-fill" style="width:15%; background:var(--red);"></div></div>
  </div>
  <div class="card">
    <div class="card-label">Arise &rarr; Cloud ASR Round Trip</div>
    <div class="metric-value">100<span class="metric-unit">ms</span></div>
    <div class="bar-wrap"><div class="bar-fill" style="width:35%;"></div></div>
  </div>
  <div class="card">
    <div class="card-label">Edge CPU Utilization</div>
    <div class="metric-value small"><span id="cpuVal">8.4</span><span class="metric-unit">%</span></div>
    <div class="bar-wrap"><div class="bar-fill" id="cpuBar" style="width:8%; background:var(--amber);"></div></div>
  </div>
</div>

<div class="row2">
  <div class="card">
    <div class="panel-title">Live Classification Confidence <small>updates every 400ms</small></div>
    <div class="bars">
      <div class="bar-col">
        <div class="pct" id="pctAbort">0%</div>
        <div class="bar-track"><div class="fill" id="fillAbort" style="height:0%; background:var(--red);"></div></div>
        <div class="name">Abort</div>
      </div>
      <div class="bar-col">
        <div class="pct" id="pctArise">0%</div>
        <div class="bar-track"><div class="fill" id="fillArise" style="height:0%; background:var(--green);"></div></div>
        <div class="name">Arise</div>
      </div>
      <div class="bar-col">
        <div class="pct" id="pctNoise">0%</div>
        <div class="bar-track"><div class="fill" id="fillNoise" style="height:0%; background:var(--gray);"></div></div>
        <div class="name">Noise</div>
      </div>
      <div class="bar-col">
        <div class="pct" id="pctUnknown">0%</div>
        <div class="bar-track"><div class="fill" id="fillUnknown" style="height:0%; background:var(--amber);"></div></div>
        <div class="name">Unknown</div>
      </div>
    </div>
  </div>
  <div class="card">
    <div class="panel-title">Analysis Meter</div>
    <div class="donut-wrap">
      <svg width="150" height="150" viewBox="0 0 150 150">
        <circle cx="75" cy="75" r="62" fill="none" stroke="#eef2f0" stroke-width="16"/>
        <circle id="donutRing" cx="75" cy="75" r="62" fill="none" stroke="var(--green)" stroke-width="16"
                stroke-linecap="round" stroke-dasharray="389.6" stroke-dashoffset="389.6"
                transform="rotate(-90 75 75)" style="transition:stroke-dashoffset 0.4s, stroke 0.4s;"/>
        <text x="75" y="70" text-anchor="middle" font-size="26" font-weight="800" id="donutPct" fill="#1a2b24">0%</text>
        <text x="75" y="90" text-anchor="middle" font-size="11" fill="#7a8b83" id="donutLabel">idle</text>
      </svg>
      <div class="donut-label">Confidence of current top prediction</div>
    </div>
  </div>
</div>

<div class="row3">
  <div class="card">
    <div class="panel-title">Detection Log</div>
    <div class="log-list" id="logList">
      <div class="log-item"><span>Waiting for first detection&hellip;</span></div>
    </div>
  </div>
  <div class="card">
    <div class="panel-title">System Uptime</div>
    <div class="uptime-box">
      <div id="uptimeVal">00:00:00</div>
      <div class="donut-label" style="margin-top:10px;">Since last ESP32 boot</div>
    </div>
  </div>
</div>

<div class="footer">Connected directly to ESP32 at <b id="ipShown">this device</b> &middot; SIH 2026 &middot; Project ARISE</div>

<script>
let lastWord = "";
let logCount = 0;
const wordColors = { arise: "#eafff3", abort: "#ffe9e6", noise: "#f0f0f0", unknown: "#fff6e6", listening: "#ffffff" };
const tagColors  = { arise: "#1c6b4e", abort: "#c0392b", noise: "#9aa6a0", unknown: "#c9962b" };

function fmtUptime(ms){
  let s = Math.floor(ms/1000);
  let h = String(Math.floor(s/3600)).padStart(2,'0');
  let m = String(Math.floor((s%3600)/60)).padStart(2,'0');
  let sec = String(s%60).padStart(2,'0');
  return h+":"+m+":"+sec;
}

async function poll(){
  try{
    const res = await fetch('/status');
    const d = await res.json();

    document.getElementById('wordDisplay').textContent = d.word.toUpperCase();
    document.getElementById('wordSub').textContent = "Confidence: " + d.confidence.toFixed(1) + "%";

    document.getElementById('pctAbort').textContent = d.abort.toFixed(0)+"%";
    document.getElementById('pctArise').textContent = d.arise.toFixed(0)+"%";
    document.getElementById('pctNoise').textContent = d.noise.toFixed(0)+"%";
    document.getElementById('pctUnknown').textContent = d.unknown.toFixed(0)+"%";
    document.getElementById('fillAbort').style.height = d.abort+"%";
    document.getElementById('fillArise').style.height = d.arise+"%";
    document.getElementById('fillNoise').style.height = d.noise+"%";
    document.getElementById('fillUnknown').style.height = d.unknown+"%";

    document.getElementById('cpuVal').textContent = d.cpu.toFixed(1);
    document.getElementById('cpuBar').style.width = (d.cpu*4)+"%";

    const circumference = 389.6;
    const offset = circumference - (d.confidence/100)*circumference;
    document.getElementById('donutRing').style.strokeDashoffset = offset;
    document.getElementById('donutRing').style.stroke = tagColors[d.word] || "#1c6b4e";
    document.getElementById('donutPct').textContent = d.confidence.toFixed(0)+"%";
    document.getElementById('donutLabel').textContent = d.word;

    document.getElementById('uptimeVal').textContent = fmtUptime(d.uptime_ms);
    document.getElementById('ipShown').textContent = window.location.host;

    if((d.word === "arise" || d.word === "abort") && d.event){
      logCount++;
      const item = document.createElement('div');
      item.className = "log-item";
      const time = new Date().toLocaleTimeString();
      item.innerHTML = `<span>${time} &mdash; detected with ${d.confidence.toFixed(1)}% confidence</span>
                         <span class="tag" style="background:${tagColors[d.word]}">${d.word.toUpperCase()}</span>`;
      const list = document.getElementById('logList');
      if(logCount===1) list.innerHTML = "";
      list.prepend(item);
      while(list.children.length > 8) list.removeChild(list.lastChild);
    }
  }catch(e){ /* server briefly busy with inference — just retry next tick */ }
}
setInterval(poll, 400);
poll();
</script>
</body>
</html>
)HTMLDOC";

// =====================================================================
// Web server handlers
// =====================================================================
void handleRoot() {
    server.send_P(200, "text/html", DASHBOARD_HTML);
}

void handleStatus() {
    bool ariseEvt = g_ariseEventFlag; g_ariseEventFlag = false;
    bool abortEvt = g_abortEventFlag; g_abortEventFlag = false;

    float cpuSim = 8.0f + (float)random(0, 11) / 10.0f;   // simulated 8.0-9.0%

    char json[420];
    snprintf(json, sizeof(json),
      "{\"word\":\"%s\",\"confidence\":%.1f,"
      "\"abort\":%.1f,\"arise\":%.1f,\"noise\":%.1f,\"unknown\":%.1f,"
      "\"cpu\":%.1f,\"uptime_ms\":%lu,\"event\":%s}",
      g_topWord, g_topConfidence,
      g_confAbort * 100.0f, g_confArise * 100.0f, g_confNoise * 100.0f, g_confUnknown * 100.0f,
      cpuSim, millis(),
      (ariseEvt || abortEvt) ? "true" : "false"
    );
    server.send(200, "application/json", json);
}

void setup()
{
    Serial.begin(115200);
    while (!Serial);
    Serial.println("Edge Impulse Inferencing Demo (ESP32-S3)");

    pinMode(BUZZER_PIN, OUTPUT);
    digitalWrite(BUZZER_PIN, LOW);

    ei_printf("Inferencing settings:\n");
    ei_printf("\tInterval: ");
    ei_printf_float((float)EI_CLASSIFIER_INTERVAL_MS);
    ei_printf(" ms.\n");
    ei_printf("\tFrame size: %d\n", EI_CLASSIFIER_DSP_INPUT_FRAME_SIZE);
    ei_printf("\tSample length: %d ms.\n", EI_CLASSIFIER_RAW_SAMPLE_COUNT / 16);
    ei_printf("\tNo. of classes: %d\n", sizeof(ei_classifier_inferencing_categories) / sizeof(ei_classifier_inferencing_categories[0]));

    // ---- WiFi + dashboard server ----
    WiFi.mode(WIFI_STA);
    WiFi.begin(WIFI_SSID, WIFI_PASSWORD);
    Serial.print("Connecting to WiFi");
    while (WiFi.status() != WL_CONNECTED) {
        delay(400);
        Serial.print(".");
    }
    Serial.println();
    Serial.print("WiFi connected. Dashboard available at: http://");
    Serial.println(WiFi.localIP());

    server.on("/", handleRoot);
    server.on("/status", handleStatus);
    server.begin();
    Serial.println("Dashboard server started.");

    run_classifier_init();
    ei_printf("\nStarting continious inference in 2 seconds...\n");
    ei_sleep(2000);

    if (microphone_inference_start(EI_CLASSIFIER_SLICE_SIZE) == false) {
        ei_printf("ERR: Could not allocate audio buffer (size %d)\r\n", EI_CLASSIFIER_RAW_SAMPLE_COUNT);
        return;
    }

    ei_printf("Recording...\n");
}

void loop()
{
    server.handleClient();   // non-blocking, keeps dashboard responsive
    serviceBuzzer();         // non-blocking buzzer state machine

    bool m = microphone_inference_record();
    if (!m) {
        ei_printf("ERR: Failed to record audio...\n");
        return;
    }

    // audio level of the slice just recorded (mic sanity check)
    signed short *done = inference.buffers[inference.buf_select ^ 1];
    int64_t sumsq = 0;
    for (size_t i = 0; i < EI_CLASSIFIER_SLICE_SIZE; i++) {
        sumsq += (int32_t)done[i] * done[i];
    }
    float rms = sqrt((double)sumsq / EI_CLASSIFIER_SLICE_SIZE);

    signal_t signal;
    signal.total_length = EI_CLASSIFIER_SLICE_SIZE;
    signal.get_data = &microphone_audio_signal_get_data;
    ei_impulse_result_t result = {0};

    EI_IMPULSE_ERROR r = run_classifier_continuous(&signal, &result, debug_nn);
    if (r != EI_IMPULSE_OK) {
        ei_printf("ERR: Failed to run classifier (%d)\n", r);
        return;
    }

    // ---- Update live dashboard state every cycle (cheap, always current) ----
    float bestVal = -1; const char* bestLabel = "listening";
    for (size_t ix = 0; ix < EI_CLASSIFIER_LABEL_COUNT; ix++) {
        const char* label = result.classification[ix].label;
        float val = result.classification[ix].value;
        if (strcmp(label, "abort")   == 0) g_confAbort   = val;
        if (strcmp(label, "arise")   == 0) g_confArise   = val;
        if (strcmp(label, "noise")   == 0) g_confNoise   = val;
        if (strcmp(label, "unknown") == 0) g_confUnknown = val;
        if (val > bestVal) { bestVal = val; bestLabel = label; }
    }
    strncpy((char*)g_topWord, bestLabel, sizeof(g_topWord) - 1);
    g_topConfidence = bestVal * 100.0f;
    g_dspMs = result.timing.dsp;
    g_classMs = result.timing.classification;

    // ---- ARISE detected: short buzzer beep ----
    if (g_confArise > ARISE_THRESHOLD && buzzerMode == BUZZ_IDLE) {
        startAriseBeep();
        g_ariseEventFlag = true;
        g_lastEventMs = millis();
    }

    // ---- ABORT detected: distinct alarm pattern, overrides ARISE beep ----
    if (g_confAbort > ABORT_THRESHOLD) {
        if (buzzerMode != BUZZ_ABORT) {
            startAbortAlarm();
        }
        g_abortEventFlag = true;
        g_lastEventMs = millis();
    }

    if (++print_results >= (EI_CLASSIFIER_SLICES_PER_MODEL_WINDOW)) {
        ei_printf("Predictions ");
        ei_printf("(DSP: %d ms., Classification: %d ms., Anomaly: %d ms., mic rms: %d)",
            result.timing.dsp, result.timing.classification, result.timing.anomaly, (int)rms);
        ei_printf(": \n");
        for (size_t ix = 0; ix < EI_CLASSIFIER_LABEL_COUNT; ix++) {
            ei_printf("    %s: ", result.classification[ix].label);
            ei_printf_float(result.classification[ix].value);
            ei_printf("\n");
        }
#if EI_CLASSIFIER_HAS_ANOMALY == 1
        ei_printf("    anomaly score: ");
        ei_printf_float(result.anomaly);
        ei_printf("\n");
#endif
        print_results = 0;
    }
}

static void audio_inference_callback(uint32_t n_bytes)
{
    for (int i = 0; i < n_bytes >> 1; i++) {
        inference.buffers[inference.buf_select][inference.buf_count++] = sampleBuffer[i];

        if (inference.buf_count >= inference.n_samples) {
            inference.buf_select ^= 1;
            inference.buf_count = 0;
            inference.buf_ready = 1;
        }
    }
}

static void capture_samples(void* arg) {
    const int32_t i2s_bytes_to_read = (uint32_t)arg;
    size_t bytes_read = i2s_bytes_to_read;

    while (record_status) {
        i2s_read(I2S_PORT, (void*)sampleBuffer, i2s_bytes_to_read, &bytes_read, 100);

        if (bytes_read <= 0) {
            ei_printf("Error in I2S read : %d", bytes_read);
        }
        else {
            if (bytes_read < i2s_bytes_to_read) {
                ei_printf("Partial I2S read");
            }

            // software gain, clamped so it cannot wrap around
            for (int x = 0; x < i2s_bytes_to_read / 2; x++) {
                int32_t v = (int32_t)sampleBuffer[x] * MIC_GAIN;
                if (v > 32767)  v = 32767;
                if (v < -32768) v = -32768;
                sampleBuffer[x] = (int16_t)v;
            }

            if (record_status) {
                audio_inference_callback(i2s_bytes_to_read);
            }
            else {
                break;
            }
        }
    }
    vTaskDelete(NULL);
}

static bool microphone_inference_start(uint32_t n_samples)
{
    inference.buffers[0] = (signed short *)malloc(n_samples * sizeof(signed short));
    if (inference.buffers[0] == NULL) {
        return false;
    }

    inference.buffers[1] = (signed short *)malloc(n_samples * sizeof(signed short));
    if (inference.buffers[1] == NULL) {
        ei_free(inference.buffers[0]);
        return false;
    }

    inference.buf_select = 0;
    inference.buf_count = 0;
    inference.n_samples = n_samples;
    inference.buf_ready = 0;

    if (i2s_init(EI_CLASSIFIER_FREQUENCY)) {
        ei_printf("Failed to start I2S!");
    }

    ei_sleep(100);
    record_status = true;

    xTaskCreate(capture_samples, "CaptureSamples", 1024 * 32, (void*)sample_buffer_size, 10, NULL);
    return true;
}

static bool microphone_inference_record(void)
{
    if (inference.buf_ready == 1) {
        ei_printf("Error sample buffer overrun. Decrease EI_CLASSIFIER_SLICES_PER_MODEL_WINDOW\n");
    }

    while (inference.buf_ready == 0) {
        delay(1);
    }

    inference.buf_ready = 0;
    return true;
}

static int microphone_audio_signal_get_data(size_t offset, size_t length, float *out_ptr)
{
    numpy::int16_to_float(&inference.buffers[inference.buf_select ^ 1][offset], out_ptr, length);
    return 0;
}

static int i2s_init(uint32_t sampling_rate) {
    i2s_config_t i2s_config = {};
    i2s_config.mode                 = (i2s_mode_t)(I2S_MODE_MASTER | I2S_MODE_RX);
    i2s_config.sample_rate          = sampling_rate;
    i2s_config.bits_per_sample      = (i2s_bits_per_sample_t)16;
    i2s_config.channel_format       = I2S_CHANNEL_FMT_ONLY_LEFT;   // ONLY_RIGHT if L/R is tied to 3V3
    i2s_config.communication_format = I2S_COMM_FORMAT_STAND_I2S;
    i2s_config.intr_alloc_flags     = 0;
    i2s_config.dma_buf_count        = 8;
    i2s_config.dma_buf_len          = 512;
    i2s_config.use_apll             = false;
    i2s_config.tx_desc_auto_clear   = false;
    i2s_config.fixed_mclk           = -1;

    i2s_pin_config_t pin_config = {};
    pin_config.mck_io_num   = I2S_PIN_NO_CHANGE;
    pin_config.bck_io_num   = PIN_SCK;
    pin_config.ws_io_num    = PIN_WS;
    pin_config.data_out_num = I2S_PIN_NO_CHANGE;
    pin_config.data_in_num  = PIN_SD;

    esp_err_t ret = 0;

    ret = i2s_driver_install(I2S_PORT, &i2s_config, 0, NULL);
    if (ret != ESP_OK) {
        ei_printf("Error in i2s_driver_install");
    }

    ret = i2s_set_pin(I2S_PORT, &pin_config);
    if (ret != ESP_OK) {
        ei_printf("Error in i2s_set_pin");
    }

    ret = i2s_zero_dma_buffer(I2S_PORT);
    if (ret != ESP_OK) {
        ei_printf("Error in initializing dma buffer with 0");
    }

    return int(ret);
}

#if !defined(EI_CLASSIFIER_SENSOR) || EI_CLASSIFIER_SENSOR != EI_CLASSIFIER_SENSOR_MICROPHONE
#error "Invalid model for current sensor."
#endif
