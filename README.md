# Vibes — Bearing Health Monitor

Binary classification of bearing vibration signals as `h` (healthy) or `f` (faulty), using per-file CSVs of `t, x, y, z` accelerometer data sampled at 1 kHz. Signals are fed to a trained `StandardScaler → LogisticRegression` pipeline, results are persisted to PostgreSQL, and a real-time dashboard tracks machine health status.

---

## Architecture

```
┌──────────────────────────────────┐
│  MicroPico  (vibes-simulation    │
│  simulates this)                 │
│                                  │
│  Sample accelerometer @ 1 kHz   │
│  Compute max/min/mean per axis   │
│  POST features as JSON           │
└──────────────────┬───────────────┘
                   │ POST /api/predict
                   ▼
┌──────────────────────────────────┐
│  vibes  (port 8000)              │
│                                  │
│  Run inference (LogReg)          │
│  Persist result to PostgreSQL    │
│  Serve dashboard + REST API      │
└──────────────────────────────────┘
                   ▲
                   │ GET /api/machines/status
                   │ GET /api/machines/{id}/history
                   │
                Browser
```

`vibes-simulation` is a browser-based stand-in for the Pico: it picks a random test CSV, extracts the same features the Pico would, and calls `POST /api/predict`. It shares the `data/` directory with this repo (`../vibes/data/`).

---

## Repository Layout

```
vibes/
├── api/
│   ├── predict.py          # POST /api/predict
│   └── status.py           # GET  /api/machines/status  +  history
├── core/
│   ├── features.py         # Feature extraction (time-domain + optional FFT)
│   └── model.py            # Model loading and inference
├── db/
│   └── database.py         # PostgreSQL connection + queries
├── sql/
│   ├── schema.sql
│   ├── insert_prediction.sql
│   ├── get_history.sql
│   └── get_machine_status.sql
├── sandbox/
│   └── create_train_test_data.py
├── static/css/style.css
├── templates/
│   ├── base.html
│   └── dashboard.html
├── data/
│   ├── archive/Healthy/    # Raw .mat files
│   ├── archive/Faulty/
│   ├── htrain/ htest/      # Healthy CSVs
│   └── ftrain/ ftest/      # Faulty CSVs
├── main.py
├── train_classifier.py
├── constants.py
└── requirements.txt
```

---

## Setup

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

Create a `.env` file:

```
SUPABASE_URL=postgresql://postgres.[project-ref]:[password]@aws-0-[region].pooler.supabase.com:6543/postgres
```

Run the database schema once:

```bash
psql $SUPABASE_URL -f sql/schema.sql
```

---

## Data Preparation

### `sandbox/create_train_test_data.py`

Reads `.mat` files from `data/archive/Healthy/` and `data/archive/Faulty/`, converts each to a DataFrame with a `t` column (1 kHz), and splits it into 5 time-ordered chunks written as CSVs. Each `.mat` file is assigned entirely to train or test (default 80/20 via `TEST_FRACTION`) so chunks from one recording never straddle the split.

```bash
.venv/bin/python sandbox/create_train_test_data.py
```

---

## Training

### `train_classifier.py`

Builds a feature table from the train/test CSV folders, fits a `StandardScaler → LogisticRegression` pipeline on train, and reports F1 + confusion matrix on test.

Features per axis (`x`, `y`, `z`): `max`, `min`, `mean` (9 total). Set `USE_TOP_FREQS = True` to add the top-3 FFT peak angular frequencies per axis (18 features). Time-domain only wins on this dataset: macro F1 ≈ 0.995 vs ≈ 0.945 with FFT features.

Set `SAVE_MODEL = True` to persist a `model.pkl` bundle `{model, feature_cols, use_top_freqs, fs, top_k}`.

```bash
.venv/bin/python train_classifier.py
```

---

## Running the Server

```bash
.venv/bin/uvicorn main:app --reload --port 8000
```

Dashboard: `http://localhost:8000`

---

## API Reference

### `POST /api/predict`

Submit a feature vector for a machine. Returns the predicted label and class probabilities.

**Request body (JSON):**

```json
{
  "machine_id": "machine_1",
  "filename": "recording_042.csv",
  "features": {
    "x_max": 1.23,
    "x_min": -0.87,
    "x_mean": 0.03,
    "y_max": 0.95,
    "y_min": -1.01,
    "y_mean": -0.01,
    "z_max": 9.91,
    "z_min": 9.72,
    "z_mean": 9.81
  }
}
```

| Field        | Type   | Description                                      |
|--------------|--------|--------------------------------------------------|
| `machine_id` | string | Identifier for the machine (e.g. `"machine_1"`) |
| `filename`   | string | Source filename, stored for traceability         |
| `features`   | object | Dict of feature name → float value              |

**Response (200):**

```json
{
  "machine_id": "machine_1",
  "filename": "recording_042.csv",
  "predicted_label": "h",
  "probabilities": {
    "f": 0.0021,
    "h": 0.9979
  }
}
```

**Error (400):** returned when feature keys don't match the trained model's expected columns.

---

### `GET /api/machines/status`

Returns the current health status of all registered machines, derived from consecutive fault predictions.

**Response (200):**

```json
[
  {
    "machine_id": "machine_1",
    "status": "green",
    "consecutive_faults": 0,
    "last_prediction": "h",
    "last_seen": "2025-06-10T14:22:01Z"
  },
  {
    "machine_id": "machine_2",
    "status": "red",
    "consecutive_faults": 7,
    "last_prediction": "f",
    "last_seen": "2025-06-10T14:23:45Z"
  }
]
```

**Status logic:**

| `consecutive_faults` | `status`  |
|----------------------|-----------|
| < 3                  | `"green"` |
| 3 – 4                | `"yellow"`|
| ≥ 5                  | `"red"`   |

---

### `GET /api/machines/{machine_id}/history`

Returns the prediction history for a single machine.

**Path parameter:** `machine_id` — e.g. `machine_1`

**Query parameter:** `limit` (int, default 100) — number of records to return, ordered newest-first.

**Response (200):**

```json
[
  {
    "id": "uuid-...",
    "machine_id": "machine_1",
    "filename": "recording_042.csv",
    "predicted_label": "h",
    "probabilities": { "f": 0.0021, "h": 0.9979 },
    "features": { "x_max": 1.23, "..." : "..." },
    "created_at": "2025-06-10T14:22:01Z"
  }
]
```

---

## Database Schema

```sql
CREATE TABLE predictions (
    id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    machine_id  TEXT NOT NULL,
    filename    TEXT,
    features    JSONB,
    predicted_label TEXT NOT NULL,
    probabilities   JSONB,
    created_at  TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_predictions_machine_time
    ON predictions (machine_id, created_at DESC);
```

---

## Dashboard

The web dashboard at `http://localhost:8000` shows:

- **Status cards** — one per machine, colour-coded green / yellow / red with consecutive fault count.
- **History tab** — last 100 predictions per machine with probability bars.

No build step: Tailwind CSS is loaded via CDN. The frontend polls the status endpoint periodically using vanilla `fetch`.

---

---

## vibes-simulation

`vibes-simulation` is a companion test harness that lets you fire synthetic signals at the vibes API without needing physical hardware.  It runs as a separate FastAPI service on **port 8001** and shares the `data/` directory from this repo.

### Setup

```bash
cd ../vibes-simulation
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

Create a `.env` (see `.env.example`):

```
VIBES_API_URL=https://vibes-lemon.vercel.app
```

```bash
.venv/bin/uvicorn main:app --reload --port 8001
```

Open `http://localhost:8001` for the interactive simulator UI.

---

### `POST /api/simulate`

Picks a random CSV from `../vibes/data/ftest/` or `../vibes/data/htest/`, extracts features, forwards them to the vibes `/api/predict` endpoint, and returns the result plus a rendered signal plot.

**Request body (JSON):**

```json
{
  "machine_id": "machine_1",
  "signal_type": "f"
}
```

| Field         | Type   | Values         | Description                        |
|---------------|--------|----------------|------------------------------------|
| `machine_id`  | string | any            | Passed through to vibes API        |
| `signal_type` | string | `"f"` or `"h"` | Which test pool to sample from     |

**Response (200):**

```json
{
  "filename": "Faulty_bearing_003_chunk2.csv",
  "predicted_label": "f",
  "probabilities": {
    "f": 0.9873,
    "h": 0.0127
  },
  "plot_base64": "<base64-encoded PNG>"
}
```

The `plot_base64` field is a matplotlib figure showing all three accelerometer channels (x = red, y = green, z = blue) rendered as a dark-theme PNG, ready to drop into an `<img src="data:image/png;base64,…">` tag.

---

### Simulator UI

The browser UI at `http://localhost:8001` has two buttons per configured machine:

- **Send Faulty** — picks a random CSV from `ftest/` and submits it.
- **Send Healthy** — picks a random CSV from `htest/` and submits it.

Results appear inline: prediction badge, probability bars, and the 3-channel signal plot.

---

---

## MicroPico + Vibration Sensor Integration

> **Planned hardware path.** This section describes the intended integration once a MicroPico with an I²C/SPI accelerometer is available.

### How it will work

A [MicroPico](https://www.raspberrypi.com/products/raspberry-pi-pico/) running MicroPython will:

1. **Sample** the accelerometer (e.g. MPU-6050, ADXL345, or LIS3DH) at 1 kHz on all three axes.
2. **Buffer** a fixed-length window (e.g. 1 000 samples = 1 second).
3. **Extract features** on-chip — per axis: `max`, `min`, `mean` (9 floats total). These mirror the feature set in `core/features.py`.
4. **POST** the feature vector to `POST /api/predict` over Wi-Fi (Pico W) or via a serial-to-HTTP bridge.
5. Repeat on a configurable interval (e.g. every 5 seconds).

No raw waveform needs to leave the device; only the 9 feature values travel over the wire.

### Example MicroPython sketch (pseudocode)

```python
import network, urequests, ujson, time
from machine import I2C, Pin
from imu import MPU6050  # community driver

VIBES_URL = "http://192.168.1.100:8000/api/predict"
MACHINE_ID = "machine_pico_1"
FS = 1000          # Hz
WINDOW = 1000      # samples

wlan = network.WLAN(network.STA_IF)
wlan.active(True)
wlan.connect("SSID", "PASSWORD")
while not wlan.isconnected():
    time.sleep(0.1)

i2c = I2C(0, sda=Pin(4), scl=Pin(5))
imu = MPU6050(i2c)

def collect_window():
    xs, ys, zs = [], [], []
    for _ in range(WINDOW):
        ax, ay, az = imu.accel.xyz
        xs.append(ax); ys.append(ay); zs.append(az)
        time.sleep_us(1000)          # ~1 kHz
    return xs, ys, zs

def extract_features(xs, ys, zs):
    def stats(v):
        return max(v), min(v), sum(v) / len(v)
    xmx,xmn,xmu = stats(xs)
    ymx,ymn,ymu = stats(ys)
    zmx,zmn,zmu = stats(zs)
    return {
        "x_max":xmx,"x_min":xmn,"x_mean":xmu,
        "y_max":ymx,"y_min":ymn,"y_mean":ymu,
        "z_max":zmx,"z_min":zmn,"z_mean":zmu,
    }

while True:
    xs, ys, zs = collect_window()
    features = extract_features(xs, ys, zs)
    payload = {
        "machine_id": MACHINE_ID,
        "filename": "live",
        "features": features,
    }
    res = urequests.post(VIBES_URL, json=payload)
    print(res.json())
    time.sleep(5)
```

### Wiring (MPU-6050 example)

| MPU-6050 pin | Pico W pin      |
|--------------|-----------------|
| VCC          | 3V3 (pin 36)    |
| GND          | GND (pin 38)    |
| SDA          | GP4 (pin 6)     |
| SCL          | GP5 (pin 7)     |
| AD0          | GND (addr 0x68) |

### Notes

- **Feature parity is critical.** The MicroPico must compute the same 9 features in the same order as `core/features.py`. If the model is retrained with FFT features (`USE_TOP_FREQS = True`), the device firmware must be updated accordingly.
- **Sampling jitter.** `time.sleep_us(1000)` is approximate on MicroPython. For high-accuracy results, use a hardware timer interrupt to trigger reads at exactly 1 kHz.
- **Offline resilience.** If the POST fails (network drop), buffer the feature vector in flash and retry on the next cycle.
- **TLS.** For production, use HTTPS and add the server's certificate to `urequests`'s CA bundle, or run an NGINX proxy on the server side.
