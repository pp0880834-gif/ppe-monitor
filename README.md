# PPE Violation Monitoring System

A real-time Personal Protective Equipment (PPE) monitoring system that detects helmet and safety vest violations using YOLOv11 object detection, multi-object tracking, and automated Telegram alerting.

## What This Does

This system ingests one or more video streams (files, webcams, or RTSP feeds) and continuously checks whether every detected person is wearing a hard hat and a safety vest. When someone is found violating PPE rules for a configurable duration (default 3 seconds), the system captures a snapshot, logs the incident to a CSV file with a unique incident ID, and sends a photo alert to a configured Telegram chat.

The system supports multi-camera setups by running one thread per source, shares a single CSV log and Telegram worker across all threads, and implements a per-person cooldown mechanism to avoid alert spam. It can be run in **dry-run** mode to validate detection and logging without sending any Telegram messages — ideal for initial tuning.

---

## Installation

```bash
pip install -r requirements.txt
```

> **Python 3.9+** is recommended. A CUDA-capable GPU will significantly improve throughput but is not required.

---

## Telegram Bot Setup

Follow these steps to receive alerts on Telegram:

1. Open Telegram and search for **@BotFather**.
2. Send the command `/newbot` and follow the prompts to choose a name for your bot.
3. BotFather will reply with your **bot token** (looks like `123456789:ABC-DEFghiJKLmnopQRSTuvwxyz`).
4. **Get your chat ID:**
   - Send any message to your new bot from the account/group you want alerts delivered to.
   - Open this URL in a browser (replace `{TOKEN}` with your actual token):
     ```
     https://api.telegram.org/bot{TOKEN}/getUpdates
     ```
   - Find the `message.chat.id` field in the JSON response. For groups it will be a negative number.
5. **Set environment variables** (PowerShell — preferred over hardcoding credentials):
   ```powershell
   $env:TELEGRAM_BOT_TOKEN = "123456789:ABC-DEFghiJKLmnopQRSTuvwxyz"
   $env:TELEGRAM_CHAT_ID   = "-100123456789"
   ```
   These override any values in `config.yaml` and are read at startup.

---

## Configuration

Open `config.yaml` and adjust as needed:

| Section | Key | Description |
|---|---|---|
| `model` | `path` | Path to the YOLO `.pt` model file. Auto-downloaded from HuggingFace if missing. |
| `model` | `conf` | Detection confidence threshold (0–1). Lower = more detections, more false positives. |
| `model` | `imgsz` | Inference image size in pixels. 640 is a good balance of speed and accuracy. |
| `model` | `device` | Compute device: `""` = auto, `"cpu"`, `"0"` = first GPU, `"cuda:0"`. |
| `classes` | `person` | Model class name for persons. |
| `classes` | `helmet` | Model class name for a worn hard hat. |
| `classes` | `no_helmet` | Model class name for the absence of a hard hat (if your model provides it). |
| `classes` | `vest` | Model class name for a worn safety vest. |
| `classes` | `no_vest` | Model class name for the absence of a vest. Remove if your model lacks this class. |
| `violation` | `duration` | Seconds a person must be in violation before an alert fires. |
| `violation` | `grace_period` | Seconds to tolerate a missed detection before resetting the violation timer. |
| `violation` | `cooldown` | Seconds before the same person can trigger another alert for the same violation type. |
| `sources` | — | List of `{screen, source}` entries. `source` can be a file path, webcam index, or RTSP URL. |
| `telegram` | `bot_token` | Bot token (prefer environment variable). |
| `telegram` | `chat_id` | Destination chat or group ID (prefer environment variable). |
| `output` | `csv_path` | Path to the output CSV log file. |
| `output` | `snapshots_dir` | Directory where violation snapshot images are saved. |

---

## Running

```bash
# Dry run (no Telegram, still saves CSV + snapshots)
python main.py --dry-run

# Dry run with live display window
python main.py --dry-run --show

# Production mode
python main.py

# With live display
python main.py --show

# Custom config file
python main.py --config production.yaml
```

Press **Q** or **Escape** in any display window to gracefully stop all streams.  
Press **Ctrl+C** in the terminal to stop.

---

## Output Files

### `violations.csv`

| Column | Description |
|---|---|
| `incident_id` | Auto-incrementing unique ID for each alert. Resumes from the last ID on restart. |
| `track_id` | ByteTrack persistent ID of the detected person. |
| `screen` | Source screen number from config. |
| `violation_type` | `NO_HELMET` or `NO_VEST`. |
| `timestamp` | UTC ISO-8601 timestamp when the alert fired. |
| `image_path` | Absolute path to the saved snapshot image. |
| `telegram_sent` | Status: `pending` → `sent` / `dry_run` / `failed` / `no_credentials` / `error`. |

### `snapshots/`

Each violation snapshot is saved as a JPEG:

```
snapshots/incident_00001_screen1_tid3_NO_HELMET.jpg
```

The image shows the annotated frame at the moment of alert, with the offending person highlighted in red.

---

## Switching Models

1. Set `model.path` in `config.yaml` to your new `.pt` file.
2. Update the `classes` section to match your model's **exact class names** (case-insensitive matching is applied automatically).
3. If your model doesn't have explicit `no_helmet` or `no_vest` classes, remove those keys — the system will infer violations from the absence of the positive class.

---

## Multi-Source Setup

Add entries to the `sources` list in `config.yaml`:

```yaml
sources:
  - screen: 1
    source: "rtsp://192.168.1.10:554/cam1"   # IP camera
  - screen: 2
    source: "rtsp://192.168.1.11:554/cam2"   # Second IP camera
  - screen: 3
    source: 0                                 # Local webcam (index 0)
  - screen: 4
    source: "C:/Videos/site_recording.mp4"   # Pre-recorded file
```

Each source runs in its own thread and shares the same CSV logger, Telegram worker, and violation cooldown tracking.
