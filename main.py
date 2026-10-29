#!/usr/bin/env python3
"""PPE Violation Monitoring System — Helmet & Safety Vest detection with real-time alerts."""

import argparse
import csv
import os
import queue
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

import cv2
import requests
import yaml
from ultralytics import YOLO


# ---------------------------------------------------------------------------
# Thread-safe print
# ---------------------------------------------------------------------------
print_lock = threading.Lock()


def tprint(*args, **kwargs):
    """Thread-safe print."""
    with print_lock:
        print(*args, **kwargs)


# ---------------------------------------------------------------------------
# Config loader
# ---------------------------------------------------------------------------

def load_config(path: str) -> dict:
    """Load YAML config and apply environment variable overrides."""
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # Environment variable overrides for Telegram credentials
    bot_token_env = os.environ.get("TELEGRAM_BOT_TOKEN", "")
    chat_id_env = os.environ.get("TELEGRAM_CHAT_ID", "")
    if bot_token_env:
        cfg.setdefault("telegram", {})["bot_token"] = bot_token_env
    if chat_id_env:
        cfg.setdefault("telegram", {})["chat_id"] = chat_id_env

    # Auto-download model if path doesn't exist
    model_path = cfg.get("model", {}).get("path", "best.pt")
    if not Path(model_path).exists():
        tprint(f"[Config] Model '{model_path}' not found. Auto-downloading from HuggingFace...")
        try:
            from huggingface_hub import hf_hub_download
            downloaded = hf_hub_download(
                "wesjos/Yolo-hard-hat-safety-vest",
                "yolo11m_safety.pt"
            )
            cfg["model"]["path"] = downloaded
            tprint(f"[Config] Model downloaded to: {downloaded}")
        except Exception as e:
            tprint(f"[Config] WARNING: Could not auto-download model: {e}")
            tprint(f"[Config] Will attempt to use '{model_path}' anyway.")

    return cfg


# ---------------------------------------------------------------------------
# CSV Logger
# ---------------------------------------------------------------------------

class CSVLogger:
    """Thread-safe CSV logger for violation incidents."""

    fieldnames = [
        "incident_id", "track_id", "screen", "violation_type",
        "timestamp", "image_path", "telegram_sent"
    ]

    def __init__(self, csv_path: str):
        self._path = csv_path
        self._lock = threading.Lock()
        self._next_id = 1

        p = Path(csv_path)
        if p.exists() and p.stat().st_size > 0:
            # Read the last incident_id to resume numbering
            try:
                with open(csv_path, "r", newline="") as f:
                    reader = csv.DictReader(f)
                    rows = list(reader)
                    if rows:
                        last_id = max(int(r["incident_id"]) for r in rows if r["incident_id"].isdigit())
                        self._next_id = last_id + 1
            except Exception:
                pass
        else:
            # Create file with header
            with open(csv_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=self.fieldnames)
                writer.writeheader()

    def next_id(self) -> int:
        """Atomically increment and return the next incident ID."""
        with self._lock:
            _id = self._next_id
            self._next_id += 1
            return _id

    def log(self, record: dict):
        """Append a row to the CSV."""
        with self._lock:
            with open(self._path, "a", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=self.fieldnames)
                writer.writerow(record)

    def update_telegram_sent(self, incident_id: int, status: str):
        """Rewrite the row matching incident_id with updated telegram_sent status."""
        with self._lock:
            try:
                with open(self._path, "r", newline="") as f:
                    rows = list(csv.DictReader(f))
                for row in rows:
                    if row.get("incident_id") == str(incident_id):
                        row["telegram_sent"] = status
                with open(self._path, "w", newline="") as f:
                    writer = csv.DictWriter(f, fieldnames=self.fieldnames)
                    writer.writeheader()
                    writer.writerows(rows)
            except Exception as e:
                tprint(f"[CSVLogger] WARNING: Could not update incident #{incident_id}: {e}")


# ---------------------------------------------------------------------------
# Telegram Worker
# ---------------------------------------------------------------------------

class TelegramWorker:
    """Background worker that sends Telegram photo alerts."""

    def __init__(self, bot_token: str, chat_id: str, dry_run: bool = False):
        self._token = bot_token
        self._chat_id = chat_id
        self._dry_run = dry_run
        self._queue: queue.Queue = queue.Queue()
        self._thread = threading.Thread(target=self._run, daemon=True, name="TelegramWorker")
        self._thread.start()

    def _run(self):
        while True:
            item = self._queue.get()
            if item is None:
                self._queue.task_done()
                break
            image_path, caption, incident_id, csv_logger = item
            try:
                self._send_one(image_path, caption, incident_id, csv_logger)
            except Exception as e:
                tprint(f"[Telegram] Unexpected error for incident #{incident_id}: {e}")
                csv_logger.update_telegram_sent(incident_id, "error")
            finally:
                self._queue.task_done()

    def _send_one(self, image_path: str, caption: str, incident_id: int, csv_logger):
        if self._dry_run:
            tprint(f"[DRY-RUN] Telegram skipped for incident #{incident_id}")
            csv_logger.update_telegram_sent(incident_id, "dry_run")
            return

        if not self._token or not self._chat_id:
            tprint(f"[Telegram] No credentials configured; skipping incident #{incident_id}")
            csv_logger.update_telegram_sent(incident_id, "no_credentials")
            return

        url = f"https://api.telegram.org/bot{self._token}/sendPhoto"
        for attempt in range(3):
            try:
                with open(image_path, "rb") as photo:
                    resp = requests.post(
                        url,
                        data={"chat_id": self._chat_id, "caption": caption},
                        files={"photo": photo},
                        timeout=15,
                    )
                if resp.status_code == 200:
                    csv_logger.update_telegram_sent(incident_id, "sent")
                    tprint(f"[Telegram] ✅ Sent incident #{incident_id}")
                    return
                else:
                    tprint(f"[Telegram] HTTP {resp.status_code} for incident #{incident_id}, attempt {attempt+1}/3")
            except Exception as e:
                tprint(f"[Telegram] Error on attempt {attempt+1}/3 for incident #{incident_id}: {e}")
            sleep_time = 2 ** attempt
            time.sleep(sleep_time)

        csv_logger.update_telegram_sent(incident_id, "failed")
        tprint(f"[Telegram] ❌ Failed to send incident #{incident_id} after 3 attempts")

    def send(self, image_path: str, caption: str, incident_id: int, csv_logger):
        """Enqueue a send job."""
        self._queue.put((image_path, caption, incident_id, csv_logger))

    def stop(self):
        """Signal the worker to stop after draining the queue."""
        self._queue.put(None)
        self._thread.join(timeout=30)


# ---------------------------------------------------------------------------
# Violation State Machine
# ---------------------------------------------------------------------------

class ViolationState:
    """Tracks violation timing state per (track_id, violation_type)."""

    def __init__(self):
        self._states: dict = {}

    def _get(self, track_id: int, vtype: str) -> dict:
        key = (track_id, vtype)
        if key not in self._states:
            self._states[key] = {
                "start": None,
                "grace_start": None,
                "alerted": False,
                "cooldown_until": 0.0,
            }
        return self._states[key]

    def process(
        self,
        track_id: int,
        vtype: str,
        is_violating: bool,
        now: float,
        duration: float,
        grace: float,
        cooldown: float,
    ) -> bool:
        """
        Update state machine and return True if an alert should fire.
        """
        state = self._get(track_id, vtype)

        if is_violating:
            state["grace_start"] = None
            if state["start"] is None:
                state["start"] = now
            elapsed = now - state["start"]
            if elapsed >= duration and not state["alerted"] and now >= state["cooldown_until"]:
                state["alerted"] = True
                state["cooldown_until"] = now + cooldown
                return True
        else:
            if state["start"] is None:
                return False
            if state["grace_start"] is None:
                state["grace_start"] = now
            if now - state["grace_start"] >= grace:
                state.update({"start": None, "alerted": False, "grace_start": None})
            else:
                elapsed = now - state["start"]
                if elapsed >= duration and not state["alerted"] and now >= state["cooldown_until"]:
                    state["alerted"] = True
                    state["cooldown_until"] = now + cooldown
                    return True

        return False

    def cleanup(self, active_ids: set):
        """Remove states for track IDs no longer active."""
        keys_to_remove = [k for k in self._states if k[0] not in active_ids]
        for k in keys_to_remove:
            del self._states[k]


# ---------------------------------------------------------------------------
# Class resolution
# ---------------------------------------------------------------------------

def resolve_classes(model_names: dict, class_map_cfg: dict) -> dict:
    """
    Build a mapping from semantic role → class index in the model.

    Returns dict with keys: person, helmet, no_helmet, vest, no_vest
    Values are int (class index) or None if not found.
    """
    reverse_map = {name.lower(): idx for idx, name in model_names.items()}

    roles = ["person", "helmet", "no_helmet", "vest", "no_vest"]
    result = {}
    for role in roles:
        cfg_name = class_map_cfg.get(role)
        if cfg_name is None:
            result[role] = None
        else:
            idx = reverse_map.get(cfg_name.lower())
            result[role] = idx  # None if not found

    tprint(f"[Classes] Resolved mapping: {result}")
    tprint(f"[Classes] Model classes: {dict(model_names)}")
    return result


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def box_center(xyxy) -> tuple:
    """Return center (cx, cy) of a bounding box."""
    x1, y1, x2, y2 = xyxy
    return (x1 + x2) / 2, (y1 + y2) / 2


def inside_box(cx: float, cy: float, xyxy) -> bool:
    """Return True if point (cx, cy) is inside the bounding box."""
    x1, y1, x2, y2 = xyxy
    return x1 <= cx <= x2 and y1 <= cy <= y2


# ---------------------------------------------------------------------------
# Frame drawing
# ---------------------------------------------------------------------------

def draw_frame(frame, person_data: list, highlight_tids: set = None):
    """
    Draw bounding boxes and labels on frame in-place.
    Returns the annotated frame.
    """
    if highlight_tids is None:
        highlight_tids = set()

    overlay = frame.copy()

    for person in person_data:
        tid = person["track_id"]
        violations = person.get("violations", [])
        box = person["box"]  # [x1, y1, x2, y2]
        x1, y1, x2, y2 = [int(v) for v in box]

        has_violation = bool(violations) or tid in highlight_tids
        color = (0, 0, 255) if has_violation else (0, 255, 0)
        thickness = 3 if (highlight_tids and tid in highlight_tids) else 2

        # Semi-transparent red overlay for highlighted persons
        if highlight_tids and tid in highlight_tids:
            cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 0, 255), -1)
            cv2.addWeighted(overlay, 0.15, frame, 0.85, 0, frame)

        # Person bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

        # Label
        label = f"ID:{tid}"
        if violations:
            label += " " + " ".join(violations)
        label_y = y1 - 8 if y1 > 20 else y1 + 18
        cv2.putText(
            frame, label, (x1, label_y),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA
        )

        # Equipment sub-boxes
        for equip in person.get("equipment", []):
            ex1, ey1, ex2, ey2 = [int(v) for v in equip["box"]]
            equip_cls = equip["cls_name"]
            # green for worn items, red for missing/violation items
            if equip_cls in ("helmet", "vest"):
                eq_color = (0, 255, 0)
            else:
                eq_color = (0, 0, 255)
            cv2.rectangle(frame, (ex1, ey1), (ex2, ey2), eq_color, 1)

    return frame


# ---------------------------------------------------------------------------
# Per-source worker
# ---------------------------------------------------------------------------

def source_worker(
    screen_num: int,
    source,
    cfg: dict,
    csv_logger: CSVLogger,
    tg_worker: TelegramWorker,
    show: bool,
    stop_event: threading.Event,
    snapshots_dir: str,
):
    """Process a single video source for PPE violations."""

    tprint(f"[Screen {screen_num}] Starting source: {source}")

    # Load model
    model_path = cfg["model"]["path"]
    model = YOLO(model_path)

    class_idx = resolve_classes(model.names, cfg.get("classes", {}))

    person_idx   = class_idx.get("person")
    helmet_idx   = class_idx.get("helmet")
    no_helmet_idx = class_idx.get("no_helmet")
    vest_idx     = class_idx.get("vest")
    no_vest_idx  = class_idx.get("no_vest")

    vstates = ViolationState()

    model_cfg   = cfg.get("model", {})
    conf        = model_cfg.get("conf", 0.35)
    imgsz       = model_cfg.get("imgsz", 640)
    device      = model_cfg.get("device", "")

    viol_cfg    = cfg.get("violation", {})
    duration    = float(viol_cfg.get("duration", 3.0))
    grace       = float(viol_cfg.get("grace_period", 0.5))
    cooldown    = float(viol_cfg.get("cooldown", 60.0))

    # Open video capture
    src = source
    if isinstance(src, str) and src.isdigit():
        src = int(src)

    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        tprint(f"[Screen {screen_num}] ERROR: Cannot open source '{source}'")
        return

    window_name = f"PPE Monitor — Screen {screen_num}"
    highlight_tids: set = set()

    try:
        while not stop_event.is_set():
            ret, frame = cap.read()
            if not ret:
                tprint(f"[Screen {screen_num}] End of stream or read error.")
                break

            now = time.time()

            # Run tracking
            try:
                results = model.track(
                    source=frame,
                    conf=conf,
                    imgsz=imgsz,
                    device=device,
                    tracker="bytetrack.yaml",
                    persist=True,
                    verbose=False,
                )[0]
            except Exception as e:
                tprint(f"[Screen {screen_num}] Tracking error: {e}")
                continue

            if results.boxes is None or results.boxes.id is None:
                annotated = draw_frame(frame, [], highlight_tids)
                if show:
                    cv2.imshow(window_name, annotated)
                    key = cv2.waitKey(1) & 0xFF
                    if key in (ord("q"), 27):
                        stop_event.set()
                continue

            # Parse detections
            boxes_data = []
            for box_obj, cls_tensor, conf_tensor, tid_tensor in zip(
                results.boxes.xyxy,
                results.boxes.cls,
                results.boxes.conf,
                results.boxes.id,
            ):
                boxes_data.append({
                    "cls": int(cls_tensor.item()),
                    "conf": float(conf_tensor.item()),
                    "box": box_obj.tolist(),
                    "track_id": int(tid_tensor.item()),
                })

            # Separate detections by class
            persons    = [b for b in boxes_data if b["cls"] == person_idx]
            helmets    = [b for b in boxes_data if helmet_idx   is not None and b["cls"] == helmet_idx]
            no_helmets = [b for b in boxes_data if no_helmet_idx is not None and b["cls"] == no_helmet_idx]
            vests      = [b for b in boxes_data if vest_idx     is not None and b["cls"] == vest_idx]
            no_vests   = [b for b in boxes_data if no_vest_idx  is not None and b["cls"] == no_vest_idx]

            # Build person_data with associated equipment & violations
            person_data = []
            for person in persons:
                pbox = person["box"]
                tid = person["track_id"]

                assoc_helmets    = []
                assoc_no_helmets = []
                assoc_vests      = []
                assoc_no_vests   = []

                for h in helmets:
                    cx, cy = box_center(h["box"])
                    if inside_box(cx, cy, pbox):
                        assoc_helmets.append(h)

                for nh in no_helmets:
                    cx, cy = box_center(nh["box"])
                    if inside_box(cx, cy, pbox):
                        assoc_no_helmets.append(nh)

                for v in vests:
                    cx, cy = box_center(v["box"])
                    if inside_box(cx, cy, pbox):
                        assoc_vests.append(v)

                for nv in no_vests:
                    cx, cy = box_center(nv["box"])
                    if inside_box(cx, cy, pbox):
                        assoc_no_vests.append(nv)

                violations = []
                if assoc_no_helmets or not assoc_helmets:
                    violations.append("NO_HELMET")
                if assoc_no_vests or not assoc_vests:
                    violations.append("NO_VEST")

                # Build equipment list for drawing
                equipment = []
                for item in assoc_helmets:
                    equipment.append({"box": item["box"], "cls_name": "helmet"})
                for item in assoc_no_helmets:
                    equipment.append({"box": item["box"], "cls_name": "no_helmet"})
                for item in assoc_vests:
                    equipment.append({"box": item["box"], "cls_name": "vest"})
                for item in assoc_no_vests:
                    equipment.append({"box": item["box"], "cls_name": "no_vest"})

                person_data.append({
                    "track_id": tid,
                    "box": pbox,
                    "violations": violations,
                    "equipment": equipment,
                })

            # Cleanup stale track states
            vstates.cleanup({p["track_id"] for p in person_data})

            # Process violations & fire alerts
            new_highlight_tids: set = set()
            for person in person_data:
                tid = person["track_id"]
                violations = person["violations"]

                for vtype in ["NO_HELMET", "NO_VEST"]:
                    is_violating = vtype in violations
                    should_alert = vstates.process(
                        tid, vtype, is_violating, now, duration, grace, cooldown
                    )

                    if should_alert:
                        incident_id = csv_logger.next_id()
                        ts_iso = datetime.fromtimestamp(now, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")  # noqa: DTZ004
                        snap_name = f"incident_{incident_id:05d}_screen{screen_num}_tid{tid}_{vtype}.jpg"
                        snap_path = str(Path(snapshots_dir) / snap_name)

                        # Draw highlighted snapshot
                        snap_frame = draw_frame(frame.copy(), person_data, highlight_tids={tid})
                        cv2.imwrite(snap_path, snap_frame)

                        caption = (
                            f"🚨 PPE VIOLATION\n"
                            f"Screen: {screen_num}\n"
                            f"Type: {vtype}\n"
                            f"Track ID: {tid}\n"
                            f"Time: {ts_iso}\n"
                            f"Incident: #{incident_id}"
                        )

                        csv_logger.log({
                            "incident_id": incident_id,
                            "track_id": tid,
                            "screen": screen_num,
                            "violation_type": vtype,
                            "timestamp": ts_iso,
                            "image_path": snap_path,
                            "telegram_sent": "pending",
                        })

                        tg_worker.send(snap_path, caption, incident_id, csv_logger)

                        tprint(
                            f"[Screen {screen_num}] 🚨 ALERT #{incident_id} "
                            f"| Track {tid} | {vtype}"
                        )

                        new_highlight_tids.add(tid)

            highlight_tids = new_highlight_tids

            # Annotate and display frame
            annotated = draw_frame(frame, person_data, highlight_tids)
            if show:
                cv2.imshow(window_name, annotated)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    stop_event.set()

    finally:
        cap.release()
        tprint(f"[Screen {screen_num}] Source closed.")


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="PPE Violation Monitoring System — Helmet & Safety Vest detection"
    )
    parser.add_argument(
        "--config", default="config.yaml",
        help="Path to YAML config file (default: config.yaml)"
    )
    parser.add_argument(
        "--show", action="store_true",
        help="Display live video windows"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Skip Telegram sending; still save CSV and snapshots"
    )
    args = parser.parse_args()

    # Load config
    cfg = load_config(args.config)

    # Resolve paths relative to the config file's directory
    config_dir = Path(args.config).resolve().parent

    model_cfg  = cfg.get("model", {})
    output_cfg = cfg.get("output", {})
    tg_cfg     = cfg.get("telegram", {})
    sources    = cfg.get("sources", [])

    csv_raw       = output_cfg.get("csv_path", "violations.csv")
    snap_raw      = output_cfg.get("snapshots_dir", "snapshots")
    csv_path      = str(config_dir / csv_raw) if not Path(csv_raw).is_absolute() else csv_raw
    snapshots_dir = str(config_dir / snap_raw) if not Path(snap_raw).is_absolute() else snap_raw

    # Determine show flag: True if passed via CLI or set in config.yaml
    show_window = args.show or cfg.get("display", {}).get("show", False)

    banner = (
        "\n" + "=" * 60 + "\n"
        + "  PPE Violation Monitoring System\n"
        + "=" * 60
    )
    print(banner)
    print(f"  Config     : {args.config}")
    print(f"  Model      : {model_cfg.get('path', 'N/A')}")
    print(f"  Sources    : {len(sources)}")
    print(f"  CSV        : {csv_path}")
    print(f"  Snapshots  : {snapshots_dir}")
    print(f"  Show       : {show_window}")
    print(f"  Dry-run    : {args.dry_run}")
    print("=" * 60 + "\n")

    # Create output directories
    Path(snapshots_dir).mkdir(parents=True, exist_ok=True)

    # Initialise shared services
    csv_logger = CSVLogger(csv_path)
    tg_worker = TelegramWorker(
        bot_token=tg_cfg.get("bot_token", ""),
        chat_id=tg_cfg.get("chat_id", ""),
        dry_run=args.dry_run,
    )

    if args.dry_run:
        print("[DRY-RUN] Telegram notifications disabled. CSV and snapshots will still be saved.\n")

    stop_event = threading.Event()

    # Launch one thread per source
    threads = []
    for src_cfg in sources:
        screen_num = src_cfg.get("screen", len(threads) + 1)
        source     = src_cfg.get("source")
        show_source = src_cfg.get("show", show_window)
        t = threading.Thread(
            target=source_worker,
            args=(screen_num, source, cfg, csv_logger, tg_worker, show_source, stop_event, snapshots_dir),
            name=f"Screen-{screen_num}",
            daemon=True,
        )
        t.start()
        threads.append(t)

    # Wait for all threads
    try:
        for t in threads:
            while t.is_alive():
                t.join(timeout=1.0)
    except KeyboardInterrupt:
        tprint("\n[Main] KeyboardInterrupt — stopping all sources...")
        stop_event.set()
        for t in threads:
            t.join(timeout=5.0)

    # Cleanup
    tg_worker.stop()
    cv2.destroyAllWindows()
    print(f"\nDone. Check {csv_path}")


if __name__ == "__main__":
    main()
