"""
Firebase Realtime DB Streamer & Local Telemetry Logger
Lead Researcher: Tejaswi E. (24BCE5518)

Streams real-time event packets to Firebase Realtime Database with asynchronous
background batching and persistent local JSONL/CSV logging for publication data.
"""

import json
import csv
import time
import os
import threading
from queue import Queue, Empty
from pathlib import Path
from typing import Dict, Any, List

from config import (
    FIREBASE_CREDENTIALS_PATH,
    FIREBASE_DATABASE_URL,
    FIREBASE_BATCH_INTERVAL_S,
    LOCAL_TELEMETRY_FILE,
    LOCAL_LOG_DIR,
)


class FirebaseTelemetryLogger:
    """Asynchronous telemetry logger streaming to Firebase and local disk."""

    def __init__(self, use_firebase: bool = True):
        self.use_firebase = use_firebase
        self.fb_app = None
        self.fb_db = None
        self.event_queue: Queue = Queue(maxsize=1000)
        self.running = True

        LOCAL_LOG_DIR.mkdir(parents=True, exist_ok=True)
        self.csv_file = LOCAL_LOG_DIR / "telemetry_events.csv"
        self._init_csv()

        if self.use_firebase:
            self._init_firebase()

        # Start asynchronous background worker
        self.worker_thread = threading.Thread(target=self._batch_worker, daemon=True)
        self.worker_thread.start()

    def _init_csv(self):
        """Ensure CSV file exists with standard research column headers."""
        if not self.csv_file.exists():
            with open(self.csv_file, mode="w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "timestamp",
                    "frame_id",
                    "head_yaw_deg",
                    "motor_channel",
                    "distance_m",
                    "ttc_seconds",
                    "approach_velocity_mps",
                    "urgency_level",
                    "hazard_class",
                    "waveform_id",
                    "waveform_name",
                    "is_depth_interrupt",
                    "ai_gated",
                    "roi_area_ratio",
                    "latency_ms",
                ])

    def _init_firebase(self):
        """Attempt to initialize Firebase Admin SDK or REST client."""
        try:
            import firebase_admin
            from firebase_admin import credentials, db

            if os.path.exists(FIREBASE_CREDENTIALS_PATH):
                cred = credentials.Certificate(FIREBASE_CREDENTIALS_PATH)
                self.fb_app = firebase_admin.initialize_app(cred, {
                    "databaseURL": FIREBASE_DATABASE_URL
                })
                self.fb_db = db.reference("live_telemetry")
                print(f"[Firebase] Connected to Realtime DB at {FIREBASE_DATABASE_URL}")
            else:
                print("[Firebase] Credentials not found. Operating with local research logger.")
        except Exception as e:
            print(f"[Firebase] Initialization bypassed ({e}). Using local research logger.")

    def log_event(self, event_data: Dict[str, Any]):
        """Non-blocking queue insertion of event telemetry packet."""
        try:
            self.event_queue.put_nowait(event_data)
        except Exception:
            pass

    def _batch_worker(self):
        """Background thread aggregating and flushing telemetry batches."""
        batch: List[Dict[str, Any]] = []
        last_flush = time.time()

        while self.running:
            try:
                # Wait for items up to interval
                item = self.event_queue.get(timeout=FIREBASE_BATCH_INTERVAL_S)
                batch.append(item)
            except Empty:
                pass

            now = time.time()
            if batch and (now - last_flush >= FIREBASE_BATCH_INTERVAL_S or len(batch) >= 20):
                self._flush_batch(batch)
                batch = []
                last_flush = now

    def _flush_batch(self, batch: List[Dict[str, Any]]):
        """Write batch to local CSV/JSONL and stream to Firebase."""
        # 1. Local JSONL append
        try:
            with open(LOCAL_TELEMETRY_FILE, "a", encoding="utf-8") as f:
                for item in batch:
                    f.write(json.dumps(item) + "\n")
        except Exception as e:
            print(f"[Logger] Local JSONL write error: {e}")

        # 2. Local CSV append for 1-click paper graphs
        try:
            with open(self.csv_file, "a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                for item in batch:
                    writer.writerow([
                        item.get("timestamp", 0.0),
                        item.get("frame_id", 0),
                        item.get("head_yaw_deg", 0.0),
                        item.get("motor_channel", ""),
                        item.get("distance_m", 0.0),
                        item.get("ttc_seconds", 0.0),
                        item.get("approach_velocity_mps", 0.0),
                        item.get("urgency_level", ""),
                        item.get("hazard_class", ""),
                        item.get("waveform_id", 0),
                        item.get("waveform_name", ""),
                        item.get("is_depth_interrupt", False),
                        item.get("ai_gated", False),
                        item.get("roi_area_ratio", 0.0),
                        item.get("latency_ms", 0.0),
                    ])
        except Exception as e:
            print(f"[Logger] Local CSV write error: {e}")

        # 3. Firebase push if configured
        if self.fb_db is not None:
            try:
                latest = batch[-1]
                self.fb_db.child("latest_state").set(latest)
            except Exception as e:
                pass

    def stop(self):
        self.running = False
