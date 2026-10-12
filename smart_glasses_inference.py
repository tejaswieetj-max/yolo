"""
Context-Aware Smart Glasses - Module 2: AI Vision & Time-To-Collision (TTC) Hazard Alert Engine
Patent Implementation & Embedded Systems Project
Lead Researcher: Dhivyashree G J. (24BCE1516)
Collaborators: Andrea A. (24BCE1142) · Tejaswi E. (24BCE5518)

Hardware Targets:
- ESP32-S3 Edge AI (INT8 TFLite via ESP-NN SIMD vector acceleration)
- Raspberry Pi 5 / Jetson Nano / PC (ONNX Runtime / PyTorch)

8-Class Granular Taxonomy:
  0: person      - Pedestrian navigation & crowd avoidance
  1: bicycle     - Approaching micromobility hazard
  2: car         - Moving/parked road vehicle
  3: motorcycle  - High-speed dynamic threat
  4: bus         - Heavy public transit collision hazard
  5: truck       - Large heavy freight hazard
  6: stairs      - Ground elevation drop-off / step descent/ascent hazard
  7: branch      - Head-height overhanging branch / awning / low obstacle
"""

import sys
import time
import argparse
from pathlib import Path

CLASS_NAMES = {
    0: "person",
    1: "bicycle",
    2: "car",
    3: "motorcycle",
    4: "bus",
    5: "truck",
    6: "stairs",
    7: "branch"
}

# Safety priority & specific auditory warning templates
HAZARD_PROFILES = {
    "stairs": {
        "priority": "HIGH",
        "action": "HALT / STEP DESCENT",
        "message": "Caution: Stairs / drop-off detected directly ahead."
    },
    "branch": {
        "priority": "HIGH",
        "action": "DUCK / HEAD HAZARD",
        "message": "Warning: Overhanging branch / low obstacle at head level."
    },
    "motorcycle": {
        "priority": "CRITICAL",
        "action": "RAPID DODGE",
        "message": "Alert: Fast-moving motorcycle approaching."
    },
    "bus": {
        "priority": "CRITICAL",
        "action": "STOP",
        "message": "Danger: Heavy bus in trajectory."
    },
    "truck": {
        "priority": "CRITICAL",
        "action": "STOP",
        "message": "Danger: Heavy truck in trajectory."
    },
    "car": {
        "priority": "WARNING",
        "action": "YIELD",
        "message": "Notice: Vehicle ahead."
    },
    "bicycle": {
        "priority": "WARNING",
        "action": "STEP ASIDE",
        "message": "Notice: Approaching bicycle."
    },
    "person": {
        "priority": "INFO",
        "action": "NAVIGATE",
        "message": "Pedestrian in navigation path."
    }
}

class SmartGlassesTTCEngine:
    """
    Time-To-Collision (TTC) & Spatial Hazard Engine:
    Calculates threat zones (Left, Center, Right) and estimates optical expansion rate
    to trigger prioritized haptic and acoustic warnings for visually impaired wearers.
    """
    def __init__(self, frame_width=416, frame_height=416):
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.center_zone_min = frame_width * 0.30
        self.center_zone_max = frame_width * 0.70

    def evaluate_detection(self, class_id, conf, box_xyxy, prev_box_xyxy=None, dt=0.1):
        x1, y1, x2, y2 = box_xyxy
        w = x2 - x1
        h = y2 - y1
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        cname = CLASS_NAMES.get(class_id, "unknown")
        profile = HAZARD_PROFILES.get(cname, {"priority": "INFO", "action": "ALERT", "message": "Obstacle"})

        # 1. Determine Spatial Direction (Lateral angle)
        if cx < self.center_zone_min:
            direction = "LEFT"
        elif cx > self.center_zone_max:
            direction = "RIGHT"
        else:
            direction = "CENTER"

        # 2. Distance Estimation via Normalized Bounding Box Area
        normalized_area = (w * h) / (self.frame_width * self.frame_height)
        if normalized_area > 0.25:
            proximity = "IMMINENT (< 1.5m)"
        elif normalized_area > 0.10:
            proximity = "CLOSE (1.5m - 3.5m)"
        elif normalized_area > 0.03:
            proximity = "MEDIUM (3.5m - 6.0m)"
        else:
            proximity = "DISTANT (> 6.0m)"

        # 3. Optical Expansion TTC Calculation
        ttc_sec = None
        if prev_box_xyxy is not None and dt > 0:
            prev_w = prev_box_xyxy[2] - prev_box_xyxy[0]
            dw_dt = (w - prev_w) / dt
            if dw_dt > 0.5:  # Object is expanding (approaching wearer)
                ttc_sec = round(w / dw_dt, 2)

        # 4. Synthesize Hazard Urgency
        is_imminent = (proximity == "IMMINENT (< 1.5m)") or (ttc_sec is not None and ttc_sec < 2.0)
        if is_imminent or cname in ["stairs", "branch"]:
            alert_level = "CRITICAL" if profile["priority"] in ["HIGH", "CRITICAL"] else "WARNING"
        else:
            alert_level = profile["priority"]

        return {
            "class_name": cname,
            "confidence": round(float(conf), 3),
            "direction": direction,
            "proximity": proximity,
            "ttc_seconds": ttc_sec,
            "alert_level": alert_level,
            "recommended_action": profile["action"],
            "voice_prompt": f"[{alert_level}] {profile['message']} ({direction}, {proximity})"
        }


def run_pipeline(model_path="best.pt", source="kaggle_outputs/test_predictions"):
    print("=" * 72)
    print("CONTEXT-AWARE SMART GLASSES: AI VISION & TTC ENGINE")
    print("Module 2 Patent Implementation | Real-Time Edge Architecture")
    print("=" * 72)
    print(f"Loading Model:  {model_path}")
    print(f"Target Classes: {len(CLASS_NAMES)} ({', '.join(CLASS_NAMES.values())})")

    p = Path(model_path)
    if not p.exists():
        print(f"[!] Error: Model file not found at {model_path}")
        return

    engine = SmartGlassesTTCEngine()

    try:
        from ultralytics import YOLO
        model = YOLO(model_path)
        print("[OK] PyTorch YOLOv8n engine initialized successfully.")
    except ImportError:
        print("[!] Ultralytics not installed in this environment. Showing simulated TTC parser output.")
        # Demonstration run
        sample_hazards = [
            (6, 0.94, (120, 250, 290, 410)), # Stairs
            (7, 0.81, (150, 20, 260, 140)),  # Branch
            (3, 0.96, (200, 100, 380, 360)), # Motorcycle
            (0, 0.92, (40, 60, 140, 350))    # Person
        ]
        print("\n--- Module 2 Hazard Evaluation Sample ---")
        for cid, conf, bbox in sample_hazards:
            eval_res = engine.evaluate_detection(cid, conf, bbox)
            print(f"Class: {eval_res['class_name']:10s} | Level: {eval_res['alert_level']:8s} | Action: {eval_res['recommended_action']:18s} | Prompt: {eval_res['voice_prompt']}")
        return

    # If ultralytics is present, run real inference on input
    src_path = Path(source)
    if src_path.exists():
        imgs = list(src_path.glob("*.jpg"))[:5] if src_path.is_dir() else [src_path]
        print(f"\nProcessing {len(imgs)} test inputs through Vision & TTC Engine:")
        for img in imgs:
            results = model.predict(source=str(img), imgsz=416, conf=0.25, verbose=False)
            print(f"\n--- Frame: {img.name} ---")
            for r in results:
                for b in r.boxes:
                    cid = int(b.cls[0])
                    conf = float(b.conf[0])
                    box = b.xyxy[0].tolist()
                    eval_res = engine.evaluate_detection(cid, conf, box)
                    print(f"-> {eval_res['voice_prompt']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Smart Glasses Inference & Hazard Alert Engine")
    parser.add_argument("--model", type=str, default="best.pt", help="Path to best.pt or best.onnx")
    parser.add_argument("--source", type=str, default="kaggle_outputs/test_predictions", help="Image, directory, or video source")
    args = parser.parse_args()

    run_pipeline(args.model, args.source)
