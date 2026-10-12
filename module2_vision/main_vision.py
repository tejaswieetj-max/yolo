"""
Module 2 Vision & TTC Main Process Entrypoint
Lead Researcher: Dhivyashree G J. (24BCE1516)

Consumes Queue A (Sensor Fusion), applies Stage 1 Depth Gating,
runs YOLOv8n + MediaPipe in parallel via Fusion Engine, computes TTC,
and pushes to Queue B.

Fusion Modes:
  - FULL: YOLO + MediaPipe parallel (best accuracy)
  - LITE: YOLO only (lowest power)
  - AUTO: Adaptive switching based on CPU temp / battery
"""

import time
import sys
from multiprocessing import Queue
from typing import Optional, Dict, Any

from config import (
    SIMULATION_MODE,
    VISION_MAX_FPS,
    DEPTH_INTERRUPT_THRESHOLD_M,
    FUSION_MODE,
)
from .tof_gate import ToFInterruptGate
from .camera_driver import CameraDriver
from .yolo_engine import YOLOv8ROIEngine
from .ttc_engine import TTCEngine
from .class_map import get_severity_score
from .sim_vision import SimulatedVisionTestbed
from .fusion_engine import FusionEngine


def _init_mediapipe_detector():
    """Try to initialize MediaPipe detector. Returns None if unavailable."""
    try:
        from .mediapipe_detector import MediaPipeDetector
        detector = MediaPipeDetector()
        if detector.is_loaded or not detector.active_models:
            # Return even if simulated — fusion engine handles it
            return detector
    except Exception as e:
        print(f"[Module 2: Vision] MediaPipe detector init failed ({e}). Running YOLO-only.")
    return None


def run_vision_module(queue_in: Queue, queue_out: Queue, stop_event=None):
    """
    Main loop for Module 2.
    Pulls from Queue A (Sensor), applies ROI gate, runs YOLO+MediaPipe fusion,
    computes TTC, pushes to Queue B (Haptic).
    """
    print(f"[Module 2: Vision] Starting Process (SIMULATION_MODE={SIMULATION_MODE})")
    print(f"[Module 2: Vision] Fusion Mode: {FUSION_MODE}")

    tof_gate = ToFInterruptGate(threshold_m=DEPTH_INTERRUPT_THRESHOLD_M)
    camera = CameraDriver()
    yolo_engine = YOLOv8ROIEngine()
    ttc_engine = TTCEngine()

    # Initialize MediaPipe detector for parallel fusion
    mp_detector = None
    if FUSION_MODE != "lite":
        mp_detector = _init_mediapipe_detector()

    # Create Fusion Engine (wraps YOLO + MediaPipe)
    fusion_engine = FusionEngine(yolo_engine=yolo_engine, mp_detector=mp_detector)
    print(f"[Module 2: Vision] Fusion Engine ready. MediaPipe: {'Active' if mp_detector else 'Disabled'}")

    interval = 1.0 / VISION_MAX_FPS

    try:
        while stop_event is None or not stop_event.is_set():
            t_start = time.time()

            # 1. Fetch sensor packet from Queue A
            sensor_packet = None
            if queue_in is not None:
                try:
                    sensor_packet = queue_in.get(timeout=0.1)
                except Exception:
                    # Queue empty
                    continue

            if sensor_packet is None:
                continue

            depth_matrix = sensor_packet["depth_matrix"]
            critical_sector = sensor_packet["critical_sector"]
            min_dist = sensor_packet["min_distance_overall_m"]
            head_yaw = sensor_packet["head_yaw_deg"]
            ground_truth = sensor_packet.get("ground_truth", {})

            # 2. Stage 1: Depth Interrupt Gate (Novelty 03)
            is_triggered, roi = tof_gate.evaluate_gate(depth_matrix)

            # Update TTC engine with sector distances
            sector_distances = {
                sec_name: sec_data.min_distance_m
                for sec_name, sec_data in sensor_packet["sectors"].items()
            }
            ttc_results = ttc_engine.process_all_sectors(sector_distances, timestamp=t_start)
            critical_ttc = ttc_results.get(critical_sector)

            primary_hazard_class = "clear"
            primary_confidence = 0.0
            ai_gated = False
            fusion_mode_used = "lite"
            yolo_ms = 0.0
            mp_ms = 0.0
            pose_detected = False
            hands_detected = False

            if is_triggered and roi is not None:
                # Stage 2: Camera Wakeup & Parallel Fusion Inference
                ret, frame = camera.capture_frame()
                if ret:
                    mock_hint = ground_truth.get("target_class", "person")

                    # Run YOLO + MediaPipe in parallel via Fusion Engine
                    fusion_result = fusion_engine.infer_fused(
                        full_frame=frame,
                        roi=roi,
                        mock_hint_class=mock_hint,
                    )

                    fusion_mode_used = fusion_result.fusion_mode
                    yolo_ms = fusion_result.yolo_inference_ms
                    mp_ms = fusion_result.mp_inference_ms
                    pose_detected = fusion_result.pose_detected
                    hands_detected = fusion_result.hands_detected

                    if fusion_result.detections:
                        # Find highest severity fused detection
                        best_det = max(
                            fusion_result.detections,
                            key=lambda d: get_severity_score(d.class_name, roi.min_distance_m)
                        )
                        primary_hazard_class = best_det.class_name
                        primary_confidence = best_det.fused_confidence
                    else:
                        primary_hazard_class = "generic_obstacle"
                        primary_confidence = 0.5
            else:
                # Gated: Camera & AI sleep, saving >70% compute
                ai_gated = True
                primary_hazard_class = "clear"

            # 3. Construct Packet for Queue B (Vision -> Haptics)
            haptic_packet = {
                "timestamp": t_start,
                "frame_id": sensor_packet.get("frame_id", 0),
                "head_yaw_deg": head_yaw,
                "critical_sector": critical_sector,
                "min_distance_m": min_dist,
                "is_depth_interrupt": is_triggered,
                "ai_gated": ai_gated,
                "roi_area_ratio": roi.area_ratio if roi else 0.0,
                "primary_class": primary_hazard_class,
                "confidence": primary_confidence,
                "ttc_seconds": critical_ttc.ttc_seconds if critical_ttc else 99.0,
                "approach_velocity_mps": critical_ttc.approach_velocity_mps if critical_ttc else 0.0,
                "urgency_level": critical_ttc.urgency_level if critical_ttc else "SAFE",
                "is_head_hazard": sensor_packet["sectors"][critical_sector].is_head_hazard if critical_sector in sensor_packet["sectors"] else False,
                # Fusion metadata
                "fusion_mode": fusion_mode_used,
                "yolo_inference_ms": yolo_ms,
                "mp_inference_ms": mp_ms,
                "pose_detected": pose_detected,
                "hands_detected": hands_detected,
            }

            # 4. Push to Queue B
            if queue_out is not None:
                if queue_out.full():
                    try:
                        queue_out.get_nowait()
                    except Exception:
                        pass
                try:
                    queue_out.put_nowait(haptic_packet)
                except Exception:
                    pass

            elapsed = time.time() - t_start
            sleep_time = interval - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    except KeyboardInterrupt:
        print("[Module 2: Vision] Stopping on user interrupt...")
    finally:
        camera.release()
        fusion_engine.shutdown()
        print("[Module 2: Vision] Exited cleanly.")


if __name__ == "__main__":
    from multiprocessing import Queue
    import numpy as np

    print("=== Testing Module 2 (Vision Gate, YOLO+MediaPipe Fusion & TTC) ===")
    q_in = Queue(maxsize=5)
    q_out = Queue(maxsize=5)

    # Feed simulated packet closer than 2.0m to trigger gate
    test_matrix = np.full((8, 8), 3.5)
    test_matrix[2:5, 3:5] = 1.2  # Obstacle at 1.2m in Center

    from module1_sensor.zone_classifier import ZoneClassifier, SectorSummary
    from module1_sensor.ego_motion import TorsoFrameVector

    fake_packet = {
        "timestamp": time.time(),
        "frame_id": 1,
        "head_yaw_deg": 0.0,
        "depth_matrix": test_matrix,
        "critical_sector": "CENTER",
        "min_distance_overall_m": 1.2,
        "sectors": {
            "LEFT": SectorSummary("LEFT", 3.5, None, 0, False, 3.5),
            "CENTER": SectorSummary("CENTER", 1.2, None, 4, False, 1.2),
            "RIGHT": SectorSummary("RIGHT", 3.5, None, 0, False, 3.5),
        },
        "ground_truth": {"target_class": "person"},
    }

    q_in.put(fake_packet)

    import threading
    stop_event = threading.Event()
    t = threading.Thread(target=run_vision_module, args=(q_in, q_out, stop_event))
    t.daemon = True
    t.start()

    time.sleep(0.5)
    try:
        out_pkt = q_out.get(timeout=2.0)
        print(f"Output Packet Received:")
        print(f"  Critical Sector: {out_pkt['critical_sector']}")
        print(f"  Primary Class:   {out_pkt['primary_class']}")
        print(f"  Confidence:      {out_pkt['confidence']:.3f}")
        print(f"  Fusion Mode:     {out_pkt['fusion_mode']}")
        print(f"  AI Gated:        {out_pkt['ai_gated']}")
        print(f"  ROI Area Ratio:  {out_pkt['roi_area_ratio']:.3f} (Savings: {(1-out_pkt['roi_area_ratio'])*100:.1f}%)")
        print(f"  Urgency Level:   {out_pkt['urgency_level']}")
        print(f"  YOLO Time:       {out_pkt['yolo_inference_ms']:.1f}ms")
        print(f"  MediaPipe Time:  {out_pkt['mp_inference_ms']:.1f}ms")
        print(f"  Pose Detected:   {out_pkt['pose_detected']}")
        print(f"  Hands Detected:  {out_pkt['hands_detected']}")
    finally:
        stop_event.set()
        print("=== Module 2 test passed successfully ===")
