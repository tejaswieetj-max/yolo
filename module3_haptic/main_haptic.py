"""
Module 3 Spatio-Tactile Haptics & Telemetry Process Entrypoint
Lead Researcher: Tejaswi E. (24BCE5518)

Consumes Queue B (Vision/TTC), encodes 3-axis spatio-tactile signals,
triggers TI DRV2605L LRA motors, renders terminal visualizer, and logs telemetry.
"""

import time
import sys
from multiprocessing import Queue
from typing import Optional, Callable

from config import SIMULATION_MODE
from .haptic_encoder import SpatioTactileEncoder
from .drv2605l_driver import DRV2605LDriver
from .sim_haptic import TerminalHapticVisualizer
from .firebase_logger import FirebaseTelemetryLogger


def run_haptic_module(queue_in: Queue, stop_event=None, event_callback: Optional[Callable] = None):
    """
    Main loop for Module 3.
    Consumes from Queue B, actuates LRAs, logs to Firebase & disk, updates visualizer.
    """
    print(f"\n[Module 3: Haptic] Starting Process (SIMULATION_MODE={SIMULATION_MODE})")

    encoder = SpatioTactileEncoder()
    drv_driver = DRV2605LDriver()
    visualizer = TerminalHapticVisualizer()
    logger = FirebaseTelemetryLogger(use_firebase=True)

    try:
        while stop_event is None or not stop_event.is_set():
            vision_packet = None
            if queue_in is not None:
                try:
                    vision_packet = queue_in.get(timeout=0.1)
                except Exception:
                    continue

            if vision_packet is None:
                continue

            now = time.time()
            # Calculate end-to-end perception-to-action latency
            t_orig = vision_packet.get("timestamp", now)
            latency_ms = (now - t_orig) * 1000.0

            # 1. Novelty 02: 3-Axis Spatio-Tactile Encoding
            cmd = encoder.encode(vision_packet)

            # 2. Hardware Actuation or Virtual Pulse
            if cmd.should_pulse:
                drv_driver.trigger_waveform(cmd.motor_channel, cmd.waveform_id)

            # 3. Console Terminal Visualization
            head_yaw = vision_packet.get("head_yaw_deg", 0.0)
            visualizer.render(cmd, head_yaw, latency_ms)

            # 4. Construct Full Telemetry Packet
            telemetry_event = {
                "timestamp": now,
                "frame_id": vision_packet.get("frame_id", 0),
                "head_yaw_deg": head_yaw,
                "motor_channel": cmd.motor_channel,
                "distance_m": cmd.distance_m,
                "ttc_seconds": cmd.ttc_seconds,
                "approach_velocity_mps": vision_packet.get("approach_velocity_mps", 0.0),
                "urgency_level": cmd.urgency_level,
                "hazard_class": cmd.hazard_class,
                "waveform_id": cmd.waveform_id,
                "waveform_name": cmd.waveform_name,
                "is_depth_interrupt": vision_packet.get("is_depth_interrupt", False),
                "ai_gated": vision_packet.get("ai_gated", False),
                "roi_area_ratio": vision_packet.get("roi_area_ratio", 0.0),
                "latency_ms": latency_ms,
                "should_pulse": cmd.should_pulse,
                # Fusion metadata
                "fusion_mode": vision_packet.get("fusion_mode", "lite"),
                "yolo_inference_ms": vision_packet.get("yolo_inference_ms", 0.0),
                "mp_inference_ms": vision_packet.get("mp_inference_ms", 0.0),
                "pose_detected": vision_packet.get("pose_detected", False),
                "hands_detected": vision_packet.get("hands_detected", False),
            }

            # 5. Push to Cloud / Local Research Logger
            logger.log_event(telemetry_event)

            # 6. Optional Direct Callback for Local Web Dashboard Server
            if event_callback is not None:
                try:
                    event_callback(telemetry_event)
                except Exception:
                    pass

    except KeyboardInterrupt:
        print("\n[Module 3: Haptic] Stopping on user interrupt...")
    finally:
        drv_driver.close()
        logger.stop()
        print("\n[Module 3: Haptic] Exited cleanly.")


if __name__ == "__main__":
    from multiprocessing import Queue
    print("=== Testing Module 3 (3-Axis Haptic & Telemetry) ===")
    q_test = Queue(maxsize=5)

    import threading
    stop_event = threading.Event()
    t = threading.Thread(target=run_haptic_module, args=(q_test, stop_event))
    t.daemon = True
    t.start()

    # Feed test packets with varying urgency
    try:
        scenarios = [
            {"critical_sector": "CENTER", "min_distance_m": 2.2, "ttc_seconds": 3.5, "primary_class": "person", "is_head_hazard": False},
            {"critical_sector": "LEFT", "min_distance_m": 1.4, "ttc_seconds": 1.2, "primary_class": "car", "is_head_hazard": False},
            {"critical_sector": "CENTER", "min_distance_m": 0.6, "ttc_seconds": 0.4, "primary_class": "overhead_hazard", "is_head_hazard": True},
        ]

        for s in scenarios:
            s["timestamp"] = time.time()
            s["head_yaw_deg"] = 12.5
            q_test.put(s)
            time.sleep(0.3)

        time.sleep(0.5)
    finally:
        stop_event.set()
        print("\n=== Module 3 test passed successfully ===")
