"""
Module 1 Sensor Fusion Main Process Entrypoint
Lead Researcher: Andrea A. (24BCE1142)

Orchestrates 30 Hz VL53L5CX acquisition, 100 Hz MPU6050 orientation tracking,
quaternion coordinate re-projection, and pushes packets to Queue A.
"""

import time
import sys
from multiprocessing import Queue
from typing import Optional

from config import (
    SIMULATION_MODE,
    USE_MEDIAPIPE_CAMERA,
    TOF_SAMPLE_RATE_HZ,
    IMU_SAMPLE_RATE_HZ,
)
from .tof_driver import VL53L7CXDriver
from .imu_driver import MPU6050Driver
from .ego_motion import EgoMotionEngine
from .zone_classifier import ZoneClassifier
from .sim_sensor import SimulatedSensorSystem


def run_sensor_module(queue_out: Queue, stop_event=None, queue_video_feed=None):
    """
    Main loop for Module 1.
    Reads sensors, applies ego-motion transform, bins sectors, pushes to queue_out.
    """
    print(f"[Module 1: Sensor] Starting Process (SIMULATION_MODE={SIMULATION_MODE})")

    ego_engine = EgoMotionEngine()
    zone_classifier = ZoneClassifier()

    mp_sensor = None  # MediaPipe sensor (if used)

    if SIMULATION_MODE:
        if USE_MEDIAPIPE_CAMERA:
            try:
                from .mediapipe_sensor import MediaPipeSensorSystem
                mp_sensor = MediaPipeSensorSystem(queue_video_feed=queue_video_feed)
                sim_sensors = None
                print("[Module 1: Sensor] Using MEDIAPIPE CAMERA mode (real webcam + hand tracking)")
            except Exception as e:
                print(f"[Module 1: Sensor] MediaPipe init failed ({e}), falling back to synthetic mode.")
                mp_sensor = None
                sim_sensors = SimulatedSensorSystem()
        else:
            sim_sensors = SimulatedSensorSystem()
        tof_hw = None
        imu_hw = None
    else:
        sim_sensors = None
        tof_hw = VL53L7CXDriver()
        imu_hw = MPU6050Driver()

    interval = 1.0 / TOF_SAMPLE_RATE_HZ
    last_loop_time = time.time()
    frame_count = 0

    try:
        while stop_event is None or not stop_event.is_set():
            t_start = time.time()

            # 1. Acquire depth array & orientation
            if mp_sensor is not None:
                # MEDIAPIPE CAMERA MODE: real webcam data
                yaw_deg, pitch_deg, roll_deg = mp_sensor.get_simulated_head_orientation(t_start)
                depth_matrix, ground_truth = mp_sensor.generate_depth_matrix(t_start)
            elif SIMULATION_MODE:
                yaw_deg, pitch_deg, roll_deg = sim_sensors.get_simulated_head_orientation(t_start)
                depth_matrix, ground_truth = sim_sensors.generate_depth_matrix(t_start)
            else:
                yaw_deg, pitch_deg, roll_deg = imu_hw.update_orientation()
                depth_matrix = tof_hw.read_distance_matrix()
                ground_truth = {}

            # 2. Novelty 01: Coordinate re-projection to Torso Frame
            sensor_vecs, torso_vecs = ego_engine.process_frame(depth_matrix, yaw_deg, pitch_deg)

            # 3. Spatial Classification & Depth Gating Assessment
            classif_result = zone_classifier.classify(torso_vecs)

            # 4. Construct Packet for Queue A (Sensor -> Vision)
            packet = {
                "timestamp": t_start,
                "frame_id": frame_count,
                "head_yaw_deg": yaw_deg,
                "head_pitch_deg": pitch_deg,
                "depth_matrix": depth_matrix,
                "sensor_vectors": sensor_vecs,
                "torso_vectors": torso_vecs,
                "sectors": classif_result.sectors,
                "critical_sector": classif_result.critical_sector,
                "min_distance_overall_m": classif_result.min_distance_overall_m,
                "interrupt_triggered": classif_result.interrupt_triggered,
                "ground_truth": ground_truth,
            }

            # 5. Push to Queue A (drop oldest if queue is congested to guarantee real-time latency)
            if queue_out is not None:
                if queue_out.full():
                    try:
                        queue_out.get_nowait()
                    except Exception:
                        pass
                try:
                    queue_out.put_nowait(packet)
                except Exception:
                    pass

            frame_count += 1

            # Sleep to match target rate
            elapsed = time.time() - t_start
            sleep_time = interval - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    except KeyboardInterrupt:
        print("[Module 1: Sensor] Stopping on user interrupt...")
    finally:
        if mp_sensor is not None:
            mp_sensor.release()
        if tof_hw is not None:
            tof_hw.close()
        if imu_hw is not None:
            imu_hw.close()
        print("[Module 1: Sensor] Exited cleanly.")


if __name__ == "__main__":
    from multiprocessing import Queue
    print("=== Testing Module 1 (Sensor Fusion & Ego-Motion) ===")
    test_queue = Queue(maxsize=5)

    import threading
    stop_event = threading.Event()

    t = threading.Thread(target=run_sensor_module, args=(test_queue, stop_event))
    t.daemon = True
    t.start()

    try:
        for i in range(10):
            pkt = test_queue.get(timeout=2.0)
            print(f"[{i+1}/10] Frame #{pkt['frame_id']} | "
                  f"Yaw: {pkt['head_yaw_deg']:+5.1f}° | "
                  f"Critical Sector: {pkt['critical_sector']:<6} | "
                  f"Min Dist: {pkt['min_distance_overall_m']:.2f}m | "
                  f"Interrupt Active: {pkt['interrupt_triggered']}")
            time.sleep(0.1)
    finally:
        stop_event.set()
        print("=== Module 1 test passed successfully ===")
