"""
End-to-End Latency & Performance Profiling Benchmark
Lead Researcher: Andrea A. / Dhivyashree G J. / Tejaswi E.

Profiles execution latencies across all 3 modules to verify real-time performance
on embedded ARM (Raspberry Pi 5) and PC hosts.
"""

import sys
import time
import csv
from pathlib import Path
import numpy as np

# Ensure workspace root is in python path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from config import (
    TOF_GRID_ROWS,
    TOF_GRID_COLS,
    LOCAL_LOG_DIR,
)
from module1_sensor.ego_motion import EgoMotionEngine
from module1_sensor.zone_classifier import ZoneClassifier
from module2_vision.tof_gate import ToFInterruptGate
from module2_vision.ttc_engine import TTCEngine
from module3_haptic.haptic_encoder import SpatioTactileEncoder


def run_latency_profiler(num_iterations: int = 150):
    print("=" * 80)
    print(" PIPELINE LATENCY PROFILING & REAL-TIME EMBEDDED BENCHMARK")
    print(f" Profiling over {num_iterations} synthetic frame cycles")
    print("=" * 80)

    ego_engine = EgoMotionEngine()
    zone_classifier = ZoneClassifier()
    tof_gate = ToFInterruptGate()
    ttc_engine = TTCEngine()
    haptic_encoder = SpatioTactileEncoder()

    timings = {
        "sensor_fusion_ms": [],
        "zone_classification_ms": [],
        "tof_gating_ms": [],
        "ttc_computation_ms": [],
        "haptic_encoding_ms": [],
        "total_pipeline_ms": [],
    }

    dummy_depth = np.full((TOF_GRID_ROWS, TOF_GRID_COLS), 3.5, dtype=np.float64)
    dummy_depth[2:5, 3:5] = 1.3  # Active obstacle at 1.3m

    for i in range(num_iterations):
        t0 = time.perf_counter()

        # Step 1: Ego-Motion Quaternion Re-projection
        t_step0 = time.perf_counter()
        sensor_vecs, torso_vecs = ego_engine.process_frame(dummy_depth, head_yaw_deg=22.5, head_pitch_deg=2.0)
        t_step1 = time.perf_counter()

        # Step 2: Spatial Zone Binning
        classif = zone_classifier.classify(torso_vecs)
        t_step2 = time.perf_counter()

        # Step 3: Stage 1 Depth Gate & Dynamic ROI
        is_triggered, roi = tof_gate.evaluate_gate(dummy_depth)
        t_step3 = time.perf_counter()

        # Step 4: TTC Kinematics
        sec_dists = {s: classif.sectors[s].min_distance_m for s in ("LEFT", "CENTER", "RIGHT")}
        ttc_res = ttc_engine.process_all_sectors(sec_dists, timestamp=t_step3)
        t_step4 = time.perf_counter()

        # Step 5: 3-Axis Spatio-Tactile Haptic Encoding
        fake_vis_packet = {
            "critical_sector": classif.critical_sector,
            "min_distance_m": classif.min_distance_overall_m,
            "ttc_seconds": ttc_res[classif.critical_sector].ttc_seconds,
            "primary_class": "person",
            "is_head_hazard": False,
        }
        cmd = haptic_encoder.encode(fake_vis_packet)
        t_step5 = time.perf_counter()

        total_ms = (t_step5 - t0) * 1000.0
        timings["sensor_fusion_ms"].append((t_step1 - t_step0) * 1000.0)
        timings["zone_classification_ms"].append((t_step2 - t_step1) * 1000.0)
        timings["tof_gating_ms"].append((t_step3 - t_step2) * 1000.0)
        timings["ttc_computation_ms"].append((t_step4 - t_step3) * 1000.0)
        timings["haptic_encoding_ms"].append((t_step5 - t_step4) * 1000.0)
        timings["total_pipeline_ms"].append(total_ms)

    # Print summary statistics
    print(f"\n{'Subsystem Pipeline Stage':<36} | {'Mean (ms)':<10} | {'P95 (ms)':<10} | {'Target':<10}")
    print("-" * 75)
    for stage, vals in timings.items():
        mean_v = np.mean(vals)
        p95_v = np.percentile(vals, 95)
        target = "< 10.0 ms" if "total" in stage else "< 2.0 ms"
        print(f"{stage:<36} | {mean_v:7.3f} ms | {p95_v:7.3f} ms | {target:<10}")

    total_mean = np.mean(timings["total_pipeline_ms"])
    max_throughput_hz = 1000.0 / total_mean
    print("-" * 75)
    print(f"Total Pipeline Processing Latency: {total_mean:.3f} ms (Throughput: {max_throughput_hz:.1f} Hz)")
    print(f"Real-Time Constraint (<= 30.0 ms): {'PASS' if total_mean < 30.0 else 'FAIL'}")

    # Export to CSV
    csv_file = LOCAL_LOG_DIR / "latency_benchmark.csv"
    with open(csv_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(list(timings.keys()))
        for row in zip(*timings.values()):
            writer.writerow(row)

    print(f"Detailed latency profile exported to: {csv_file}")
    print("=" * 80)


if __name__ == "__main__":
    run_latency_profiler()
