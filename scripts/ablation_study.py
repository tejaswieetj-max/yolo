"""
Ablation Study: Kinematic Ego-Motion Compensation (Novelty 01)
Lead Researcher: Andrea A. (24BCE1142)

Evaluates obstacle vector accuracy WITH vs WITHOUT quaternion rotation transform R(θ)
across head yaw sweeps from -90 deg to +90 deg. Outputs empirical metrics for paper tables.
"""

import math
import csv
import sys
from pathlib import Path
import numpy as np

# Ensure workspace root is in python path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from config import (
    TOF_GRID_ROWS,
    TOF_GRID_COLS,
    BASE_DIR,
    LOCAL_LOG_DIR,
)
from module1_sensor.ego_motion import EgoMotionEngine
from module1_sensor.zone_classifier import ZoneClassifier


def run_ablation_experiment():
    print("=" * 80)
    print(" NOVELTY 01 ABLATION STUDY: HEAD-TO-TORSO KINEMATIC RE-PROJECTION")
    print(" Comparing Torso Azimuth Tracking Error (WITH vs WITHOUT Ego-Motion)")
    print("=" * 80)

    ego_engine = EgoMotionEngine()
    zone_classifier = ZoneClassifier()

    # Head yaw angles to test (degrees)
    yaw_angles = [-90, -60, -45, -30, -15, 0, 15, 30, 45, 60, 90]

    # Fixed true obstacle in torso coordinate frame: directly ahead (Azimuth = 0 deg, Distance = 1.5m)
    true_obstacle_distance_m = 1.5
    true_obstacle_azimuth_deg = 0.0
    true_torso_sector = "CENTER"

    results = []

    LOCAL_LOG_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = LOCAL_LOG_DIR / "ablation_results.csv"

    print(f"\n{'Yaw Angle':<12} | {'WITHOUT Ego-Motion':<28} | {'WITH Ego-Motion':<28} | {'Status'}")
    print(f"{'(deg)':<12} | {'Perceived Az':<12} {'Sector':<14} | {'Corrected Az':<12} {'Sector':<14} |")
    print("-" * 84)

    for yaw in yaw_angles:
        # In sensor frame, turning head by `yaw` makes a torso-centered object appear at `-yaw`
        sensor_azimuth = -yaw

        # Create synthetic 8x8 matrix where obstacle appears at sensor_azimuth
        # Map sensor_azimuth (-45 to +45) to columns 0 to 7
        col_idx = int(np.clip(round((sensor_azimuth + 45.0) / (90.0 / 8.0) - 0.5), 0, 7))
        depth_matrix = np.full((8, 8), 3.5)
        depth_matrix[3:5, col_idx] = true_obstacle_distance_m

        # 1. Condition A: WITHOUT Ego-Motion (Raw Sensor Frame assumed as Torso Frame)
        sensor_vecs = ego_engine.raw_grid_to_sensor_vectors(depth_matrix)
        raw_azimuth = sensor_vecs[col_idx + 3 * 8].azimuth_deg
        # Raw sector classification without transform
        if raw_azimuth < -15.0:
            raw_sector = "LEFT"
        elif raw_azimuth > 15.0:
            raw_sector = "RIGHT"
        else:
            raw_sector = "CENTER"

        error_without = abs(raw_azimuth - true_obstacle_azimuth_deg)

        # 2. Condition B: WITH Ego-Motion (Quaternion R(θ) Re-projection)
        torso_vecs = ego_engine.transform_to_torso_frame(sensor_vecs, head_yaw_deg=yaw)
        classif_res = zone_classifier.classify(torso_vecs)
        corrected_sector = classif_res.critical_sector
        closest_v = classif_res.sectors[corrected_sector].closest_vector
        corrected_azimuth = closest_v.azimuth_torso_deg if closest_v else 0.0

        error_with = abs(corrected_azimuth - true_obstacle_azimuth_deg)
        corrected_ok = corrected_sector == true_torso_sector

        results.append({
            "head_yaw_deg": yaw,
            "raw_azimuth_deg": raw_azimuth,
            "raw_sector": raw_sector,
            "error_without_deg": error_without,
            "corrected_azimuth_deg": corrected_azimuth,
            "corrected_sector": corrected_sector,
            "error_with_deg": error_with,
            "correction_successful": corrected_ok,
        })

        status_str = "[OK] Preserved" if corrected_ok else "[FAIL]"
        print(f"{yaw:+6.1f} deg    | {raw_azimuth:+6.1f} deg   {raw_sector:<14} | {corrected_azimuth:+6.1f} deg   {corrected_sector:<14} | {status_str}")

    # Write ablation results to CSV
    with open(csv_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)

    print("-" * 84)
    # Compute metrics for within-FoV operational range (-45 to +45 deg)
    fov_results = [r for r in results if abs(r["head_yaw_deg"]) <= 45.0]
    avg_err_without_fov = sum(r["error_without_deg"] for r in fov_results) / len(fov_results)
    avg_err_with_fov = sum(r["error_with_deg"] for r in fov_results) / len(fov_results)
    accuracy_without = sum(1 for r in fov_results if r["raw_sector"] == true_torso_sector) / len(fov_results) * 100.0
    accuracy_with = sum(1 for r in fov_results if r["correction_successful"]) / len(fov_results) * 100.0

    print(f"\n--- OPERATIONAL ENVELOPE BENCHMARKS (Head Yaw within [-45°, +45°] FoV) ---")
    print(f"Azimuth Routing Accuracy WITHOUT Ego-Motion: {accuracy_without:5.1f}%  (Baseline)")
    print(f"Azimuth Routing Accuracy WITH Ego-Motion:    {accuracy_with:5.1f}%  (Proposed Novelty 01)")
    print(f"Mean Angular Error WITHOUT Ego-Motion:       {avg_err_without_fov:5.2f}°")
    print(f"Mean Angular Error WITH Ego-Motion:          {avg_err_with_fov:5.2f}°  (Error reduction: {(1.0 - avg_err_with_fov/avg_err_without_fov)*100:.1f}%)")
    print(f"Ablation empirical dataset exported to: {csv_path}")
    print("=" * 80)


if __name__ == "__main__":
    run_ablation_experiment()
