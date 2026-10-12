"""
Zero-Hardware Sensor Simulation Engine
Lead Researcher: Andrea A. (24BCE1142)

Synthesizes realistic 8x8 Time-of-Flight depth matrices and sinusoidal IMU
head-yaw sweeps so Modules 2 and 3 can develop and benchmark without physical sensors.
"""

import math
import time
import numpy as np
from typing import Tuple, Dict, Any

from config import (
    TOF_GRID_ROWS,
    TOF_GRID_COLS,
    TOF_MAX_RANGE_METERS,
    MAX_COMPENSATED_YAW_DEG,
)


class SimulatedSensorSystem:
    """Generates synthetic multi-zone LiDAR depth maps and head yaw sweeps."""

    def __init__(self, sweep_frequency_hz: float = 0.25, max_sweep_yaw_deg: float = 40.0):
        self.sweep_freq = sweep_frequency_hz
        self.max_yaw = min(max_sweep_yaw_deg, MAX_COMPENSATED_YAW_DEG)
        self.start_time = time.time()
        self.scenario_timer = 0.0
        self.scenario_index = 0
        self.scenarios = ["approaching_person", "approaching_vehicle_left", "overhead_branch", "safe_clearance"]

    def get_simulated_head_orientation(self, current_time: float) -> Tuple[float, float, float]:
        """
        Synthesize sinusoidal head yaw sweep:
        yaw = max_yaw * sin(2 * pi * f * t)
        """
        elapsed = current_time - self.start_time
        yaw_deg = self.max_yaw * math.sin(2.0 * math.pi * self.sweep_freq * elapsed)
        # Small natural pitch variation (+/- 5 deg)
        pitch_deg = 4.0 * math.sin(2.0 * math.pi * 0.1 * elapsed)
        roll_deg = 1.0 * math.sin(2.0 * math.pi * 0.05 * elapsed)
        return float(yaw_deg), float(pitch_deg), float(roll_deg)

    def generate_depth_matrix(self, current_time: float) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Generates dynamic 8x8 depth array (meters) simulating realistic obstacles.
        Cycles through practical outdoor navigation scenarios every 10 seconds.
        """
        elapsed = current_time - self.start_time
        # Switch scenarios every 8 seconds
        cycle_time = elapsed % 32.0
        scenario_idx = int(cycle_time // 8.0)
        scenario_progress = (cycle_time % 8.0) / 8.0  # 0.0 to 1.0

        # Background base wall / distant ground at 3.5m - 4.0m
        matrix = np.random.uniform(3.4, 3.8, size=(TOF_GRID_ROWS, TOF_GRID_COLS))

        ground_truth = {
            "scenario": self.scenarios[scenario_idx],
            "distance_m": 4.0,
            "target_class": "clear",
            "sector": "NONE",
            "is_overhead": False,
        }

        if scenario_idx == 0:
            # Scenario 1: Pedestrian directly in center, walking towards user
            # Distance decreases from 2.8m to 0.7m
            dist = 2.8 - (2.1 * scenario_progress) + np.random.normal(0, 0.03)
            dist = max(0.5, dist)
            # Center columns 3-4, torso/head rows 2-5
            matrix[2:6, 3:5] = dist
            # Fringe falloff
            matrix[1, 3:5] = dist + 0.2
            matrix[6, 3:5] = dist + 0.2
            ground_truth.update({
                "scenario": "approaching_person",
                "distance_m": dist,
                "target_class": "person",
                "sector": "CENTER",
                "is_overhead": False,
            })

        elif scenario_idx == 1:
            # Scenario 2: Rapid vehicle / cyclist approaching on the LEFT
            # Distance decreases from 3.2m to 0.9m
            dist = 3.2 - (2.3 * scenario_progress) + np.random.normal(0, 0.04)
            dist = max(0.7, dist)
            # Left columns 0-2, rows 3-6
            matrix[3:7, 0:3] = dist
            ground_truth.update({
                "scenario": "approaching_vehicle_left",
                "distance_m": dist,
                "target_class": "car",
                "sector": "LEFT",
                "is_overhead": False,
            })

        elif scenario_idx == 2:
            # Scenario 3: Overhead Tree Branch / Signboard at Head Height (Novelty 01/02 test)
            # Static overhead obstacle around 1.3m - 1.6m
            dist = 1.8 - (0.6 * scenario_progress)
            # Head-height rows 0-2 across Center and Right (cols 2-6)
            matrix[0:3, 2:7] = dist
            ground_truth.update({
                "scenario": "overhead_branch",
                "distance_m": dist,
                "target_class": "overhead",
                "sector": "CENTER",
                "is_overhead": True,
            })

        else:
            # Scenario 4: Clear hallway / open sidewalk
            ground_truth.update({
                "scenario": "safe_clearance",
                "distance_m": 3.6,
                "target_class": "clear",
                "sector": "NONE",
                "is_overhead": False,
            })

        return np.clip(matrix, 0.1, TOF_MAX_RANGE_METERS), ground_truth
