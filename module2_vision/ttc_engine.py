"""
Novelty 02: Time-to-Collision (TTC) Dynamic Kinematics Engine
Lead Researcher: Dhivyashree G J. (24BCE1516)

Calculates approach velocity (v = -dd/dt) across a 5-frame rolling window
and computes Time-to-Collision (TTC = d / v) to modulate haptic urgency.
"""

import time
from collections import deque
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

from config import (
    TTC_ROLLING_WINDOW_SIZE,
    TTC_MIN_VELOCITY_MPS,
    TTC_IMMINENT_THRESHOLD_S,
    TTC_APPROACHING_THRESHOLD_S,
    MOTOR_LEFT,
    MOTOR_CENTER,
    MOTOR_RIGHT,
    ALL_MOTORS,
)


@dataclass
class TTCAssessment:
    sector: str
    current_distance_m: float
    approach_velocity_mps: float   # Positive = closing in, Negative = moving away
    ttc_seconds: float             # Estimated seconds to collision
    urgency_level: str             # "IMMINENT" (<0.8s), "APPROACHING" (0.8-2.0s), "SAFE" (>2.0s)
    is_closing: bool


class TTCEngine:
    """Tracks distance time-series per sector to compute velocity and TTC."""

    def __init__(self, window_size: int = TTC_ROLLING_WINDOW_SIZE):
        self.window_size = window_size
        # Deque of (timestamp, distance) pairs for each sector
        self.history: Dict[str, deque] = {
            m: deque(maxlen=window_size) for m in ALL_MOTORS
        }

    def update_sector(self, sector: str, distance_m: float, timestamp: Optional[float] = None) -> TTCAssessment:
        """
        Add a distance observation and update TTC metrics for the given sector.
        """
        if timestamp is None:
            timestamp = time.time()

        buf = self.history.get(sector)
        if buf is None:
            buf = deque(maxlen=self.window_size)
            self.history[sector] = buf

        buf.append((timestamp, float(distance_m)))

        # Need at least 2 points to compute velocity
        if len(buf) < 2:
            return TTCAssessment(
                sector=sector,
                current_distance_m=distance_m,
                approach_velocity_mps=0.0,
                ttc_seconds=99.0,
                urgency_level="SAFE",
                is_closing=False,
            )

        # Compute approach velocity using first and latest point in window
        t_first, d_first = buf[0]
        t_last, d_last = buf[-1]
        dt = t_last - t_first

        if dt < 1e-4:
            velocity = 0.0
        else:
            # Positive velocity means distance is decreasing (closing in)
            velocity = (d_first - d_last) / dt

        is_closing = velocity > 0.05  # Closing threshold: > 5 cm/s

        if is_closing:
            clamped_vel = max(velocity, TTC_MIN_VELOCITY_MPS)
            ttc = d_last / clamped_vel
        else:
            # Stationary or moving away
            ttc = 99.0

        # Classify Urgency Level
        if is_closing and ttc < TTC_IMMINENT_THRESHOLD_S:
            urgency = "IMMINENT"
        elif is_closing and ttc <= TTC_APPROACHING_THRESHOLD_S:
            urgency = "APPROACHING"
        else:
            urgency = "SAFE"

        return TTCAssessment(
            sector=sector,
            current_distance_m=d_last,
            approach_velocity_mps=velocity,
            ttc_seconds=ttc,
            urgency_level=urgency,
            is_closing=is_closing,
        )

    def process_all_sectors(self, sector_distances: Dict[str, float], timestamp: Optional[float] = None) -> Dict[str, TTCAssessment]:
        """Update and return assessments for all 3 sectors."""
        results = {}
        for sector, dist in sector_distances.items():
            results[sector] = self.update_sector(sector, dist, timestamp)
        return results
