"""
Novelty 02: 3-Axis Spatio-Tactile Haptic Encoding Engine
Lead Researcher: Tejaswi E. (24BCE5518)

Concurrently modulates:
  - Axis 1 (Spatial Azimuth): Directional neckband motor (Left, Center, Right)
  - Axis 2 (Urgency via TTC): Pulse firing cadence (20 Hz buzz vs periodic ping)
  - Axis 3 (Hazard Texture): TI DRV2605L waveform library selection
"""

import time
from dataclasses import dataclass
from typing import Optional, Dict

from config import (
    MOTOR_LEFT,
    MOTOR_CENTER,
    MOTOR_RIGHT,
    TTC_IMMINENT_THRESHOLD_S,
    TTC_APPROACHING_THRESHOLD_S,
)
from .waveform_map import select_tactile_waveform, WaveformMetadata


@dataclass
class HapticCommand:
    motor_channel: str          # "LEFT", "CENTER", "RIGHT"
    waveform_id: int
    waveform_name: str
    urgency_level: str          # "IMMINENT", "APPROACHING", "SAFE"
    pulse_cadence_hz: float     # Firing rate in Hz
    should_pulse: bool          # True if cadence timer dictates firing now
    ttc_seconds: float
    distance_m: float
    hazard_class: str


class SpatioTactileEncoder:
    """Translates high-level perception packets into multi-axis tactile signals."""

    def __init__(self):
        # Track last pulse timestamp per motor channel
        self.last_pulse_times: Dict[str, float] = {
            MOTOR_LEFT: 0.0,
            MOTOR_CENTER: 0.0,
            MOTOR_RIGHT: 0.0,
        }

    def encode(self, vision_packet: dict) -> HapticCommand:
        """
        Takes vision/TTC packet and generates 3-axis spatio-tactile actuation command.
        """
        now = time.time()

        # Axis 1: Azimuth Motor Selection
        motor = vision_packet.get("critical_sector", MOTOR_CENTER)
        if motor not in (MOTOR_LEFT, MOTOR_CENTER, MOTOR_RIGHT):
            motor = MOTOR_CENTER

        ttc = float(vision_packet.get("ttc_seconds", 99.0))
        dist = float(vision_packet.get("min_distance_m", 4.0))
        hazard_class = vision_packet.get("primary_class", "generic_obstacle")
        is_overhead = vision_packet.get("is_head_hazard", False)

        # Axis 2: Urgency Classification & Cadence Modulation
        if ttc < TTC_IMMINENT_THRESHOLD_S:
            urgency = "IMMINENT"
            cadence_hz = 20.0     # 20 Hz rapid buzz
            min_interval = 0.05   # 50 ms
        elif ttc <= TTC_APPROACHING_THRESHOLD_S:
            urgency = "APPROACHING"
            cadence_hz = 4.0      # 4 Hz double-pulse cadence
            min_interval = 0.25   # 250 ms
        else:
            urgency = "SAFE"
            cadence_hz = 1.0      # 1 Hz gentle presence ping
            min_interval = 1.0    # 1000 ms

        # Axis 3: Tactile Texture Selection from DRV2605L ROM Library
        waveform_meta: WaveformMetadata = select_tactile_waveform(
            hazard_class=hazard_class,
            urgency_level=urgency,
            is_overhead=is_overhead,
        )

        # Check cadence interval
        last_fired = self.last_pulse_times[motor]
        should_pulse = (now - last_fired) >= min_interval

        if should_pulse:
            self.last_pulse_times[motor] = now

        return HapticCommand(
            motor_channel=motor,
            waveform_id=waveform_meta.id,
            waveform_name=waveform_meta.name,
            urgency_level=urgency,
            pulse_cadence_hz=cadence_hz,
            should_pulse=should_pulse,
            ttc_seconds=ttc,
            distance_m=dist,
            hazard_class=hazard_class,
        )
