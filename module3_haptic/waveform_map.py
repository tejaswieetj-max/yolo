"""
TI DRV2605L ROM Waveform Mapping & Tactile Texture Presets
Lead Researcher: Tejaswi E. (24BCE5518)

Maps 123 internal DRV2605L ROM waveforms to semantic hazard classes and TTC urgency tiers.
"""

from dataclasses import dataclass
from typing import Dict, Optional

from config import (
    WAVEFORM_SAFE_AWARENESS,
    WAVEFORM_APPROACHING_ALERT,
    WAVEFORM_IMMINENT_COLLISION,
    WAVEFORM_OVERHEAD_HAZARD,
    WAVEFORM_PERSON_SOFT,
    WAVEFORM_VEHICLE_SHARP,
)


@dataclass
class WaveformMetadata:
    id: int
    name: str
    description: str
    duration_ms: int
    sharpness: str  # "Gentle", "Sharp", "Buzz", "Click"


# Key DRV2605L ROM Library Waveform Definitions
WAVEFORM_LIBRARY: Dict[int, WaveformMetadata] = {
    1: WaveformMetadata(1, "Strong Click - 100%", "Crisp singular click for safe presence", 40, "Click"),
    2: WaveformMetadata(2, "Strong Click - 60%", "Soft singular click", 40, "Gentle"),
    12: WaveformMetadata(12, "Triple Click - 100%", "Rapid triple pulse for approaching obstacle", 120, "Sharp"),
    14: WaveformMetadata(14, "Strong Tick 1", "Brief directional tick", 25, "Sharp"),
    15: WaveformMetadata(15, "Medium Click 1 - 100%", "Distinct double-beat for overhead hazard warning", 85, "Sharp"),
    47: WaveformMetadata(47, "Buzz 1 - 100%", "Intense 20 Hz buzz for imminent collision", 250, "Buzz"),
    52: WaveformMetadata(52, "Pulsing Sharp 100%", "Repetitive urgent pulses", 180, "Buzz"),
    64: WaveformMetadata(64, "Transition Hum 1 - 100%", "Smooth sinusoidal vibration for pedestrian courtesy", 150, "Gentle"),
    70: WaveformMetadata(70, "Transition Click 1", "Soft transition click", 60, "Gentle"),
    88: WaveformMetadata(88, "Alert 750ms", "Extended alert for emergency vehicle presence", 750, "Buzz"),
    118: WaveformMetadata(118, "Long Buzz 100%", "Maximum continuous buzz warning", 500, "Buzz"),
}


def get_waveform_metadata(waveform_id: int) -> WaveformMetadata:
    """Retrieve metadata or return a generic fallback."""
    return WAVEFORM_LIBRARY.get(
        waveform_id,
        WaveformMetadata(waveform_id, f"Waveform #{waveform_id}", "Custom DRV2605L effect", 100, "Sharp")
    )


def select_tactile_waveform(
    hazard_class: str,
    urgency_level: str,
    is_overhead: bool = False,
) -> WaveformMetadata:
    """
    Novelty 02 (Axis 3): Multi-Axis Texture Selection.
    Selects distinct tactile texture based on hazard class and urgency level.
    """
    h_class = hazard_class.lower()

    # Rule 1: Imminent Collision overrides texture with maximum alert Buzz
    if urgency_level == "IMMINENT":
        return get_waveform_metadata(WAVEFORM_IMMINENT_COLLISION)

    # Rule 2: Overhead Hazard (e.g. tree branch, awning) uses distinct double-beat click
    if is_overhead or h_class in ("branch", "overhead_hazard"):
        return get_waveform_metadata(WAVEFORM_OVERHEAD_HAZARD)

    # Rule 2b: Stairs / Ground elevation drop-off uses sharp cautionary pulse
    if h_class == "stairs":
        if urgency_level == "APPROACHING":
            return get_waveform_metadata(WAVEFORM_APPROACHING_ALERT)
        return get_waveform_metadata(WAVEFORM_SAFE_AWARENESS)

    # Rule 3: Fast or motorized vehicle (Car, Bus, Truck, Motorcycle) uses sharp pulse
    if h_class in ("car", "truck", "bus", "motorcycle"):
        if urgency_level == "APPROACHING":
            return get_waveform_metadata(WAVEFORM_VEHICLE_SHARP)
        return get_waveform_metadata(WAVEFORM_APPROACHING_ALERT)

    # Rule 4: Pedestrian uses smooth soft transition
    if h_class == "person":
        if urgency_level == "APPROACHING":
            return get_waveform_metadata(WAVEFORM_APPROACHING_ALERT)
        return get_waveform_metadata(WAVEFORM_PERSON_SOFT)

    # Rule 5: Default based on urgency tier
    if urgency_level == "APPROACHING":
        return get_waveform_metadata(WAVEFORM_APPROACHING_ALERT)
    else:
        return get_waveform_metadata(WAVEFORM_SAFE_AWARENESS)
