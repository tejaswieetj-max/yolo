"""
Module 3: Spatio-Tactile Haptics & Cloud Telemetry
Lead Researcher: Tejaswi E. (24BCE5518)
Novelty 02: Multi-Axis Tactile Feedback & IoT Telemetry
"""

from .waveform_map import WAVEFORM_LIBRARY, get_waveform_metadata, select_tactile_waveform
from .drv2605l_driver import DRV2605LDriver
from .haptic_encoder import SpatioTactileEncoder, HapticCommand
from .firebase_logger import FirebaseTelemetryLogger
from .sim_haptic import TerminalHapticVisualizer

__all__ = [
    "WAVEFORM_LIBRARY",
    "get_waveform_metadata",
    "select_tactile_waveform",
    "DRV2605LDriver",
    "SpatioTactileEncoder",
    "HapticCommand",
    "FirebaseTelemetryLogger",
    "TerminalHapticVisualizer",
]
