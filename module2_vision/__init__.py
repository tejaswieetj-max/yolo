"""
Module 2: Dual-Stage AI Vision & Time-to-Collision (TTC) Engine
Lead Researcher: Dhivyashree G J. (24BCE1516)
Novelty 02 & 03: Dynamic ROI Inference & Spatio-Tactile TTC
+ YOLO + MediaPipe Parallel Fusion Engine
"""

from .tof_gate import ToFInterruptGate, RegionOfInterest
from .camera_driver import CameraDriver
from .yolo_engine import YOLOv8ROIEngine
from .ttc_engine import TTCEngine, TTCAssessment
from .class_map import CLASS_SEVERITY_WEIGHTS, get_class_name
from .sim_vision import SimulatedVisionTestbed
from .fusion_engine import FusionEngine, FusedDetection, FusionResult
from .mediapipe_detector import MediaPipeDetector, MediaPipeDetectionResult

__all__ = [
    "ToFInterruptGate",
    "RegionOfInterest",
    "CameraDriver",
    "YOLOv8ROIEngine",
    "TTCEngine",
    "TTCAssessment",
    "CLASS_SEVERITY_WEIGHTS",
    "get_class_name",
    "SimulatedVisionTestbed",
    "FusionEngine",
    "FusedDetection",
    "FusionResult",
    "MediaPipeDetector",
    "MediaPipeDetectionResult",
]
