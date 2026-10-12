"""
YOLO + MediaPipe Parallel Fusion Engine
Merges detections from YOLOv8n and MediaPipe (Object Detection, Pose, Hands)
using a weighted confidence ensemble with cross-validation bonuses.

Runs both models in parallel threads on the same camera frame,
then fuses results for higher classification accuracy.

Adaptive mode switching:
  - FULL:  YOLO + MediaPipe parallel (best accuracy, ~10-14ms on Pi 5)
  - LITE:  YOLO only (lowest power, ~5-7ms on Pi 5)
  - AUTO:  Switches based on CPU temperature and battery level
"""

import time
import threading
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Tuple, Any
from concurrent.futures import ThreadPoolExecutor, Future, TimeoutError
import numpy as np

from config import (
    FUSION_MODE,
    FUSION_CPU_TEMP_THRESHOLD_C,
    FUSION_BATTERY_LOW_PERCENT,
    FUSION_WEIGHT_YOLO,
    FUSION_WEIGHT_MP_OBJECT,
    FUSION_BONUS_POSE_CONFIRM,
    FUSION_BONUS_HANDS_CONFIRM,
    FUSION_INFERENCE_TIMEOUT_S,
    SIMULATION_MODE,
)


@dataclass
class FusedDetection:
    """A single fused detection combining YOLO and MediaPipe evidence."""
    class_name: str
    fused_confidence: float
    yolo_confidence: float
    mp_object_confidence: float
    pose_confirmed: bool
    hands_confirmed: bool
    bbox: Tuple[int, int, int, int]  # (xmin, ymin, xmax, ymax)
    center_x: float
    center_y: float
    source: str  # "yolo", "mediapipe", "fused"


@dataclass
class FusionResult:
    """Complete output from the fusion engine for one frame."""
    detections: List[FusedDetection]
    primary_class: str
    primary_confidence: float
    fusion_mode: str            # "full", "lite"
    yolo_inference_ms: float
    mp_inference_ms: float
    fusion_overhead_ms: float
    total_inference_ms: float
    yolo_detection_count: int
    mp_detection_count: int
    pose_detected: bool
    hands_detected: bool


def _get_cpu_temperature() -> float:
    """Read Raspberry Pi CPU temperature. Returns 0.0 on non-Pi systems."""
    try:
        with open("/sys/class/thermal/thermal_zone0/temp", "r") as f:
            return float(f.read().strip()) / 1000.0
    except Exception:
        return 0.0  # Not on a Pi or permission denied


def _get_battery_percent() -> float:
    """
    Read battery percentage. Returns 100.0 if no battery info available.
    Works with UPS HATs that expose battery via I2C or sysfs.
    """
    try:
        # Try standard Linux power supply sysfs
        import glob
        for path in glob.glob("/sys/class/power_supply/*/capacity"):
            with open(path, "r") as f:
                return float(f.read().strip())
    except Exception:
        pass
    return 100.0  # Assume plugged in if no battery info


def _compute_iou(box_a: Tuple[int, int, int, int], box_b: Tuple[int, int, int, int]) -> float:
    """Compute Intersection over Union between two bounding boxes."""
    xa = max(box_a[0], box_b[0])
    ya = max(box_a[1], box_b[1])
    xb = min(box_a[2], box_b[2])
    yb = min(box_a[3], box_b[3])

    inter = max(0, xb - xa) * max(0, yb - ya)
    if inter == 0:
        return 0.0

    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    union = area_a + area_b - inter

    return inter / union if union > 0 else 0.0


class FusionEngine:
    """
    Orchestrates parallel YOLO + MediaPipe inference and merges results.

    Architecture:
        Camera Frame
            ├── Thread A: YOLOv8n ONNX inference
            └── Thread B: MediaPipe (Object + Pose + Hands)
                    ↓
            Fusion Engine (weighted ensemble)
                    ↓
            FusedDetection list → TTC Engine → Haptics
    """

    def __init__(self, yolo_engine, mp_detector=None):
        """
        Args:
            yolo_engine: YOLOv8ROIEngine instance (always required)
            mp_detector: MediaPipeDetector instance (None in lite mode)
        """
        self.yolo_engine = yolo_engine
        self.mp_detector = mp_detector
        self.executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="fusion")

        # Adaptive mode state
        self._current_mode = "lite"
        self._mode_check_interval = 5.0  # Check every 5 seconds
        self._last_mode_check = 0.0

        # Performance tracking
        self._frame_count = 0
        self._total_yolo_ms = 0.0
        self._total_mp_ms = 0.0

    def should_use_full_mode(self) -> bool:
        """Determine whether to run in FULL (YOLO+MP) or LITE (YOLO-only) mode."""
        if FUSION_MODE == "full":
            return True
        elif FUSION_MODE == "lite":
            return False
        elif FUSION_MODE == "auto":
            now = time.time()
            if now - self._last_mode_check < self._mode_check_interval:
                return self._current_mode == "full"

            self._last_mode_check = now
            cpu_temp = _get_cpu_temperature()
            battery = _get_battery_percent()

            # Switch to lite if overheating or low battery
            if cpu_temp > FUSION_CPU_TEMP_THRESHOLD_C or battery < FUSION_BATTERY_LOW_PERCENT:
                if self._current_mode != "lite":
                    print(f"[Fusion] AUTO → LITE mode (temp={cpu_temp:.1f}°C, battery={battery:.0f}%)")
                self._current_mode = "lite"
                return False
            else:
                if self._current_mode != "full":
                    print(f"[Fusion] AUTO → FULL mode (temp={cpu_temp:.1f}°C, battery={battery:.0f}%)")
                self._current_mode = "full"
                return True
        else:
            return False

    def infer_fused(
        self,
        full_frame: np.ndarray,
        roi,  # RegionOfInterest from tof_gate
        mock_hint_class: Optional[str] = None,
    ) -> FusionResult:
        """
        Run parallel inference and fuse results.

        In FULL mode: YOLO and MediaPipe run concurrently on the same frame.
        In LITE mode: Only YOLO runs.

        Args:
            full_frame: BGR camera frame (H, W, 3)
            roi: RegionOfInterest from ToF depth gate
            mock_hint_class: Simulation hint for YOLO fallback

        Returns:
            FusionResult with merged detections
        """
        t_fusion_start = time.time()
        use_full = self.should_use_full_mode() and self.mp_detector is not None

        yolo_detections = []
        mp_result = None
        yolo_ms = 0.0
        mp_ms = 0.0

        if use_full:
            # PARALLEL EXECUTION: both models on separate threads
            yolo_future = self.executor.submit(
                self._run_yolo, full_frame, roi, mock_hint_class
            )
            mp_future = self.executor.submit(
                self._run_mediapipe, full_frame
            )

            # Wait for YOLO (always needed)
            try:
                yolo_detections, yolo_ms = yolo_future.result(timeout=FUSION_INFERENCE_TIMEOUT_S * 2)
            except (TimeoutError, Exception) as e:
                print(f"[Fusion] YOLO thread error: {e}")
                yolo_detections, yolo_ms = [], 0.0

            # Wait for MediaPipe (with timeout — don't block if slow)
            try:
                mp_result, mp_ms = mp_future.result(timeout=FUSION_INFERENCE_TIMEOUT_S)
            except TimeoutError:
                mp_result, mp_ms = None, 0.0
            except Exception as e:
                mp_result, mp_ms = None, 0.0

        else:
            # LITE MODE: YOLO only (sequential, no thread overhead)
            yolo_detections, yolo_ms = self._run_yolo(full_frame, roi, mock_hint_class)

        # FUSION: merge YOLO + MediaPipe results
        t_merge_start = time.time()
        fused = self._merge_detections(yolo_detections, mp_result)
        merge_ms = (time.time() - t_merge_start) * 1000.0

        # Determine primary detection
        primary_class = "clear"
        primary_conf = 0.0
        if fused:
            best = max(fused, key=lambda d: d.fused_confidence)
            primary_class = best.class_name
            primary_conf = best.fused_confidence

        total_ms = (time.time() - t_fusion_start) * 1000.0

        # Track performance
        self._frame_count += 1
        self._total_yolo_ms += yolo_ms
        self._total_mp_ms += mp_ms

        return FusionResult(
            detections=fused,
            primary_class=primary_class,
            primary_confidence=primary_conf,
            fusion_mode="full" if use_full else "lite",
            yolo_inference_ms=yolo_ms,
            mp_inference_ms=mp_ms,
            fusion_overhead_ms=merge_ms,
            total_inference_ms=total_ms,
            yolo_detection_count=len(yolo_detections),
            mp_detection_count=len(mp_result.objects) if mp_result and mp_result.objects else 0,
            pose_detected=mp_result.pose.is_person_detected if mp_result and mp_result.pose else False,
            hands_detected=(mp_result.hands.num_hands > 0) if mp_result and mp_result.hands else False,
        )

    def _run_yolo(self, frame, roi, mock_hint):
        """Execute YOLO inference and return (detections, time_ms)."""
        t = time.time()
        dets = self.yolo_engine.infer_roi(frame, roi, mock_hint_class=mock_hint)
        ms = (time.time() - t) * 1000.0
        return dets, ms

    def _run_mediapipe(self, frame):
        """Execute MediaPipe inference and return (result, time_ms)."""
        t = time.time()
        result = self.mp_detector.detect(frame)
        ms = (time.time() - t) * 1000.0
        return result, ms

    def _merge_detections(self, yolo_dets, mp_result) -> List[FusedDetection]:
        """
        Weighted ensemble fusion of YOLO and MediaPipe detections.

        Rules:
        1. Each YOLO detection gets base weighted confidence
        2. If MediaPipe Pose confirms a person skeleton → +POSE_BONUS
        3. If MediaPipe Hands detects hands near a YOLO person → +HANDS_BONUS
        4. If MediaPipe Object detects something YOLO missed → add as new detection
        5. If YOLO says "car" but Pose sees a skeleton → reclassify as "person"
        """
        fused: List[FusedDetection] = []

        # Extract MediaPipe signals
        pose = mp_result.pose if mp_result else None
        hands = mp_result.hands if mp_result else None
        mp_objects = mp_result.objects if mp_result else []

        person_pose_confirmed = pose is not None and pose.is_person_detected
        has_hands = hands is not None and hands.num_hands > 0

        # --- Step 1: Process each YOLO detection ---
        for yolo_det in yolo_dets:
            yolo_conf = yolo_det.confidence
            mp_obj_conf = 0.0
            pose_bonus = False
            hands_bonus = False
            final_class = yolo_det.class_name

            yolo_bbox = yolo_det.bbox_global

            # Check if any MediaPipe object detection overlaps this YOLO box
            for mp_obj in mp_objects:
                iou = _compute_iou(yolo_bbox, mp_obj.bbox)
                if iou > 0.3:
                    mp_obj_conf = max(mp_obj_conf, mp_obj.confidence)
                    mp_obj._matched = True  # Mark as matched

            # Cross-validation: Pose confirms person
            if person_pose_confirmed and pose.body_bbox is not None:
                iou_pose = _compute_iou(yolo_bbox, pose.body_bbox)
                if iou_pose > 0.2:
                    pose_bonus = True
                    # Reclassify correction: if YOLO says vehicle but pose says person
                    if final_class in ("car", "truck", "bus", "motorcycle"):
                        if pose.confidence > 0.6:
                            final_class = "person"

            # Cross-validation: Hands near this detection
            if has_hands:
                for hand_bbox in hands.hand_bboxes:
                    iou_hand = _compute_iou(yolo_bbox, hand_bbox)
                    if iou_hand > 0.1:
                        hands_bonus = True
                        break

            # Weighted confidence computation
            fused_conf = FUSION_WEIGHT_YOLO * yolo_conf
            if mp_obj_conf > 0:
                fused_conf += FUSION_WEIGHT_MP_OBJECT * mp_obj_conf
            if pose_bonus:
                fused_conf += FUSION_BONUS_POSE_CONFIRM
            if hands_bonus:
                fused_conf += FUSION_BONUS_HANDS_CONFIRM

            # Clamp to [0, 1]
            fused_conf = min(1.0, max(0.0, fused_conf))

            fused.append(FusedDetection(
                class_name=final_class,
                fused_confidence=fused_conf,
                yolo_confidence=yolo_conf,
                mp_object_confidence=mp_obj_conf,
                pose_confirmed=pose_bonus,
                hands_confirmed=hands_bonus,
                bbox=yolo_bbox,
                center_x=yolo_det.center_x,
                center_y=yolo_det.center_y,
                source="fused" if (mp_obj_conf > 0 or pose_bonus or hands_bonus) else "yolo",
            ))

        # --- Step 2: Add MediaPipe-only detections (YOLO missed) ---
        for mp_obj in mp_objects:
            if getattr(mp_obj, "_matched", False):
                continue  # Already matched to a YOLO detection

            # Only add if confidence is high enough to be standalone
            if mp_obj.confidence >= 0.5:
                fused_conf = FUSION_WEIGHT_MP_OBJECT * mp_obj.confidence
                if person_pose_confirmed and mp_obj.class_name == "person":
                    fused_conf += FUSION_BONUS_POSE_CONFIRM

                fused.append(FusedDetection(
                    class_name=mp_obj.class_name,
                    fused_confidence=min(1.0, fused_conf),
                    yolo_confidence=0.0,
                    mp_object_confidence=mp_obj.confidence,
                    pose_confirmed=person_pose_confirmed and mp_obj.class_name == "person",
                    hands_confirmed=False,
                    bbox=mp_obj.bbox,
                    center_x=mp_obj.center_x,
                    center_y=mp_obj.center_y,
                    source="mediapipe",
                ))

        # --- Step 3: If pose detected person but neither model had a box, add implicit detection ---
        if person_pose_confirmed and pose.body_bbox is not None:
            has_person_det = any(d.class_name == "person" for d in fused)
            if not has_person_det:
                fused.append(FusedDetection(
                    class_name="person",
                    fused_confidence=min(1.0, FUSION_BONUS_POSE_CONFIRM + pose.confidence * 0.3),
                    yolo_confidence=0.0,
                    mp_object_confidence=0.0,
                    pose_confirmed=True,
                    hands_confirmed=has_hands,
                    bbox=pose.body_bbox,
                    center_x=(pose.body_bbox[0] + pose.body_bbox[2]) / 2.0,
                    center_y=(pose.body_bbox[1] + pose.body_bbox[3]) / 2.0,
                    source="pose_only",
                ))

        return fused

    def get_stats(self) -> Dict[str, Any]:
        """Return performance statistics."""
        if self._frame_count == 0:
            return {"frames": 0}
        return {
            "frames": self._frame_count,
            "avg_yolo_ms": self._total_yolo_ms / self._frame_count,
            "avg_mp_ms": self._total_mp_ms / self._frame_count,
            "current_mode": self._current_mode,
        }

    def shutdown(self):
        """Clean up thread pool."""
        self.executor.shutdown(wait=False)
        if self.mp_detector is not None:
            self.mp_detector.release()
