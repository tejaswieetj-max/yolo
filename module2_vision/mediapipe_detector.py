"""
MediaPipe Parallel Detector Module
Runs MediaPipe Object Detection, Pose, and Hands on camera frames
alongside YOLOv8n for fusion-based classification.

Designed to run in a parallel thread — receives a camera frame,
returns structured detections that the Fusion Engine merges with YOLO results.
"""

import time
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict, Any

from config import (
    MP_ENABLE_OBJECT_DETECTION,
    MP_ENABLE_POSE,
    MP_ENABLE_HANDS,
    MP_OBJECT_CONFIDENCE_THRESHOLD,
    MP_POSE_CONFIDENCE_THRESHOLD,
    MP_HANDS_CONFIDENCE_THRESHOLD,
    CAMERA_WIDTH,
    CAMERA_HEIGHT,
)


@dataclass
class MPObjectDetection:
    """A single object detected by MediaPipe Object Detection."""
    class_name: str
    confidence: float
    bbox: Tuple[int, int, int, int]  # (xmin, ymin, xmax, ymax) in pixel coords
    center_x: float
    center_y: float


@dataclass
class MPPoseResult:
    """Pose landmark detection result."""
    is_person_detected: bool
    confidence: float
    num_visible_landmarks: int
    # Key landmark positions (normalized 0-1)
    nose: Optional[Tuple[float, float]] = None
    left_shoulder: Optional[Tuple[float, float]] = None
    right_shoulder: Optional[Tuple[float, float]] = None
    left_hip: Optional[Tuple[float, float]] = None
    right_hip: Optional[Tuple[float, float]] = None
    # Estimated body bounding box (pixel coords)
    body_bbox: Optional[Tuple[int, int, int, int]] = None


@dataclass
class MPHandsResult:
    """Hand landmark detection result."""
    num_hands: int
    confidence: float
    hand_landmarks: List[Dict[str, float]] = field(default_factory=list)
    # Bounding boxes for each detected hand (pixel coords)
    hand_bboxes: List[Tuple[int, int, int, int]] = field(default_factory=list)


@dataclass
class MediaPipeDetectionResult:
    """Combined result from all MediaPipe sub-models."""
    objects: List[MPObjectDetection]
    pose: Optional[MPPoseResult]
    hands: Optional[MPHandsResult]
    inference_time_ms: float
    models_active: List[str]  # e.g. ["object_detection", "pose", "hands"]


class MediaPipeDetector:
    """
    Runs MediaPipe models (Object Detection, Pose, Hands) on a camera frame.

    All models run on-device using TFLite — no network required.
    Designed to be called from a parallel thread alongside YOLO inference.
    """

    def __init__(self):
        self.mp_hands = None
        self.mp_pose = None
        self.mp_object = None
        self.hands_detector = None
        self.pose_detector = None
        self.object_detector = None
        self.is_loaded = False
        self.active_models: List[str] = []

        self._init_models()

    def _init_models(self):
        """Initialize all enabled MediaPipe models."""
        try:
            import mediapipe as mp
        except ImportError:
            print("[MediaPipe Detector] mediapipe not installed. Running in simulated mode.")
            return

        # 1. Hands Detection
        if MP_ENABLE_HANDS:
            try:
                self.mp_hands = mp.solutions.hands
                self.hands_detector = self.mp_hands.Hands(
                    static_image_mode=False,
                    max_num_hands=2,
                    min_detection_confidence=MP_HANDS_CONFIDENCE_THRESHOLD,
                    min_tracking_confidence=0.4,
                )
                self.active_models.append("hands")
                print("[MediaPipe Detector] Hands model loaded")
            except Exception as e:
                print(f"[MediaPipe Detector] Hands init failed: {e}")

        # 2. Pose Detection
        if MP_ENABLE_POSE:
            try:
                self.mp_pose = mp.solutions.pose
                self.pose_detector = self.mp_pose.Pose(
                    static_image_mode=False,
                    model_complexity=0,  # Lite model for Pi 5 speed
                    min_detection_confidence=MP_POSE_CONFIDENCE_THRESHOLD,
                    min_tracking_confidence=0.4,
                )
                self.active_models.append("pose")
                print("[MediaPipe Detector] Pose model loaded (complexity=0, Lite)")
            except Exception as e:
                print(f"[MediaPipe Detector] Pose init failed: {e}")

        # 3. Object Detection (using MediaPipe's ObjectDetector task API if available)
        if MP_ENABLE_OBJECT_DETECTION:
            try:
                from mediapipe.tasks import python as mp_tasks
                from mediapipe.tasks.python import vision as mp_vision

                base_options = mp_tasks.BaseOptions(
                    model_asset_path=self._find_object_model()
                )
                options = mp_vision.ObjectDetectorOptions(
                    base_options=base_options,
                    score_threshold=MP_OBJECT_CONFIDENCE_THRESHOLD,
                    max_results=10,
                    running_mode=mp_vision.RunningMode.IMAGE,
                )
                self.object_detector = mp_vision.ObjectDetector.create_from_options(options)
                self.active_models.append("object_detection")
                print("[MediaPipe Detector] Object Detection model loaded (EfficientDet-Lite)")
            except Exception as e:
                print(f"[MediaPipe Detector] Object Detection init skipped ({e}). "
                      "Pose + Hands will still contribute to fusion.")

        self.is_loaded = len(self.active_models) > 0
        print(f"[MediaPipe Detector] Active models: {self.active_models}")

    def _find_object_model(self) -> str:
        """Locate or download the EfficientDet-Lite0 .tflite model."""
        from pathlib import Path
        import urllib.request

        model_dir = Path(__file__).resolve().parent.parent / "models"
        model_dir.mkdir(exist_ok=True)
        model_path = model_dir / "efficientdet_lite0.tflite"

        if not model_path.exists():
            url = ("https://storage.googleapis.com/mediapipe-models/"
                   "object_detector/efficientdet_lite0/int8/latest/"
                   "efficientdet_lite0.tflite")
            print("[MediaPipe Detector] Downloading EfficientDet-Lite0 model...")
            try:
                urllib.request.urlretrieve(url, str(model_path))
                print(f"[MediaPipe Detector] Model saved to {model_path}")
            except Exception as e:
                raise RuntimeError(f"Failed to download model: {e}")

        return str(model_path)

    def detect(self, frame: np.ndarray) -> MediaPipeDetectionResult:
        """
        Run all active MediaPipe models on the given BGR frame.
        Returns combined detection result.
        """
        t_start = time.time()

        objects: List[MPObjectDetection] = []
        pose_result: Optional[MPPoseResult] = None
        hands_result: Optional[MPHandsResult] = None

        if not self.is_loaded:
            return self._simulated_detect(frame, t_start)

        h, w = frame.shape[:2]

        # Convert BGR -> RGB once for all models
        import cv2
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # --- Object Detection ---
        if self.object_detector is not None:
            try:
                import mediapipe as mp
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
                det_result = self.object_detector.detect(mp_image)

                for detection in det_result.detections:
                    bbox = detection.bounding_box
                    category = detection.categories[0]
                    xmin = max(0, bbox.origin_x)
                    ymin = max(0, bbox.origin_y)
                    xmax = min(w, xmin + bbox.width)
                    ymax = min(h, ymin + bbox.height)

                    objects.append(MPObjectDetection(
                        class_name=category.category_name.lower(),
                        confidence=category.score,
                        bbox=(xmin, ymin, xmax, ymax),
                        center_x=(xmin + xmax) / 2.0,
                        center_y=(ymin + ymax) / 2.0,
                    ))
            except Exception:
                pass  # Non-fatal: other models still contribute

        # --- Pose Detection ---
        if self.pose_detector is not None:
            try:
                pose_results = self.pose_detector.process(rgb_frame)
                if pose_results.pose_landmarks:
                    landmarks = pose_results.pose_landmarks.landmark
                    visible_count = sum(1 for lm in landmarks if lm.visibility > 0.5)
                    avg_visibility = sum(lm.visibility for lm in landmarks) / len(landmarks)

                    nose = (landmarks[0].x, landmarks[0].y) if landmarks[0].visibility > 0.3 else None
                    l_shoulder = (landmarks[11].x, landmarks[11].y) if landmarks[11].visibility > 0.3 else None
                    r_shoulder = (landmarks[12].x, landmarks[12].y) if landmarks[12].visibility > 0.3 else None
                    l_hip = (landmarks[23].x, landmarks[23].y) if landmarks[23].visibility > 0.3 else None
                    r_hip = (landmarks[24].x, landmarks[24].y) if landmarks[24].visibility > 0.3 else None

                    vis_xs = [lm.x for lm in landmarks if lm.visibility > 0.3]
                    vis_ys = [lm.y for lm in landmarks if lm.visibility > 0.3]
                    body_bbox = None
                    if vis_xs and vis_ys:
                        body_bbox = (
                            int(min(vis_xs) * w), int(min(vis_ys) * h),
                            int(max(vis_xs) * w), int(max(vis_ys) * h),
                        )

                    pose_result = MPPoseResult(
                        is_person_detected=visible_count >= 5,
                        confidence=avg_visibility,
                        num_visible_landmarks=visible_count,
                        nose=nose,
                        left_shoulder=l_shoulder,
                        right_shoulder=r_shoulder,
                        left_hip=l_hip,
                        right_hip=r_hip,
                        body_bbox=body_bbox,
                    )
                else:
                    pose_result = MPPoseResult(
                        is_person_detected=False, confidence=0.0, num_visible_landmarks=0,
                    )
            except Exception:
                pass

        # --- Hands Detection ---
        if self.hands_detector is not None:
            try:
                hands_results = self.hands_detector.process(rgb_frame)
                if hands_results.multi_hand_landmarks:
                    num_hands = len(hands_results.multi_hand_landmarks)
                    hand_bboxes = []
                    hand_lm_list = []

                    for hand_landmarks in hands_results.multi_hand_landmarks:
                        xs = [lm.x for lm in hand_landmarks.landmark]
                        ys = [lm.y for lm in hand_landmarks.landmark]
                        xmin = int(min(xs) * w)
                        ymin = int(min(ys) * h)
                        xmax = int(max(xs) * w)
                        ymax = int(max(ys) * h)
                        hand_bboxes.append((xmin, ymin, xmax, ymax))

                        hand_lm_list.append({
                            "wrist_x": hand_landmarks.landmark[0].x,
                            "wrist_y": hand_landmarks.landmark[0].y,
                            "index_tip_x": hand_landmarks.landmark[8].x,
                            "index_tip_y": hand_landmarks.landmark[8].y,
                        })

                    avg_conf = sum(
                        hands_results.multi_handedness[i].classification[0].score
                        for i in range(num_hands)
                    ) / num_hands if hands_results.multi_handedness else 0.5

                    hands_result = MPHandsResult(
                        num_hands=num_hands,
                        confidence=avg_conf,
                        hand_landmarks=hand_lm_list,
                        hand_bboxes=hand_bboxes,
                    )
                else:
                    hands_result = MPHandsResult(num_hands=0, confidence=0.0)
            except Exception:
                pass

        elapsed_ms = (time.time() - t_start) * 1000.0

        return MediaPipeDetectionResult(
            objects=objects,
            pose=pose_result,
            hands=hands_result,
            inference_time_ms=elapsed_ms,
            models_active=self.active_models,
        )

    def _simulated_detect(self, frame: np.ndarray, t_start: float) -> MediaPipeDetectionResult:
        """Generate simulated MediaPipe detections for testing without mediapipe installed."""
        h, w = frame.shape[:2]

        pose_result = MPPoseResult(
            is_person_detected=True,
            confidence=0.72,
            num_visible_landmarks=12,
            nose=(0.5, 0.3),
            left_shoulder=(0.4, 0.4),
            right_shoulder=(0.6, 0.4),
            body_bbox=(int(0.3 * w), int(0.2 * h), int(0.7 * w), int(0.9 * h)),
        )

        hands_result = MPHandsResult(num_hands=0, confidence=0.0)

        elapsed_ms = (time.time() - t_start) * 1000.0

        return MediaPipeDetectionResult(
            objects=[],
            pose=pose_result,
            hands=hands_result,
            inference_time_ms=elapsed_ms,
            models_active=["pose_simulated"],
        )

    def release(self):
        """Release all MediaPipe resources."""
        if self.hands_detector is not None:
            self.hands_detector.close()
        if self.pose_detector is not None:
            self.pose_detector.close()
        if self.object_detector is not None:
            self.object_detector.close()
        print("[MediaPipe Detector] Resources released.")
