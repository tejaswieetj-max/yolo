"""
MediaPipe Camera-Based Sensor System
Replaces zero-hardware synthetic simulation with real webcam + MediaPipe Hands.

Uses hand landmark tracking to derive:
  - Distance (from hand bounding box size relative to frame)
  - Velocity & classification (fast hand = vehicle, slow hand = person/wall)
  - Spatial sector (hand horizontal position -> Left/Center/Right)
"""

import math
import time
import cv2
import numpy as np
from typing import Tuple, Dict, Any, Optional
from collections import deque
from multiprocessing import Queue

try:
    import mediapipe as mp
except ImportError:
    raise ImportError("mediapipe is required. Install with: pip install mediapipe")

from config import (
    TOF_GRID_ROWS,
    TOF_GRID_COLS,
    TOF_MAX_RANGE_METERS,
    CAMERA_WIDTH,
    CAMERA_HEIGHT,
    MP_DISTANCE_CALIBRATION_K,
    MP_VEHICLE_VELOCITY_THRESHOLD,
    MP_VELOCITY_SMOOTHING_WINDOW,
    MP_MIN_HAND_SIZE_RATIO,
)


class MediaPipeSensorSystem:
    """
    Generates real 8x8 depth matrices and classification from webcam hand tracking.
    
    How it works:
    - MediaPipe Hands detects the user's hand in the webcam frame
    - Hand bounding box size is inversely proportional to distance:
        distance ≈ K / bbox_diagonal_ratio
    - Hand movement velocity between frames determines classification:
        fast movement (>threshold) → 'car' (vehicle)
        slow movement (≤threshold) → 'person' (static wall/obstacle)
    - Hand horizontal position maps to Left/Center/Right azimuth sectors
    """

    def __init__(self, camera_index: int = 0, show_preview: bool = True, queue_video_feed: Optional[Queue] = None):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.6,
            min_tracking_confidence=0.5,
        )
        self.mp_drawing = mp.solutions.drawing_utils

        # Camera setup
        self.cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)

        if not self.cap.isOpened():
            raise RuntimeError(
                f"[MediaPipe Sensor] Cannot open webcam at index {camera_index}. "
                "Check that your camera is connected and not in use by another app."
            )

        print(f"[MediaPipe Sensor] Webcam opened successfully (index={camera_index})")
        self.show_preview = show_preview
        self.queue_video_feed = queue_video_feed

        # Calibration constant: distance ≈ K / hand_bbox_diagonal_ratio
        self.K = MP_DISTANCE_CALIBRATION_K
        self.vehicle_velocity_threshold = MP_VEHICLE_VELOCITY_THRESHOLD
        self.min_hand_size = MP_MIN_HAND_SIZE_RATIO

        # State tracking for velocity computation
        self.prev_hand_center = None       # (cx, cy) normalized 0-1
        self.prev_time = None
        self.velocity_history = deque(maxlen=MP_VELOCITY_SMOOTHING_WINDOW)
        self.distance_history = deque(maxlen=5)

        # Last known state (for smooth transitions)
        self.last_distance = TOF_MAX_RANGE_METERS
        self.last_class = "clear"
        self.frames_without_hand = 0

    def get_simulated_head_orientation(self, current_time: float) -> Tuple[float, float, float]:
        """
        Returns fixed head orientation (no IMU available).
        The ego-motion math still runs but applies identity transform.
        """
        return 0.0, 0.0, 0.0

    def _compute_hand_bbox(self, hand_landmarks) -> Tuple[float, float, float, float, float, float]:
        """
        Compute bounding box of hand landmarks in normalized coordinates.
        Returns: (xmin, ymin, xmax, ymax, center_x, center_y) all in [0, 1]
        """
        xs = [lm.x for lm in hand_landmarks.landmark]
        ys = [lm.y for lm in hand_landmarks.landmark]

        xmin, xmax = min(xs), max(xs)
        ymin, ymax = min(ys), max(ys)
        cx = (xmin + xmax) / 2.0
        cy = (ymin + ymax) / 2.0

        return xmin, ymin, xmax, ymax, cx, cy

    def _estimate_distance(self, xmin, ymin, xmax, ymax) -> float:
        """
        Estimate distance from hand bounding box size.
        Larger hand in frame = closer to camera.
        Uses inverse proportional mapping: distance ≈ K / bbox_diagonal_ratio
        """
        width = xmax - xmin
        height = ymax - ymin
        diagonal = math.sqrt(width ** 2 + height ** 2)

        if diagonal < self.min_hand_size:
            return TOF_MAX_RANGE_METERS

        distance = self.K / diagonal
        distance = max(0.2, min(distance, TOF_MAX_RANGE_METERS))

        # Smooth distance with history
        self.distance_history.append(distance)
        smoothed = sum(self.distance_history) / len(self.distance_history)

        return smoothed

    def _compute_velocity(self, cx: float, cy: float, current_time: float) -> float:
        """
        Compute hand movement velocity in normalized units/second.
        Uses frame-to-frame landmark center displacement.
        """
        if self.prev_hand_center is None or self.prev_time is None:
            self.prev_hand_center = (cx, cy)
            self.prev_time = current_time
            return 0.0

        dt = current_time - self.prev_time
        if dt < 0.001:
            return 0.0

        dx = cx - self.prev_hand_center[0]
        dy = cy - self.prev_hand_center[1]
        pixel_velocity = math.sqrt(dx ** 2 + dy ** 2) / dt

        self.prev_hand_center = (cx, cy)
        self.prev_time = current_time

        # Smooth velocity
        self.velocity_history.append(pixel_velocity)
        smoothed_vel = sum(self.velocity_history) / len(self.velocity_history)

        return smoothed_vel

    def _classify_by_velocity(self, velocity: float) -> str:
        """
        Classify the detected object based on hand movement speed.
        Fast hand movement → vehicle (car)
        Slow/stationary hand → person (wall/static obstacle)
        """
        if velocity > self.vehicle_velocity_threshold:
            return "car"
        else:
            return "person"

    def _determine_sector(self, cx: float) -> str:
        """
        Map hand horizontal center to spatial sector.
        Left third → LEFT, Middle third → CENTER, Right third → RIGHT
        """
        if cx < 0.33:
            return "LEFT"
        elif cx > 0.66:
            return "RIGHT"
        else:
            return "CENTER"

    def _hand_to_grid_zones(self, xmin, ymin, xmax, ymax) -> Tuple[int, int, int, int]:
        """
        Map hand bounding box (normalized 0-1) to 8x8 grid zone indices.
        Returns: (col_start, col_end, row_start, row_end)
        """
        col_start = max(0, int(xmin * TOF_GRID_COLS))
        col_end = min(TOF_GRID_COLS, int(math.ceil(xmax * TOF_GRID_COLS)))
        row_start = max(0, int(ymin * TOF_GRID_ROWS))
        row_end = min(TOF_GRID_ROWS, int(math.ceil(ymax * TOF_GRID_ROWS)))
        return col_start, col_end, row_start, row_end

    def generate_depth_matrix(self, current_time: float) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Captures a webcam frame, runs MediaPipe Hands, and generates:
          - 8x8 depth matrix with real distance estimates
          - Ground truth dict with classification and sector info
        """
        # Background: all zones safe at far distance
        matrix = np.random.uniform(3.4, 3.8, size=(TOF_GRID_ROWS, TOF_GRID_COLS))

        ground_truth = {
            "scenario": "mediapipe_live",
            "distance_m": TOF_MAX_RANGE_METERS,
            "target_class": "clear",
            "sector": "NONE",
            "is_overhead": False,
            "hand_velocity": 0.0,
            "source": "mediapipe_camera",
        }

        ret, frame = self.cap.read()
        if not ret or frame is None:
            self.frames_without_hand += 1
            return np.clip(matrix, 0.1, TOF_MAX_RANGE_METERS), ground_truth

        # Flip horizontally for natural mirror view
        frame = cv2.flip(frame, 1)

        # Convert BGR to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_frame)

        if results.multi_hand_landmarks:
            self.frames_without_hand = 0
            hand_landmarks = results.multi_hand_landmarks[0]

            # 1. Compute hand bounding box
            xmin, ymin, xmax, ymax, cx, cy = self._compute_hand_bbox(hand_landmarks)

            # 2. Estimate distance from hand size
            distance = self._estimate_distance(xmin, ymin, xmax, ymax)
            self.last_distance = distance

            # 3. Compute hand velocity
            velocity = self._compute_velocity(cx, cy, current_time)

            # 4. Classify: fast = vehicle, slow = person/wall
            target_class = self._classify_by_velocity(velocity)
            self.last_class = target_class

            # 5. Determine spatial sector
            sector = self._determine_sector(cx)

            # 6. Fill 8x8 grid zones covered by the hand with estimated distance
            c_start, c_end, r_start, r_end = self._hand_to_grid_zones(xmin, ymin, xmax, ymax)

            # Expand zone coverage slightly for robustness
            c_start = max(0, c_start - 1)
            c_end = min(TOF_GRID_COLS, c_end + 1)
            r_start = max(0, r_start - 1)
            r_end = min(TOF_GRID_ROWS, r_end + 1)

            matrix[r_start:r_end, c_start:c_end] = distance
            # Add slight noise for realism
            noise = np.random.normal(0, 0.02, size=(r_end - r_start, c_end - c_start))
            matrix[r_start:r_end, c_start:c_end] += noise

            ground_truth.update({
                "scenario": f"mediapipe_{target_class}",
                "distance_m": distance,
                "target_class": target_class,
                "sector": sector,
                "is_overhead": ymin < 0.25,  # Hand in upper quarter = overhead
                "hand_velocity": velocity,
                "hand_bbox": [xmin, ymin, xmax, ymax],
                "hand_center": [cx, cy],
            })

            if self.show_preview:
                # Show preview window with landmarks (for debugging / demo)
                self.mp_drawing.draw_landmarks(
                    frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS
                )

                # Draw info overlay
                h, w = frame.shape[:2]
                label = f"Class: {target_class.upper()} | Dist: {distance:.2f}m | Vel: {velocity:.2f}"
                color = (0, 0, 255) if target_class == "car" else (0, 255, 0)
                cv2.rectangle(frame,
                              (int(xmin * w), int(ymin * h)),
                              (int(xmax * w), int(ymax * h)),
                              color, 2)
                cv2.putText(frame, label, (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                cv2.putText(frame, f"Sector: {sector}", (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

        else:
            # No hand detected
            self.frames_without_hand += 1
            # Reset velocity tracking after hand disappears
            if self.frames_without_hand > 5:
                self.prev_hand_center = None
                self.prev_time = None
                self.velocity_history.clear()
                self.distance_history.clear()
                self.last_distance = TOF_MAX_RANGE_METERS

        # Try to stream frame to Dashboard if queue is provided
        if self.queue_video_feed is not None:
            # Compress to JPEG
            ret_enc, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 60])
            if ret_enc:
                jpg_bytes = buffer.tobytes()
                # Non-blocking put to avoid slowing down sensor loop
                if self.queue_video_feed.full():
                    try:
                        self.queue_video_feed.get_nowait() # pop oldest
                    except Exception:
                        pass
                try:
                    self.queue_video_feed.put_nowait(jpg_bytes)
                except Exception:
                    pass

        return np.clip(matrix, 0.1, TOF_MAX_RANGE_METERS), ground_truth

    def release(self):
        """Release webcam and MediaPipe resources."""
        if self.cap is not None and self.cap.isOpened():
            self.cap.release()
        self.hands.close()
        cv2.destroyAllWindows()
        print("[MediaPipe Sensor] Resources released.")
