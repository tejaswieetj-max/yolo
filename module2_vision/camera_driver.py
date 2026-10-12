"""
Camera Acquisition Driver (Pi Camera CSI / OpenCV USB / Synthetic Stream)
Lead Researcher: Dhivyashree G J. (24BCE1516)
"""

import time
import numpy as np
from typing import Optional, Tuple

from config import (
    CAMERA_WIDTH,
    CAMERA_HEIGHT,
    CAMERA_FPS,
    SIMULATION_MODE,
    USE_MEDIAPIPE_CAMERA,
)


class CameraDriver:
    """Wrapper supporting Raspberry Pi Camera CSI, OpenCV USB/webcam, and synthetic fallback."""

    def __init__(self, camera_index: int = 0, width: int = CAMERA_WIDTH, height: int = CAMERA_HEIGHT):
        self.width = width
        self.height = height
        self.cap = None
        self.is_opened = False
        self.using_synthetic = False

        self._init_camera(camera_index)

    def _init_camera(self, camera_index: int):
        """Try initializing physical camera (OpenCV VideoCapture or PiCamera2)."""
        # When MediaPipe mode is active, Module 1 owns the webcam exclusively.
        # Module 2 uses synthetic frames; classification comes from ground_truth in the queue.
        if SIMULATION_MODE and USE_MEDIAPIPE_CAMERA:
            print("[CameraDriver] MediaPipe mode active — camera owned by Module 1. Using synthetic frames.")
            self.using_synthetic = True
            self.is_opened = True
            return

        try:
            import cv2
            self.cap = cv2.VideoCapture(camera_index)
            if self.cap.isOpened():
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
                self.cap.set(cv2.CAP_PROP_FPS, CAMERA_FPS)
                self.is_opened = True
                print(f"[CameraDriver] Physical camera initialized at index {camera_index} ({self.width}x{self.height})")
                return
        except Exception as e:
            print(f"[CameraDriver] OpenCV capture not available: {e}")

        # Fallback to synthetic frame generator
        print("[CameraDriver] Physical camera unavailable. Using synthetic frame generator.")
        self.using_synthetic = True
        self.is_opened = True

    def capture_frame(self) -> Tuple[bool, np.ndarray]:
        """Captures a BGR image frame (H, W, 3)."""
        if not self.using_synthetic and self.cap is not None:
            try:
                ret, frame = self.cap.read()
                if ret and frame is not None:
                    if frame.shape[0] != self.height or frame.shape[1] != self.width:
                        import cv2
                        frame = cv2.resize(frame, (self.width, self.height))
                    return True, frame
            except Exception:
                pass

        # Generate synthetic frame with realistic gradient and timestamp
        synthetic_frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        # Deep urban street scene gradient
        for y in range(self.height):
            intensity = int(40 + (y / self.height) * 80)
            synthetic_frame[y, :, :] = [intensity, intensity + 10, intensity + 20]

        # Draw a simulated sidewalk horizon line
        synthetic_frame[int(self.height * 0.6):, :] = [60, 60, 65]
        return True, synthetic_frame

    def release(self):
        if self.cap is not None:
            try:
                self.cap.release()
            except Exception:
                pass
            self.cap = None
        self.is_opened = False
