"""
Novelty 03: Dual-Stage Spatial Interrupt Gate & Dynamic ROI Mapper
Lead Researcher: Dhivyashree G J. (24BCE1516)

Evaluates 8x8 ToF depth matrices as a low-power hardware depth interrupt.
If distance < 2.0m, calculates optimal Camera Region-of-Interest (ROI) bounding box,
reducing inference compute on Raspberry Pi 5 by >70%.
"""

from dataclasses import dataclass
from typing import List, Tuple, Optional
import numpy as np

from config import (
    TOF_GRID_ROWS,
    TOF_GRID_COLS,
    DEPTH_INTERRUPT_THRESHOLD_M,
    CAMERA_WIDTH,
    CAMERA_HEIGHT,
    ROI_PADDING_PX,
)


@dataclass
class RegionOfInterest:
    """Bounding box for dynamic camera frame cropping."""
    xmin: int
    ymin: int
    xmax: int
    ymax: int
    width: int
    height: int
    area_ratio: float           # Cropped area / full frame area (e.g., 0.22 = 78% savings)
    active_zone_count: int
    min_distance_m: float


class ToFInterruptGate:
    """
    Stage 1 Spatial Pre-Filter.
    Decides whether camera & YOLO neural networks should wake up, and computes ROI crop coordinates.
    """

    def __init__(
        self,
        threshold_m: float = DEPTH_INTERRUPT_THRESHOLD_M,
        camera_width: int = CAMERA_WIDTH,
        camera_height: int = CAMERA_HEIGHT,
        padding_px: int = ROI_PADDING_PX,
    ):
        self.threshold_m = threshold_m
        self.cam_w = camera_width
        self.cam_h = camera_height
        self.padding_px = padding_px
        self.total_pixels = camera_width * camera_height

    def evaluate_gate(self, depth_matrix_m: np.ndarray) -> Tuple[bool, Optional[RegionOfInterest]]:
        """
        Check if any zone in 8x8 matrix is closer than depth threshold.
        Returns:
            (is_triggered, roi_or_none)
        """
        # Boolean mask of hazardous zones
        hazard_mask = depth_matrix_m < self.threshold_m

        if not np.any(hazard_mask):
            # All obstacles are > 2.0m: Keep camera & AI idle
            return False, None

        # Find row and col bounds of active zones
        rows, cols = np.where(hazard_mask)
        min_r, max_r = int(np.min(rows)), int(np.max(rows))
        min_c, max_c = int(np.min(cols)), int(np.max(cols))

        min_dist = float(np.min(depth_matrix_m[hazard_mask]))
        zone_count = int(np.sum(hazard_mask))

        # Map 8x8 grid coordinates to Camera Resolution
        # Note: col 0 is left, col 7 is right; row 0 is top, row 7 is bottom
        x_step = self.cam_w / float(TOF_GRID_COLS)
        y_step = self.cam_h / float(TOF_GRID_ROWS)

        raw_xmin = int(min_c * x_step)
        raw_xmax = int((max_c + 1) * x_step)
        raw_ymin = int(min_r * y_step)
        raw_ymax = int((max_r + 1) * y_step)

        # Apply safety padding
        xmin = max(0, raw_xmin - self.padding_px)
        ymin = max(0, raw_ymin - self.padding_px)
        xmax = min(self.cam_w, raw_xmax + self.padding_px)
        ymax = min(self.cam_h, raw_ymax + self.padding_px)

        width = xmax - xmin
        height = ymax - ymin
        area_ratio = (width * height) / float(self.total_pixels)

        roi = RegionOfInterest(
            xmin=xmin,
            ymin=ymin,
            xmax=xmax,
            ymax=ymax,
            width=width,
            height=height,
            area_ratio=area_ratio,
            active_zone_count=zone_count,
            min_distance_m=min_dist,
        )

        return True, roi
