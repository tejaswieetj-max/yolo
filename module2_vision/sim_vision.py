"""
Laptop Webcam & Simulation Testbed for Vision Pipeline
Lead Researcher: Dhivyashree G J. (24BCE1516)

Provides visual feedback with dynamic ROI overlays, detection bounding boxes,
and HUD telemetry for laptop-based evaluation.
"""

from typing import List, Optional
import numpy as np
from .tof_gate import RegionOfInterest
from .yolo_engine import DetectedObject


class SimulatedVisionTestbed:
    """Renders debug overlays onto video frames."""

    @staticmethod
    def draw_debug_overlay(
        frame: np.ndarray,
        roi: Optional[RegionOfInterest],
        detections: List[DetectedObject],
        ttc_info: dict,
        head_yaw_deg: float = 0.0,
    ) -> np.ndarray:
        """Draws HUD overlays for visual inspection."""
        try:
            import cv2
        except ImportError:
            return frame

        vis = frame.copy()
        h, w = vis.shape[:2]

        # 1. Draw HUD Header
        cv2.rectangle(vis, (0, 0), (w, 40), (15, 23, 42), -1)
        yaw_str = f"Head Yaw: {head_yaw_deg:+5.1f} deg"
        cv2.putText(vis, yaw_str, (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 242, 254), 2)

        if roi is not None:
            # 2. Draw Dynamic ROI bounding box (Novelty 03)
            # Amber/Cyan box indicating the cropped sub-image passed to YOLO
            cv2.rectangle(vis, (roi.xmin, roi.ymin), (roi.xmax, roi.ymax), (0, 215, 255), 2)
            savings_pct = (1.0 - roi.area_ratio) * 100.0
            roi_label = f"DYNAMIC ROI (Savings: {savings_pct:.1f}%)"
            cv2.putText(vis, roi_label, (roi.xmin, max(20, roi.ymin - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 215, 255), 1)

            # 3. Draw Detections
            for det in detections:
                gx1, gy1, gx2, gy2 = det.bbox_global
                # Color code: Red for vehicle, Green for person, Magenta for overhead
                box_color = (0, 0, 255) if det.class_name in ("car", "bus", "truck") else (0, 255, 0)
                if det.class_name == "overhead_hazard":
                    box_color = (255, 0, 255)

                cv2.rectangle(vis, (gx1, gy1), (gx2, gy2), box_color, 2)
                label = f"{det.class_name.upper()} {det.confidence:.2f}"
                cv2.putText(vis, label, (gx1, max(15, gy1 - 5)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 2)
        else:
            # Gating idle
            cv2.putText(vis, "STAGE 1 GATE: IDLE (>2.0m) - AI SLEEPING", (180, 26),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (16, 185, 129), 2)

        # 4. Draw TTC Urgency Footer
        urgency = ttc_info.get("urgency", "SAFE")
        ttc_val = ttc_info.get("ttc", 99.0)
        sector = ttc_info.get("sector", "CENTER")

        footer_color = (16, 185, 129)
        if urgency == "IMMINENT":
            footer_color = (0, 0, 239)
        elif urgency == "APPROACHING":
            footer_color = (0, 165, 255)

        cv2.rectangle(vis, (0, h - 35), (w, h), (15, 23, 42), -1)
        ttc_str = f"URGENCY: {urgency} | Sector: {sector} | TTC: {ttc_val:.2f}s"
        cv2.putText(vis, ttc_str, (10, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.55, footer_color, 2)

        return vis
