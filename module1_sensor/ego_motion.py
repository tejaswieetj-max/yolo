"""
Novelty 01: Head-to-Torso Kinematic Transform Engine
Lead Researcher: Andrea A. (24BCE1142)

Implements real-time quaternion and rotation matrix operations to re-project
64 Time-of-Flight depth vectors from the head-mounted frame into the fixed
torso reference frame at 100 Hz.
"""

import math
from dataclasses import dataclass
from typing import List, Tuple, Dict, Any
import numpy as np

from config import (
    TOF_GRID_ROWS,
    TOF_GRID_COLS,
    TOF_FOV_HORIZONTAL_DEG,
    TOF_FOV_VERTICAL_DEG,
)


@dataclass
class Quaternion:
    """Hamiltonian Quaternion q = w + xi + yj + zk for 3D rotations."""
    w: float = 1.0
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0

    @classmethod
    def from_euler(cls, yaw_rad: float, pitch_rad: float = 0.0, roll_rad: float = 0.0) -> "Quaternion":
        """
        Construct quaternion from Tait-Bryan Euler angles (Yaw-Pitch-Roll: Z-Y-X sequence).
        yaw: rotation around Z-axis (head turn left/right)
        pitch: rotation around X-axis (head nod up/down)
        roll: rotation around Y-axis (head tilt)
        """
        cy = math.cos(yaw_rad * 0.5)
        sy = math.sin(yaw_rad * 0.5)
        cp = math.cos(pitch_rad * 0.5)
        sp = math.sin(pitch_rad * 0.5)
        cr = math.cos(roll_rad * 0.5)
        sr = math.sin(roll_rad * 0.5)

        w = cr * cp * cy + sr * sp * sy
        x = sr * cp * cy - cr * sp * sy
        y = cr * sp * cy + sr * cp * sy
        z = cr * cp * sy - sr * sp * cy
        return cls(w, x, y, z)

    def to_rotation_matrix(self) -> np.ndarray:
        """Derive 3x3 orthonormal rotation matrix R from quaternion."""
        w, x, y, z = self.w, self.x, self.y, self.z
        return np.array([
            [1.0 - 2.0 * (y*y + z*z),  2.0 * (x*y - z*w),        2.0 * (x*z + y*w)],
            [2.0 * (x*y + z*w),        1.0 - 2.0 * (x*x + z*z),  2.0 * (y*z - x*w)],
            [2.0 * (x*z - y*w),        2.0 * (y*z + x*w),        1.0 - 2.0 * (x*x + y*y)],
        ], dtype=np.float64)

    def rotate_vector(self, v: np.ndarray) -> np.ndarray:
        """Rotate 3D Cartesian vector v = [x, y, z] by quaternion."""
        R = self.to_rotation_matrix()
        return np.dot(R, v)


@dataclass
class SensorFrameVector:
    """Obstacle vector in head-mounted sensor coordinate frame."""
    row: int
    col: int
    raw_distance_m: float
    azimuth_deg: float      # Horizontal angle relative to head line-of-sight
    elevation_deg: float    # Vertical angle relative to head plane
    x_head: float           # Lateral (positive = right)
    y_head: float           # Forward (positive = forward)
    z_head: float           # Vertical (positive = upward)


@dataclass
class TorsoFrameVector:
    """Obstacle vector re-projected to torso reference frame."""
    row: int
    col: int
    distance_m: float
    x_torso: float
    y_torso: float
    z_torso: float
    azimuth_torso_deg: float   # Azimuth relative to walking trajectory (torso frame)
    elevation_torso_deg: float
    head_yaw_deg: float        # Applied compensation yaw


class EgoMotionEngine:
    """
    Ego-Motion Kinematic Transform Engine.
    Converts 64 raw VL53L5CX zones to torso-centered Cartesian and spherical vectors.
    """

    def __init__(self):
        # Pre-compute beam optical angles for the 8x8 ToF grid
        self._init_beam_geometry()

    def _init_beam_geometry(self):
        """Compute nominal azimuth (phi) and elevation (theta) for each zone center."""
        self.zone_azimuth_deg = np.zeros((TOF_GRID_ROWS, TOF_GRID_COLS), dtype=np.float64)
        self.zone_elevation_deg = np.zeros((TOF_GRID_ROWS, TOF_GRID_COLS), dtype=np.float64)

        h_step = TOF_FOV_HORIZONTAL_DEG / TOF_GRID_COLS
        v_step = TOF_FOV_VERTICAL_DEG / TOF_GRID_ROWS

        for r in range(TOF_GRID_ROWS):
            for c in range(TOF_GRID_COLS):
                # Columns 0 to 7: -45 deg to +45 deg (left to right)
                az = - (TOF_FOV_HORIZONTAL_DEG / 2.0) + (c + 0.5) * h_step
                # Rows 0 to 7: +45 deg to -45 deg (top to bottom)
                el = (TOF_FOV_VERTICAL_DEG / 2.0) - (r + 0.5) * v_step
                self.zone_azimuth_deg[r, c] = az
                self.zone_elevation_deg[r, c] = el

    def raw_grid_to_sensor_vectors(self, depth_matrix_m: np.ndarray) -> List[SensorFrameVector]:
        """Convert an 8x8 depth matrix into 64 3D sensor-frame vectors."""
        vectors = []
        for r in range(TOF_GRID_ROWS):
            for c in range(TOF_GRID_COLS):
                dist = float(depth_matrix_m[r, c])
                az_deg = float(self.zone_azimuth_deg[r, c])
                el_deg = float(self.zone_elevation_deg[r, c])

                az_rad = math.radians(az_deg)
                el_rad = math.radians(el_deg)

                # Spherical to Cartesian (Head Frame)
                # x: lateral (right), y: forward, z: elevation (up)
                x_head = dist * math.cos(el_rad) * math.sin(az_rad)
                y_head = dist * math.cos(el_rad) * math.cos(az_rad)
                z_head = dist * math.sin(el_rad)

                vectors.append(SensorFrameVector(
                    row=r,
                    col=c,
                    raw_distance_m=dist,
                    azimuth_deg=az_deg,
                    elevation_deg=el_deg,
                    x_head=x_head,
                    y_head=y_head,
                    z_head=z_head,
                ))
        return vectors

    def transform_to_torso_frame(
        self,
        sensor_vectors: List[SensorFrameVector],
        head_yaw_deg: float,
        head_pitch_deg: float = 0.0,
    ) -> List[TorsoFrameVector]:
        """
        Re-project sensor-frame vectors to torso-frame using quaternion rotation.
        
        Head yaw rotation: turning head right (+yaw) means the head-fixed sensor
        line-of-sight is at +yaw relative to torso. Obstacles on that line-of-sight
        are therefore at +yaw relative to the user's torso walking direction.
        """
        yaw_rad = math.radians(head_yaw_deg)
        pitch_rad = math.radians(head_pitch_deg)

        # Quaternion representing head-to-torso coordinate transformation
        # For a coordinate frame rotated by +yaw relative to torso,
        # transforming coordinates from head frame to torso frame requires rotation by +yaw
        # (p_torso = R_z(+yaw) * p_head with x_t = x_h*cos(yaw) + y_h*sin(yaw))
        q_head_to_torso = Quaternion.from_euler(yaw_rad=-yaw_rad, pitch_rad=-pitch_rad)
        R = q_head_to_torso.to_rotation_matrix()

        torso_vectors = []
        for sv in sensor_vectors:
            p_head = np.array([sv.x_head, sv.y_head, sv.z_head], dtype=np.float64)
            p_torso = np.dot(R, p_head)

            x_t = float(p_torso[0])
            y_t = float(p_torso[1])
            z_t = float(p_torso[2])

            dist_t = math.sqrt(x_t*x_t + y_t*y_t + z_t*z_t)
            # Torso-frame azimuth: angle from torso forward axis (y) in horizontal plane (x-y)
            az_torso_deg = math.degrees(math.atan2(x_t, max(y_t, 1e-6)))
            # Torso-frame elevation
            ground_dist = math.sqrt(x_t*x_t + y_t*y_t)
            el_torso_deg = math.degrees(math.atan2(z_t, max(ground_dist, 1e-6)))

            torso_vectors.append(TorsoFrameVector(
                row=sv.row,
                col=sv.col,
                distance_m=dist_t,
                x_torso=x_t,
                y_torso=y_t,
                z_torso=z_t,
                azimuth_torso_deg=az_torso_deg,
                elevation_torso_deg=el_torso_deg,
                head_yaw_deg=head_yaw_deg,
            ))

        return torso_vectors

    def process_frame(
        self,
        depth_matrix_m: np.ndarray,
        head_yaw_deg: float,
        head_pitch_deg: float = 0.0,
    ) -> Tuple[List[SensorFrameVector], List[TorsoFrameVector]]:
        """Complete pipeline: Raw 8x8 matrix -> Sensor vectors -> Torso-corrected vectors."""
        sensor_vecs = self.raw_grid_to_sensor_vectors(depth_matrix_m)
        torso_vecs = self.transform_to_torso_frame(sensor_vecs, head_yaw_deg, head_pitch_deg)
        return sensor_vecs, torso_vecs
