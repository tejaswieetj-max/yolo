"""
Spatial Binning & Zone Classification Engine
Lead Researcher: Andrea A. (24BCE1142)

Classifies torso-reprojected obstacle vectors into 3 horizontal azimuth sectors
(Left, Center, Right) and 2 vertical elevation bands (Head-height vs Ground-level).
"""

from dataclasses import dataclass
from typing import List, Dict, Any
from .ego_motion import TorsoFrameVector
from config import (
    MOTOR_LEFT,
    MOTOR_CENTER,
    MOTOR_RIGHT,
    DEPTH_INTERRUPT_THRESHOLD_M,
)


@dataclass
class SectorSummary:
    name: str                # "LEFT", "CENTER", "RIGHT"
    min_distance_m: float    # Distance to closest obstacle in this sector
    closest_vector: Any      # TorsoFrameVector of the closest object
    active_hazard_count: int # Count of zones with distance < threshold
    is_head_hazard: bool     # True if closest hazard is at head/overhead level
    mean_distance_m: float

# Backwards compatible alias
SpatialSector = SectorSummary


@dataclass
class SpatialClassificationResult:
    sectors: Dict[str, SectorSummary]
    critical_sector: str      # Sector with highest urgency / lowest distance
    min_distance_overall_m: float
    interrupt_triggered: bool # True if any sector < DEPTH_INTERRUPT_THRESHOLD_M
    all_torso_vectors: List[TorsoFrameVector]
    head_yaw_deg: float


class ZoneClassifier:
    """
    Partitions 64 torso-frame vectors into Left, Center, and Right sectors,
    classifying overhead vs ground hazards.
    """

    def __init__(
        self,
        azimuth_boundary_deg: float = 15.0,
        elevation_head_ground_split_deg: float = 0.0,
        interrupt_threshold_m: float = DEPTH_INTERRUPT_THRESHOLD_M,
    ):
        self.azimuth_boundary_deg = azimuth_boundary_deg
        self.elevation_split_deg = elevation_head_ground_split_deg
        self.interrupt_threshold_m = interrupt_threshold_m

    def classify(self, torso_vectors: List[TorsoFrameVector]) -> SpatialClassificationResult:
        """Partition torso vectors and determine critical sector."""
        sector_vectors = {
            MOTOR_LEFT: [],
            MOTOR_CENTER: [],
            MOTOR_RIGHT: [],
        }

        head_yaw = torso_vectors[0].head_yaw_deg if torso_vectors else 0.0

        for vec in torso_vectors:
            az = vec.azimuth_torso_deg
            if az < -self.azimuth_boundary_deg:
                sector_vectors[MOTOR_LEFT].append(vec)
            elif az > self.azimuth_boundary_deg:
                sector_vectors[MOTOR_RIGHT].append(vec)
            else:
                sector_vectors[MOTOR_CENTER].append(vec)

        sectors_summary: Dict[str, SectorSummary] = {}
        min_overall_dist = 999.0
        critical_sec = MOTOR_CENTER
        interrupt_active = False

        for sec_name, vecs in sector_vectors.items():
            if not vecs:
                sectors_summary[sec_name] = SectorSummary(
                    name=sec_name,
                    min_distance_m=4.0,
                    closest_vector=None,
                    active_hazard_count=0,
                    is_head_hazard=False,
                    mean_distance_m=4.0,
                )
                continue

            distances = [v.distance_m for v in vecs]
            min_d = min(distances)
            mean_d = sum(distances) / len(distances)
            closest_v = vecs[distances.index(min_d)]
            haz_count = sum(1 for d in distances if d < self.interrupt_threshold_m)

            # Elevation check: positive elevation angle indicates head/overhead hazard
            is_head = closest_v.elevation_torso_deg > self.elevation_split_deg

            if haz_count > 0:
                interrupt_active = True

            if min_d < min_overall_dist:
                min_overall_dist = min_d
                critical_sec = sec_name

            sectors_summary[sec_name] = SectorSummary(
                name=sec_name,
                min_distance_m=min_d,
                closest_vector=closest_v,
                active_hazard_count=haz_count,
                is_head_hazard=is_head,
                mean_distance_m=mean_d,
            )

        return SpatialClassificationResult(
            sectors=sectors_summary,
            critical_sector=critical_sec,
            min_distance_overall_m=min_overall_dist if min_overall_dist < 999.0 else 4.0,
            interrupt_triggered=interrupt_active,
            all_torso_vectors=torso_vectors,
            head_yaw_deg=head_yaw,
        )
