"""
Obstacle Semantic Class Mapping & Severity Hierarchy
Lead Researcher: Dhivyashree G J. (24BCE1516)
"""

from typing import Dict

# 8-Class Granular Taxonomy for Context-Aware Smart Glasses
CLASS_NAMES = {
    0: "person",
    1: "bicycle",
    2: "car",
    3: "motorcycle",
    4: "bus",
    5: "truck",
    6: "stairs",
    7: "branch",
    99: "overhead_hazard",   # Custom extension: low-hanging branch/awning
    100: "generic_obstacle",
}

# Severity weighting for multi-object prioritisation (Higher = more hazardous)
CLASS_SEVERITY_WEIGHTS: Dict[str, float] = {
    "truck": 1.0,
    "bus": 0.95,
    "car": 0.90,
    "motorcycle": 0.85,
    "stairs": 0.85,          # Critical ground elevation drop-off
    "bicycle": 0.75,
    "branch": 0.75,          # Head-height obstacle
    "overhead_hazard": 0.70,
    "person": 0.60,
    "generic_obstacle": 0.50,
    "clear": 0.0,
}


def get_class_name(class_id: int) -> str:
    return CLASS_NAMES.get(class_id, "generic_obstacle")


def get_severity_score(class_name: str, distance_m: float) -> float:
    """Combines class threat with physical proximity."""
    weight = CLASS_SEVERITY_WEIGHTS.get(class_name.lower(), 0.5)
    # Proximity factor: 1.0 at 0m, decaying to 0.0 at 4m
    proximity = max(0.0, (4.0 - distance_m) / 4.0)
    return weight * 0.6 + proximity * 0.4
