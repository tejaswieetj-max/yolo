"""
Global Configuration and Constants for Context-Aware Smart Glasses System.
Supports both Hardware (Raspberry Pi 5) and Zero-Hardware Simulation modes.
"""

import os
from pathlib import Path

# Base Directory
BASE_DIR = Path(__file__).resolve().parent

# ==============================================================================
# OPERATION MODE
# Set SIMULATION_MODE = True for laptop/PC testing without physical sensors.
# Set SIMULATION_MODE = False when deployed on Raspberry Pi 5 with I2C/CSI buses.
# ==============================================================================
SIMULATION_MODE = os.getenv("SIMULATION_MODE", "True").lower() in ("true", "1", "yes")

# ==============================================================================
# MEDIAPIPE CAMERA MODE (Real Webcam + Hand Tracking)
# When True AND SIMULATION_MODE is True, uses webcam + MediaPipe Hands
# instead of synthetic random data. Set False to revert to pure synthetic mode.
# ==============================================================================
USE_MEDIAPIPE_CAMERA = os.getenv("USE_MEDIAPIPE_CAMERA", "True").lower() in ("true", "1", "yes")

# Distance calibration: distance_meters ≈ K / hand_bbox_diagonal_ratio
# Tune this for your camera setup. Higher K = hand appears farther.
MP_DISTANCE_CALIBRATION_K = float(os.getenv("MP_DISTANCE_CALIBRATION_K", "0.55"))

# Hand velocity threshold (normalized units/sec) for vehicle classification.
# Hand moving faster than this → classified as "car" (vehicle).
# Hand moving slower → classified as "person" (wall/static obstacle).
MP_VEHICLE_VELOCITY_THRESHOLD = float(os.getenv("MP_VEHICLE_VELOCITY_THRESHOLD", "1.5"))

# Smoothing window size for velocity computation (number of frames)
MP_VELOCITY_SMOOTHING_WINDOW = int(os.getenv("MP_VELOCITY_SMOOTHING_WINDOW", "5"))

# Minimum hand bounding box diagonal (normalized) to consider as valid detection
MP_MIN_HAND_SIZE_RATIO = float(os.getenv("MP_MIN_HAND_SIZE_RATIO", "0.03"))

# ==============================================================================
# HARDWARE I2C & PIN CONFIGURATION (Raspberry Pi 5)
# ==============================================================================
I2C_BUS_ID = 1

# I2C Addresses (Standard default 7-bit addresses)
VL53L7CX_I2C_ADDR = 0x29   # 8x8 Multizone Time-of-Flight LiDAR (VL53L7CX)
MPU6050_I2C_ADDR = 0x68    # 6-Axis IMU (Gyroscope + Accelerometer)
DRV2605L_I2C_ADDR = 0x5A   # TI Haptic Driver (123 ROM Waveforms)
PCA9548A_I2C_ADDR = 0x70   # TCA/PCA9548A I2C Multiplexer for 3× DRV2605L

# PCA9548A Mux Channel → Motor Mapping
MUX_CHANNEL_LEFT = 0       # SD0/SC0 → DRV2605L #1 (Left LRA)
MUX_CHANNEL_CENTER = 1     # SD1/SC1 → DRV2605L #2 (Center LRA)
MUX_CHANNEL_RIGHT = 2      # SD2/SC2 → DRV2605L #3 (Right LRA)

# Actuator Channels / LRA Motor Identifiers
MOTOR_LEFT = "LEFT"
MOTOR_CENTER = "CENTER"
MOTOR_RIGHT = "RIGHT"
ALL_MOTORS = [MOTOR_LEFT, MOTOR_CENTER, MOTOR_RIGHT]

# GPIO Pinout for DRV2605L IN/TRIG pins (BCM numbering)
GPIO_LRA_LEFT = 17         # RPi Pin 11
GPIO_LRA_CENTER = 27       # RPi Pin 13
GPIO_LRA_RIGHT = 22        # RPi Pin 15

# Additional GPIO pins for sensor interrupts
GPIO_IMU_INT = 4           # RPi Pin 7  — MPU6050 INT
GPIO_TOF_LPN = 24          # RPi Pin 18 — VL53L7CX LPn (low-power enable)
GPIO_TOF_INT = 25          # RPi Pin 22 — VL53L7CX INT (data-ready interrupt)

# ==============================================================================
# SENSOR & SAMPLING FREQUENCIES
# ==============================================================================
IMU_SAMPLE_RATE_HZ = 100    # 100 Hz kinematic tracking
TOF_SAMPLE_RATE_HZ = 30     # 30 Hz multizone depth matrix
VISION_MAX_FPS = 15         # Dynamic ROI YOLOv8n inference cap on Pi 5

TOF_GRID_ROWS = 8
TOF_GRID_COLS = 8
TOF_TOTAL_ZONES = 64
TOF_FOV_HORIZONTAL_DEG = 90.0  # Horizontal Field of View (45 deg either side)
TOF_FOV_VERTICAL_DEG = 90.0    # Vertical Field of View (45 deg elevation span)
TOF_MAX_RANGE_METERS = 4.0     # Maximum physical measurement range

# ==============================================================================
# NOVELTY 01: KINEMATICS & EGO-MOTION PARAMETERS
# ==============================================================================
COMPLEMENTARY_FILTER_ALPHA = 0.98   # Weight for gyro vs accelerometer integration
MAX_COMPENSATED_YAW_DEG = 90.0      # Maximum expected head rotation angle

# Spatial Azimuth Binning (Columns 0 to 7)
# Columns 0-2: Left, Columns 3-4: Center, Columns 5-7: Right
SECTOR_LEFT_COLS = (0, 1, 2)
SECTOR_CENTER_COLS = (3, 4)
SECTOR_RIGHT_COLS = (5, 6, 7)

# Vertical Elevation Binning (Rows 0 to 7)
# Rows 0-3: Head-height & overhanging hazards
# Rows 4-7: Ground-level & descending obstacles
ELEVATION_HEAD_ROWS = (0, 1, 2, 3)
ELEVATION_GROUND_ROWS = (4, 5, 6, 7)

# ==============================================================================
# NOVELTY 03: DUAL-STAGE SPATIAL INTERRUPT GATING
# ==============================================================================
# Depth interrupt threshold: If all zones > 2.0m, camera & YOLOv8 remain idle
DEPTH_INTERRUPT_THRESHOLD_M = 2.0

# Camera Resolution
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
CAMERA_FPS = 30

# Dynamic ROI Margin (padding added to cropped box in pixels)
ROI_PADDING_PX = 20

# Context-Aware Smart Glasses 8-Class Hazard Taxonomy (Custom Fine-Tuned Model)
# 0: person, 1: bicycle, 2: car, 3: motorcycle, 4: bus, 5: truck, 6: stairs, 7: branch
YOLO_TARGET_CLASSES = {
    0: "person",
    1: "bicycle",
    2: "car",
    3: "motorcycle",
    4: "bus",
    5: "truck",
    6: "stairs",
    7: "branch",
}
YOLO_MODEL_NAME = os.getenv("YOLO_MODEL_NAME", "best.onnx")
YOLO_CONFIDENCE_THRESHOLD = 0.35

# ==============================================================================
# YOLO + MEDIAPIPE PARALLEL FUSION ENGINE
# ==============================================================================
# Adaptive Mode: "full" (YOLO+MediaPipe), "lite" (YOLO-only), "auto" (adaptive)
FUSION_MODE = os.getenv("FUSION_MODE", "auto")

# Adaptive switching thresholds (only used when FUSION_MODE="auto")
FUSION_CPU_TEMP_THRESHOLD_C = float(os.getenv("FUSION_CPU_TEMP_THRESHOLD_C", "75.0"))
FUSION_BATTERY_LOW_PERCENT = float(os.getenv("FUSION_BATTERY_LOW_PERCENT", "20.0"))

# Fusion confidence weights: Final = w_yolo*YOLO + w_mp_obj*MP_Object + bonuses
FUSION_WEIGHT_YOLO = float(os.getenv("FUSION_WEIGHT_YOLO", "0.55"))
FUSION_WEIGHT_MP_OBJECT = float(os.getenv("FUSION_WEIGHT_MP_OBJECT", "0.25"))
FUSION_BONUS_POSE_CONFIRM = float(os.getenv("FUSION_BONUS_POSE_CONFIRM", "0.12"))
FUSION_BONUS_HANDS_CONFIRM = float(os.getenv("FUSION_BONUS_HANDS_CONFIRM", "0.08"))

# MediaPipe sub-model toggles (all enabled by default for Full mode)
MP_ENABLE_OBJECT_DETECTION = os.getenv("MP_ENABLE_OBJECT_DETECTION", "True").lower() in ("true", "1", "yes")
MP_ENABLE_POSE = os.getenv("MP_ENABLE_POSE", "True").lower() in ("true", "1", "yes")
MP_ENABLE_HANDS = os.getenv("MP_ENABLE_HANDS", "True").lower() in ("true", "1", "yes")

# MediaPipe detection confidence thresholds
MP_OBJECT_CONFIDENCE_THRESHOLD = float(os.getenv("MP_OBJECT_CONFIDENCE_THRESHOLD", "0.40"))
MP_POSE_CONFIDENCE_THRESHOLD = float(os.getenv("MP_POSE_CONFIDENCE_THRESHOLD", "0.50"))
MP_HANDS_CONFIDENCE_THRESHOLD = float(os.getenv("MP_HANDS_CONFIDENCE_THRESHOLD", "0.50"))

# Parallel inference timeout (seconds) — max time to wait for slower model
FUSION_INFERENCE_TIMEOUT_S = float(os.getenv("FUSION_INFERENCE_TIMEOUT_S", "0.1"))

# ==============================================================================
# NOVELTY 02: TIME-TO-COLLISION (TTC) & SPATIO-TACTILE ENCODING
# ==============================================================================
TTC_ROLLING_WINDOW_SIZE = 5      # Rolling frame window for approach rate dv/dt
TTC_MIN_VELOCITY_MPS = 0.01      # Minimum approach velocity clamp to avoid div by zero

# TTC Urgency Boundaries
TTC_IMMINENT_THRESHOLD_S = 0.8    # < 0.8s: Imminent collision -> High-rate sharp buzz
TTC_APPROACHING_THRESHOLD_S = 2.0 # 0.8s - 2.0s: Approaching -> Double-click pulse
                                  # > 2.0s: Safe situational awareness -> Gentle pulse

# DRV2605L ROM Waveform Presets (from 123-effect library)
WAVEFORM_SAFE_AWARENESS = 1       # "Strong Click - 100%"
WAVEFORM_APPROACHING_ALERT = 12   # "Triple Click - 100%"
WAVEFORM_IMMINENT_COLLISION = 47  # "Buzz 1 - 100%"
WAVEFORM_OVERHEAD_HAZARD = 15     # "Medium Click 1 - 100%"
WAVEFORM_PERSON_SOFT = 64         # "Transition Hum 1 - 100%"
WAVEFORM_VEHICLE_SHARP = 47       # "Buzz 1 - 100%"

# ==============================================================================
# TELEMETRY & CLOUD CONFIGURATION
# ==============================================================================
LOCAL_LOG_DIR = BASE_DIR / "logs"
LOCAL_TELEMETRY_FILE = LOCAL_LOG_DIR / "telemetry_events.jsonl"
LOCAL_LOG_DIR.mkdir(exist_ok=True)

# Firebase Realtime Database
FIREBASE_CREDENTIALS_PATH = os.getenv("FIREBASE_CREDENTIALS_PATH", str(BASE_DIR / "firebase_credentials.json"))
FIREBASE_DATABASE_URL = os.getenv("FIREBASE_DATABASE_URL", "https://smart-glasses-telemetry-default-rtdb.firebaseio.com")
FIREBASE_BATCH_INTERVAL_S = 0.2  # 200 ms batch push interval

# Dashboard Local Server
DASHBOARD_HOST = "127.0.0.1"
DASHBOARD_PORT = 8080

# Motor channel → Mux channel mapping (for PCA9548A routing)
MUX_CHANNEL_MAP = {
    MOTOR_LEFT: MUX_CHANNEL_LEFT,
    MOTOR_CENTER: MUX_CHANNEL_CENTER,
    MOTOR_RIGHT: MUX_CHANNEL_RIGHT,
}
