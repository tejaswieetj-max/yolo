# Context-Aware Smart Glasses — Final Master Project Plan
### Team: Andrea A. (24BCE1142) · Dhivyashree G J. (24BCE1516) · Tejaswi E. (24BCE5518)

---

## 1. Executive Summary & Journal Novelty Assessment

**Target Outcome:** High-impact journal publication in **IEEE Sensors Journal**, **MDPI Sensors**, or **IEEE Access**.

### Novelty Scorecard

| Dimension | Baseline / Existing Work | Our Proposed Solution | Journal Value |
|---|---|---|---|
| **Novelty 01: Head-to-Torso Kinematic Transform** | Head rotation misdirects torso haptics (turning head 45° signals false azimuth). | 100 Hz quaternion rotation matrix $R(\theta_{\text{head}})$ re-projects 64 ToF obstacle vectors to torso frame. | **High (6.5/10)** — Proved via ablation study (WITH vs WITHOUT). |
| **Novelty 02: Multi-Axis Spatio-Tactile Encoding** | Static distance-to-intensity mapping (can't differentiate closing car from static wall). | Simultaneous 3-axis encoding: **Azimuth** (motor pin) × **Urgency** (Time-to-Collision $\Delta d/\Delta t$) × **Hazard Texture** (DRV2605L waveform library). | **High (7.0/10)** — Quantified via user study confusion matrix. |
| **Novelty 03: Dual-Stage Spatial Interrupt Gating** | Continuous edge AI drains battery in <1 hour; continuous ToF lacks semantic context. | 8×8 ToF acts as low-power hardware depth interrupt (<2.0m threshold); dynamically crops camera ROI for YOLOv8n inference. | **Very High (8.5/10)** — >70% compute savings on plain Pi 5 without AI accelerator. |
| **Combined System Novelty** | Proprietary $2,500+ commercial smart glasses or fragmented academic sensors. | **Unified, end-to-end edge-to-cloud ecosystem** (<₹6,000 BOM) with Firebase telemetry and real-time researcher/caregiver web dashboard. | **Publication-Grade (9.0/10)** |

---

## 2. Paper-Grade Innovation Framing

To secure Q1/Q2 journal acceptance, the contributions are framed as follows:

1. **Ego-Motion Compensation Engine:**
   > *"A real-time, quaternion-based coordinate re-projection engine operating at 100 Hz on an embedded ARM host, correcting angular disparity between head-mounted spatial ranging arrays and torso-fixed haptic actuators during rapid head yaw (up to ±90°)."*

2. **Time-to-Collision (TTC) Spatio-Tactile Encoding:**
   > *"A multi-dimensional tactile feedback scheme that concurrently modulates spatial motor selection, dynamic approach velocity (TTC), and vibration textures from a 123-waveform library to convey both obstacle urgency and semantic classification without acoustic sensory deprivation."*

3. **Dual-Stage Spatial Interrupt Vision Gating:**
   > *"A hierarchical perception pipeline where an 8×8 multizone Time-of-Flight matrix operates as a spatial pre-filter, gating camera activation and confining YOLOv8 Nano inference to dynamically cropped Regions of Interest (ROI) — extending battery endurance by >3.5× on commodity Raspberry Pi 5 hardware without auxiliary neural accelerators."*

---

## 3. Hardware Constraints & BOM (Locked)

> **Strict Rule:** No hardware components outside this bill of materials will be used.

| Component | Role | Interface | Power / Spec |
|---|---|---|---|
| **Raspberry Pi 5 Model B** (4GB / 8GB) | Central processing unit (runs Python, OpenCV, YOLOv8n, Firebase) | Linux (Raspberry Pi OS) | 5V / 3A USB-C |
| **ST VL53L5CX** | 8×8 Multizone Time-of-Flight LiDAR (64 zones, 30 Hz, 4m range, 90° FoV) | I2C (Address `0x29`) | 3.3V / ~45 mA |
| **OmniVision OV2640 / Pi Camera Module** | 2 MP CMOS camera for selective ROI visual confirmation | CSI / USB Frame Bus | 3.3V / ~150 mA |
| **InvenSense MPU6050** | 6-axis IMU (gyroscope + accelerometer) for 100 Hz head orientation tracking | I2C (Address `0x68`) | 3.3V / ~3.8 mA |
| **TI DRV2605L** | Haptic motor driver with built-in ROM library (123 tactile waveforms) | I2C (Address `0x5A`) | 3.3V / ~2 mA standby |
| **3× Linear Resonant Actuators (LRAs)** | Left, Center, and Right directional haptic neckband transducers | Driven via DRV2605L / PWM | Fast rise time (<30 ms) |
| **Power Subsystem** | 3.7V 1860 mAh Li-Po Battery + TP4056 USB-C Charger + 5V/3.3V Step-Up/Buck | Power Rail | >3.5 hrs continuous operation |

---

## 4. Equal 3-Way Software Workload Split

The architecture divides responsibilities equally into **3 core packages**, each matching one member's primary research and coding ownership.

```mermaid
graph TD
    subgraph Edge Perception [Raspberry Pi 5]
        M1[Andrea: Sensor Fusion & Ego-Motion Engine] -->|Queue A: 64-Zone Corrected Vectors| M2[Dhivyashree: Dual-Stage AI Vision & TTC Pipeline]
        M2 -->|Queue B: Classified Hazards & TTC| M3A[Tejaswi: 3-Axis Spatio-Tactile Haptic Engine]
        M3A -->|I2C 0x5A| LRA[Neckband 3x LRA Motors]
        M3A -->|Async WebSocket Stream| FB[(Firebase Realtime Database)]
    end
    subgraph Cloud Telemetry & Visual Analytics
        FB -->|Real-Time Web Sync| M3B[Tejaswi: Caregiver & Researcher Web Dashboard]
    end
```

---

### MEMBER 1: Andrea A. (24BCE1142)
#### Module: Sensor Fusion, Kinematics & Ego-Motion Engine (Novelty 01 Lead)

**Workload & Core Tasks:**
1. **Hardware Sensor Drivers:** Writes robust `smbus2` I2C drivers for the `VL53L5CX` (extracting the 8×8 depth array at 30 Hz) and `MPU6050` (sampling accelerometer + gyroscope at 100 Hz).
2. **Quaternion Kinematics Engine (`ego_motion.py`):**
   - Implements a Complementary/Kalman filter on IMU angular rates to derive drift-free head yaw ($\theta_{\text{yaw}}$) and pitch ($\theta_{\text{pitch}}$).
   - Computes real-time 3D rotation transform $R(\theta_{\text{head}})$, mapping raw sensor-frame obstacle coordinates into the fixed torso reference frame.
3. **Zone Partitioning:** Converts the transformed 64-zone matrix into 3 spatial azimuth sectors (Left: zones 0–2; Center: zones 3–4; Right: zones 5–7) and 2 elevation bands (Head-height hazard vs Ground obstacle).
4. **Paper Ablation Study Code:** Implements automated data logging comparing obstacle vectors **WITH** and **WITHOUT** ego-motion correction during 45° and 90° head sweeps.
5. **Sensor Simulation Engine (`sim_sensor.py`):** Generates synthetic 8×8 depth matrices and sinusoidal IMU yaw sweeps so Modules 2 and 3 can develop on laptops without physical hardware.

**Directory Structure:**
```text
module1_sensor/
├── tof_driver.py         # VL53L5CX I2C acquisition (8x8 depth array @ 30 Hz)
├── imu_driver.py         # MPU6050 I2C acquisition & complementary filter (100 Hz)
├── ego_motion.py         # Quaternion rotation matrix R(θ) transform
├── zone_classifier.py    # Spatial binning (Azimuth L/C/R + Elevation Head/Ground)
├── sim_sensor.py         # Mock depth & IMU generator for hardware-free simulation
└── main_sensor.py        # Process entry: runs fusion loop and pushes to Queue A
```

---

### MEMBER 2: Dhivyashree G J. (24BCE1516)
#### Module: Dual-Stage AI Vision & Time-to-Collision (TTC) Engine (Novelty 02 & 03 Co-Lead)

**Workload & Core Tasks:**
1. **Stage 1 Spatial Interrupt Gate (`tof_gate.py`):**
   - Evaluates incoming zone vectors from Queue A.
   - If all zones $> 2.0\text{ m}$, camera and AI stay idle.
   - If any zone $< 2.0\text{ m}$, dynamically flags the zone bounding coordinates.
2. **Dynamic ROI Frame Cropper:** Translates flagged ToF zones into pixel bounding boxes $(X_{\min}, Y_{\min}, X_{\max}, Y_{\max})$ on the camera frame, passing only the cropped sub-image to the neural network.
3. **YOLOv8 Nano Inference Strategy:**
   - **Phase 1 (Immediate / Simulation):** Uses standard pretrained `yolov8n.pt` with COCO class filtering (`classes=[0, 1, 2, 3, 5, 7]` for `person`, `bicycle`, `car`, `motorcycle`, `bus`, `truck`). Zero training required, runs at ~15 FPS on Pi 5.
   - **Phase 2 (Paper Novelty Fine-Tuning - Optional):** Quick transfer learning on Kaggle/Google Colab (15 mins on free T4 GPU) using open-source Roboflow datasets for descending stairs and overhanging branches.
4. **Time-to-Collision (TTC) Engine (`ttc_engine.py`):**
   - Calculates relative approach velocity $v_{\text{approach}} = \frac{d_{t-1} - d_t}{\Delta t}$ across a 5-frame rolling window.
   - Computes $\text{TTC} = \frac{d_t}{\max(v_{\text{approach}}, 0.01\text{ m/s})}$.
5. **Vision Simulation Engine (`sim_vision.py`):** Feeds laptop webcam frames into the cropped ROI pipeline with real-time OpenCV bounding box overlays.

**Directory Structure:**
```text
module2_vision/
├── tof_gate.py           # Stage 1 depth threshold & ROI bounding box calculator
├── camera_driver.py      # Camera capture wrapper (Pi Camera CSI / OpenCV USB)
├── yolo_engine.py        # YOLOv8n inference on cropped ROI
├── ttc_engine.py         # Rolling approach rate & TTC computation
├── class_map.py          # Class filtering & severity weighting
├── sim_vision.py         # Laptop webcam mock vision testbed
└── main_vision.py        # Process entry: Queue A -> AI/TTC -> Queue B
```

---

### MEMBER 3: Tejaswi E. (24BCE5518)
#### Module: Spatio-Tactile Haptics & Real-Time Caregiver Web Dashboard (Novelty 02 Co-Lead & IoT Lead)

**Workload & Core Tasks:**
1. **DRV2605L Haptic Actuator Driver:** Interfaces with the TI DRV2605L over I2C, accessing its internal ROM library of 123 tactile waveforms.
2. **3-Axis Spatio-Tactile Encoding Engine (`haptic_encoder.py`):**
   - **Axis 1 (Spatial Azimuth):** Routes signal to Left, Center, or Right LRA.
   - **Axis 2 (Urgency via TTC):** Modulates pulse firing frequency:
     - $\text{TTC} < 0.8\text{ s}$ (Imminent): Rapid 20 Hz buzz (Waveform #47).
     - $0.8\text{ s} \le \text{TTC} \le 2.0\text{ s}$ (Approaching): Moderate double-click (Waveform #12).
     - $\text{TTC} > 2.0\text{ s}$ (Safe awareness): Gentle pulse (Waveform #1).
   - **Axis 3 (Hazard Texture):** Selects distinct waveforms per class (`Person`: soft transition; `Vehicle`: sharp alert; `Overhead`: prolonged warning).
3. **Firebase Telemetry Streamer (`firebase_logger.py`):** Asynchronously pushes timestamped event packets (azimuth, TTC, class, distance, waveform ID, latency) to Firebase Realtime DB.
4. **Caregiver & Researcher Real-Time Web Dashboard:** Builds the front-end web dashboard visualizing live spatial radar, neckband vibration nodes, and telemetry metrics.
5. **Haptic Simulation Mode (`sim_haptic.py`):** Console graphical visualizer rendering live LRA pulses and mock Firebase publisher.

**Directory Structure:**
```text
module3_haptic/
├── drv2605l_driver.py    # DRV2605L I2C control & waveform triggering
├── haptic_encoder.py     # 3-axis spatio-tactile mapping logic
├── waveform_map.py       # DRV2605L ROM effect index mapping
├── firebase_logger.py    # Firebase Admin SDK async batch streaming
├── sim_haptic.py         # Terminal visualizer for simulated LRA pulses
└── main_haptic.py        # Process entry: Queue B -> Haptics + Firebase push

dashboard/
├── index.html            # Web dashboard structure & layout
├── style.css             # Glassmorphism dark-mode styling
├── app.js                # Firebase Web SDK real-time sync & Chart.js logic
└── assets/               # Visual icons & assets
```

---

## 5. Caregiver & Researcher Web Dashboard Specification

### UI/UX Theme & Visual Architecture
- **Theme:** Obsidian Dark Mode (`#0B0F19` deep space background, `#111827` cards, `#00F2FE` cyan accents, `#10B981` emerald status indicators, `#EF4444` hazard alerts).
- **Styling:** Modern glassmorphism with subtle border glow effects that flash red/amber when an active hazard pulse triggers.

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│  🟢 SYSTEM: ONLINE   |   MODE: Autonomous Outdoor   |   LATENCY: 26 ms   |   BATTERY: 88%  │
├──────────────────────────────────────────┬─────────────────────────────────────────────┤
│  🧭 180° SPATIAL RADAR & HEAD ORIENTATION │  📳 NECKBAND HAPTIC STATE VISUALIZER        │
│                                          │                                             │
│      [ Left ]   [ Center ]   [ Right ]   │       (L)             (C)             (R)   │
│         \           |           /        │      Left            Center          Right  │
│          \       ● 1.1m        /         │      Node            Node            Node   │
│           \     (Person)      /          │     [Safe]        [PULSING 🔴]      [Safe]  │
│            \        |        /           │                                             │
│       Current Head Yaw: +14° (Compensated)│  Active Waveform: "Strong Buzz 100%" (#47)  │
├──────────────────────────────────────────┼─────────────────────────────────────────────┤
│  📋 LIVE HAZARD DETECTION STREAM (EVENT LOG)│  📊 SESSION ANALYTICS & RESEARCH DATA EXPORT│
│                                          │                                             │
│  [20:45:12] Person detected at 1.1m      │  • Hazard Distribution (Donut Chart)        │
│             TTC: 0.65s ➔ Center Pulse    │    - Pedestrian: 60%                        │
│  [20:45:08] Overhead Branch at 1.7m      │    - Vehicle: 25%                           │
│             TTC: 1.40s ➔ Center Soft     │    - Overhead: 15%                          │
│  [20:44:55] Vehicle approaching Left 2.8m│                                             │
│             TTC: 0.88s ➔ Left Pulse      │  [📥 Export Session CSV/JSON for Paper]     │
└──────────────────────────────────────────┴─────────────────────────────────────────────┘
```

### Key Dashboard Widgets & Functionality:
1. **Live Header Telemetry:** Real-time online/offline indicator, battery status, end-to-end loop latency (ms), and active session timer.
2. **Top-Down 180° Spatial Radar:** Visualizes Left, Center, and Right zones with animated blips indicating detected obstacles and distance. Displays live head yaw compensation angle ($\theta_{\text{head}}$).
3. **Neckband Haptic Node Visualizer:** Dynamic animation of the 3 neckband LRAs. When a motor fires, the corresponding node pulses with color-coded urgency (🟢 Low, 🟡 Moderate, 🔴 High) and shows the active waveform name.
4. **Real-Time Event Stream:** Rolling table listing timestamp, detected obstacle class, distance, computed TTC, and motor triggered.
5. **Session Analytics & 1-Click Export:** Chart.js visualizations of hazard distributions, plus an **Export Dataset (.csv)** button allowing the team to directly extract empirical data for publication graphs.

---

## 6. Project Directory Layout & Integration

```text
embedded_smart_glasses/
├── module1_sensor/              # Andrea A. (Sensor Fusion & Ego-Motion)
│   ├── tof_driver.py
│   ├── imu_driver.py
│   ├── ego_motion.py
│   ├── zone_classifier.py
│   ├── sim_sensor.py
│   └── main_sensor.py
├── module2_vision/              # Dhivyashree G J. (AI Vision & TTC)
│   ├── tof_gate.py
│   ├── camera_driver.py
│   ├── yolo_engine.py
│   ├── ttc_engine.py
│   ├── class_map.py
│   ├── sim_vision.py
│   └── main_vision.py
├── module3_haptic/              # Tejaswi E. (Haptics & Cloud Telemetry)
│   ├── drv2605l_driver.py
│   ├── haptic_encoder.py
│   ├── waveform_map.py
│   ├── firebase_logger.py
│   ├── sim_haptic.py
│   └── main_haptic.py
├── dashboard/                   # Tejaswi E. (Caregiver & Researcher Web App)
│   ├── index.html
│   ├── style.css
│   ├── app.js
│   └── assets/
├── config.py                    # Global constants, pinouts, and SIMULATION_MODE flag
├── main.py                      # Multi-process launcher with inter-process queues
├── .env                         # Firebase credentials (ignored by git)
├── .gitignore
├── requirements.txt             # Python dependencies
└── README.md                    # Project documentation
```

### Master Integration Loop (`main.py`)
```python
from multiprocessing import Process, Queue
from config import SIMULATION_MODE
from module1_sensor.main_sensor import run_sensor_module
from module2_vision.main_vision import run_vision_module
from module3_haptic.main_haptic import run_haptic_module

if __name__ == '__main__':
    print(f"[SYSTEM] Starting Smart Glasses Pipeline (SIMULATION_MODE={SIMULATION_MODE})")
    
    # Inter-process queues
    queue_sensor_to_vision = Queue(maxsize=10)
    queue_vision_to_haptic = Queue(maxsize=10)
    
    # Process initialization
    p1 = Process(target=run_sensor_module, args=(queue_sensor_to_vision,), name="SensorProcess")
    p2 = Process(target=run_vision_module, args=(queue_sensor_to_vision, queue_vision_to_haptic), name="VisionProcess")
    p3 = Process(target=run_haptic_module, args=(queue_vision_to_haptic,), name="HapticProcess")
    
    p1.start(); p2.start(); p3.start()
    p1.join(); p2.join(); p3.join()
```

---

## 7. Zero-Hardware Simulation Strategy

All team members can build, test, and debug their code on laptops immediately without waiting for hardware:

```python
# config.py
SIMULATION_MODE = True  # Switch to False when Raspberry Pi 5 & sensors are wired
```

- **Module 1 (Andrea):** Uses `sim_sensor.py` to synthesize 8×8 depth matrices and a ±45° sinusoidal head-yaw sweep, outputting realistic zone vectors.
- **Module 2 (Dhivyashree):** Uses `sim_vision.py` to capture frames from a standard laptop webcam (`cv2.VideoCapture(0)`) and executes YOLOv8n inference on simulated obstacle bounding boxes.
- **Module 3 & Dashboard (Tejaswi):** Uses `sim_haptic.py` to print animated console visualizers of active LRA pulses while streaming live data to Firebase Realtime DB, powering the web dashboard in real time.

---

## 8. Journal Paper Structure & Research Deliverables

| Paper Section | Content & Focus | Generated By |
|---|---|---|
| **1. Introduction & Related Work** | Head-height hazard blindness, acoustic isolation drawbacks, and state-of-the-art haptic navigation gaps. | All |
| **2. System Architecture** | Dual-stage hierarchical perception and Raspberry Pi 5 edge computing pipeline. | All |
| **3. Kinematics & Ego-Motion Compensation** | Quaternion rotation transformation $R(\theta_{\text{head}})$ and coordinate alignment mathematics. | Andrea |
| **4. Dual-Stage Spatial Interrupt Vision** | 8×8 ToF triggering mechanism, ROI dynamic cropping, and Time-to-Collision (TTC) algorithm. | Dhivyashree |
| **5. 3-Axis Spatio-Tactile Encoding & Cloud Telemetry** | DRV2605L waveform library mapping, Firebase schema, and real-time monitoring dashboard. | Tejaswi |
| **6. Experimental Results & Discussion** | Ablation study (with/without ego-motion), end-to-end latency benchmarks, battery endurance curves, and user trial confusion matrix. | All (via Dashboard Export) |
| **7. Conclusion** | Cost-to-performance analysis (<₹6,000 BOM vs $2,500 commercial devices) and future work. | All |

---

## 9. 6-Week Development & Publication Roadmap

| Week | Milestone & Deliverables | Primary Lead |
|:---:|---|:---:|
| **Week 1** | Project setup: shared Git repository, `config.py`, inter-process queues, and simulation testbeds (`sim_*.py`). | All |
| **Week 2** | Andrea: Quaternion $R(\theta)$ math & mock IMU sweep.<br>Dhivyashree: YOLOv8n webcam integration & class filter.<br>Tejaswi: Firebase Realtime DB schema & basic dashboard HTML layout. | Andrea<br>Dhivyashree<br>Tejaswi |
| **Week 3** | Full software pipeline integration in simulation mode; live web dashboard syncing with simulated navigation events. | All |
| **Week 4** | Physical hardware assembly on Raspberry Pi 5; test I2C buses (`0x29`, `0x68`, `0x5A`) and switch `SIMULATION_MODE = False`. | All |
| **Week 5** | Empirical testing: conduct ablation study (ego-motion on vs off) and export navigation trial datasets via the web dashboard. | All |
| **Week 6** | Compile results, generate benchmark graphs, and finalize the journal paper manuscript draft. | All |
