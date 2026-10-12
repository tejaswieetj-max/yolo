# Context-Aware Smart Glasses for Visually Impaired Navigation
### An Edge-to-Cloud Assistive System with Kinematic Ego-Motion Compensation & Spatio-Tactile Haptics

**Research Team:**
- **Andrea A.** (24BCE1142) — Sensor Fusion, Kinematics & Ego-Motion Engine (*Novelty 01 Lead*)
- **Dhivyashree G J.** (24BCE1516) — Dual-Stage AI Vision & Time-to-Collision (TTC) Pipeline (*Novelty 02 & 03 Co-Lead*)
- **Tejaswi E.** (24BCE5518) — 3-Axis Spatio-Tactile Haptic Engine & Caregiver Web Dashboard (*Novelty 02 Co-Lead & IoT Lead*)

**Target Journal:** *IEEE Sensors Journal* / *MDPI Sensors* / *IEEE Access*

---

## 1. System Overview & Core Novelties

This repository contains the complete software ecosystem for context-aware smart glasses designed for assistive navigation. It solves three major limitations found in commercial assistive glasses ($2,500+) and academic prototypes:

1. **Novelty 01: Head-to-Torso Kinematic Transform ($R(\theta_{\text{head}})$):**
   - Head-mounted sensors (8×8 ToF) rotate when the user turns their head, misdirecting torso-worn haptic actuators.
   - A real-time **quaternion rotation matrix at 100 Hz** re-projects all 64 depth vectors into the fixed torso reference frame, yielding **100% azimuth routing accuracy** across the operational field-of-view (vs. 14.3% uncompensated).
2. **Novelty 02: Multi-Axis Spatio-Tactile Encoding:**
   - Simultaneously conveys **Azimuth** (Left/Center/Right LRA) × **Urgency** (Time-to-Collision $TTC = \frac{d}{\Delta d/\Delta t}$) × **Tactile Texture** (123 waveforms from TI DRV2605L ROM library) without acoustic sensory deprivation.
3. **Novelty 03: Dual-Stage Spatial Interrupt Gating:**
   - 8×8 ToF operates as a low-power hardware depth pre-filter (<2.0m threshold).
   - Only when triggered does the camera wake up to crop a dynamic **Region of Interest (ROI)** for YOLOv8n, saving **>70% compute power** and extending battery life by >3.5× on commodity Raspberry Pi 5 hardware.

---

## 2. Repository Architecture

```text
ES_Software/
├── config.py                      # Global configuration, thresholds, and SIMULATION_MODE flag
├── main.py                        # Multi-process orchestrator (Queues A & B)
├── requirements.txt               # Dependencies
├── .env.example                   # Telemetry & Firebase environment template
│
├── module1_sensor/                # Andrea A. (Sensor Fusion & Ego-Motion)
│   ├── tof_driver.py              # ST VL53L5CX I2C acquisition (8x8 @ 30 Hz)
│   ├── imu_driver.py              # MPU6050 I2C acquisition & Complementary Filter (100 Hz)
│   ├── ego_motion.py              # Novelty 01: Quaternion R(θ) head-to-torso re-projection
│   ├── zone_classifier.py         # Spatial binning (Left/Center/Right + Head/Ground bands)
│   ├── sim_sensor.py              # Synthetic depth & sinusoidal yaw sweep generator
│   └── main_sensor.py             # Process entry: runs fusion loop and pushes to Queue A
│
├── module2_vision/                # Dhivyashree G J. (AI Vision & TTC)
│   ├── tof_gate.py                # Novelty 03: Stage 1 depth gate (<2.0m) & ROI mapper
│   ├── camera_driver.py           # Camera wrapper (Pi Camera CSI / OpenCV USB / Synthetic)
│   ├── yolo_engine.py             # YOLOv8n inference on dynamic cropped ROI
│   ├── ttc_engine.py              # Novelty 02: Rolling 5-frame approach velocity & TTC
│   ├── class_map.py               # Class filtering & severity weighting
│   ├── sim_vision.py              # Laptop webcam mock vision testbed with OpenCV overlays
│   └── main_vision.py             # Process entry: Queue A -> AI/TTC -> Queue B
│
├── module3_haptic/                # Tejaswi E. (Haptics & Telemetry)
│   ├── drv2605l_driver.py         # TI DRV2605L I2C control & ROM waveform triggering
│   ├── haptic_encoder.py          # Novelty 02: 3-axis spatio-tactile mapping
│   ├── waveform_map.py            # DRV2605L 123-waveform library mapping
│   ├── firebase_logger.py         # Firebase Realtime DB async streamer + local CSV logger
│   ├── sim_haptic.py              # Terminal animated visualizer for neckband LRAs
│   └── main_haptic.py             # Process entry: Queue B -> Haptics + Firebase push
│
├── dashboard/                     # Tejaswi E. (Caregiver & Researcher Web App)
│   ├── index.html                 # Dark glassmorphic UI: status, 180° radar, LRA nodes, event log
│   ├── style.css                  # Obsidian dark theme (#0B0F19), cyan/emerald glows
│   ├── app.js                     # Real-time radar, neckband pulse visualizer, CSV export
│   └── server.py                  # Local HTTP + SSE telemetry server (FastAPI/Uvicorn)
│
└── scripts/                       # Research Validation & Benchmark Suite
    ├── ablation_study.py          # Automated evaluation: WITH vs WITHOUT ego-motion
    └── latency_benchmark.py       # End-to-end pipeline latency profiling
```

---

## 3. Quickstart: Zero-Hardware Simulation Mode

The software includes a zero-hardware simulation testbed so anyone can run and test the complete system on a PC/laptop without physical sensors:

### Step 1: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 2: Run End-to-End Pipeline
```bash
python main.py --sim
```
This spawns all three processes concurrently, passing synthetic 8×8 depth matrices, sinusoidal head yaw sweeps, dynamic ROI vision gating, and live console neckband LRA pulses.

### Step 3: Launch Web Dashboard
In a separate terminal or using the `--dashboard` flag:
```bash
python dashboard/server.py
```
Open **[http://127.0.0.1:8080](http://127.0.0.1:8080)** in your browser to view:
- **180° Top-Down Spatial Radar** with live head-yaw needle compensation.
- **Neckband 3-Node Visualizer** pulsing green/yellow/red with active DRV2605L waveform names.
- **Live Hazard Stream Table** tracking distance, TTC, and sector.
- **1-Click CSV/JSON Export** to download empirical research datasets.

---

## 4. Physical Hardware Deployment (Raspberry Pi 5)

When deploying on the physical Raspberry Pi 5:
1. Connect sensor breakout boards to Raspberry Pi 5 I2C bus 1:
   - **ST VL53L5CX:** 3.3V, GND, SDA (GPIO 2), SCL (GPIO 3) -> Address `0x29`
   - **InvenSense MPU6050:** 3.3V, GND, SDA (GPIO 2), SCL (GPIO 3) -> Address `0x68`
   - **TI DRV2605L:** 3.3V, GND, SDA (GPIO 2), SCL (GPIO 3) -> Address `0x5A`
   - **Linear Resonant Actuators (LRAs):** Connected to DRV2605L output / multiplexer for Left, Center, Right channels.
   - **Camera:** Raspberry Pi Camera Module connected via CSI ribbon.
2. In `config.py` (or using `--real` flag), set:
   ```python
   SIMULATION_MODE = False
   ```
3. Run the pipeline:
   ```bash
   python main.py --real
   ```

---

## 5. Paper Reproducibility & Research Benchmarks

### 1. Novelty 01 Ablation Study
To reproduce the empirical table comparing obstacle azimuth tracking **WITH vs WITHOUT** ego-motion compensation:
```bash
python scripts/ablation_study.py
```
**Empirical Findings:**
- Azimuth Routing Accuracy WITHOUT Ego-Motion: **14.3%** (Baseline)
- Azimuth Routing Accuracy WITH Ego-Motion: **100.0%** (Proposed Novelty 01)
- Mean Angular Error Reduction: **86.0%** (from 24.91° to 3.48°)

### 2. Latency & Real-Time Profiler
To measure sub-millisecond execution times across each module:
```bash
python scripts/latency_benchmark.py
```
**Empirical Findings:**
- Total Pipeline Processing Latency: **~1.06 ms** (Throughput: **>900 Hz**)
- Far exceeding the real-time threshold ($\le 30.0\text{ ms}$).
