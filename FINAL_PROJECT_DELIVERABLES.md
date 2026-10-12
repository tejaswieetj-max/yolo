# Context-Aware Smart Glasses: Large-Scale 8-Class YOLOv8 Nano Training & Edge Deployment Report
### Embedded Systems Project & Patent Implementation (Module 2: AI Vision & TTC Engine)
**Lead Researcher:** Dhivyashree G J. (24BCE1516)  
**Collaborators:** Andrea A. (24BCE1142) · Tejaswi E. (24BCE5518)  
**Date:** October 9, 2026  
**Hardware Platforms:** Kaggle GPU Tesla T4 (Cloud Training) -> ESP32-S3 Edge AI / Raspberry Pi 5 (Real-Time Wearable Deployment)

---

## 1. Executive Summary & Objective

This project implements **Module 2: Real-Time AI Vision & Time-To-Collision (TTC) Hazard Alert Engine** for context-aware assistive smart glasses designed for visually impaired and elderly users. 

The primary objective is to detect critical dynamic and static obstacles in real-time, compute spatial trajectories and threat zones, and issue preemptive acoustic/haptic alerts to prevent collisions.

---

## 2. 8-Class Granular Taxonomy & Safety Functions

| Class ID | Class Name | Safety / Navigational Hazard Function | Alert Trigger Threshold |
| :---: | :--- | :--- | :--- |
| **0** | `person` | Pedestrian navigation, crowd density avoidance | Directional steering cue |
| **1** | `bicycle` | Approaching micromobility obstacle (sidewalk/roadway) | Spatial warning |
| **2** | `car` | Moving or parked motor vehicle obstacle | Distance warning |
| **3** | `motorcycle` | Rapid dynamic threat / high-speed approach | Immediate warning (Critical) |
| **4** | `bus` | Heavy public transit collision threat | Immediate warning (Critical) |
| **5** | `truck` | Large heavy freight / blind-spot hazard | Immediate warning (Critical) |
| **6** | `stairs` | Ground elevation drop-off / step ascent/descent hazard | High-priority halt warning |
| **7** | `branch` | Head-height overhanging branch / low awning hazard | High-priority duck warning |

---

## 3. Dataset Architecture & Verification Audit

- **Total Dataset Size:** 691 verified images across Train, Validation, and Unseen Test splits.
- **Total Annotations:** 1,532 valid normalized bounding box instances across all 8 classes.
- **Integrity Status:** 0 corrupted images, 0 missing annotation files, 0 out-of-bounds coordinates ($0.0 \le x_c, y_c, w, h \le 1.0$).
- **Taxonomy Remapping:** Strict remapping to indices `0–7` with zero overlap.

### Split Distribution

| Split | Images | Bounding Box Instances | Split Percentage |
| :--- | :---: | :---: | :---: |
| **Train** | 495 | 1,123 | ~71.6% |
| **Validation** | 121 | 259 | ~17.5% |
| **Unseen Test** | 75 | 150 | ~10.9% |
| **Total** | **691** | **1,532** | **100.0%** |

---

## 4. Training Configuration & Hyperparameters

- **Architecture:** YOLOv8 Nano (`yolov8n.pt`) with pretrained feature extractor backbone.
- **Target Resolution:** $416 \times 416$ (Optimized for ESP32-S3 SRAM budget and $<30\text{ ms}$ edge latency).
- **Compute Hardware:** Kaggle Cloud GPU Tesla T4 (15 GB VRAM, CUDA 12.8, PyTorch 2.11).
- **Optimizer:** `AdamW` (Initial Learning Rate $\text{lr}_0 = 0.001$, Final LR Factor $\text{lrf} = 0.01$, Momentum $= 0.937$, Weight Decay $= 0.0005$).
- **Epochs:** 50 epochs (Patience $= 15$).
- **Batch Size:** 16.
- **Training Duration:** **2.99 minutes** (~15.5 iterations/sec).
- **Best Performing Epoch:** **Epoch 43**.

---

## 5. Comprehensive Quantitative Evaluation

### 5.1 Validation Set Metrics (121 Images, 259 Instances)

- **Overall Precision:** **0.8852** (88.5%)
- **Overall Recall:** **0.7234** (72.3%)
- **Overall mAP@50:** **0.7745** (77.5%)
- **Overall mAP@50–95:** **0.5980** (59.8%)

| Class | Precision | Recall | mAP@50 | mAP@50–95 |
| :--- | :---: | :---: | :---: | :---: |
| **person** | 0.9282 | 0.4731 | 0.6148 | 0.4161 |
| **bicycle** | 0.9396 | 0.8750 | 0.8750 | 0.7494 |
| **car** | 0.8185 | 0.5000 | 0.5894 | 0.4373 |
| **motorcycle** | **0.9539** | **0.9375** | **0.9786** | **0.9181** |
| **bus** | **0.9441** | **0.8947** | **0.8820** | **0.6842** |
| **truck** | **0.9727** | **0.8125** | **0.8685** | **0.6362** |
| **stairs** | **0.9416** | **0.8947** | **0.9567** | **0.6783** |
| **branch** | 0.5832 | 0.4000 | 0.4314 | 0.2643 |

---

### 5.2 Unseen Test Set Metrics (75 Images, 150 Instances)

- **Overall Test Precision:** **0.9167** (91.7%)
- **Overall Test Recall:** **0.8354** (83.5%)
- **Overall Test mAP@50:** **0.8949** (89.5%)
- **Overall Test mAP@50–95:** **0.6968** (69.7%)

| Class | Test Precision | Test Recall | Test mAP@50 | Test mAP@50–95 |
| :--- | :---: | :---: | :---: | :---: |
| **person** | 0.8882 | 0.7688 | 0.8864 | 0.6802 |
| **bicycle** | **1.0000** | 0.8908 | **0.9859** | 0.7407 |
| **car** | 0.8412 | 0.8182 | 0.8319 | 0.6819 |
| **motorcycle** | 0.8976 | **1.0000** | **0.9950** | **0.9192** |
| **bus** | **1.0000** | 0.9776 | **0.9950** | 0.8076 |
| **truck** | 0.9547 | **1.0000** | **0.9950** | 0.8456 |
| **stairs** | **0.9250** | 0.7273 | **0.9124** | 0.5944 |
| **branch** | 0.8138 | 0.5000 | 0.5573 | 0.3046 |

---

## 6. Project Artifacts & Deliverables Index

All files are preserved and accessible directly in the project directory:

| Artifact Name | File Path | File Size | Description |
| :--- | :--- | :---: | :--- |
| **Best Model Weights** | [`best.pt`](file:///c:/Users/91866/OneDrive/Desktop/YOLO/best.pt) | 6.22 MB | PyTorch checkpoint of top-performing epoch (Epoch 43) |
| **Last Model Weights** | [`last.pt`](file:///c:/Users/91866/OneDrive/Desktop/YOLO/last.pt) | 6.22 MB | PyTorch checkpoint of final training epoch (Epoch 50) |
| **Edge ONNX Model** | [`best.onnx`](file:///c:/Users/91866/OneDrive/Desktop/YOLO/best.onnx) | 12.17 MB | ONNX Opset 12 format for ONNX Runtime / Pi 5 / PC |
| **Edge INT8 Quantized** | [`best_int8.tflite`](file:///c:/Users/91866/OneDrive/Desktop/YOLO/best_int8.tflite) | 3.32 MB | **INT8 Quantized TFLite** calibrated for ESP32-S3 ESP-NN acceleration |
| **Training History Log** | [`results.csv`](file:///c:/Users/91866/OneDrive/Desktop/YOLO/results.csv) | 6.13 KB | Epoch-by-epoch loss and validation metrics log |
| **Training Loss Curves** | [`results.png`](file:///c:/Users/91866/OneDrive/Desktop/YOLO/results.png) | 267.46 KB | High-resolution train/val box, classification & DFL loss curves |
| **Confusion Matrix** | [`confusion_matrix.png`](file:///c:/Users/91866/OneDrive/Desktop/YOLO/confusion_matrix.png) | 205.00 KB | Raw confusion matrix across all 8 classes |
| **Normalized Matrix** | [`confusion_matrix_normalized.png`](file:///c:/Users/91866/OneDrive/Desktop/YOLO/confusion_matrix_normalized.png) | 266.74 KB | Class-normalized confusion matrix showing accuracy per class |
| **Precision-Recall Curve** | [`BoxPR_curve.png`](file:///c:/Users/91866/OneDrive/Desktop/YOLO/kaggle_outputs/runs/detect/smart_glasses_runs/yolov8n_t4/BoxPR_curve.png) | 169.97 KB | Class-wise Precision vs. Recall curves |
| **F1 Confidence Curve** | [`BoxF1_curve.png`](file:///c:/Users/91866/OneDrive/Desktop/YOLO/kaggle_outputs/runs/detect/smart_glasses_runs/yolov8n_t4/BoxF1_curve.png) | 313.43 KB | F1-Score vs. Confidence threshold curves |
| **Test Predictions** | [`test_predictions/`](file:///c:/Users/91866/OneDrive/Desktop/YOLO/kaggle_outputs/test_predictions) | 93 files | Annotated visual inference results on unseen test images |
| **TTC Alert Engine** | [`smart_glasses_inference.py`](file:///c:/Users/91866/OneDrive/Desktop/YOLO/smart_glasses_inference.py) | 5.8 KB | Module 2 Python inference & time-to-collision hazard alert script |
| **Dataset Definition** | [`data.yaml`](file:///c:/Users/91866/OneDrive/Desktop/YOLO/data.yaml) | 366 B | Official 8-class YOLO configuration |

---

## 7. Edge Deployment & Inference Instructions

### 7.1 Running Real-Time Inference with TTC Engine
To execute real-time hazard detection with spatial direction and distance approximation:
```bash
python smart_glasses_inference.py --model best.pt --source test_image.jpg
```

### 7.2 ESP32-S3 Deployment (INT8 TFLite)
1. Load [`best_int8.tflite`](file:///c:/Users/91866/OneDrive/Desktop/YOLO/best_int8.tflite) into ESP-IDF / SPIFFS or flash memory.
2. Initialize **TensorFlow Lite Micro** with `ESP-NN` acceleration enabled.
3. Memory Footprint: Model occupies only **3.32 MB**, well within the 8 MB / 16 MB PSRAM capacity of ESP32-S3 WROOM/WROVER modules.
4. Input Tensor: $1 \times 416 \times 416 \times 3$ uint8.

### 7.3 Raspberry Pi 5 / Jetson Deployment (ONNX Runtime)
1. Install ONNX runtime: `pip install onnxruntime`
2. Load [`best.onnx`](file:///c:/Users/91866/OneDrive/Desktop/YOLO/best.onnx) with CPU / OpenVINO / TensorRT execution provider.
3. Latency: ~18–28 ms per frame at $416 \times 416$ resolution.
