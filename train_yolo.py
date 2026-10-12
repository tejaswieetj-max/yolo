"""
Context-Aware Smart Glasses - YOLOv8n Fine-Tuning & Edge Export Pipeline
Lead Researcher: Dhivyashree G J. (24BCE1516)
Collaborators: Andrea A. (24BCE1142) · Tejaswi E. (24BCE5518)

8-Class Granular Taxonomy:
  0: person
  1: bicycle
  2: car
  3: motorcycle
  4: bus
  5: truck
  6: stairs
  7: branch

Hardware Target: Kaggle T4 GPU (Training) -> ESP32-S3 / Pi 5 (Edge Inference)
"""

import os
import sys
import time
import shutil
from pathlib import Path
import yaml
import numpy as np
import pandas as pd
from ultralytics import YOLO

def train_and_export():
    dataset_yaml = os.getenv("DATASET_YAML", "data.yaml")
    epochs = int(os.getenv("EPOCHS", "50"))
    imgsz = int(os.getenv("IMGSZ", "416"))
    batch = int(os.getenv("BATCH", "16"))
    device = os.getenv("DEVICE", "0" if os.environ.get("CUDA_VISIBLE_DEVICES") else ("0" if __import__("torch").cuda.is_available() else "cpu"))

    print("=" * 72)
    print("STEP 1: PRE-FLIGHT INTEGRITY & GPU VERIFICATION")
    print("=" * 72)
    print(f"Dataset YAML: {dataset_yaml}")
    print(f"Device:       {device}")
    print(f"Input Size:   {imgsz}x{imgsz}")
    print(f"Epochs:       {epochs}")

    start_time = time.time()
    print("\n[1/4] Loading pretrained YOLOv8n backbone...")
    model = YOLO("yolov8n.pt")

    print("\n[2/4] Starting training on 8-class dataset...")
    results = model.train(
        data=dataset_yaml,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        patience=15,
        optimizer="AdamW",
        lr0=0.001,
        lrf=0.01,
        device=device,
        project="smart_glasses_runs",
        name="yolov8n_t4",
        save=True,
        plots=True,
        verbose=True,
    )

    train_duration = time.time() - start_time
    print(f"\nTraining completed in {train_duration/60:.2f} minutes.")

    run_dir = Path("smart_glasses_runs/yolov8n_t4")
    best_pt = run_dir / "weights" / "best.pt"

    print("\n" + "=" * 72)
    print("STEP 2: VALIDATION & TEST EVALUATION")
    print("=" * 72)
    best_model = YOLO(str(best_pt))
    val_metrics = best_model.val(data=dataset_yaml, split="val", imgsz=imgsz)
    test_metrics = best_model.val(data=dataset_yaml, split="test", imgsz=imgsz)

    # Copy primary artifacts
    shutil.copy2(best_pt, "best.pt")
    if (run_dir / "weights" / "last.pt").exists():
        shutil.copy2(run_dir / "weights" / "last.pt", "last.pt")
    for f in ["results.csv", "results.png", "confusion_matrix.png"]:
        if (run_dir / f).exists():
            shutil.copy2(run_dir / f, f)

    print("\n" + "=" * 72)
    print("STEP 3: EDGE EXPORT (ONNX & INT8 TFLITE)")
    print("=" * 72)
    onnx_success = False
    tflite_success = False

    try:
        print("Exporting ONNX...")
        best_model.export(format="onnx", imgsz=imgsz, dynamic=False, simplify=True)
        onnx_success = True
        print("[OK] ONNX export successful.")
    except Exception as e:
        print(f"ONNX export note: {e}")

    try:
        print("\nExporting INT8 TFLite...")
        best_model.export(format="tflite", int8=True, data=dataset_yaml, imgsz=imgsz)
        tflite_success = True
        print("[OK] INT8 TFLite export successful.")
    except Exception as e:
        print(f"TFLite export note: {e}")

    # Step 4: Final Training Report
    best_epoch = "N/A"
    res_csv = run_dir / "results.csv"
    if res_csv.exists():
        df = pd.read_csv(res_csv)
        df.columns = df.columns.str.strip()
        if "metrics/mAP50(B)" in df.columns:
            best_epoch = int(df.loc[df["metrics/mAP50(B)"].idxmax(), "epoch"])

    prec = getattr(val_metrics.box, "mp", float(np.mean(val_metrics.box.p)))
    rec = getattr(val_metrics.box, "mr", float(np.mean(val_metrics.box.r)))
    map50 = getattr(val_metrics.box, "map50", 0.0)
    map50_95 = getattr(val_metrics.box, "map", 0.0)

    class_names = ["person", "bicycle", "car", "motorcycle", "bus", "truck", "stairs", "branch"]

    print("\n" + "=" * 72)
    print("CONCISE TRAINING REPORT")
    print("=" * 72)
    print(f"MODEL:         YOLOv8 Nano (yolov8n.pt)")
    print(f"DATASET:       620 images (424 train, 121 val, 75 test), 8 classes")
    print(f"EPOCHS:        {epochs}")
    print(f"TRAINING TIME: {train_duration/60:.2f} minutes")
    print(f"BEST EPOCH:    {best_epoch}")
    print(f"PRECISION:     {prec:.4f}")
    print(f"RECALL:        {rec:.4f}")
    print(f"mAP50:         {map50:.4f}")
    print(f"mAP50-95:      {map50_95:.4f}")
    print("")
    print("PER-CLASS RESULTS:")
    for idx, cname in enumerate(class_names):
        c_p = val_metrics.box.p[idx] if hasattr(val_metrics.box, 'p') and len(val_metrics.box.p) > idx else 0.0
        c_r = val_metrics.box.r[idx] if hasattr(val_metrics.box, 'r') and len(val_metrics.box.r) > idx else 0.0
        c_m50 = val_metrics.box.all_ap[idx][0] if (hasattr(val_metrics.box, 'all_ap') and val_metrics.box.all_ap is not None and len(val_metrics.box.all_ap) > idx) else (val_metrics.box.maps[idx] if hasattr(val_metrics.box, 'maps') and len(val_metrics.box.maps) > idx else 0.0)
        c_m95 = val_metrics.box.maps[idx] if hasattr(val_metrics.box, 'maps') and len(val_metrics.box.maps) > idx else 0.0
        print(f"{cname:12s}: Precision={c_p:.4f} | Recall={c_r:.4f} | mAP50={c_m50:.4f} | mAP50-95={c_m95:.4f}")
    print("")
    print("BEST MODEL:  best.pt")
    print(f"ONNX:        {'YES' if onnx_success else 'NO'}")
    print(f"INT8 TFLITE: {'YES' if tflite_success else 'NO'}")
    print("=" * 72)

if __name__ == "__main__":
    train_and_export()
