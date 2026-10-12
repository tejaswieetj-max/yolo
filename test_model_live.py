"""
Context-Aware Smart Glasses - Real-Time Test & Validation Script
Run this script to test your trained YOLOv8 model on any image, test folder, or webcam!

Usage:
  # Test on sample test images:
  python test_model_live.py --source sample_test_results

  # Test on a specific image:
  python test_model_live.py --source sample_test_results/test_IMG_01616.jpg

  # Test on live webcam (if connected):
  python test_model_live.py --source 0
"""

import sys
import argparse
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="Test Smart Glasses YOLOv8 Model")
    parser.add_argument("--model", type=str, default="best.pt", help="Path to best.pt or best.onnx")
    parser.add_argument("--source", type=str, default="sample_test_results", help="Path to image, folder, or 0 for webcam")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--save", action="store_true", default=True, help="Save annotated images")
    args = parser.parse_args()

    print("=" * 65)
    print("CONTEXT-AWARE SMART GLASSES: MODEL TEST RUNNER")
    print("=" * 65)
    print(f"Loading model:  {args.model}")
    
    from ultralytics import YOLO
    model = YOLO(args.model)
    print(f"Model classes:  {model.names}")
    print(f"Input source:   {args.source}")
    print(f"Confidence:     {args.conf}")

    results = model.predict(
        source=args.source,
        conf=args.conf,
        imgsz=416,
        save=args.save,
        project="live_test_output",
        name="run",
        exist_ok=True
    )

    print("\n" + "=" * 65)
    print("DETECTION RESULTS")
    print("=" * 65)
    for r in results:
        img_name = Path(r.path).name if hasattr(r, 'path') else "Input Frame"
        boxes = r.boxes
        print(f"\nImage: {img_name} -> {len(boxes)} detection(s):")
        for b in boxes:
            cls_id = int(b.cls[0])
            conf_val = float(b.conf[0])
            cls_name = model.names[cls_id]
            xyxy = [round(x, 1) for x in b.xyxy[0].tolist()]
            print(f"  * [{cls_name.upper():10s}] Confidence: {conf_val:.2%} | Box: {xyxy}")

    print("\n" + "=" * 65)
    print("All annotated images with bounding boxes saved to:")
    print("  c:\\Users\\91866\\OneDrive\\Desktop\\YOLO\\live_test_output\\run\\")
    print("=" * 65)

if __name__ == "__main__":
    main()
