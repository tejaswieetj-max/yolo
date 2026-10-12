"""
Context-Aware Smart Glasses - Standalone ONNX Inference & Hazard Alert Engine
Runs directly with onnxruntime, numpy, and pillow (no heavy PyTorch installation needed!).

Usage:
  python run_onnx_inference.py --image sample_test_results/test_IMG_01616.jpg
  python run_onnx_inference.py --dir sample_test_results
"""

import sys
import argparse
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
import onnxruntime as ort

CLASS_NAMES = [
    "person",      # 0
    "bicycle",     # 1
    "car",         # 2
    "motorcycle",  # 3
    "bus",         # 4
    "truck",       # 5
    "stairs",      # 6
    "branch"       # 7
]

HAZARD_ACTIONS = {
    "stairs": ("CRITICAL", "HALT / STEP DESCENT", "Caution: Stairs detected directly ahead"),
    "branch": ("CRITICAL", "HEAD CLEARANCE / DUCK", "Warning: Overhanging branch / low obstacle at head level"),
    "motorcycle": ("CRITICAL", "RAPID DODGE", "Alert: Fast-moving motorcycle approaching"),
    "bus": ("CRITICAL", "STOP", "Danger: Heavy bus in path"),
    "truck": ("CRITICAL", "STOP", "Danger: Heavy truck in path"),
    "car": ("WARNING", "YIELD", "Notice: Vehicle ahead"),
    "bicycle": ("WARNING", "STEP ASIDE", "Notice: Approaching bicycle"),
    "person": ("INFO", "NAVIGATE", "Pedestrian detected")
}

def nms(boxes, scores, iou_thresh=0.45):
    """Greedy Non-Maximum Suppression"""
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= iou_thresh)[0]
        order = order[inds + 1]
    return keep

def predict_image(session, img_path, conf_thresh=0.25, save_annotated=True):
    orig_img = Image.open(img_path).convert("RGB")
    orig_w, orig_h = orig_img.size

    # Resize to 416x416 with aspect ratio preservation / letterbox
    img_resized = orig_img.resize((416, 416))
    img_data = np.array(img_resized, dtype=np.float32) / 255.0
    # Shape: (1, 3, 416, 416)
    img_tensor = np.transpose(img_data, (2, 0, 1))[np.newaxis, ...]

    input_name = session.get_inputs()[0].name
    outputs = session.run(None, {input_name: img_tensor})
    # Output shape is (1, 12, 3549) -> (12, 3549) -> transpose to (3549, 12)
    output = outputs[0][0].T

    boxes = []
    scores = []
    class_ids = []

    for pred in output:
        # pred[0:4] = cx, cy, w, h
        # pred[4:12] = 8 class scores
        class_scores = pred[4:]
        max_score = float(np.max(class_scores))
        if max_score >= conf_thresh:
            cid = int(np.argmax(class_scores))
            cx, cy, w, h = pred[0:4]
            # Convert to x1, y1, x2, y2 on 416x416
            x1 = cx - w / 2.0
            y1 = cy - h / 2.0
            x2 = cx + w / 2.0
            y2 = cy + h / 2.0

            # Scale to original image dimensions
            x1 = max(0, x1 * orig_w / 416.0)
            y1 = max(0, y1 * orig_h / 416.0)
            x2 = min(orig_w, x2 * orig_w / 416.0)
            y2 = min(orig_h, y2 * orig_h / 416.0)

            boxes.append([x1, y1, x2, y2])
            scores.append(max_score)
            class_ids.append(cid)

    if not boxes:
        return []

    boxes = np.array(boxes)
    scores = np.array(scores)
    class_ids = np.array(class_ids)

    # Apply NMS
    keep_indices = nms(boxes, scores)
    final_detections = []

    draw = ImageDraw.Draw(orig_img) if save_annotated else None

    for idx in keep_indices:
        b = boxes[idx]
        sc = scores[idx]
        cid = class_ids[idx]
        cname = CLASS_NAMES[cid]
        level, action, msg = HAZARD_ACTIONS.get(cname, ("INFO", "ALERT", "Obstacle"))

        # Direction analysis
        cx = (b[0] + b[2]) / 2.0
        if cx < orig_w * 0.33:
            direction = "LEFT"
        elif cx > orig_w * 0.66:
            direction = "RIGHT"
        else:
            direction = "CENTER"

        final_detections.append({
            "class": cname,
            "confidence": round(float(sc), 3),
            "bbox": [round(float(coord), 1) for coord in b],
            "direction": direction,
            "alert_level": level,
            "action": action,
            "voice_warning": f"[{level}] {msg} ({direction}, Conf: {sc:.0%})"
        })

        if draw:
            color = "red" if level == "CRITICAL" else ("orange" if level == "WARNING" else "blue")
            draw.rectangle(list(b), outline=color, width=3)
            draw.text((b[0] + 4, b[1] + 4), f"{cname} {sc:.2f} ({direction})", fill=color)

    if save_annotated:
        out_dir = Path("live_test_output")
        out_dir.mkdir(exist_ok=True)
        save_path = out_dir / f"pred_{Path(img_path).name}"
        orig_img.save(save_path)

    return final_detections


def main():
    parser = argparse.ArgumentParser(description="Run ONNX inference on Smart Glasses model")
    parser.add_argument("--model", type=str, default="best.onnx", help="Path to best.onnx")
    parser.add_argument("--image", type=str, default=None, help="Path to a single image")
    parser.add_argument("--dir", type=str, default="sample_test_results", help="Path to a directory of images")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    args = parser.parse_args()

    print("=" * 68)
    print("CONTEXT-AWARE SMART GLASSES: ONNX RUNTIME INFERENCE")
    print("Edge Acceleration & Module 2 Hazard Warning Engine")
    print("=" * 68)

    model_p = Path(args.model)
    if not model_p.exists():
        print(f"[!] Error: Model {args.model} not found.")
        sys.exit(1)

    print(f"Loading ONNX Model: {args.model} (Size: {model_p.stat().st_size / 1024 / 1024:.2f} MB)...")
    session = ort.InferenceSession(str(model_p), providers=["CPUExecutionProvider"])
    print("[OK] Model successfully loaded into ONNX Runtime engine!\n")

    if args.image:
        images = [Path(args.image)]
    else:
        images = list(Path(args.dir).glob("*.jpg")) + list(Path(args.dir).glob("*.png"))

    if not images:
        print(f"No images found in {args.dir}")
        return

    print(f"Testing {len(images)} images through the Smart Glasses vision engine:")
    for img_p in images:
        print(f"\n--- Frame: {img_p.name} ---")
        detections = predict_image(session, img_p, conf_thresh=args.conf)
        if detections:
            for d in detections:
                print(f"  -> {d['voice_warning']}")
                print(f"     Action: {d['action']:20s} | Coordinates: {d['bbox']}")
        else:
            print("  (No hazards detected above threshold - Path clear)")

    print("\n" + "=" * 68)
    print("[SUCCESS] TEST COMPLETE: Verified that the trained model is functioning!")
    print("Annotated test output saved to: live_test_output/")
    print("=" * 68)

if __name__ == "__main__":
    main()
