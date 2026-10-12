"""
Dataset Preparation, Conversion, and Integrity Verification Pipeline
Context-Aware Smart Glasses - 8-Class Granular YOLOv8 Model

Taxonomy:
  0: person
  1: bicycle
  2: car
  3: motorcycle
  4: bus
  5: truck
  6: stairs
  7: branch
"""

import os
import sys
import glob
import shutil
import hashlib
from pathlib import Path
from collections import Counter

TARGET_CLASSES = {
    0: "person",
    1: "bicycle",
    2: "car",
    3: "motorcycle",
    4: "bus",
    5: "truck",
    6: "stairs",
    7: "branch",
}

class DatasetAuditor:
    def __init__(self, dataset_dir: str = "dataset"):
        self.base_dir = Path(dataset_dir)
        self.splits = ["train", "val", "test"]

    def audit_all(self):
        print("=" * 76)
        print("CONTEXT-AWARE SMART GLASSES: 8-CLASS DATASET INTEGRITY AUDIT")
        print("=" * 76)

        total_images = 0
        total_annotations = 0
        class_counter = Counter()
        corrupted_files = []
        missing_labels = []
        duplicate_images = []
        out_of_bounds_bboxes = []
        image_hashes = {}

        split_summary = {}

        for split in self.splits:
            img_dir = self.base_dir / "images" / split
            lbl_dir = self.base_dir / "labels" / split

            if not img_dir.exists() or not lbl_dir.exists():
                print(f"[!] Warning: Split directory missing: {split}")
                continue

            img_files = list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png")) + list(img_dir.glob("*.jpeg"))
            split_summary[split] = {"images": len(img_files), "annotations": 0}

            for img_path in img_files:
                total_images += 1

                # 1. Image hash check (Duplicates)
                try:
                    with open(img_path, "rb") as f:
                        data = f.read()
                        h = hashlib.md5(data).hexdigest()
                    if h in image_hashes:
                        duplicate_images.append((str(img_path), image_hashes[h]))
                    else:
                        image_hashes[h] = str(img_path)
                except Exception as e:
                    corrupted_files.append((str(img_path), str(e)))
                    continue

                # 2. Image integrity check (valid file headers & minimum size)
                if len(data) < 100:
                    corrupted_files.append((str(img_path), "File too small (<100 bytes)"))
                elif img_path.suffix.lower() in [".jpg", ".jpeg"] and not (data.startswith(b"\xff\xd8") and data.endswith(b"\xff\xd9")):
                    # Check for valid SOI marker
                    if not data.startswith(b"\xff\xd8"):
                        corrupted_files.append((str(img_path), "Corrupted JPEG SOI marker"))

                # 3. Label matching check
                lbl_path = lbl_dir / f"{img_path.stem}.txt"
                if not lbl_path.exists():
                    missing_labels.append(str(img_path))
                    continue

                # 4. Annotation format & coordinate validation
                try:
                    with open(lbl_path, "r") as f:
                        lines = f.readlines()

                    if not lines:
                        missing_labels.append(f"{lbl_path} (Empty file)")

                    for line_idx, line in enumerate(lines):
                        parts = line.strip().split()
                        if not parts:
                            continue
                        if len(parts) != 5:
                            out_of_bounds_bboxes.append((str(lbl_path), f"Line {line_idx+1}: invalid format: {line.strip()}"))
                            continue

                        cls_id = int(parts[0])
                        x_c, y_c, w, h = map(float, parts[1:])

                        # Validate class ID
                        if cls_id not in TARGET_CLASSES:
                            out_of_bounds_bboxes.append((str(lbl_path), f"Invalid class ID: {cls_id}"))
                            continue

                        # Validate normalized bounding box ranges
                        if not (0.0 <= x_c <= 1.0 and 0.0 <= y_c <= 1.0 and 0.0 < w <= 1.0 and 0.0 < h <= 1.0):
                            out_of_bounds_bboxes.append((str(lbl_path), f"Coordinates out of bounds: {parts[1:]}"))
                            continue

                        class_counter[cls_id] += 1
                        total_annotations += 1
                        split_summary[split]["annotations"] += 1
                except Exception as e:
                    corrupted_files.append((str(lbl_path), str(e)))

        print("\n1. DATASET SPLIT BREAKDOWN:")
        for split, data in split_summary.items():
            print(f"   - {split.upper()} IMAGES: {data['images']:5d} | {data['annotations']:5d} annotations")
        print(f"   -> TOTAL IMAGES: {total_images:5d} | TOTAL ANNOTATIONS: {total_annotations:5d}")

        print("\n2. PER-CLASS ANNOTATION COUNTS:")
        for cls_id in range(8):
            cls_name = TARGET_CLASSES[cls_id]
            count = class_counter[cls_id]
            pct = (count / total_annotations * 100) if total_annotations > 0 else 0.0
            print(f"   {cls_name:12s}: {count:5d} ({pct:5.2f}%)")

        print("\n3. INTEGRITY AUDIT RESULTS:")
        print(f"   Missing labels:          {len(missing_labels)}")
        print(f"   Corrupted images:        {len(corrupted_files)}")
        print(f"   Duplicate images:        {len(duplicate_images)}")
        print(f"   Invalid bounding boxes:  {len(out_of_bounds_bboxes)}")

        is_clean = (len(corrupted_files) == 0 and len(missing_labels) == 0 and len(out_of_bounds_bboxes) == 0 and total_images > 0)
        print("\n" + "=" * 76)
        print(f"KAGGLE DATASET READY: {'YES' if is_clean else 'NO'}")
        print("=" * 76)

        return {
            "split_summary": split_summary,
            "class_counter": class_counter,
            "corrupted": len(corrupted_files),
            "missing": len(missing_labels),
            "duplicates": len(duplicate_images),
            "invalid_bboxes": len(out_of_bounds_bboxes),
            "ready": is_clean
        }

if __name__ == "__main__":
    auditor = DatasetAuditor(r"C:\Users\91866\OneDrive\Desktop\YOLO\dataset")
    auditor.audit_all()
