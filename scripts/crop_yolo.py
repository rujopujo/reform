"""
crop_yolo.py — Crop bounding boxes from YOLO object detection datasets.
Usage:
    python scripts/crop_yolo.py <yolo_dir> crops/<name>
"""

import argparse
import os
import sys
from pathlib import Path
from PIL import Image
import yaml

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def find_data_yaml(yolo_dir: Path):
    candidates = list(yolo_dir.rglob("data.yaml")) + list(yolo_dir.rglob("*.yaml"))
    for c in candidates:
        try:
            with open(c, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
                if isinstance(data, dict) and "names" in data:
                    return c, data
        except Exception:
            continue
    return None, None


def get_class_names(yaml_data):
    if not yaml_data or "names" not in yaml_data:
        return {}
    names = yaml_data["names"]
    if isinstance(names, list):
        return {i: str(n).strip() for i, n in enumerate(names)}
    elif isinstance(names, dict):
        return {int(k): str(v).strip() for k, v in names.items()}
    return {}


def main():
    parser = argparse.ArgumentParser(description="Crop bounding boxes from a YOLOv8 dataset")
    parser.add_argument("yolo_dir", type=str, help="Path to YOLO dataset root directory")
    parser.add_argument("out_dir", type=str, help="Destination directory for cropped images")
    parser.add_argument("--min-size", type=int, default=32, help="Minimum width and height of crop in pixels")

    args = parser.parse_args()
    yolo_dir = Path(args.yolo_dir)
    out_dir = Path(args.out_dir)

    if not yolo_dir.exists():
        print(f"Error: YOLO directory not found: {yolo_dir}")
        sys.exit(1)

    yaml_path, yaml_data = find_data_yaml(yolo_dir)
    class_names = get_class_names(yaml_data)
    if yaml_path:
        print(f"Loaded class names from {yaml_path}: {class_names}")
    else:
        print("Warning: data.yaml not found. Using class indices as class names.")

    out_dir.mkdir(parents=True, exist_ok=True)

    # Search for all image files
    all_images = [p for p in yolo_dir.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS]
    print(f"Found {len(all_images)} images in {yolo_dir}")

    cropped_counts = {}
    skipped_small = 0
    total_crops = 0

    for img_path in all_images:
        # Corresponding label file can be in labels/ sibling or same folder
        # e.g., images/train/foo.jpg -> labels/train/foo.txt or train/images/foo.jpg -> train/labels/foo.txt
        label_candidates = [
            img_path.with_suffix(".txt"),
            Path(str(img_path).replace(os.sep + "images" + os.sep, os.sep + "labels" + os.sep)).with_suffix(".txt"),
            img_path.parent.parent / "labels" / img_path.parent.name / f"{img_path.stem}.txt",
            img_path.parent.parent / "labels" / f"{img_path.stem}.txt",
        ]

        label_path = None
        for cand in label_candidates:
            if cand.exists():
                label_path = cand
                break

        if not label_path:
            continue

        try:
            with Image.open(img_path) as img:
                img_w, img_h = img.size
                rgb_img = img.convert("RGB")

                with open(label_path, "r", encoding="utf-8") as f:
                    lines = [line.strip().split() for line in f if line.strip()]

                for idx, parts in enumerate(lines):
                    if len(parts) < 5:
                        continue
                    cls_id = int(parts[0])
                    xc, yc, w, h = map(float, parts[1:5])

                    cls_name = class_names.get(cls_id, f"class_{cls_id}")
                    # Sanitise folder name
                    cls_name_safe = cls_name.replace(" ", "_").lower()

                    xmin = max(0, int((xc - w / 2.0) * img_w))
                    ymin = max(0, int((yc - h / 2.0) * img_h))
                    xmax = min(img_w, int((xc + w / 2.0) * img_w))
                    ymax = min(img_h, int((yc + h / 2.0) * img_h))

                    crop_w = xmax - xmin
                    crop_h = ymax - ymin

                    if crop_w < args.min_size or crop_h < args.min_size:
                        skipped_small += 1
                        continue

                    crop_img = rgb_img.crop((xmin, ymin, xmax, ymax))

                    cls_dir = out_dir / cls_name_safe
                    cls_dir.mkdir(parents=True, exist_ok=True)

                    out_path = cls_dir / f"{img_path.stem}_crop_{idx}.jpg"
                    crop_img.save(out_path, quality=92)

                    cropped_counts[cls_name_safe] = cropped_counts.get(cls_name_safe, 0) + 1
                    total_crops += 1
        except Exception as e:
            print(f"Error processing {img_path}: {e}")

    print("\n" + "="*50)
    print("CROP SUMMARY")
    print("="*50)
    for c, cnt in sorted(cropped_counts.items()):
        print(f"  {c}: {cnt} crops")
    print(f"Total crops saved: {total_crops}")
    if skipped_small > 0:
        print(f"Skipped crops < {args.min_size}px: {skipped_small}")
    print(f"Output saved to: {out_dir}")


if __name__ == "__main__":
    main()
