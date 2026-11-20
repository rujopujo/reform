"""
check_data.py — Dataset validation and inspection tool for ReForm.
Usage:
    python scripts/check_data.py
    python scripts/check_data.py --min-train 300 --verify-images
    python scripts/check_data.py --inspect <folder> --depth 3
"""

import argparse
import hashlib
import os
import sys
from collections import defaultdict
from pathlib import Path
from PIL import Image

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
ANNOTATION_EXTENSIONS = {".txt", ".xml", ".json", ".yaml", ".yml", ".csv"}

STAGE0_EXPECTED = ["plastic", "metal", "PCB", "glass_screen", "battery", "unknown"]
STAGE1_EXPECTED = ["keyboard", "mouse", "remote", "printer_casing", "tv_panel", "other_plastic", "unknown"]


def hash_file(filepath: Path) -> str:
    """Compute MD5 hash of file content to detect leakage and duplicates."""
    h = hashlib.md5()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def inspect_directory(root_path: Path, max_depth: int = 3):
    print(f"\n[INSPECT] Scanning '{root_path}' (depth <= {max_depth})...\n")
    if not root_path.exists():
        print(f"Error: Directory '{root_path}' does not exist.")
        sys.exit(1)

    root_level = len(root_path.parts)
    folder_stats = []
    has_annotations = False
    total_images = 0

    for current_dir, dirs, files in os.walk(root_path):
        current_path = Path(current_dir)
        depth = len(current_path.parts) - root_level
        if depth > max_depth:
            continue

        images = [f for f in files if Path(f).suffix.lower() in IMAGE_EXTENSIONS]
        annotations = [f for f in files if Path(f).suffix.lower() in ANNOTATION_EXTENSIONS and f != "data.yaml"]
        if Path(current_dir, "data.yaml").exists():
            has_annotations = True
        if annotations:
            has_annotations = True

        img_count = len(images)
        total_images += img_count
        if img_count > 0 or depth <= max_depth:
            folder_stats.append((current_path, depth, img_count, len(annotations)))

    for path, depth, img_count, annot_count in folder_stats:
        indent = "  " * depth
        rel = path.relative_to(root_path) if path != root_path else "."
        note = f" ({annot_count} annotations)" if annot_count > 0 else ""
        print(f"{indent}- {rel}: {img_count} images{note}")

    print(f"\nTotal images found: {total_images}")
    if has_annotations:
        print("[WARNING] Annotations/YAML detected! This appears to be an object-detection set. Crop first with crop_yolo.py.")


def check_stage(stage_dir: Path, stage_num: int, min_train: int = 0, verify_images: bool = False):
    errors = []
    warnings = []

    print(f"\n{'='*70}\nSTAGE {stage_num} DATASET CHECK: {stage_dir}\n{'='*70}")
    if not stage_dir.exists():
        errors.append(f"Stage directory missing: {stage_dir}")
        return errors, warnings

    splits = ["train", "val", "test"]
    for s in splits:
        if not (stage_dir / s).exists():
            errors.append(f"Split directory missing: {stage_dir / s}")

    expected_classes = STAGE0_EXPECTED if stage_num == 0 else STAGE1_EXPECTED

    # Discover all classes in train, val, test
    found_classes = set()
    split_classes = {s: set() for s in splits}
    for s in splits:
        s_dir = stage_dir / s
        if s_dir.exists():
            for c_dir in s_dir.iterdir():
                if c_dir.is_dir() and not c_dir.name.startswith("."):
                    found_classes.add(c_dir.name)
                    split_classes[s].add(c_dir.name)

    sorted_classes = sorted(found_classes, key=lambda x: (x not in expected_classes, x))

    # Check for empty folders and missing splits
    class_counts = {c: {s: 0 for s in splits} for c in sorted_classes}
    file_hashes = defaultdict(lambda: defaultdict(list)) # hash -> split -> list of (class, path)

    for c in sorted_classes:
        for s in splits:
            c_dir = stage_dir / s / c
            if not c_dir.exists():
                errors.append(f"Class '{c}' missing in '{s}' split ({c_dir})")
                continue

            imgs = [p for p in c_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS]
            count = len(imgs)
            class_counts[c][s] = count
            if count == 0:
                errors.append(f"Empty class folder: {c_dir}")

            # Verify images and compute hashes
            for p in imgs:
                if verify_images:
                    try:
                        with Image.open(p) as img:
                            img.verify()
                    except Exception as e:
                        errors.append(f"Corrupt/unreadable image: {p} ({e})")
                try:
                    f_hash = hash_file(p)
                    file_hashes[f_hash][s].append((c, p))
                except Exception as e:
                    errors.append(f"Failed to read file: {p} ({e})")

    # Leakage check: same hash in multiple splits
    leakage_count = 0
    for f_hash, splits_present in file_hashes.items():
        if len(splits_present) > 1:
            leakage_count += 1
            details = ", ".join(f"{s}: {[str(p[1].name) for p in paths]}" for s, paths in splits_present.items())
            if leakage_count <= 10:
                errors.append(f"Leakage detected across splits for same image: {details}")

    if leakage_count > 10:
        errors.append(f"... and {leakage_count - 10} more cross-split leakage instances.")

    # Cross-class check: same hash in multiple classes within the same split
    cross_class_count = 0
    for f_hash, splits_present in file_hashes.items():
        for s, class_paths in splits_present.items():
            distinct_classes = set(c for c, _ in class_paths)
            if len(distinct_classes) > 1:
                cross_class_count += 1
                if cross_class_count <= 5:
                    warnings.append(f"Duplicate image filed under multiple classes in '{s}': {distinct_classes}")

    # Print Table
    header = f"{'Class':<20} | {'Train':>8} | {'Val':>8} | {'Test':>8} | {'Total':>8}"
    print(header)
    print("-" * len(header))
    total_train = 0
    total_val = 0
    total_test = 0

    train_numbers = []
    for c in sorted_classes:
        tr = class_counts[c]["train"]
        va = class_counts[c]["val"]
        te = class_counts[c]["test"]
        tot = tr + va + te
        total_train += tr
        total_val += va
        total_test += te
        if tr > 0:
            train_numbers.append((c, tr))
        print(f"{c:<20} | {tr:>8} | {va:>8} | {te:>8} | {tot:>8}")

        if min_train > 0 and tr < min_train:
            warnings.append(f"Class '{c}' train count ({tr}) is below --min-train ({min_train})")

    print("-" * len(header))
    grand_total = total_train + total_val + total_test
    print(f"{'TOTAL':<20} | {total_train:>8} | {total_val:>8} | {total_test:>8} | {grand_total:>8}")

    # Class imbalance check
    if train_numbers:
        train_sorted = sorted(train_numbers, key=lambda x: x[1])
        min_c, min_val = train_sorted[0]
        max_c, max_val = train_sorted[-1]
        if min_val > 0 and (max_val / min_val) >= 10.0:
            allowed_high = {"plastic", "other_plastic", "unknown"}
            if max_c in allowed_high:
                warnings.append(f"Class imbalance >= 10:1 ({max_c}={max_val} vs {min_c}={min_val}), acceptable for background/collector classes.")
            else:
                warnings.append(f"High class imbalance >= 10:1 ({max_c}={max_val} vs {min_c}={min_val}). Consider adding images to '{min_c}'.")

    return errors, warnings


def main():
    parser = argparse.ArgumentParser(description="ReForm Dataset Checker & Inspector")
    parser.add_argument("--inspect", type=str, help="Inspect a raw/downloaded folder without checking train/val/test splits")
    parser.add_argument("--depth", type=int, default=3, help="Max depth for inspect")
    parser.add_argument("--min-train", type=int, default=0, help="Flag warning if train images per class < min-train")
    parser.add_argument("--verify-images", action="store_true", help="Verify image integrity with PIL")
    parser.add_argument("--data-dir", type=str, default="data", help="Path to data directory")

    args = parser.parse_args()

    if args.inspect:
        inspect_directory(Path(args.inspect), max_depth=args.depth)
        return

    data_dir = Path(args.data_dir)
    all_errors = []
    all_warnings = []

    for stage_num in [0, 1]:
        stage_dir = data_dir / f"stage{stage_num}"
        errs, warns = check_stage(stage_dir, stage_num, min_train=args.min_train, verify_images=args.verify_images)
        all_errors.extend(errs)
        all_warnings.extend(warns)

    print("\n" + "="*70)
    print("CHECK DATA SUMMARY")
    print("="*70)
    if all_warnings:
        print(f"\nWARNINGS ({len(all_warnings)}):")
        for w in all_warnings:
            print(f"  [WARN] {w}")
    if all_errors:
        print(f"\nERRORS ({len(all_errors)}):")
        for e in all_errors:
            print(f"  [ERROR] {e}")

    print(f"\nResult: {len(all_errors)} error(s), {len(all_warnings)} warning(s).")
    if all_errors:
        sys.exit(1)
    else:
        sys.exit(0)


if __name__ == "__main__":
    main()
