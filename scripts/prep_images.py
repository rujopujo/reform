"""
prep_images.py — Clean, stage, inspect, and ingest images for ReForm dataset.
Usage:
    python scripts/prep_images.py stage --out temp_datasets/stg/<class> --max 600 <src1> [<src2> ...]
    python scripts/prep_images.py sheet --source <folder> --out sheets/<class>.jpg
    python scripts/prep_images.py ingest --stage 1 --class keyboard --source temp_datasets/stg/keyboard --mirror-plastic
"""

import argparse
import hashlib
import math
import os
import random
import shutil
import sys
from pathlib import Path
from PIL import Image

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

STAGE0_CLASSES = ["plastic", "metal", "PCB", "glass_screen", "battery", "unknown"]
STAGE1_CLASSES = ["keyboard", "mouse", "remote", "printer_casing", "tv_panel", "other_plastic", "unknown"]


def hash_file(filepath: Path) -> str:
    h = hashlib.md5()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def cmd_stage(args):
    out_dir = Path(args.out)
    if out_dir.exists():
        if args.overwrite:
            shutil.rmtree(out_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
        else:
            print(f"Destination '{out_dir}' exists. Use --overwrite to replace it.")
    else:
        out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[STAGE] Gathering images from {len(args.sources)} source path(s) into '{out_dir}' (max={args.max})...")

    candidate_files = []
    skipped_augmented = 0

    for src_str in args.sources:
        src_path = Path(src_str)
        if not src_path.exists():
            print(f"[WARN] Source path does not exist: {src_path}")
            continue

        if src_path.is_file():
            all_files = [src_path]
        else:
            all_files = list(src_path.rglob("*"))

        for p in all_files:
            if not p.is_file() or p.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            # Rule: NEVER use *Augmented* folders
            parts_lower = [part.lower() for part in p.parts]
            if any("augmented" in part for part in parts_lower):
                skipped_augmented += 1
                continue
            candidate_files.append(p)

    print(f"Found {len(candidate_files)} raw candidate image files (skipped {skipped_augmented} in augmented folders).")

    valid_images = []
    seen_hashes = set()
    skipped_small = 0
    skipped_corrupt = 0
    skipped_duplicates = 0

    for p in candidate_files:
        try:
            with Image.open(p) as img:
                w, h = img.size
                if w < 96 or h < 96:
                    skipped_small += 1
                    continue
                # Verify reading pixels
                img.verify()
        except Exception:
            skipped_corrupt += 1
            continue

        try:
            f_hash = hash_file(p)
            if f_hash in seen_hashes:
                skipped_duplicates += 1
                continue
            seen_hashes.add(f_hash)
            valid_images.append(p)
        except Exception:
            skipped_corrupt += 1
            continue

    print(f"Validation: {len(valid_images)} valid unique images (skipped {skipped_small} < 96px, {skipped_corrupt} corrupt, {skipped_duplicates} duplicate).")

    if not valid_images:
        print("[ERROR] No valid images found to stage.")
        sys.exit(1)

    random.seed(42)
    random.shuffle(valid_images)

    if args.max and len(valid_images) > args.max:
        valid_images = valid_images[:args.max]
        print(f"Capped to {args.max} images per --max argument.")

    copied = 0
    for idx, p in enumerate(valid_images, 1):
        ext = p.suffix.lower()
        if ext not in [".jpg", ".jpeg"]:
            # Convert non-jpeg to jpg for uniform dataset
            dest_name = f"stg_{idx:05d}.jpg"
            dest_path = out_dir / dest_name
            try:
                with Image.open(p) as img:
                    img.convert("RGB").save(dest_path, "JPEG", quality=95)
                copied += 1
            except Exception as e:
                print(f"[WARN] Failed converting {p}: {e}")
        else:
            dest_name = f"stg_{idx:05d}{ext}"
            dest_path = out_dir / dest_name
            shutil.copy2(p, dest_path)
            copied += 1

    print(f"[STAGE COMPLETE] Successfully staged {copied} clean images in '{out_dir}'.\n")


def cmd_sheet(args):
    source_dir = Path(args.source)
    out_file = Path(args.out)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    if not source_dir.exists():
        print(f"[ERROR] Source folder does not exist: {source_dir}")
        sys.exit(1)

    images = [p for p in source_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS]
    if not images:
        # Check subdirectories
        images = [p for p in source_dir.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS]

    if not images:
        print(f"[ERROR] No images found in {source_dir}")
        sys.exit(1)

    count = min(args.count, len(images))
    random.seed(42)
    selected = random.sample(images, count)

    grid_size = int(math.ceil(math.sqrt(count)))
    thumb_size = 200
    sheet = Image.new("RGB", (grid_size * thumb_size, grid_size * thumb_size), color=(240, 240, 240))

    for idx, img_path in enumerate(selected):
        try:
            with Image.open(img_path) as img:
                thumb = img.convert("RGB")
                thumb.thumbnail((thumb_size, thumb_size))
                x = (idx % grid_size) * thumb_size + (thumb_size - thumb.width) // 2
                y = (idx // grid_size) * thumb_size + (thumb_size - thumb.height) // 2
                sheet.paste(thumb, (x, y))
        except Exception as e:
            print(f"[WARN] Failed loading {img_path}: {e}")

    sheet.save(out_file, quality=88)
    print(f"[SHEET] Created {count}-image contact sheet at '{out_file}'. Open and inspect it before ingesting.")


def cmd_ingest(args):
    stage = args.stage
    cls_name = args.class_name
    src_dir = Path(args.source)
    data_dir = Path(args.data_dir)

    expected = STAGE0_CLASSES if stage == 0 else STAGE1_CLASSES
    if cls_name not in expected:
        if not args.allow_new_class:
            lower_matches = [c for c in expected if c.lower() == cls_name.lower()]
            if lower_matches:
                print(f"[ERROR] '{cls_name}' is not an existing Stage {stage} class. Did you mean exact casing '{lower_matches[0]}'?")
            else:
                print(f"[ERROR] '{cls_name}' is not an existing Stage {stage} class (expected one of: {expected}).")
                print("Pass --allow-new-class if you intend to create a new class.")
            sys.exit(1)
        else:
            print(f"[INFO] Ingesting brand-new class '{cls_name}' into Stage {stage}.")

    if not src_dir.exists():
        print(f"[ERROR] Source folder does not exist: {src_dir}")
        sys.exit(1)

    staged_images = sorted([p for p in src_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS])
    if not staged_images:
        print(f"[ERROR] No images found in source: {src_dir}")
        sys.exit(1)

    print(f"[INGEST] Processing {len(staged_images)} images from '{src_dir}' for Stage {stage} class '{cls_name}'...")

    # Load existing hashes to prevent re-ingesting duplicates
    existing_hashes = set()
    target_stage_dir = data_dir / f"stage{stage}"
    for p in target_stage_dir.rglob("*"):
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS:
            try:
                existing_hashes.add(hash_file(p))
            except Exception:
                pass

    if args.mirror_plastic:
        plastic_dir = data_dir / "stage0"
        for p in plastic_dir.rglob("*"):
            if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS:
                try:
                    existing_hashes.add(hash_file(p))
                except Exception:
                    pass

    new_images = []
    skipped_existing = 0
    for p in staged_images:
        try:
            h = hash_file(p)
            if h in existing_hashes:
                skipped_existing += 1
                continue
            existing_hashes.add(h)
            new_images.append(p)
        except Exception:
            continue

    if skipped_existing > 0:
        print(f"Skipped {skipped_existing} images already existing in dataset.")

    if not new_images:
        print("[INFO] All images are already present in the dataset. Nothing to ingest.")
        return

    random.seed(42)
    random.shuffle(new_images)

    total = len(new_images)
    n_train = int(total * 0.70)
    n_val = int(total * 0.15)
    n_test = total - n_train - n_val

    splits = {
        "train": new_images[:n_train],
        "val": new_images[n_train:n_train + n_val],
        "test": new_images[n_train + n_val:]
    }

    print(f"Split plan (70/15/15): train={len(splits['train'])} val={len(splits['val'])} test={len(splits['test'])} (Total: {total})")

    if args.dry_run:
        print("[DRY-RUN] Preview complete. No files were copied.")
        return

    copied_counts = {"train": 0, "val": 0, "test": 0}
    mirrored_counts = {"train": 0, "val": 0, "test": 0}

    for split_name, img_list in splits.items():
        dst_folder = data_dir / f"stage{stage}" / split_name / cls_name
        dst_folder.mkdir(parents=True, exist_ok=True)

        mirror_folder = (data_dir / "stage0" / split_name / "plastic") if args.mirror_plastic else None
        if mirror_folder:
            mirror_folder.mkdir(parents=True, exist_ok=True)

        for p in img_list:
            # Generate unique name in destination
            h = hash_file(p)[:8]
            fname = f"{cls_name}_{h}_{p.name}"
            target_file = dst_folder / fname
            shutil.copy2(p, target_file)
            copied_counts[split_name] += 1

            if mirror_folder:
                mirror_file = mirror_folder / fname
                shutil.copy2(p, mirror_file)
                mirrored_counts[split_name] += 1

    print(f"[INGEST COMPLETE] Copied {copied_counts['train']} train, {copied_counts['val']} val, {copied_counts['test']} test into 'data/stage{stage}' class '{cls_name}'.")
    if args.mirror_plastic:
        print(f"[MIRROR COMPLETE] Mirrored {mirrored_counts['train']} train, {mirrored_counts['val']} val, {mirrored_counts['test']} test into 'data/stage0/*/plastic'.")


def main():
    parser = argparse.ArgumentParser(description="ReForm Image Preparation & Staging Utility")
    subparsers = parser.add_subparsers(dest="subcommand", required=True)

    # stage
    p_stage = subparsers.add_parser("stage", help="Clean and stage raw images")
    p_stage.add_argument("--out", required=True, help="Destination directory for staged images")
    p_stage.add_argument("--max", type=int, default=600, help="Max random images to keep")
    p_stage.add_argument("--overwrite", action="store_true", help="Overwrite destination directory")
    p_stage.add_argument("sources", nargs="+", help="One or more source folders")

    # sheet
    p_sheet = subparsers.add_parser("sheet", help="Create contact sheet grid")
    p_sheet.add_argument("--source", required=True, help="Source image folder")
    p_sheet.add_argument("--out", required=True, help="Output image file (e.g. sheets/keyboard.jpg)")
    p_sheet.add_argument("--count", type=int, default=25, help="Number of images in grid")

    # ingest
    p_ingest = subparsers.add_parser("ingest", help="Ingest staged images into dataset splits (70/15/15)")
    p_ingest.add_argument("--stage", type=int, choices=[0, 1], required=True, help="Target stage (0 or 1)")
    p_ingest.add_argument("--class", dest="class_name", required=True, help="Target class name (case sensitive)")
    p_ingest.add_argument("--source", required=True, help="Staged source directory")
    p_ingest.add_argument("--mirror-plastic", action="store_true", help="Mirror Stage 1 plastic items into Stage 0 plastic")
    p_ingest.add_argument("--allow-new-class", action="store_true", help="Allow creating a new class not in defaults")
    p_ingest.add_argument("--dry-run", action="store_true", help="Calculate split plan without copying")
    p_ingest.add_argument("--data-dir", default="data", help="Root data directory")

    args = parser.parse_args()
    if args.subcommand == "stage":
        cmd_stage(args)
    elif args.subcommand == "sheet":
        cmd_sheet(args)
    elif args.subcommand == "ingest":
        cmd_ingest(args)


if __name__ == "__main__":
    main()
