"""
eval_test.py — Evaluate ReForm models on held-out test splits.
Usage:
    python scripts/eval_test.py --stage all
    python scripts/eval_test.py --stage 0
    python scripts/eval_test.py --stage 1
"""

import argparse
import json
import os
import sys
from pathlib import Path
import numpy as np

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from sklearn.metrics import classification_report, confusion_matrix

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def evaluate_stage(stage: int, data_dir: Path, models_dir: Path):
    stage_dir = data_dir / f"stage{stage}"
    test_dir = stage_dir / "test"
    model_path = models_dir / f"stage{stage}_model.keras"
    classes_path = models_dir / f"stage{stage}_classes.json"
    report_path = models_dir / f"stage{stage}_test_report.txt"

    print(f"\n{'='*70}\nEVALUATING STAGE {stage} ON TEST SPLIT\n{'='*70}")

    if not test_dir.exists():
        print(f"[ERROR] Test directory not found: {test_dir}")
        return False

    if not model_path.exists():
        print(f"[ERROR] Model file not found: {model_path}")
        return False

    if classes_path.exists():
        with open(classes_path, "r", encoding="utf-8") as f:
            class_names = json.load(f)
    else:
        class_names = sorted([d.name for d in test_dir.iterdir() if d.is_dir() and not d.name.startswith(".")])

    print(f"Model: {model_path.name}")
    print(f"Classes ({len(class_names)}): {class_names}")

    test_ds = tf.keras.utils.image_dataset_from_directory(
        str(test_dir),
        labels="inferred",
        label_mode="categorical",
        class_names=class_names,
        color_mode="rgb",
        batch_size=32,
        image_size=(224, 224),
        shuffle=False
    )

    test_ds_prep = test_ds.map(lambda x, y: (preprocess_input(x), y))

    model = tf.keras.models.load_model(str(model_path), compile=False)

    y_true_list = []
    for _, y_batch in test_ds:
        y_true_list.append(np.argmax(y_batch.numpy(), axis=1))
    y_true = np.concatenate(y_true_list) if y_true_list else np.array([])

    if len(y_true) == 0:
        print("[ERROR] No test samples found.")
        return False

    probs = model.predict(test_ds_prep, verbose=1)
    y_pred = probs.argmax(axis=1)

    acc = float((y_pred == y_true).mean())
    cm = confusion_matrix(y_true, y_pred, labels=range(len(class_names)))
    pairs = sorted(
        ((cm[i, j], class_names[i], class_names[j])
         for i in range(len(class_names)) for j in range(len(class_names))
         if i != j and cm[i, j] > 0),
        reverse=True
    )[:5]

    lines = [
        f"STAGE {stage} TEST RESULTS ({len(y_true)} images)",
        f"Model: {model_path.name}",
        f"Accuracy: {acc * 100:.2f}%",
        "",
        classification_report(y_true, y_pred, labels=range(len(class_names)), target_names=class_names, zero_division=0),
        "Confusion matrix (rows = true, columns = predicted):",
        "classes: " + ", ".join(class_names)
    ]
    lines += [" ".join(f"{v:5d}" for v in row) + f"   <- {class_names[i]}" for i, row in enumerate(cm)]
    lines += [
        "",
        "Most common mistakes (count: true -> predicted):"
    ] + ([f"  {n}: {t} -> {p}" for n, t, p in pairs] or ["  none"])

    report_text = "\n".join(lines)
    print("\n" + report_text + "\n")

    report_path.write_text(report_text, encoding="utf-8")
    print(f"Report saved to: {report_path}")
    return True


def main():
    parser = argparse.ArgumentParser(description="Evaluate ReForm models on test split")
    parser.add_argument("--stage", choices=["0", "1", "all"], default="all", help="Stage to evaluate (0, 1, or all)")
    parser.add_argument("--data-dir", default="data", help="Data directory path")
    parser.add_argument("--models-dir", default="models", help="Models directory path")

    args = parser.parse_args()
    data_dir = Path(args.data_dir)
    models_dir = Path(args.models_dir)

    stages = [0, 1] if args.stage == "all" else [int(args.stage)]
    all_ok = True
    for s in stages:
        ok = evaluate_stage(s, data_dir, models_dir)
        if not ok:
            all_ok = False

    if not all_ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
