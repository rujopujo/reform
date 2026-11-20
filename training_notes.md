# ReForm Training Notes - Run 1

**Date:** 2026-11-20  
**Branch:** `training-run-1`  
**Repository:** `c:\Ruhaan model train\Git folderr\reform`  

---

## 1. Environment & Hardware Status

- **Git Version:** `git version 2.55.0.windows.5`
- **Python Environment:**
  - System Python: `Python 3.12.3` (64-bit)
  - Installed Python 3.11 via `winget`: `Python 3.11.9`
  - Created `.venv` with: `py -3.11 -m venv .venv`
  - Upgraded `pip` to 26.3
  - Installed pinned dependencies: `tensorflow-cpu==2.15.1`, `numpy==1.26.4`, `scikit-learn==1.9.1`, `pillow==13.0.0`, `pyyaml==6.0.3`, `kaggle==2.2.5`, `roboflow==1.7.3`
- **Hardware & GPU Status (WSL2 Environment Activated):**
  - **Environment:** Switched execution to WSL2 Ubuntu (`.venv_gpu` located at `.venv_gpu/`).
  - **GPU Detected:** `NVIDIA GeForce RTX 5070 Ti` (16,303 MiB VRAM, Driver 621.24, CUDA 13.5).
  - **TensorFlow Configuration:** `tensorflow[and-cuda]==2.15.1` detecting `[PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')]`.
  - **Compute Capability & PTX Note:** The RTX 5070 Ti features Blackwell architecture (`sm_120` / compute capability 12.0). TensorFlow 2.15 was compiled against older CUDA compute architectures (up to sm_89/90), logging:
    `W tensorflow/core/common_runtime/gpu/gpu_device.cc:2348] TensorFlow was not built with CUDA kernel binaries compatible with compute capability 12.0. CUDA kernels will be jit-compiled from PTX...`
    CUDA driver JIT compiles the required kernels from embedded PTX to run natively on the RTX 5070 Ti.

---

## 2. Setup Actions Executed

- Checked out new git branch:
  ```powershell
  git checkout -b training-run-1
  ```
- Created directories:
  ```powershell
  mkdir temp_datasets
  mkdir models_backup
  ```
- Backed up existing models to `models_backup/`:
  - `models/model_comparison.json` -> `models_backup/model_comparison.json`
  - `models/stage0_classes.json` -> `models_backup/stage0_classes.json`
  - `models/stage0_model.keras` -> `models_backup/stage0_model.keras`
  - `models/stage1_classes.json` -> `models_backup/stage1_classes.json`
  - `models/stage1_model.keras` -> `models_backup/stage1_model.keras`
- Implemented and verified the 4 helper scripts in `scripts/`:
  - `scripts/check_data.py`: Multi-stage data validation, leakage check, empty folders, inspect mode
  - `scripts/prep_images.py`: Staging with deduping/capping, contact sheet generator, 70/15/15 ingestion with `--mirror-plastic`
  - `scripts/crop_yolo.py`: YOLOv8 detection dataset parser and bounding-box cropper
  - `scripts/eval_test.py`: Full evaluation of models on held-out test splits with classification reports and confusion matrices

---

## 3. Dataset Changes & Ingestion Summary

### Remediations & Staging
1. **Remote Remediation (Option A):** Moved all 451 legacy media-player images from `data/stage1/{train,val,test}/remote` into `data/stage1/{split}/other_plastic`. Replaced with 339 pure, cropped handheld remote control boxes from Roboflow `tahsin/tv-remote`.
2. **Printer Casing:** Added 550 crops from Roboflow `2-olter/office-printer-bbogo`.
3. **Keyboards:** Added 550 raw images from Mendeley Accessories, EWasteNet, and Laptop Components.
4. **Mice:** Added 515 raw images from Mendeley Accessories and EWasteNet.
5. **TV Panels:** Added 496 raw images from EWasteNet and Mendeley Accessories.
6. **PCBs:** Added 140 raw motherboard images from Laptop Components.
7. **Batteries:** Added 280 raw battery pack and CMOS battery images from Laptop Components.
8. **Glass Screens:** Added 140 raw LCD screen images from Laptop Components.
9. **Metal:** Added 121 unique metal images from Kaggle Garbage Classification.

---

## 4. Final Pre-Training Dataset Gate Check (Part 6)

Command executed:
```powershell
.venv\Scripts\python.exe scripts/check_data.py --min-train 300 --verify-images
```

### Stage 0 Final Counts
| Class | Train | Val | Test | Total |
|---|---|---|---|---|
| PCB | 700 | 166 | 124 | 990 |
| battery | 744 | 169 | 125 | 1,038 |
| glass_screen | 601 | 138 | 101 | 840 |
| metal | 637 | 162 | 121 | 920 |
| plastic | 4,860 | 1,264 | 865 | 6,989 |
| unknown | 1,399 | 382 | 216 | 1,997 |
| **TOTAL** | **8,941** | **2,281** | **1,552** | **12,774** |

### Stage 1 Final Counts
| Class | Train | Val | Test | Total |
|---|---|---|---|---|
| keyboard | 696 | 190 | 123 | 1,009 |
| mouse | 691 | 157 | 116 | 964 |
| other_plastic | 1,517 | 460 | 250 | 2,227 |
| printer_casing | 700 | 170 | 130 | 1,000 |
| remote | 237 | 50 | 52 | 339 |
| tv_panel | 669 | 162 | 119 | 950 |
| unknown | 2,555 | 690 | 359 | 3,604 |
| **TOTAL** | **7,065** | **1,879** | **1,149** | **10,093** |

**Gate Check Result:** `0 error(s), 2 warning(s)` (remote train=237 < 300, and expected collector imbalance for unknown).
All images verified with PIL: **0 corrupt files, 0 split leakage, 0 empty folders.**

---

## 5. Old Model Baseline Scores on Updated Test Split (Part 6)

Command executed:
```powershell
.venv\Scripts\python.exe scripts/eval_test.py --stage all
```

### Stage 0 Old Model Baseline
- **Overall Test Accuracy:** **65.66%** (1,552 test images)
- **Per-Class Breakdown:**
  - `PCB`: precision 0.49, recall 0.78, f1 0.60 (support: 124)
  - `battery`: precision 0.35, recall 0.79, f1 0.49 (support: 125)
  - `glass_screen`: precision 0.61, recall 0.60, f1 0.61 (support: 101)
  - `metal`: precision 0.45, recall 0.93, f1 0.61 (support: 121)
  - `plastic`: precision 0.96, recall 0.59, f1 0.73 (support: 865)
  - `unknown`: precision 0.74, recall 0.62, f1 0.68 (support: 216)
- **Top Errors:**
  - 151: plastic -> battery
  - 94: plastic -> metal
  - 48: plastic -> PCB

### Stage 1 Old Model Baseline
- **Overall Test Accuracy:** **74.67%** (1,149 test images)
- **Per-Class Breakdown:**
  - `keyboard`: precision 0.95, recall 0.85, f1 0.90 (support: 123)
  - `mouse`: precision 0.92, recall 0.78, f1 0.85 (support: 116)
  - `other_plastic`: precision 0.83, recall 0.38, f1 0.52 (support: 250)
  - `printer_casing`: precision 0.91, recall 0.97, f1 0.94 (support: 130)
  - `remote`: precision 0.16, recall 0.44, f1 0.24 (support: 52) *(poor performance due to legacy training on media players)*
  - `tv_panel`: precision 0.86, recall 0.68, f1 0.76 (support: 119)
  - `unknown`: precision 0.75, recall 0.94, f1 0.83 (support: 359)
- **Top Errors:**
  - 69: other_plastic -> remote
  - 61: other_plastic -> unknown
  - 22: remote -> unknown
  - 20: mouse -> remote

---

## 6. Dataset Licensing Attributions

- **Electronic Accessories Classification Dataset:** Mendeley Data, CC BY 4.0 (Attribution required)
- **Laptop Components Dataset:** Mendeley Data, CC BY 4.0 (Attribution required)
- **EWasteNet Dataset:** Research dataset (Niful Islam et al.)
- **Garbage Classification Dataset:** Kaggle (mostafaabla/garbage-classification)
- **Roboflow Detection Sets:** CC BY 4.0 / Open Source where specified


---

## 7. Part 7.1 Dry Run Results (GPU on WSL2)

Executed with `train.ipynb` (`DRY_RUN = True`, batch size 64, mixed precision float16, in-memory caching):
- **Hardware Verified:** `NVIDIA GeForce RTX 5070 Ti` (compute capability 12.0)
- **Mixed Precision:** Enabled (`mixed_float16`)
- **PTX JIT Compilation:** Executed smoothly during initial epoch graph construction; no crashes or memory overflow.

### Per-Epoch GPU Timing
- **Stage 0 (8,941 train images):**
  - Phase 1 (Head): Epoch 1 = 63s (cache fill + PTX JIT), Epoch 2 = **14s** (warm cache, 100ms/step)
  - Phase 2 (Fine-tune): Epoch 1 = 73s
  - Estimated Stage 0 full run: **~16 minutes**
- **Stage 1 (7,065 train images):**
  - Phase 1 (Head): Epoch 1 = 13s, Epoch 2 = **7s** (warm cache, 63ms/step)
  - Phase 2 (Fine-tune): Epoch 1 = **10s** (91ms/step)
  - Estimated Stage 1 full run: **~3 minutes**

### Total Estimated Full-Run Duration
- **Combined full training duration:** **~19 minutes** (far below the 4-hour threshold!).
- Outputs were safely saved to `models/_dryrun/` without touching `models/`.


---

## 8. Full Training Pipeline Execution & Comparative Evaluation (Parts 7 & 8)

Full training executed on **NVIDIA GeForce RTX 5070 Ti** (`mixed_float16`, batch size 64, in-memory caching):
- Automatic pre-training backup created at: `models_backup/20261120_104925/` (7 files).
- Total training duration for both stages: **6.5 minutes** (Stage 0: 4.5 min, Stage 1: 2.0 min).

### Training Progression & EarlyStopping
- **Stage 0 (Material Sorter):**
  - Phase 1 (Top Layer): EarlyStopping restored best weights from Epoch 4/15.
  - Phase 2 (Fine-Tuning): EarlyStopping restored best weights from Epoch 1/10.
  - Final Validation Accuracy: **42.31%** (loss 1.5456).
- **Stage 1 (Item Classifier):**
  - Phase 1 (Top Layer): EarlyStopping restored best weights from Epoch 4/15.
  - Phase 2 (Fine-Tuning): EarlyStopping restored best weights from Epoch 1/10.
  - Final Validation Accuracy: **63.17%** (loss 1.2953).

### Test Split Comparison: New Models vs. Old Baseline

#### Stage 0 (Material Sorter)
- **Old Model Accuracy:** **65.66%**
- **New Model Accuracy:** **42.07%**
- **Per-Class Metrics (New vs. Old):**
  - `PCB`: precision 0.23 vs 0.49 | recall 0.35 vs 0.78 | f1 0.28 vs 0.60
  - `battery`: precision 0.29 vs 0.35 | recall 0.59 vs 0.79 | f1 0.39 vs 0.49
  - `glass_screen`: precision 0.20 vs 0.61 | recall 0.50 vs 0.60 | f1 0.29 vs 0.61
  - `metal`: precision 0.28 vs 0.45 | recall 0.64 vs 0.93 | f1 0.39 vs 0.61
  - `plastic`: precision 0.89 vs 0.96 | recall 0.36 vs 0.59 | f1 0.52 vs 0.73
  - `unknown`: precision 0.43 vs 0.74 | recall 0.42 vs 0.62 | f1 0.43 vs 0.68

#### Stage 1 (Item Classifier)
- **Old Model Accuracy:** **74.67%**
- **New Model Accuracy:** **65.19%**
- **Per-Class Metrics (New vs. Old):**
  - `remote`: **precision 0.72 vs 0.16 | recall 0.85 vs 0.44 | f1 0.78 vs 0.24**  
    *(Huge breakthrough: replacing media-player contamination with genuine remote control crops tripled F1 performance from 0.24 to 0.78!)*
  - `other_plastic`: precision 0.56 vs 0.83 | recall 0.76 vs 0.38 | f1 0.64 vs 0.52
  - `printer_casing`: precision 0.77 vs 0.91 | recall 0.82 vs 0.97 | f1 0.80 vs 0.94
  - `keyboard`: precision 0.61 vs 0.95 | recall 0.59 vs 0.85 | f1 0.60 vs 0.90
  - `mouse`: precision 0.55 vs 0.92 | recall 0.61 vs 0.78 | f1 0.58 vs 0.85
  - `tv_panel`: precision 0.65 vs 0.86 | recall 0.43 vs 0.68 | f1 0.52 vs 0.76
  - `unknown`: precision 0.75 vs 0.75 | recall 0.59 vs 0.94 | f1 0.66 vs 0.83


---

## 9. Stage 0 Retraining (Option 2) with Tuned Settings

### Motivation & Root Cause
In Run 1, Stage 0 test accuracy dropped to **42.07%** (vs 65.66% old baseline). Root cause: Sklearn's default `compute_class_weight('balanced')` assigned the dominant `plastic` class (4,860 images) a tiny weight of **0.31** while assigning minority classes (`glass_screen`, `metal`, `PCB`, `battery`) weights between **2.00 and 2.48**. This created an **8:1 loss penalty ratio**, which heavily penalized the model for missing any minority class and caused it to hedge ambiguous plastic surfaces into minority classes (misclassifying 475 out of 865 true plastic images, dropping plastic recall to 36%). Furthermore, `EarlyStopping(patience=4)` and `ReduceLROnPlateau(patience=2)` prematurely aborted training in Phase 1 (Epoch 8) and Phase 2 (Epoch 5).

### Tuned Configuration (`train.ipynb` CONFIG)
- **Target:** `STAGES = [0]` (only retrained Stage 0; `models/stage1_model.keras` and `models/stage1_classes.json` preserved intact).
- **Class Weighting:** `CLASS_WEIGHT_MODE = "sqrt"` (Normalized square-root class weights: $w = \sqrt{w_{\text{balanced}}} / \text{mean}(\sqrt{w_{\text{balanced}}})$).
  - `plastic`: **0.44**
  - `unknown`: **0.82**
  - `battery`: **1.12**
  - `PCB`: **1.16**
  - `metal`: **1.21**
  - `glass_screen`: **1.25**
  - **Penalty Ratio:** Max/Min ratio reduced from **8.0x** down to **2.8x**.
- **Early Stopping & Learning Rate:**
  - `EARLY_STOPPING_PATIENCE = 6` (increased from 4)
  - `REDUCE_LR_PATIENCE = 3` (increased from 2)
- **Pipeline:** `tf.data.AUTOTUNE` active on all maps/prefetch, in-memory uint8 caching.
- **Hardware:** RTX 5070 Ti (16 GB, `sm_120`), CUDA 13.5, `mixed_float16`, batch size 64.

### Training Progression & Epoch Metrics
- Automatic pre-run backup created: `models_backup/20261120_112240/` (previous backups untouched).
- **Phase 1 (Top Layer, 15 epochs):**
  - Epoch 1: loss 1.4880, val_loss 1.3005, val_acc 57.12% (137ms/step)
  - Epoch 6: loss 1.2309, val_loss 1.2490, val_acc 63.61%
  - Epoch 9: ReduceLR -> 0.0005
  - Epoch 10: loss 1.1860, val_loss 1.2207, val_acc 66.16%
  - Epoch 14: ReduceLR -> 0.00025
  - Epoch 15: loss 1.1680, val_loss 1.2082, val_acc 65.72%
- **Phase 2 (Fine-Tuning last 30 layers, 10 epochs):**
  - Epoch 1: loss 1.2356, val_loss 1.1698, val_acc 66.11%
  - Epoch 3: loss 1.1991, val_loss 1.1424, val_acc 66.90%
  - Epoch 6: loss 1.1676, val_loss 1.1330, val_acc 68.26%
  - Epoch 9: loss 1.1434, val_loss 1.1270, val_acc 68.96%
  - Epoch 10: loss 1.1382, val_loss 1.1356, val_acc **69.09%**
- **Total Duration:** 7.3 minutes.
- **Saved Artifacts:** `models/stage0_model.keras` (9.58 MB float32), `models/stage0_classes.json`.

### Three-Way Comparative Evaluation (1,552 Held-Out Test Images)

| Metric | Old Model Baseline | Retrained Run 1 (Severe Weights) | Retrained Run 2 (Tuned Sqrt Weights) |
|---|---|---|---|
| **Overall Test Accuracy** | **65.66%** | **42.07%** | **66.24%** (+0.58% over Old, +24.17% over Run 1) |
| **Stage 0 Val Accuracy** | N/A | 42.31% | **69.09%** |
| `plastic` Precision | 0.96 | 0.89 | 0.68 |
| `plastic` Recall | 0.59 | 0.36 | **0.93** (808 / 865 correct!) |
| `plastic` F1-Score | 0.73 | 0.52 | **0.79** |
| `PCB` Precision / Recall / F1 | 0.49 / 0.78 / 0.60 | 0.23 / 0.35 / 0.28 | **0.70** / 0.32 / 0.44 |
| `battery` Precision / Recall / F1 | 0.35 / 0.79 / 0.49 | 0.29 / 0.59 / 0.39 | **0.61** / 0.32 / 0.42 |
| `glass_screen` Precision / Recall / F1 | 0.61 / 0.60 / 0.61 | 0.20 / 0.50 / 0.29 | 0.46 / 0.26 / 0.33 |
| `metal` Precision / Recall / F1 | 0.45 / 0.93 / 0.61 | 0.28 / 0.64 / 0.39 | **0.77** / 0.28 / 0.41 |
| `unknown` Precision / Recall / F1 | 0.74 / 0.62 / 0.68 | 0.43 / 0.42 / 0.43 | 0.56 / 0.37 / 0.45 |

### Confusion Matrix (Run 2 Retrained Stage 0)
```
classes: PCB, battery, glass_screen, metal, plastic, unknown
   40     1     1     0    62    20   <- PCB
    2    40     3     0    78     2   <- battery
    2     2    26     2    64     5   <- glass_screen
    2     5     5    34    63    12   <- metal
    3    10    15     5   808    24   <- plastic
    8     8     7     3   110    80   <- unknown
```

### Trade-off Analysis: Minority Class Recall
While the tuned normalized square-root weighting resolved the plastic misclassification failure (rebounding `plastic` recall from 36% to **93%**, raising overall Stage 0 test accuracy to **66.24%**, and lifting plastic F1 to **0.79**), it introduced a known trade-off:
- Lower recall on minority Stage 0 classes (`glass_screen`: 26%, `PCB`: 32%, `metal`: 28%, `battery`: 32%).
- **Why this occurred:** Under the original 8:1 loss penalty, the network was coerced into predicting minority classes on any ambiguous boundary to avoid the heavy loss penalty, artificially inflating minority recall at the expense of plastic. With the balanced 2.8x ratio, the network requires higher confidence to predict a minority class, and ambiguous items with plastic backings or frames are more frequently attributed to plastic.
- **Future Recommendation:** To raise minority class recall without penalizing plastic, future iterations should ingest more diverse raw non-plastic samples (especially uncased PCBs and cracked glass panels) or apply a slight inference-time probability threshold adjustment in `src/server.py` for minority materials.


---

## 10. Final Verification & Post-Training Cleanup

### Final Gate Check
- Ran `python scripts/check_data.py --verify-images` across all 22,867 dataset images.
- **Results:** `0 error(s), 1 warning(s)`.
  - 0 corrupt files, 0 decoding errors, 0 leakage, 0 empty folders.
  - Expected collector imbalance warning on `unknown` class.

### App & Streamlit Startup Verification
- Confirmed primary dashboard entry point is `app.py` in the project root (which delegates cleanly to `src/app.py`).
- Direct invocation `streamlit run app.py` (port 8501) and backend `python -m uvicorn server:app --host 127.0.0.1 --port 7860` verified.
- FastAPI health endpoint `/api/health` reports `status: "healthy"` with both models loaded:
  - `stage0_loaded: True` (6 classes)
  - `stage1_loaded: True` (7 classes)
- End-to-end cascading inference tested on a real e-waste image (`remote_06de8612_stg_00328.jpg`): correctly predicted Stage 0 `plastic` (69.7%) and cascaded to Stage 1 `remote` (68.9%).

### Workspace Cleanup (Reclaimed ~19.44 GB)
Approved cleanup of temporary working folders executed:
- `temp_datasets/`: Deleted 67,671 raw download files and TF cache (~19.34 GB).
- `crops/`: Deleted 2,604 intermediate Roboflow YOLO crops (80.48 MB).
- `sheets/`: Deleted 10 visual review contact sheets (1.19 MB).
- `models/_dryrun/`: Deleted 9 dry-run model artifacts and test outputs (18.38 MB).

### Preserved Deliverables & Git Safety
- `reform_results/` and `reform_results.zip` (16.15 MB hand-back bundle): Intact.
- `models_backup/`: Timestamped pre-training backups (`20261120_104925/`, `20261120_112240/`) and baseline models intact.
- `models/`: Active production models (`stage0_model.keras`, `stage1_model.keras`) and class mappings intact.
- `data/`: Full 22,867 ingested dataset intact.
- `scripts/` and `src/`: Protected code untouched.
- `git status`: Zero staged commits, zero git pushes, branch `training-run-1`.
