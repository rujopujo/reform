# ReForm Training Workflow

Step-by-step guide for training the ReForm e-waste classifier with Google Antigravity. Written so that a person or an AI agent can follow it without prior knowledge of the project. Version of 19 Nov 2026. Repo: https://github.com/rujopujo/reform


## How to use this document

- **With the Antigravity agent:** put this file (the PDF, or the .md copy, which agents read more reliably) in the project folder, open the folder in Antigravity, and in the agent chat attach it with `@` (or give its path) and paste the **Agent brief** below.
- **By hand:** follow Parts 2 to 10 in order and run every command in a terminal opened in the repo folder.
- Every step ends with a **CHECK**. Do not move on until the check passes.
- A **STOP** means: pause and ask the project owner (the person who gave you this) before continuing.
- The four helper scripts mentioned here (`prep_images.py`, `check_data.py`, `eval_test.py`, `crop_yolo.py`) come in a separate zip file. They were tested on sample data with the repo's own `train_models.py`.


### Agent brief (paste this into the agent)

```
You are helping me train the ReForm e-waste image classifier. Follow the attached
workflow document step by step, in order. Rules:
1. Work on a new git branch. Never push. Never commit images, temp_datasets/, models_backup/
   or any token or API key.
2. Do not edit existing files in scripts/ or src/. You may ADD the four helper scripts
   from the zip (prep_images.py, check_data.py, eval_test.py, crop_yolo.py) to scripts/.
3. Do NOT run scripts/download_datasets.py. It contains a hard-coded Kaggle key.
4. After every step, run its CHECK and show me the output. If a check fails, fix it or
   ask me. Do not skip ahead.
5. Stop and ask me at every STOP.
6. Ask before deleting anything. Back up models/ before training.
7. Keep a log in training_notes.md: commands run, image counts, problems, final metrics.
```


## Part 0. What you are doing

ReForm identifies e-waste from a photo in two steps. **Stage 0** sorts the material (plastic, metal, PCB, glass_screen, battery, unknown). **Stage 1** names the plastic item (keyboard, mouse, remote, printer_casing, tv_panel, other_plastic, unknown). Each stage is a separate image classifier (MobileNetV2). You will add more training images, retrain both stages, test them and hand the new model files back to the project owner.

| Thing | Detail |
|---|---|
| Final output | `models/stage0_model.keras`, `models/stage1_model.keras` and their `*_classes.json` files, plus test reports and a log. |
| How training works | `scripts/train_models.py` starts from ImageNet weights every time (no resume). Images are resized to 224 x 224. Phase 1 trains only the new top layer for up to 15 epochs, Phase 2 fine-tunes the last 30 layers for up to 10 epochs. Training stops early if validation loss stops improving for 4 epochs. Class weights are balanced automatically. |
| Data layout | `data/stage0/{train,val,test}/<class>/` and `data/stage1/{train,val,test}/<class>/`, one folder per class. Class names are case-sensitive (`PCB` is upper-case, all others lower-case). |
| Existing data | The repo already contains thousands of images (see Part 3). You are adding to them, not starting from scratch. |
| First run needs internet | TensorFlow downloads the small ImageNet weight file the first time. It is cached afterwards. |
| Approximate time | Setup 30 to 60 min. Downloads: depends on connection. Cleaning and staging 10 to 30 min. Training: 15 to 45 min with a working GPU, or several hours on CPU only (Part 2.6 explains how to measure it). Testing 5 min. |


## Part 1. Ground rules

| Do | Do not |
|---|---|
| Work on a new branch: `git checkout -b training-run-1`. | Do not push, and do not run `git add .` (new images and datasets would be added). |
| Add the new helper scripts to `scripts/`. | Do not edit `train_models.py`, `add_images.py` or anything in `src/`. If a change is truly needed, STOP and ask. |
| Use your own Kaggle and Roboflow tokens, stored in `~/.kaggle/kaggle.json` and an environment variable. | Never paste a token into a file, a command that gets saved, or a commit. Never run `scripts/download_datasets.py`: it has someone else's Kaggle key written into it. |
| Train on **raw** images only. | Do not use folders named *Augmented*. Copies of one photo can land in both train and test and fake a high accuracy. |
| Look at sample images from every class you add (Part 5). | Do not add a class of images you have not looked at. |
| Back up `models/` before training. | Do not delete anything without asking. |


## Part 2. Set up the computer


### 2.1 Check the tools

```
git --version
python --version
```

> **CHECK.** Git prints a version. Python must be **3.9, 3.10 or 3.11** (3.11 preferred). TensorFlow 2.15, which the repo pins, does not install on Python 3.12 or newer. If Python is 3.12+, STOP and ask the owner, or install Python 3.11 from python.org and use it for the next steps (on Windows: `py -3.11 -m venv .venv`).


### 2.2 Get the code and make a branch

```
git clone https://github.com/rujopujo/reform.git
cd reform
git checkout -b training-run-1
```

> **CHECK.** `git branch` shows `* training-run-1`. The folder has `scripts/`, `src/`, `data/`, `models/`.


### 2.3 Create the Python environment

```
python -m venv .venv
# Windows PowerShell:   .venv\Scripts\Activate.ps1
# Windows cmd:          .venv\Scripts\activate.bat
# Mac / Linux:          source .venv/bin/activate
python -m pip install --upgrade pip
pip install "tensorflow-cpu>=2.15.0,<2.16.0" "numpy>=1.23.5,<2.0.0" pillow scikit-learn pyyaml
pip install kaggle roboflow
```

> **NOTE.** This is the minimum for **training**. It deliberately skips the rest of `requirements.txt` (EasyOCR pulls in a very large PyTorch download that training does not need). Note that `requirements.txt` does not list scikit-learn, but `train_models.py` imports it, so install it as shown. The full `requirements.txt` is installed later in Part 9, only for trying the app.

> **CHECK.** This prints `2.15.x` and no error: `python -c "import tensorflow as tf, sklearn, PIL; print(tf.__version__)"`


### 2.4 Add the helper scripts

Unzip `reform_helper_scripts.zip` and copy `prep_images.py`, `check_data.py`, `eval_test.py` and `crop_yolo.py` into the repo's `scripts/` folder. These are new files; nothing existing is changed. Then create the working folders:

```
mkdir temp_datasets
mkdir models_backup
```

> **CHECK.** `python scripts/check_data.py --help` prints usage text. (`temp_datasets/` is already git-ignored. `models_backup/` is not, so never run `git add .`: see Part 10.)


### 2.5 Back up the current models

```
# Windows PowerShell:  Copy-Item models\* models_backup\
# Mac / Linux:         cp models/* models_backup/
```

> **CHECK.** `models_backup/` contains `stage0_model.keras`, `stage1_model.keras` and the `*_classes.json` files.


### 2.6 Find out whether training will use the GPU

```
python -c "import tensorflow as tf; print(tf.__version__, tf.config.list_physical_devices('GPU'))"
```

If it prints a GPU in the list, training will be fast. If it prints `[]`, training runs on the CPU, which is much slower. This is common, for these reasons:

- The repo installs `tensorflow-cpu`, which never uses a GPU.
- Per TensorFlow's own install guide, native Windows has not supported GPU since TensorFlow 2.10. Newer TensorFlow needs Linux or WSL2.
- NVIDIA RTX 50-series cards (such as the 5070 Ti) are newer than TensorFlow 2.15. Community reports show they need newer CUDA and TensorFlow builds or workarounds. No tested recipe is included here.

> **STOP.** If the list is empty, STOP and tell the owner. The choices are: (a) run the whole workflow on CPU and leave it running for several hours or overnight; (b) spend up to 30 minutes trying to get GPU working (WSL2 with a current TensorFlow GPU install, per TensorFlow's current docs). Changing the TensorFlow version can break `train_models.py`, which uses the older Keras image API, so if you change it you MUST re-run the dry run in Part 7.1 and tell the owner exactly what you changed; (c) use Google Colab (untested here: it needs a TensorFlow version compatible with this script).


### 2.7 Baseline: check the data that is already in the repo

```
python scripts/check_data.py
```

This prints one table per stage with the number of images per class in train, val and test, and flags problems: empty class folders, images that appear in two splits (leakage), and images filed under two classes.

> **CHECK.** Save the printed tables into `training_notes.md` as the **before** counts. Warnings (such as class imbalance) are normal. If it prints **ERROR** lines, the problem is already in the repo: STOP and report the exact output to the owner. Do not fix it silently.


## Part 3. Decide what to add (before downloading anything)


### 3.1 What is already there

| Stage | Class | Images now (train / all splits) |
|---|---|---|
| 0 | PCB | 602 / 850 |
| 0 | battery | 548 / 758 |
| 0 | glass_screen | 503 / 700 |
| 0 | metal | 553 / 799 |
| 0 | plastic | 3,146 / 4,539 |
| 0 | unknown | 1,399 / 1,997 |
| 1 | keyboard | 311 / 459 |
| 1 | mouse | 331 / 449 |
| 1 | remote | 298 / 451 |
| 1 | printer_casing | 315 / 450 |
| 1 | tv_panel | 322 / 454 |
| 1 | other_plastic | 1,219 / 1,776 |
| 1 | unknown | 2,555 / 3,604 |

Counts were read from the repo on 19 Nov 2026 and may differ slightly from your baseline. Use your own baseline from Part 2.7 as the truth.


### 3.2 The principle

- **Add images to the small classes** (keyboard, mouse, remote, printer_casing, tv_panel, PCB, battery, glass_screen, metal). Do **not** add more to plastic, other_plastic or unknown: they are already large, and a class that is 10 times bigger than another biases the model.
- Aim for roughly 1,000 images per small class in total (existing plus new). A few hundred clean images beat thousands of mislabelled ones.
- A class that does not look like its label is worse than a missing class. If you are unsure about a source folder, leave it out.


### 3.3 Where each class can come from

| ReForm class | Source (folder name inside the dataset) | Suggested --max |
|---|---|---|
| Stage 1 keyboard | Electronic Accessories: Keyboard; EWasteNet: Keyboards | 600 |
| Stage 1 mouse | Electronic Accessories: Mouse; EWasteNet: Mouses | 600 |
| Stage 1 tv_panel | EWasteNet: TV (about 162 images). Electronic Accessories: Monitor is optional: only if the owner agrees monitors count as tv_panel. | 600 |
| Stage 1 remote | No ready-made classification set. Crop from the Roboflow detection sets: TV Remote (tahsin), remote-controll-classifier, Remote Control Detection (Part 4.5). | 600 |
| Stage 1 printer_casing | Crop from Roboflow: Office Printer (2-olter), printer (Part 4.5). | 600 |
| Stage 0 PCB | Laptop Components: Motherboard (140 raw images). | 500 |
| Stage 0 battery | Laptop Components: Battery, CMOSBattery. Garbage Classification: battery. | 500 |
| Stage 0 glass_screen | Laptop Components: LCDScreen (140 raw). Check they look like the existing glass_screen images. | 500 |
| Stage 0 metal | Garbage Classification: metal. | 500 |


### 3.4 Optional: brand-new classes

Possible new Stage 1 classes: `charger_casing` (Electronic Accessories: Charger), `laptop`, `phone`, `headset`. A new class needs at least a few hundred clean images, goes through the same steps, and uses `--allow-new-class` when ingesting (Part 5). The app looks items up in `data/knowledge_base.json`. If a class has no entry there, `src/server.py` falls back to the `other_plastic` entry, so the app does not crash, but the advice shown will be generic. `charger_casing` already has an entry.


### 3.5 What not to use

- Any folder named *Augmented* (the two Mendeley datasets contain them).
- Garbage Classification folders `brown-glass`, `green-glass`, `white-glass` (bottles, not screens) and `plastic` (bottles and packaging, not device casings).
- The Datacluster mobile phone set on Hugging Face: its license bars training commercial models.

> **STOP.** Show the owner your class plan (which classes, which sources, how many images) and wait for a yes before downloading.


## Part 4. Download the datasets

Everything goes under `temp_datasets/`. After each download, run the inspect command. It prints every folder with its image count and warns if the data has annotation files (a sign of an object-detection set that must be cropped first).


### 4.1 Electronic Accessories Classification Dataset (Mendeley)

1. Open https://data.mendeley.com/datasets/9njzry9f6m in a browser and click **Download All**.
2. Unzip into `temp_datasets/accessories`. If Windows complains about long file names, unzip to a short path such as `C:\z` and move the folder.

```
python scripts/check_data.py --inspect temp_datasets/accessories --depth 3
```

> **CHECK.** Expected: 14,027 images in total, in 10 class folders (Charger, Game Controller, Headphone, Keyboard, Laptop, Monitor, Mouse, Smartphone, Smartwatch, Speaker), split between raw and augmented (about 4,027 raw and 10,000 augmented). Write down the exact path of the raw folder, for example `temp_datasets/accessories/<RAW>/Keyboard`. If the numbers are very different, STOP and ask.


### 4.2 Laptop Components (Mendeley)

Open https://data.mendeley.com/datasets/pwctygncrj, click **Download All**, unzip into `temp_datasets/laptop_components`.

```
python scripts/check_data.py --inspect temp_datasets/laptop_components --depth 3
```

> **CHECK.** Expected: folders `Raw Data` and `Augmented Data`, each with 26 class folders. `Raw Data` has 140 images per class (3,640 in total). Only `Raw Data` is used.


### 4.3 EWasteNet

```
cd temp_datasets
git clone https://github.com/NifulIslam/EWasteNet-A-Two-Stream-DeiT-Approach-for-E-Waste-Classification ewastenet
cd ..
python scripts/check_data.py --inspect temp_datasets/ewastenet --depth 3
```

> **CHECK.** Expected: about 1,053 images in 8 classes (Mobile, Laptop, Keyboards, Mouses, Microwave, TV, Smartwatch, Camera). If the repo has no images (only code), or the license is unclear, STOP and tell the owner.


### 4.4 Garbage Classification (Kaggle) for metal and battery

Sign in to Kaggle with your own account. With the current Kaggle command line tool (version 2.x), the easiest way is `kaggle auth login`, which opens a browser. Alternatively create a token at https://www.kaggle.com/settings/api and either set the environment variable `KAGGLE_API_TOKEN` or save the token in `~/.kaggle/access_token`. Never save a token inside the repo.

```
kaggle auth login
kaggle datasets download mostafaabla/garbage-classification -p temp_datasets/garbage --unzip
python scripts/check_data.py --inspect temp_datasets/garbage --depth 3
```

> **CHECK.** Expected: about 15,515 images in 12 class folders including `battery` and `metal`. Only those two (and optionally non-e-waste classes for `unknown`) are used. Authentication errors mean you are not signed in: see Part 11. (Older 1.x versions of the Kaggle tool use `-d mostafaabla/garbage-classification` instead of the bare name.)


### 4.5 Roboflow detection sets (remote and printer), then crop

These datasets are object detection sets: each image has boxes drawn around the objects. They must be cropped into one image per object before use.

1. Create a free Roboflow account and copy your API key from its settings. Save it in an environment variable, never in a file. Windows PowerShell: `$env:ROBOFLOW_API_KEY="your-key"`. Mac/Linux: `export ROBOFLOW_API_KEY=your-key`.
2. On each dataset's Roboflow page note the workspace and project (from the URL: `universe.roboflow.com/<workspace>/<project>`) and the version number (Versions tab). Prefer a version **without** augmentation.
3. Download in YOLOv8 format with this snippet (change the three names and version; or use the website's Download Dataset button, choose YOLOv8, unzip):

```
import os
from roboflow import Roboflow
rf = Roboflow(api_key=os.environ["ROBOFLOW_API_KEY"])
rf.workspace("tahsin").project("tv-remote").version(1).download(
    "yolov8", location="temp_datasets/rf_tv_remote")
```

Sets to fetch: `tahsin/tv-remote` (410 images, remotes), `2-olter/office-printer-bbogo` (2,004 images, printers). Optional: `camera-ysgam/remote-control-detection`, `image-processing-fsllp/remote-controll-classifier` (a detection set despite its name, 405 images), `printer/printer-ler5v`.

Then crop the boxes into one folder per class:

```
python scripts/crop_yolo.py temp_datasets/rf_tv_remote crops/rf_tv_remote
python scripts/check_data.py --inspect crops/rf_tv_remote --depth 2
```

> **CHECK.** `crops/rf_tv_remote/` has one folder per class name from the dataset's `data.yaml` (for example `remote`) with hundreds of cropped images. If a dataset has several classes, only use the folders that match what you want. Skip any box class that merges several devices.


## Part 5. Clean, stage and add the images to the dataset

Each class goes through two commands. **stage** gathers images from one or more source folders into a single clean folder: it skips unreadable images, images under 96 px, augmented folders and exact duplicates, and keeps at most `--max` random images. **ingest** splits that folder 70 / 15 / 15 into `data/stageN/{train,val,test}/<class>` and skips anything already in the dataset. Both are safe to re-run.


### 5.1 The recipe for one class (example: keyboard)

```
python scripts/prep_images.py stage --out temp_datasets/stg/keyboard --max 600 ^
    "temp_datasets/accessories/<RAW>/Keyboard" "temp_datasets/ewastenet/<FOLDER>/Keyboards"
python scripts/prep_images.py sheet --source temp_datasets/stg/keyboard --out sheets/keyboard.jpg
```

> **NOTE.** The `^` at the end of a line is the Windows cmd line-continuation. In PowerShell use a backtick instead, in Mac/Linux use `\`, or put the whole command on one line. Replace the `<...>` parts with the real paths you wrote down in Part 4.

Open `sheets/keyboard.jpg`. It is a grid of 25 random staged images.

> **STOP.** Look at every contact sheet. If more than about 1 in 10 pictures does not show the class (wrong object, a person, a diagram, a box only), do NOT ingest it. Tell the owner which source folder is bad. Make a fresh sheet after fixing the sources (`--overwrite` replaces the staging folder).

If the sheet looks right, add the images to the dataset. For Stage 1 plastic items use `--mirror-plastic` so the same images also teach Stage 0 what plastic looks like. To preview without copying anything, add `--dry-run` to the ingest command first.

```
python scripts/prep_images.py ingest --stage 1 --class keyboard --source temp_datasets/stg/keyboard --mirror-plastic
python scripts/check_data.py
```

> **CHECK.** The ingest prints the plan (for example `train=420 val=90 test=90`). The real run prints how many files were copied. `check_data.py` shows the bigger keyboard count and ends with `0 error(s)`. Class name typos such as `Keyboard` are refused automatically.


### 5.2 The same recipe for every class in your plan

| Class | stage and ingest options | Notes |
|---|---|---|
| keyboard | `--stage 1 --mirror-plastic` | As above. |
| mouse | `--stage 1 --mirror-plastic` | Sources: Accessories Mouse, EWasteNet Mouses. |
| tv_panel | `--stage 1 --mirror-plastic` | Source: EWasteNet TV. |
| remote | `--stage 1 --mirror-plastic` | Source: crops from the Roboflow remote sets. |
| printer_casing | `--stage 1 --mirror-plastic` | Source: crops from the Roboflow printer sets. |
| PCB, battery, glass_screen, metal | `--stage 0` (no mirror) | These are not plastic items, so they only go into Stage 0. |
| new class (optional) | `--stage 1 --mirror-plastic --allow-new-class` | Only if the owner approved it in Part 3. |


### 5.3 Check the existing remote class

The repo's older import script maps the source folder named `Player` to the `remote` class. That folder may contain media or DVD players, not remote controls. Look:

```
python scripts/prep_images.py sheet --source data/stage1/train/remote --out sheets/existing_remote.jpg
```

> **STOP.** If the existing remote images do not show remote controls, report it to the owner before adding more remotes.


### 5.4 Why not just use scripts/add_images.py?

It works, but it **always** copies every image into Stage 0 plastic (its `--is-plastic` flag is on and cannot be turned off), so it cannot add PCB, battery, metal or glass, and it would wrongly add a Stage 1 `unknown` folder to plastic. `prep_images.py ingest` does the same split safely, with a choice of stage.


## Part 6. The gate before training

Run the final check and read it carefully.

```
python scripts/check_data.py --min-train 300 --verify-images
```

| Check | Pass means |
|---|---|
| Result line | `0 error(s)`. Errors are: an empty class folder (crashes training), a class missing from one split, the same image in two splits (leakage), an unreadable image. |
| Counts | Each small class is roughly 800 to 1,500 images in total. Compare with your **before** tables. |
| Imbalance warning | A warning about a 10:1 or larger gap is acceptable only for `unknown`, `plastic`, `other_plastic`. For other classes, add images to the small class. |
| Sheets | You looked at a contact sheet for every class you added. |
| Backup | `models_backup/` holds the old models (Part 2.5). |
| Disk and power | At least 5 GB free. The computer is plugged in and set not to sleep. |

Optional but useful: score the **old** models on the **new** test split now, so you can compare later. This only works if the class list has not changed (it stops with a message otherwise):

```
python scripts/eval_test.py --stage all
```

Copy the two accuracy numbers into `training_notes.md` as **old model**. Then rename the reports so they are not overwritten: `models/stage0_test_report.txt` to `models/stage0_test_report_old.txt`, and the same for stage 1.

> **STOP.** Show the owner the final check output and ask for a go-ahead to train. Training overwrites `models/stage0_model.keras` and `models/stage1_model.keras`.


## Part 7. Train

> **NOTE.** **Faster option:** the separate notebook `train.ipynb` trains the same models and writes the same files, but with a cached input pipeline, mixed precision on NVIDIA GPUs, an automatic time estimate, an automatic backup and an overwrite guard. Put it in the repo root and run it instead of 7.1 and 7.2: its dry run (`DRY_RUN = True`) replaces 7.1 and its real run replaces 7.2. Parts 8 to 10 still apply. The notebook already writes the test reports from Part 8 into `models/`.


### 7.1 Dry run (one epoch of each phase)

```
python scripts/train_models.py --stage all --epochs-head 1 --epochs-fine 1
```

This proves everything works and measures the speed. Look for these lines in the output:

- `[INFO] Running on CPU` or `Hardware accelerator detected: 1 GPU(s)`.
- `Classes (N): [...]` and `Samples: train=..., val=..., test=...` for each stage.
- `[PHASE 1 COMPLETE] Duration: X minutes` and `[PHASE 2 COMPLETE] Duration: Y minutes`.
- `[SAVED ARTIFACTS]` for both stages and finally `[DONE] All requested training stages completed successfully!`.

> **CHECK.** No error. The accuracy after one epoch can be low; that is normal. **Estimate the full run:** about `X x 15 + Y x 10` minutes per stage, usually less because training stops early. Write this estimate in `training_notes.md`. If it is more than about 4 hours in total, STOP and tell the owner before continuing.


### 7.2 The full run, with a log file

```
# Mac / Linux:
python scripts/train_models.py --stage all 2>&1 | tee training_log.txt

# Windows PowerShell:
python scripts/train_models.py --stage all 2>&1 | Tee-Object -FilePath training_log.txt
```

- Do not close the terminal or let the computer sleep. There is no resume: if it stops, start again.
- Out-of-memory error: add `--batch-size 16` (or 8).
- To train one stage only: `--stage 0` or `--stage 1`.
- Watch `val_accuracy`. It should rise over the first epochs. `EarlyStopping` messages are normal and mean training finished early because it stopped improving.
- Warning sign: `accuracy` near 99% but `val_accuracy` far lower means the model memorised the training images. Add more varied images to the small classes.

> **CHECK.** The log ends with `[DONE] All requested training stages completed successfully!`. The log also contains a `[CLASSIFICATION REPORT - VALIDATION SET]` table per stage. Save the log; it is part of the hand-back.


## Part 8. Test the new models properly

`train_models.py` reports results on the validation split, which the training process itself used to pick the best epoch. The **test** split is untouched. Score it:

```
python scripts/eval_test.py --stage all
```

It prints accuracy, a per-class table (precision, recall, f1), a confusion matrix and the five most common mistakes, and saves them in `models/stage0_test_report.txt` and `models/stage1_test_report.txt`.

| Result | What to do |
|---|---|
| Overall test accuracy below about 80%, or a class with recall below about 60% (rules of thumb, not guarantees) | Open the images of that class and of the class it is confused with. Look for wrong labels, too few images, or two classes that look alike. Fix the data (Part 5) and retrain. |
| Accuracy at or above the old model's numbers from Part 6 | Good. Move on. |
| Accuracy lower than the old model | Do not hand over the new models as an upgrade. Keep the backup in `models_backup/`, report both sets of numbers to the owner, and say what you changed. |

> **CHECK.** Both report files exist. Copy the accuracy figures and the per-class table into `training_notes.md`.


## Part 9. Try the app with the new models

Install the rest of the project's packages (this is large, mainly EasyOCR and PyTorch) and start the web app:

```
pip install -r requirements.txt
python -m uvicorn server:app --host 127.0.0.1 --port 7860
```

In a second terminal check the app loaded both models and shows the same class lists as the training output:

```
curl http://127.0.0.1:7860/api/health
```

> **NOTE.** On Windows PowerShell, `curl` is a different built-in command. Type `curl.exe` instead of `curl` for both curl commands in this part.

> **CHECK.** The reply shows `"stage0_loaded": true`, `"stage1_loaded": true` and the class lists you trained. Then open http://127.0.0.1:7860 in a browser and upload at least 5 real photos of each class you changed (take new photos with a phone; do not use images from the dataset). Note every wrong answer in `training_notes.md`. You can also test one file from the terminal: `curl -F "file=@photo.jpg" http://127.0.0.1:7860/api/step1-classify`.

Stop the server with Ctrl+C when done.


## Part 10. Hand the results back

1. Run `git status`. Expect modified `models/*.keras` and `models/*_classes.json`, and many untracked new images under `data/`. Do **not** commit the images (the repo would grow by hundreds of MB). Commit nothing unless the owner asks.
2. Collect these files in one folder named `reform_results`: `models/stage0_model.keras`, `models/stage1_model.keras`, `models/stage0_classes.json`, `models/stage1_classes.json`, `models/stage0_test_report.txt`, `models/stage1_test_report.txt`, `models/stage0_test_report_old.txt` and `stage1` equivalent if made, `training_log.txt`, `training_notes.md`.
3. Zip the folder and send it to the owner, together with the before and after image counts, the training time, and any problems. Do not send tokens or API keys.
4. Do not delete `models_backup/` until the owner confirms the new models are accepted.

> **CHECK.** The zip contains the four model files, the two reports, the log and the notes. The notes include the dataset licenses: both Mendeley sets are CC BY 4.0 (attribution required).


## Part 11. When something goes wrong

| Problem | Likely cause and fix |
|---|---|
| `ModuleNotFoundError: No module named 'sklearn'` | `pip install scikit-learn` (it is missing from requirements.txt). |
| `No matching distribution found for tensorflow-cpu` | Python is 3.12 or newer. Use Python 3.11 (Part 2.1). |
| Log says `Running on CPU` but the machine has a GPU | See Part 2.6. This is expected with `tensorflow-cpu`. |
| `GatherV2` or `indices not in [0, N)` error, or a class-weight error | An empty class folder. Run `python scripts/check_data.py`, then delete or fill the empty folder it names (ask first). |
| `Found 0 images belonging to 0 classes` | Wrong folder layout. `data/stageN/train/<class>/image.jpg` is required. Run `check_data.py`. |
| `ResourceExhaustedError` or out of memory | `--batch-size 16` or `8`. |
| `Unable to open file` / download failure for weights | No internet on the first run. Connect and retry. |
| Kaggle 401 or 403, or `Authentication required` | Run `kaggle auth login`, or set `KAGGLE_API_TOKEN` to a token from your own account. Never use the key in `download_datasets.py`. |
| `unrecognized arguments: -d` | Your Kaggle tool is the new 2.x version: give the dataset name without `-d`, as in Part 4.4. |
| Roboflow `version not found` or `project not found` | Check workspace, project and version on the dataset page. Try another version number, or use the website's Download Dataset button. |
| `cannot identify image file` | A corrupt image. The `stage` command removes these. Re-stage that class. |
| Accuracy stays near 1 divided by the number of classes | Usually wrong labels or a bad source folder. Look at contact sheets for the class and re-check Part 5. |
| `prep_images.py: not an existing Stage N class` | Typo or wrong case (`pcb` instead of `PCB`). Or add `--allow-new-class` if it is a new class on purpose. |
| Unzip fails with long path errors on Windows | Unzip into a short path such as `C:\z`, or use 7-Zip. |
| Training takes far longer than the dry-run estimate | The estimate assumes all epochs; check the log. Stop and ask the owner before changing epochs. |
| Anything else | Copy the last 30 lines of output into `training_notes.md` and ask the owner. Do not edit the repo's scripts to work around it. |


## Appendix A. Command cheat sheet

```
# look inside a download
python scripts/check_data.py --inspect <folder> --depth 3
# crop a Roboflow YOLO export
python scripts/crop_yolo.py <yolo_dir> crops/<name>
# clean + cap + dedupe one class
python scripts/prep_images.py stage --out temp_datasets/stg/<class> --max 600 <src1> <src2>
# eyeball 25 random images
python scripts/prep_images.py sheet --source <folder> --out sheets/<class>.jpg
# add to the dataset (70/15/15)
python scripts/prep_images.py ingest --stage 1 --class <class> --source temp_datasets/stg/<class> --mirror-plastic
python scripts/prep_images.py ingest --stage 0 --class PCB --source temp_datasets/stg/pcb
# validate everything
python scripts/check_data.py --min-train 300 --verify-images
# train, then score on the test split
python scripts/train_models.py --stage all
python scripts/eval_test.py --stage all
```


## Appendix B. Dataset links

| Dataset | Use | Link |
|---|---|---|
| Electronic Accessories Classification (Mendeley), 14,027 images, CC BY 4.0 | keyboard, mouse, charger, headphone, laptop, monitor, smartphone | https://data.mendeley.com/datasets/9njzry9f6m |
| Laptop Components (Mendeley), 18,200 images, CC BY 4.0 | PCB (Motherboard), battery, glass_screen (LCDScreen) | https://data.mendeley.com/datasets/pwctygncrj |
| EWasteNet dataset (authors' GitHub), 1,053 images | keyboard, mouse, tv_panel, phone, laptop | https://github.com/NifulIslam/EWasteNet-A-Two-Stream-DeiT-Approach-for-E-Waste-Classification |
| Garbage Classification (Kaggle), 15,515 images | metal, battery, unknown | https://www.kaggle.com/datasets/mostafaabla/garbage-classification |
| TV Remote (tahsin), 410 images, detection | remote (crop) | https://universe.roboflow.com/tahsin/tv-remote |
| Remote Control Detection (camera-ysgam), detection | remote (crop) | https://universe.roboflow.com/camera-ysgam/remote-control-detection |
| remote-controll-classifier, 405 images, detection | remote (crop) | https://universe.roboflow.com/image-processing-fsllp/remote-controll-classifier/dataset/2 |
| Office Printer (2-olter), 2,004 images, detection | printer_casing (crop) | https://universe.roboflow.com/2-olter/office-printer-bbogo |
| printer (printer-ler5v), CC BY 4.0, detection | printer_casing (crop) | https://universe.roboflow.com/printer/printer-ler5v |

> **NOTE.** Counts and licenses were read from each dataset's page on 19 Nov 2026. Always check the license on the page before use. The exact folder layout inside the two Mendeley downloads and EWasteNet was not seen when this was written, which is why Part 4 asks you to inspect and record the real folder names.
