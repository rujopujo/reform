# ============================ CONFIG (the only cell to edit) ============================
STAGES              = [0]         # 0 = material sorter only (preserves Stage 1 model)
DRY_RUN             = True        # True: timing run, saves to models/_dryrun, never overwrites real models
CONFIRM_OVERWRITE   = False       # a human must set this True before a real run may replace models/
EPOCHS_HEAD         = 15          # phase 1 (top layer only); same default as train_models.py
EPOCHS_FINE         = 10          # phase 2 (last 30 layers);  same default as train_models.py
BATCH_SIZE          = None        # None = automatic (64 on GPU, 32 on CPU)
MIXED_PRECISION     = "auto"      # "auto" (GPU with tensor cores only), True or False
CACHE               = "auto"      # "auto", "memory", "disk" or "none"
MIN_TRAIN_PER_CLASS = 100         # the data check warns below this
WEIGHTS             = "imagenet"  # leave as is
SEED                = 42
# Fixed to match scripts/train_models.py. Do not change:
IMG_SIZE, LR_HEAD, LR_FINE, DROPOUT, FINE_TUNE_LAYERS = 224, 1e-3, 1e-5, 0.4, 30

# Stage 0 Tuning Parameters:
CLASS_WEIGHT_MODE       = "sqrt"  # "sqrt" (smoothed 2.8x max/min ratio), "none" (unweighted), or "balanced" (severe 8x)
EARLY_STOPPING_PATIENCE = 6       # increased from 4 to 6 to prevent premature cutoff
REDUCE_LR_PATIENCE      = 3       # increased from 2 to 3 to prevent rapid learning rate decay

import os, sys, json, time, shutil, subprocess, datetime, platform
from pathlib import Path
from collections import Counter
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix
from PIL import Image

def find_root():
    p = Path.cwd().resolve()
    for q in [p, *p.parents]:
        if (q / "scripts" / "train_models.py").exists():
            return q
    raise RuntimeError("Open this notebook from inside the reform repo folder "
                       "(could not find scripts/train_models.py above the current folder).")
ROOT = find_root(); os.chdir(ROOT)
DATA, MODELS = ROOT / "data", ROOT / "models"

import keras as _keras_pkg
KERAS_V = _keras_pkg.__version__
if int(KERAS_V.split(".")[0]) >= 3:
    raise RuntimeError(f"Keras {KERAS_V} found. This notebook and the repo need TensorFlow 2.15 "
                       "(Keras 2). Install: pip install 'tensorflow-cpu>=2.15,<2.16'. If you changed TensorFlow on "
                       "purpose to use a GPU, STOP and ask the owner.")

gpus = tf.config.list_physical_devices("GPU")
for g in gpus:
    tf.config.experimental.set_memory_growth(g, True)
cc = None
if gpus:
    try:
        cc = tf.config.experimental.get_device_details(gpus[0]).get("compute_capability")
    except Exception:
        cc = None

def total_ram_gb():
    try:
        import psutil; return psutil.virtual_memory().total / 1e9
    except Exception:
        pass
    try:
        if os.name == "nt":
            import ctypes
            class MS(ctypes.Structure):
                _fields_ = [("l", ctypes.c_ulong), ("m", ctypes.c_ulong), ("tp", ctypes.c_ulonglong),
                            ("ap", ctypes.c_ulonglong), ("tpf", ctypes.c_ulonglong), ("apf", ctypes.c_ulonglong),
                            ("tv", ctypes.c_ulonglong), ("av", ctypes.c_ulonglong), ("ae", ctypes.c_ulonglong)]
            s = MS(); s.l = ctypes.sizeof(MS); ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(s))
            return s.tp / 1e9
        with open("/proc/meminfo") as f:
            return int(f.readline().split()[1]) / 1e6
    except Exception:
        return None
RAM_GB = total_ram_gb()

ON_GPU = bool(gpus)
BATCH = BATCH_SIZE or (64 if ON_GPU else 32)
if MIXED_PRECISION == "auto":
    USE_MIXED = bool(ON_GPU and cc and cc[0] >= 7)
else:
    USE_MIXED = bool(MIXED_PRECISION)
AUTOTUNE = tf.data.AUTOTUNE
tf.keras.utils.set_random_seed(SEED)

print(f"Python {platform.python_version()} | TensorFlow {tf.__version__} | Keras {KERAS_V}")
print(f"Repo root : {ROOT}")
print(f"GPU       : {[g.name for g in gpus] or 'NONE'}  compute capability: {cc}")
print(f"CPU cores : {os.cpu_count()} | RAM: {RAM_GB:.0f} GB" if RAM_GB else f"CPU cores : {os.cpu_count()}")
print(f"Batch size: {BATCH} | mixed precision: {USE_MIXED} | cache: {CACHE}")
print(f"Mode      : {'DRY RUN (saves to models/_dryrun)' if DRY_RUN else 'REAL RUN (replaces models/)'}")
if not ON_GPU:
    print("\n" + "!" * 70)
    print("WARNING: TensorFlow sees NO GPU, so training runs on the CPU (several hours).")
    print("Common reasons: the repo installs 'tensorflow-cpu'; native Windows has no GPU support")
    print("after TensorFlow 2.10; RTX 50-series cards need newer CUDA/TensorFlow than 2.15.")
    print("STOP and tell the owner before a long CPU run. See the workflow document, Part 2.6.")
    print("!" * 70)

checker = ROOT / "scripts" / "check_data.py"
if not checker.exists():
    raise FileNotFoundError("scripts/check_data.py is missing. Copy the helper scripts into scripts/ first.")
r = subprocess.run([sys.executable, str(checker), "--min-train", str(MIN_TRAIN_PER_CLASS)],
                   capture_output=True, text=True)
print(r.stdout[-6000:], r.stderr[-2000:])
if r.returncode != 0:
    raise RuntimeError("Data check failed. Fix the errors listed above (see workflow Part 5/6) before training.")
if not DRY_RUN and not CONFIRM_OVERWRITE:
    raise RuntimeError("STOP: a real run replaces models/stage0_model.keras and stage1_model.keras. "
                       "Get the owner's approval, then set CONFIRM_OVERWRITE = True in CONFIG.")

OUT = MODELS / "_dryrun" if DRY_RUN else MODELS
OUT.mkdir(parents=True, exist_ok=True)
if not DRY_RUN:
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    bdir = ROOT / "models_backup" / stamp
    bdir.mkdir(parents=True, exist_ok=True)
    n = 0
    for f in MODELS.iterdir():
        if f.is_file():
            shutil.copy2(f, bdir / f.name); n += 1
    print(f"Backed up {n} files from models/ to {bdir}")
else:
    print(f"Dry run: outputs go to {OUT}. The real models are not touched.")
CACHE_DIR = ROOT / "temp_datasets" / "tf_cache"
shutil.rmtree(CACHE_DIR, ignore_errors=True)   # never reuse a cache from an older dataset
CACHE_DIR.mkdir(parents=True, exist_ok=True)

aug_layers = keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(20 / 360, fill_mode="nearest"),
    layers.RandomTranslation(0.1, 0.1, fill_mode="nearest"),
    layers.RandomZoom(0.15, fill_mode="nearest"),
], name="augmentation")

def augment(x, y):
    x = tf.cast(x, tf.float32)
    x = aug_layers(x, training=True)
    b = tf.random.uniform([tf.shape(x)[0], 1, 1, 1], 0.7, 1.3)          # brightness 0.7 to 1.3
    return tf.clip_by_value(x * b, 0.0, 255.0), y

def prep(x, y):
    return preprocess_input(tf.cast(x, tf.float32)), y

def load_split(path, class_names=None, shuffle=False):
    # Unbatched, one-hot labels, bilinear resize (same as the app). shuffle=True shuffles the file list once.
    return keras.utils.image_dataset_from_directory(
        str(path), labels="inferred", label_mode="categorical", class_names=class_names,
        color_mode="rgb", batch_size=None, image_size=(IMG_SIZE, IMG_SIZE),
        shuffle=shuffle, seed=SEED, interpolation="bilinear")

def cached(ds, n, tag):
    ds = ds.map(lambda x, y: (tf.cast(tf.round(x), tf.uint8), y), num_parallel_calls=AUTOTUNE)
    need_gb = n * IMG_SIZE * IMG_SIZE * 3 / 1e9
    mode = CACHE
    if mode == "auto":
        mode = "memory" if need_gb < (0.25 * RAM_GB if RAM_GB else 3.0) else "disk"
    if mode == "memory":
        return ds.cache(), mode
    if mode == "disk":
        return ds.cache(str(CACHE_DIR / tag)), mode
    return ds, "none"

def build_train(raw, n, tag):
    ds, mode = cached(raw, n, tag)
    ds = ds.shuffle(min(n, 4096), seed=SEED, reshuffle_each_iteration=True)
    ds = ds.batch(BATCH).map(augment, num_parallel_calls=AUTOTUNE).map(prep, num_parallel_calls=AUTOTUNE)
    opts = tf.data.Options(); opts.deterministic = False
    return ds.with_options(opts).prefetch(AUTOTUNE), mode

def build_eval(raw, n, tag):
    ds, mode = cached(raw, n, tag)
    return ds.batch(BATCH).map(prep, num_parallel_calls=AUTOTUNE).prefetch(AUTOTUNE), mode, ds

def save_aug_preview(uint8_ds, path, n=9):
    xs = [x for x, _ in uint8_ds.take(n)]
    x = tf.stack(xs); y = tf.zeros([n, 1])
    a, _ = augment(x, y)
    a = a.numpy().astype("uint8")
    side = int(np.ceil(np.sqrt(n)))
    sheet = Image.new("RGB", (side * IMG_SIZE, side * IMG_SIZE), "white")
    for i, im in enumerate(a):
        sheet.paste(Image.fromarray(im), ((i % side) * IMG_SIZE, (i // side) * IMG_SIZE))
    sheet.save(path, quality=88)

def build_model(num_classes, weights=WEIGHTS):
    inp = keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3), name="input_image")
    base = MobileNetV2(include_top=False, weights=weights, input_tensor=inp)
    base.trainable = False
    x = layers.GlobalAveragePooling2D(name="gap")(base.output)
    x = layers.Dropout(DROPOUT, name="dropout")(x)
    out = layers.Dense(num_classes, activation="softmax", name="predictions", dtype="float32")(x)
    model = Model(inp, out, name="reform_mobilenetv2")
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=LR_HEAD),
                  loss="categorical_crossentropy", metrics=["accuracy"])
    return model, base

def enable_fine_tuning(model, base):
    base.trainable = True
    for layer in base.layers[:-FINE_TUNE_LAYERS]:
        layer.trainable = False
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=LR_FINE),
                  loss="categorical_crossentropy", metrics=["accuracy"])

def save_as_float32(model, num_classes, path):
    policy = keras.mixed_precision.global_policy().name
    if policy == "float32":
        model.save(str(path)); return
    keras.mixed_precision.set_global_policy("float32")
    clean, _ = build_model(num_classes, weights=None)
    clean.set_weights(model.get_weights())
    keras.mixed_precision.set_global_policy(policy)
    clean.save(str(path))

class EpochTimes(keras.callbacks.Callback):
    def on_train_begin(self, logs=None): self.times = []
    def on_epoch_begin(self, epoch, logs=None): self._t = time.time()
    def on_epoch_end(self, epoch, logs=None): self.times.append(time.time() - self._t)

def make_callbacks():
    timer = EpochTimes()
    p_es = globals().get("EARLY_STOPPING_PATIENCE", 4)
    p_lr = globals().get("REDUCE_LR_PATIENCE", 2)
    return timer, [timer,
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=p_es, restore_best_weights=True, verbose=1),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=p_lr, min_lr=1e-7, verbose=1)]

if USE_MIXED:
    keras.mixed_precision.set_global_policy("mixed_float16")
EP_H, EP_F = (2, 1) if DRY_RUN else (EPOCHS_HEAD, EPOCHS_FINE)
RESULTS = {}

def train_stage(stage):
    stage_dir = DATA / f"stage{stage}"
    print("\n" + "=" * 72 + f"\nSTAGE {stage}  ({stage_dir})\n" + "=" * 72)
    t0 = time.time()
    train_raw = load_split(stage_dir / "train", shuffle=True)
    class_names = train_raw.class_names
    val_raw  = load_split(stage_dir / "val",  class_names=class_names)
    test_raw = load_split(stage_dir / "test", class_names=class_names)
    train_paths, val_paths, test_paths = train_raw.file_paths, val_raw.file_paths, test_raw.file_paths
    n_tr, n_va, n_te = len(train_paths), len(val_paths), len(test_paths)
    counts = Counter(Path(p).parent.name for p in train_paths)
    print(f"Classes ({len(class_names)}): {class_names}")
    print(f"Images: train={n_tr} val={n_va} test={n_te} | train per class: {dict(counts)}")

    y = np.array([class_names.index(Path(p).parent.name) for p in train_paths])
    mode = globals().get("CLASS_WEIGHT_MODE", "balanced")
    if mode == "none" or mode is None:
        class_weights = None
        print("Class weights: None (unweighted cross-entropy)")
    elif mode == "sqrt":
        w = compute_class_weight("balanced", classes=np.arange(len(class_names)), y=y)
        w = np.sqrt(w)
        w = w / np.mean(w)
        class_weights = dict(enumerate(w))
        print("Class weights (normalized sqrt):", {class_names[k]: round(float(v), 2) for k, v in class_weights.items()})
    else:
        w = compute_class_weight("balanced", classes=np.arange(len(class_names)), y=y)
        class_weights = dict(enumerate(w))
        print("Class weights (balanced):", {class_names[k]: round(float(v), 2) for k, v in class_weights.items()})

    tag = f"s{stage}"
    train_ds, m1 = build_train(train_raw, n_tr, tag + "_train")
    val_ds, m2, _ = build_eval(val_raw, n_va, tag + "_val")
    test_ds, m3, _ = build_eval(test_raw, n_te, tag + "_test")
    print(f"Cache mode: train={m1} val={m2} test={m3}")

    _, yb = next(iter(val_ds))
    got = [class_names[i] for i in np.argmax(yb.numpy(), 1)]
    want = [Path(p).parent.name for p in val_paths[:len(got)]]
    assert got == want, "Label/file mismatch in the input pipeline. STOP and report this."
    save_aug_preview(cached(train_raw, n_tr, tag + "_prev")[0], OUT / f"aug_preview_stage{stage}.jpg")
    print(f"Saved an augmentation preview: {OUT / f'aug_preview_stage{stage}.jpg'}  (look at it once)")

    model, base = build_model(len(class_names))
    th, cb = make_callbacks()
    print(f"\n[PHASE 1] top layer, up to {EP_H} epoch(s)")
    model.fit(train_ds, epochs=EP_H, validation_data=val_ds, class_weight=class_weights, callbacks=cb, verbose=2)
    head_times = th.times
    fine_times = []
    if EP_F > 0:
        enable_fine_tuning(model, base)
        tf_, cb = make_callbacks()
        print(f"\n[PHASE 2] fine-tune last {FINE_TUNE_LAYERS} layers, up to {EP_F} epoch(s)")
        model.fit(train_ds, epochs=EP_F, validation_data=val_ds, class_weight=class_weights, callbacks=cb, verbose=2)
        fine_times = tf_.times
    val_loss, val_acc = model.evaluate(val_ds, verbose=0)
    print(f"\n[VALIDATION] accuracy {val_acc*100:.2f}%  loss {val_loss:.4f}")

    mpath, cpath = OUT / f"stage{stage}_model.keras", OUT / f"stage{stage}_classes.json"
    save_as_float32(model, len(class_names), mpath)
    cpath.write_text(json.dumps(class_names, indent=2), encoding="utf-8")
    print(f"Saved {mpath} ({mpath.stat().st_size/1e6:.1f} MB) and {cpath.name}")
    RESULTS[stage] = dict(class_names=class_names, n_train=n_tr, n_val=n_va, n_test=n_te, val_acc=float(val_acc),
                          head_times=head_times, fine_times=fine_times, minutes=(time.time() - t0) / 60,
                          test_ds=test_ds, model_path=mpath)
    keras.backend.clear_session()

for s in STAGES:
    train_stage(s)

keras.mixed_precision.set_global_policy("float32")
SUMMARY = {}
for s in STAGES:
    R = RESULTS[s]; names = R["class_names"]
    model = keras.models.load_model(str(R["model_path"]), compile=False)
    y_true = np.concatenate([np.argmax(y, 1) for _, y in R["test_ds"]])
    probs = model.predict(R["test_ds"], verbose=0)
    y_pred = probs.argmax(1)
    acc = float((y_pred == y_true).mean())
    cm = confusion_matrix(y_true, y_pred, labels=range(len(names)))
    pairs = sorted(((cm[i, j], names[i], names[j]) for i in range(len(names)) for j in range(len(names))
                    if i != j and cm[i, j] > 0), reverse=True)[:5]
    lines = [f"STAGE {s} TEST RESULTS ({len(y_true)} images)", f"Accuracy: {acc*100:.2f}%", "",
             classification_report(y_true, y_pred, labels=range(len(names)), target_names=names, zero_division=0),
             "Confusion matrix (rows = true, columns = predicted):", "classes: " + ", ".join(names)]
    lines += [" ".join(f"{v:5d}" for v in row) + f"   <- {names[i]}" for i, row in enumerate(cm)]
    lines += ["", "Most common mistakes (count: true -> predicted):"] + \
             ([f"  {n}: {t} -> {p}" for n, t, p in pairs] or ["  none"])
    text = "\n".join(lines)
    print(text + "\n")
    (OUT / f"stage{s}_test_report.txt").write_text(text, encoding="utf-8")
    SUMMARY[s] = dict(test_accuracy=acc, val_accuracy=R["val_acc"], classes=names, n_train=R["n_train"],
                      n_val=R["n_val"], n_test=R["n_test"], minutes=round(R["minutes"], 1))
(OUT / "training_summary.json").write_text(json.dumps(
    dict(dry_run=DRY_RUN, tensorflow=tf.__version__, gpu=[g.name for g in gpus], mixed_precision=USE_MIXED,
         batch_size=BATCH, stages=SUMMARY), indent=2), encoding="utf-8")
print("Saved training_summary.json and test reports in", OUT)

print(f"{'DRY RUN' if DRY_RUN else 'REAL RUN'} finished. Files are in {OUT}\n")
for s in STAGES:
    S_ = SUMMARY[s]
    print(f"Stage {s}: val acc {S_['val_accuracy']*100:.1f}% | test acc {S_['test_accuracy']*100:.1f}% | "
          f"{S_['n_train']} train images | {S_['minutes']:.1f} min")
if DRY_RUN:
    total = 0.0
    for s in STAGES:
        R = RESULTS[s]
        warm_head = R["head_times"][-1] if R["head_times"] else 0
        fine = np.mean(R["fine_times"]) if R["fine_times"] else warm_head
        est = (EPOCHS_HEAD * warm_head + EPOCHS_FINE * fine) / 60
        total += est
        print(f"Stage {s}: epoch times head={['%.0fs' % t for t in R['head_times']]} fine={['%.0fs' % t for t in R['fine_times']]}"
              f"  ->  full run at most about {est:.0f} min (early stopping often ends sooner)")
    print(f"\nESTIMATED FULL RUN, all stages: at most about {total:.0f} min ({total/60:.1f} h), plus about 1 min of loading per stage.")
    print("\nSTOP: show this estimate to the user. If approved, set DRY_RUN = False and CONFIRM_OVERWRITE = True "
          "in CONFIG and run the notebook again from the top.")
