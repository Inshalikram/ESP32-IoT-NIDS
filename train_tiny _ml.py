# ============================================================
#  TINYML IDS TRAINING — FINAL
#  Dataset  : IoMT 2024
#  Model    : Keras Dense → TFLite (ESP32)
#  Classes  : 0=BenignTraffic 1=DDoS 2=DoS 3=Mirai 4=Recon
#
#  Best validated result (v6, batch=256):
#    Accuracy : 95.5%   F1 Macro : 0.82
#    DoS F1   : 0.92    Mirai F1 : 0.30 → targeting 0.45+ here
#
#  Key design decisions (hard-won across 6 versions):
#    - Sample weights via tf.data  (not class_weight arg)
#    - Weighted focal loss gamma=2.5
#    - DOS_BOOST=6.0  → fixes DoS/DDoS confusion (was 45%→11%)
#    - MIRAI_BOOST=1.2 → balanced Mirai weight without FP explosion
#    - BATCH_SIZE=256  (1024 breaks LR schedule, hurts DoS recall)
#    - Warmup+cosine LR (ReduceLROnPlateau caused oscillation)
#    - EarlyStopping on val macro-F1  (val_loss dominated by DDoS)
#    - PATIENCE=8  (saves ~20 min vs 15, catches best epoch)
#    - Architecture: 26→128→64→32→16→16→5  (~60KB TFLite)
#
#  Outputs (models/):
#    tinyml_final.keras
#    tinyml_final.tflite   ← flash to ESP32
#    tinyml_final_scaler.pkl
#    tinyml_final_features.pkl
#    tinyml_final_curves.png
# ============================================================

import os
import pickle

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight

print("=" * 55)
print("  TinyML IDS — FINAL  |  IoMT 2024  |  26 Features")
print("=" * 55)

os.makedirs("models", exist_ok=True)

CLASS_NAMES = ["BenignTraffic", "DDoS", "DoS", "Mirai", "Recon"]

# ============================================================
#  TUNING KNOBS
#  Edit these between runs; everything else is fixed.
#
#  DOS_BOOST   : multiplier on balanced weight for DoS (class 2)
#    6.0 → DoS F1=0.92, DoS→DDoS confusion=11%  ✔ best so far
#    If DoS precision drops below 0.85 → lower to 5.0
#
#  MIRAI_BOOST : multiplier for Mirai (class 3)
#    0.5 → Mirai F1=0.55 but regressed when DOS_BOOST raised
#    1.2 → targeting Mirai F1≥0.45 with DOS_BOOST=6.0
#    If Mirai precision < 0.35 (too many FP) → lower to 0.8
#    If Mirai recall   < 0.35 (too many FN) → raise to 1.5
#
#  BATCH_SIZE  : MUST stay at 256
#    1024 breaks the warmup/cosine LR schedule — DoS collapses
#
#  PATIENCE    : 8 balances early stopping vs training time
#    Best epoch in v6 was epoch 29 → patience=8 catches it
# ============================================================
DOS_BOOST    = 6.0
MIRAI_BOOST  = 1.2
GAMMA        = 2.5

EPOCHS        = 60
BATCH_SIZE    = 256     # DO NOT change — see note above
WARMUP_EPOCHS = 5
LR_BASE       = 1e-3
PATIENCE      = 15

# ============================================================
#  STEP 1 — LOAD
# ============================================================
print("\n[1/7] Loading data...")

X = pd.read_csv("data/X_train.csv")
y = pd.read_csv("data/y_train.csv")["Label"]

print(f"  Raw shape   : {X.shape}")
print(f"  Label counts:\n{y.value_counts()}")

# ============================================================
#  STEP 2 — FEATURE SELECTION  (same 26 as LightGBM)
# ============================================================
print("\n[2/7] Feature selection...")

drop_cols = ["Rate", "Srate", "Tot sum"]
X = X.drop(columns=[c for c in drop_cols if c in X.columns])

try:
    selected_feats = pickle.load(open("models/selected_features.pkl", "rb"))
    available = [f for f in selected_feats if f in X.columns]
    assert len(available) > 0
    X = X[available]
    print(f"  Using selector: {len(available)} features")
except Exception:
    FEATURES_26 = [
        "AVG", "Covariance", "Duration", "Header_Length", "IAT",
        "ICMP", "Magnitue", "Max", "Min", "Number",
        "Protocol_Type", "Radius", "Std", "TCP", "Tot_size",
        "UDP", "Variance", "Weight", "ack_count", "ack_flag_number",
        "fin_count", "psh_flag_number", "rst_count", "rst_flag_number",
        "syn_count", "syn_flag_number"
    ]
    available = [f for f in FEATURES_26 if f in X.columns]
    X = X[available]
    print(f"  Using hardcoded: {len(available)} features")

print(f"  Features ({len(X.columns)}): {list(X.columns)}")
pickle.dump(list(X.columns), open("models/tinyml_final_features.pkl", "wb"))

# ============================================================
#  STEP 3 — CLEAN
# ============================================================
print("\n[3/7] Cleaning...")
X = X.astype(np.float32)
X.replace([np.inf, -np.inf], np.nan, inplace=True)
X.fillna(0, inplace=True)
print(f"  Shape: {X.shape}")

# ============================================================
#  STEP 4 — SCALE
# ============================================================
print("\n[4/7] Scaling...")
scaler   = StandardScaler()
X_scaled = scaler.fit_transform(X)
pickle.dump(scaler, open("models/tinyml_final_scaler.pkl", "wb"))
print(f"  Mean  (first 5): {scaler.mean_[:5]}")
print(f"  Scale (first 5): {scaler.scale_[:5]}")

# ============================================================
#  STEP 5 — SPLIT
# ============================================================
print("\n[5/7] Splitting (80/20 stratified)...")
X_train, X_val, y_train, y_val = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)
print(f"  Train: {X_train.shape}  Val: {X_val.shape}")

# ── Sample weights ──────────────────────────────────────────
print("\n  Computing sample weights...")
classes          = np.unique(y_train)
base_weights     = compute_class_weight("balanced", classes=classes,
                                        y=y_train)
base_weight_dict = dict(zip(classes, base_weights))
base_weight_dict[2] *= DOS_BOOST
base_weight_dict[3] *= MIRAI_BOOST

sample_weights_train = np.array(
    [base_weight_dict[c] for c in y_train.values], dtype=np.float32
)

print("  Effective weights:")
for c, w in base_weight_dict.items():
    print(f"    {CLASS_NAMES[c]:15s}: {w:.2f}")

# ============================================================
#  STEP 6 — BUILD & TRAIN
# ============================================================
print("\n[6/7] Building and training...")

n_features = X_train.shape[1]   # 26
n_classes  = len(np.unique(y))  # 5

# ── Macro-F1 metric ─────────────────────────────────────────
class MacroF1(tf.keras.metrics.Metric):
    def __init__(self, num_classes, name="macro_f1", **kwargs):
        super().__init__(name=name, **kwargs)
        self.num_classes = num_classes
        self.conf = self.add_weight(
            name="conf", shape=(num_classes, num_classes),
            initializer="zeros", dtype=tf.float32
        )

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred_cls = tf.cast(tf.argmax(y_pred, axis=-1), tf.int32)
        y_true     = tf.cast(tf.reshape(y_true, [-1]), tf.int32)
        self.conf.assign_add(tf.math.confusion_matrix(
            y_true, y_pred_cls,
            num_classes=self.num_classes, dtype=tf.float32
        ))

    def result(self):
        tp = tf.linalg.diag_part(self.conf)
        fp = tf.reduce_sum(self.conf, axis=0) - tp
        fn = tf.reduce_sum(self.conf, axis=1) - tp
        p  = tp / (tp + fp + 1e-7)
        r  = tp / (tp + fn + 1e-7)
        return tf.reduce_mean(2 * p * r / (p + r + 1e-7))

    def reset_state(self):
        self.conf.assign(tf.zeros_like(self.conf))


# ── Early stopping on val macro-F1 ──────────────────────────
class F1EarlyStopping(tf.keras.callbacks.Callback):
    def __init__(self, patience=8, min_delta=1e-3):
        super().__init__()
        self.patience     = patience
        self.min_delta    = min_delta
        self.best         = -np.inf
        self.wait         = 0
        self.best_weights = None

    def on_epoch_end(self, epoch, logs=None):
        val_f1 = logs.get("val_macro_f1", 0.0)
        if val_f1 > self.best + self.min_delta:
            self.best         = val_f1
            self.wait         = 0
            self.best_weights = self.model.get_weights()
            print(f"  ✔ val_macro_f1 improved to {val_f1:.4f}")
        else:
            self.wait += 1
            print(f"  · no improvement ({self.wait}/{self.patience})")
            if self.wait >= self.patience:
                print(f"\n  ⏹ Early stopping at epoch {epoch+1}")
                self.model.stop_training = True
                self.model.set_weights(self.best_weights)


# ── Warmup + cosine decay LR ─────────────────────────────────
total_steps  = (len(X_train) // BATCH_SIZE + 1) * EPOCHS
warmup_steps = (len(X_train) // BATCH_SIZE + 1) * WARMUP_EPOCHS

class WarmupCosineDecay(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, warmup_steps, total_steps, lr_base, alpha=1e-5):
        self.warmup_steps = tf.cast(warmup_steps, tf.float32)
        self.lr_base      = lr_base
        self.cosine       = tf.keras.optimizers.schedules.CosineDecay(
            initial_learning_rate=lr_base,
            decay_steps=total_steps - warmup_steps,
            alpha=alpha
        )

    def __call__(self, step):
        step   = tf.cast(step, tf.float32)
        warmup = self.lr_base * (step / (self.warmup_steps + 1e-7))
        cosine = self.cosine(tf.maximum(step - self.warmup_steps, 0))
        return tf.where(step < self.warmup_steps, warmup, cosine)

    def get_config(self):
        return {}


# ── tf.data pipeline — sample weights embedded ──────────────
def make_train_dataset(X, y, sw, batch_size):
    ds = tf.data.Dataset.from_tensor_slices((
        tf.cast(X,  tf.float32),
        tf.cast(y,  tf.int32),
        tf.cast(sw, tf.float32)
    ))
    ds = ds.shuffle(buffer_size=100_000, seed=42).batch(batch_size)

    def mixup(xb, yb, wb, alpha=0.3):
        n   = tf.shape(xb)[0]
        lam = tf.reshape(
            tf.random.uniform([n], 0.0, alpha, dtype=tf.float32), [-1, 1]
        )
        idx = tf.random.shuffle(tf.range(n))
        return (1 - lam) * xb + lam * tf.gather(xb, idx), yb, wb

    return ds.map(mixup, num_parallel_calls=tf.data.AUTOTUNE)\
             .prefetch(tf.data.AUTOTUNE)


def make_val_dataset(X, y, batch_size):
    ds = tf.data.Dataset.from_tensor_slices((
        tf.cast(X, tf.float32),
        tf.cast(y, tf.int32)
    ))
    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)


train_ds = make_train_dataset(X_train, y_train.values,
                               sample_weights_train, BATCH_SIZE)
val_ds   = make_val_dataset(X_val, y_val.values, BATCH_SIZE)


# ── Weighted focal loss ──────────────────────────────────────
def weighted_focal_loss(gamma=2.5):
    def loss_fn(y_true, y_pred, sample_weight=None):
        y_true = tf.cast(tf.reshape(y_true, [-1]), tf.int32)
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0)
        n      = tf.shape(y_pred)[0]
        pt     = tf.gather_nd(y_pred,
                              tf.stack([tf.range(n), y_true], axis=1))
        fl     = tf.pow(1.0 - pt, gamma) * (-tf.math.log(pt))
        if sample_weight is not None:
            sw = tf.cast(tf.reshape(sample_weight, [-1]), tf.float32)
            sw = sw / (tf.reduce_mean(sw) + 1e-7)
            fl = fl * sw
        return tf.reduce_mean(fl)
    return loss_fn


# ── Model: 26→128→64→32→16→16→5 (~60KB TFLite) ──────────────
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(n_features,)),

    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.3),

    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.2),

    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.2),

    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dropout(0.1),

    tf.keras.layers.Dense(16, activation='relu'),

    tf.keras.layers.Dense(n_classes, activation='softmax')
], name="IoMT_TinyML_IDS_final")

model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=WarmupCosineDecay(warmup_steps, total_steps, LR_BASE)
    ),
    loss=weighted_focal_loss(gamma=GAMMA),
    metrics=['accuracy', MacroF1(n_classes)]
)

model.summary()

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=[F1EarlyStopping(patience=PATIENCE)],
    verbose=1
)

# ============================================================
#  STEP 7 — EVALUATE
# ============================================================
print("\n[7/7] Evaluating...")

y_prob = model.predict(val_ds, verbose=0)
y_pred = np.argmax(y_prob, axis=1)

acc = np.mean(y_pred == y_val.values)
f1  = f1_score(y_val, y_pred, average="macro", zero_division=0)
print(f"\n  Validation Accuracy : {acc:.4f}")
print(f"  F1 Macro            : {f1:.4f}")

print("\nClassification Report:")
print(classification_report(y_val, y_pred,
                             target_names=CLASS_NAMES, zero_division=0))

cm = confusion_matrix(y_val, y_pred)
print("Confusion Matrix:")
print(cm)

per_class = f1_score(y_val, y_pred, average=None, zero_division=0)
print("\nPer-class F1:")
for name, score in zip(CLASS_NAMES, per_class):
    print(f"  {name:15s}: {score:.3f}  {'█' * int(score * 20)}")

print("\nDoS/DDoS confusion:")
dos_as_ddos = cm[2][1]; dos_total  = cm[2].sum()
ddos_as_dos = cm[1][2]; ddos_total = cm[1].sum()
print(f"  DoS  → DDoS : {dos_as_ddos:,} / {dos_total:,}  ({100*dos_as_ddos/dos_total:.1f}%)")
print(f"  DDoS → DoS  : {ddos_as_dos:,} / {ddos_total:,}  ({100*ddos_as_dos/ddos_total:.1f}%)")

# ============================================================
#  TRAINING CURVES
# ============================================================
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle(f"FINAL | DOS={DOS_BOOST} MIRAI={MIRAI_BOOST} γ={GAMMA} bs={BATCH_SIZE}")

    axes[0].plot(history.history.get("loss",         []), label="train")
    axes[0].plot(history.history.get("val_loss",     []), label="val")
    axes[0].set_title("Focal Loss"); axes[0].legend()

    axes[1].plot(history.history.get("accuracy",     []), label="train")
    axes[1].plot(history.history.get("val_accuracy", []), label="val")
    axes[1].set_title("Accuracy"); axes[1].legend()

    axes[2].plot(history.history.get("macro_f1",     []), label="train")
    axes[2].plot(history.history.get("val_macro_f1", []), label="val")
    axes[2].set_title("Macro F1"); axes[2].legend()

    plt.tight_layout()
    plt.savefig("models/tinyml_final_curves.png", dpi=120)
    print("\n  Saved: models/tinyml_final_curves.png")
except ImportError:
    print("\n  (matplotlib not installed — skipping)")

# ============================================================
#  SAVE — Keras + TFLite
# ============================================================
model.save("models/tinyml_final.keras")
print("\n✅ Saved: models/tinyml_final.keras")

print("\n  Converting to TFLite...")
try:
    converter    = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()
    tflite_path  = "models/tinyml_final.tflite"
    with open(tflite_path, "wb") as f:
        f.write(tflite_model)
    print(f"  ✅ Saved: {tflite_path}  ({len(tflite_model)/1024:.1f} KB)")
    print(f"     → Flash this file to your ESP32")
except Exception as e:
    print(f"  TFLite conversion failed: {e}")

# ============================================================
#  ESP32 CONSTANTS
# ============================================================
print("\n" + "=" * 55)
print("  COPY THESE INTO YOUR ESP32 CODE:")
print("=" * 55)
print(f"\nconst int   NUM_FEATURES = {n_features};")
print(f"const int   NUM_CLASSES  = {n_classes};")
print(f"\nconst float FEATURE_MEAN[{n_features}] = {{")
print(f"  {', '.join([f'{v:.6f}f' for v in scaler.mean_])}")
print("};")
print(f"\nconst float FEATURE_STD[{n_features}] = {{")
print(f"  {', '.join([f'{v:.6f}f' for v in scaler.scale_])}")
print("};")
print(f"\nconst char* CLASS_NAMES[{n_classes}] = {{")
print(f"  {', '.join([chr(34)+n+chr(34) for n in CLASS_NAMES])}")
print("};")
print("=" * 55)
print("\nDone ✅")