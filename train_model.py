import pandas as pd
import numpy as np
import pickle
import lightgbm as lgb

from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import VarianceThreshold
from sklearn.metrics import f1_score
from sklearn.utils.class_weight import compute_class_weight

print("Loading data...")

# ---------------- LOAD ----------------
X = pd.read_csv("data/X_train.csv")
y = pd.read_csv("data/y_train.csv")["Label"]

features = pickle.load(open("models/features.pkl", "rb"))
X = X[features]

# =========================================================
# 🔥 REMOVE LEAKAGE FEATURES
# =========================================================
drop_cols = ["Rate", "Srate", "Tot sum"]
X = X.drop(columns=[c for c in drop_cols if c in X.columns])

print("Leakage features removed:", drop_cols)

# =========================================================
# ✅ FAST UNDERSAMPLING (LESS AGGRESSIVE)
# =========================================================
print("Applying balanced sampling...")

df = X.copy()
df["Label"] = y

counts = df["Label"].value_counts()

major_class = counts.idxmax()
df_major = df[df["Label"] == major_class]
df_minor = df[df["Label"] != major_class]

# less aggressive → better generalization
target_size = len(df_minor) * 3

if target_size > len(df_major):
    df_major = df_major.sample(n=target_size, replace=True, random_state=42)
else:
    df_major = df_major.sample(n=target_size, replace=False, random_state=42)

df = pd.concat([df_major, df_minor]).sample(frac=1, random_state=42)

X = df.drop("Label", axis=1)
y = df["Label"]

# =========================================================
# ✅ CLASS WEIGHTS (LESS EXTREME)
# =========================================================
classes = np.unique(y)
weights = compute_class_weight("balanced", classes=classes, y=y)
class_weights = dict(zip(classes, weights))

# soften extreme class weight (class 3)
if 3 in class_weights:
    class_weights[3] *= 1.5   # instead of 2x

print("Class weights:", class_weights)

# =========================================================
# ✅ FEATURE SELECTION (SAFE)
# =========================================================
selector = VarianceThreshold(0.005)
X_selected = selector.fit_transform(X)

selected_features = np.array(X.columns)[selector.get_support()]

# FIX WARNING (important)
X_selected = pd.DataFrame(X_selected, columns=selected_features)

pickle.dump(selector, open("models/selector.pkl", "wb"))
pickle.dump(selected_features, open("models/selected_features.pkl", "wb"))

# =========================================================
# ✅ CROSS VALIDATION + THRESHOLD SEARCH
# =========================================================
print("Starting CV with threshold tuning...")

kf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

f1_scores = []
best_thresholds = []

for fold, (train_idx, val_idx) in enumerate(kf.split(X_selected, y)):

    print(f"\n--- Fold {fold+1} ---")

    X_train, X_val = X_selected.iloc[train_idx], X_selected.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

    model = lgb.LGBMClassifier(
        objective="multiclass",
        n_estimators=400,
        learning_rate=0.04,
        num_leaves=64,
        max_depth=9,
        max_bin=127,
        class_weight=class_weights,
        subsample=0.85,
        colsample_bytree=0.85,
        n_jobs=-1,
        random_state=42,
        force_col_wise=True
    )

    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        eval_metric="multi_logloss",
        callbacks=[
            lgb.early_stopping(40),
            lgb.log_evaluation(100)
        ]
    )

    # ---------------- PREDICTIONS ----------------
    y_prob = model.predict_proba(X_val)
    y_pred = model.predict(X_val)

    # =========================================================
    # 🔥 THRESHOLD SEARCH (AUTO)
    # =========================================================
    best_f1 = 0
    best_t3, best_t4 = 0.3, 0.5

    for t3 in np.arange(0.3, 0.6, 0.05):
        for t4 in np.arange(0.5, 0.7, 0.05):

            y_adj = y_pred.copy()
            y_adj[y_prob[:, 3] > t3] = 3
            y_adj[y_prob[:, 4] > t4] = 4

            f1 = f1_score(y_val, y_adj, average="macro")

            if f1 > best_f1:
                best_f1 = f1
                best_t3, best_t4 = t3, t4

    print(f"Best thresholds → Class3: {best_t3}, Class4: {best_t4}")
    print("F1:", best_f1)

    f1_scores.append(best_f1)
    best_thresholds.append((best_t3, best_t4))

print("\nMean F1:", np.mean(f1_scores))

# =========================================================
# ✅ FINAL MODEL
# =========================================================
print("\nTraining final model...")

final_model = lgb.LGBMClassifier(
    objective="multiclass",
    n_estimators=400,
    learning_rate=0.04,
    num_leaves=64,
    max_depth=9,
    max_bin=127,
    class_weight=class_weights,
    subsample=0.85,
    colsample_bytree=0.85,
    n_jobs=-1,
    random_state=42,
    force_col_wise=True
)

final_model.fit(X_selected, y)

# ---------------- SAVE EVERYTHING ----------------
pickle.dump(final_model, open("models/model.pkl", "wb"))

# average thresholds
t3 = np.mean([t[0] for t in best_thresholds])
t4 = np.mean([t[1] for t in best_thresholds])

pickle.dump((t3, t4), open("models/thresholds.pkl", "wb"))

print(f"Saved thresholds: Class3={t3}, Class4={t4}")
print("✅ TRAINING DONE (IMPROVED)")