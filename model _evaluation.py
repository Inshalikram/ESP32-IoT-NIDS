import pandas as pd
import numpy as np
import pickle

from sklearn.metrics import classification_report, f1_score, confusion_matrix, roc_auc_score

print("🚀 Loading model...")

model = pickle.load(open("models/model.pkl", "rb"))
features = pickle.load(open("models/features.pkl", "rb"))
selected_features = pickle.load(open("models/selected_features.pkl", "rb"))
t3, t4 = pickle.load(open("models/thresholds.pkl", "rb"))

# ---------------- LOAD TEST ----------------
X_test = pd.read_csv("data/X_test.csv")
y_test = pd.read_csv("data/y_test.csv")["Label"]

# ---------------- ALIGN ----------------
X_test = X_test.reindex(columns=features, fill_value=0)

# REMOVE LEAKAGE FEATURES
drop_cols = ["Rate", "Srate", "Tot sum"]
X_test = X_test.drop(columns=drop_cols, errors="ignore")

# FEATURE SELECT
X_test = X_test[selected_features.tolist()]

# ---------------- PREDICT ----------------
y_prob = model.predict_proba(X_test)
y_pred = model.predict(X_test)

# APPLY OPTIMIZED THRESHOLDS
y_adj = y_pred.copy()
y_adj[y_prob[:, 3] > t3] = 3
y_adj[y_prob[:, 4] > t4] = 4

y_pred = y_adj

# ---------------- METRICS ----------------
print("\n===== FINAL IDS RESULTS =====\n")

print("F1 Macro:", f1_score(y_test, y_pred, average="macro"))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:\n", cm)

print("\nROC-AUC:",
      roc_auc_score(y_test, y_prob, multi_class="ovr"))

# ---------------- FAR ----------------
print("\n===== FALSE ALARM RATE =====")

fp = cm.sum(axis=0) - cm.diagonal()
tp = cm.diagonal()
far = fp / (fp + tp + 1e-9)

for i, f in enumerate(far):
    print(f"Class {i}: {f:.4f}")

print("\n✅ FINAL EVALUATION DONE (IMPROVED)")