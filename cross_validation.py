import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold, cross_val_score

# ---------------- LOAD ----------------
X = pd.read_csv("data/X_train.csv")
y = pd.read_csv("data/y_train.csv").iloc[:, 0]

# ---------------- MODEL ----------------
model = lgb.LGBMClassifier(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=10,
    num_leaves=128,
    class_weight="balanced",
    random_state=42
)

# ---------------- CROSS VALIDATION ----------------
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

scores = cross_val_score(
    model,
    X,
    y,
    cv=cv,
    scoring="f1_macro"
)

print("\n===== CROSS VALIDATION RESULTS =====")
print("F1 Scores:", scores)
print("Mean F1:", np.mean(scores))
print("Std Dev:", np.std(scores))