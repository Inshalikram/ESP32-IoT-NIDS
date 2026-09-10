import pandas as pd
import numpy as np
import pickle

print("Loading merged datasets...")

train = pd.read_csv("data/train_merged.csv", low_memory=False)
test = pd.read_csv("data/test_merged.csv", low_memory=False)

# ---------------- BASIC CLEAN ----------------
def basic_clean(df):
    df = df.copy()

    drop_cols = ["Flow ID", "Src IP", "Dst IP", "Timestamp"]
    df.drop(columns=[c for c in drop_cols if c in df.columns], inplace=True, errors="ignore")

    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.fillna(0, inplace=True)

    return df


train = basic_clean(train)
test = basic_clean(test)
# ---------------- LEAKAGE CHECK (IMPORTANT) ----------------
print("Checking feature correlation with label...")

corr = train.corr(numeric_only=True)

if "Label" in corr.columns:
    label_corr = corr["Label"].abs().sort_values(ascending=False)
    print("\nTop correlated features with Label:\n")
    print(label_corr.head(20))

# ---------------- REMOVE CONSTANT COLUMNS ----------------
print("Removing constant columns...")

common_cols = list(set(train.columns).intersection(set(test.columns)))

train = train[common_cols]
test = test[common_cols]

nunique = train.nunique()
stable_cols = nunique[nunique > 1].index

train = train[stable_cols]
test = test[stable_cols.intersection(test.columns)]

# ---------------- FINAL ALIGNMENT ----------------
print("Final alignment...")

# ensure same columns order
test = test.reindex(columns=train.columns, fill_value=0)

# ---------------- SAVE FEATURE SCHEMA (VERY IMPORTANT) ----------------
features = [c for c in train.columns if c != "Label"]

pickle.dump(features, open("models/features.pkl", "wb"))

# ---------------- SPLIT ----------------
y_train = train["Label"]
X_train = train.drop("Label", axis=1)

y_test = test["Label"]
X_test = test.drop("Label", axis=1)

# ---------------- SAVE ----------------
X_train.to_csv("data/X_train.csv", index=False)
X_test.to_csv("data/X_test.csv", index=False)
y_train.to_csv("data/y_train.csv", index=False)
y_test.to_csv("data/y_test.csv", index=False)

print("✅ DATASET PREPARED (SCHEMA LOCKED)")
print("Train shape:", X_train.shape)
print("Test shape:", X_test.shape)