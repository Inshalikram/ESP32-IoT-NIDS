import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import LabelEncoder

print("🚀 Starting preprocessing (memory-safe)...")

# ---------------- SETTINGS ----------------
CHUNK_SIZE = 200000  

# ---------------- CLEAN FUNCTION ----------------
def clean(df):

    # Drop leakage columns
    drop_cols = ["Flow ID", "Src IP", "Dst IP", "Timestamp"]
    df.drop(columns=[c for c in drop_cols if c in df.columns],
            inplace=True, errors="ignore")

    # Replace inf
    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    # Fill missing
    df.fillna(0, inplace=True)

    # Convert float64 → float32 (huge memory saving)
    for col in df.select_dtypes(include=["float64"]).columns:
        df[col] = df[col].astype("float32")

    return df


# ---------------- LOAD + PROCESS IN CHUNKS ----------------
def load_in_chunks(path):
    chunks = []

    for chunk in pd.read_csv(path, chunksize=CHUNK_SIZE, low_memory=True):
        chunk = clean(chunk)
        chunks.append(chunk)

    return pd.concat(chunks, ignore_index=True)


print("📥 Loading train...")
train = load_in_chunks("data/train_merged.csv")

print("📥 Loading test...")
test = load_in_chunks("data/test_merged.csv")

# ---------------- ALIGN COLUMNS ----------------
print("🔗 Aligning columns...")

common_cols = list(set(train.columns).intersection(set(test.columns)))
common_cols = sorted(common_cols)

if "Label" not in common_cols:
    raise ValueError("❌ Label column missing!")

train = train[common_cols]
test = test[common_cols]

# ---------------- REMOVE CONSTANT FEATURES ----------------
print("🧹 Removing constant columns...")

nunique = train.nunique()
valid_cols = nunique[nunique > 1].index

train = train[valid_cols]
test = test[valid_cols.intersection(test.columns)]

# ---------------- SPLIT ----------------
print("✂ Splitting...")

y_train = train["Label"]
X_train = train.drop("Label", axis=1)

y_test = test["Label"]
X_test = test.drop("Label", axis=1)

# ---------------- FINAL ALIGNMENT ----------------
print("🔄 Final feature alignment...")

X_test = X_test.reindex(columns=X_train.columns, fill_value=0)

# ---------------- LABEL ENCODING ----------------
print("🔤 Encoding labels...")

le = LabelEncoder()
y_train = le.fit_transform(y_train.astype(str))
y_test = le.transform(y_test.astype(str))

# ---------------- SAVE ----------------
print("💾 Saving outputs...")

features = X_train.columns.tolist()

pickle.dump(features, open("models/features.pkl", "wb"))
pickle.dump(le, open("models/label_encoder.pkl", "wb"))

X_train.to_csv("data/X_train.csv", index=False)
X_test.to_csv("data/X_test.csv", index=False)

pd.DataFrame(y_train, columns=["Label"]).to_csv("data/y_train.csv", index=False)
pd.DataFrame(y_test, columns=["Label"]).to_csv("data/y_test.csv", index=False)

print("\n✅ PREPROCESSING COMPLETE (NO CRASH + CONSISTENT)")
print("Train shape:", X_train.shape)
print("Test shape:", X_test.shape)
print("Classes:", le.classes_)