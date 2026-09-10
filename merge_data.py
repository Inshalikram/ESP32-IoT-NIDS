import os
import pandas as pd
import numpy as np

def map_label(filename):
    name = filename.lower()

    if "benign" in name:
        return "Benign"
    elif "ddos" in name:
        return "DDoS"
    elif "dos" in name:
        return "DoS"
    elif "recon" in name:
        return "Recon"
    elif "mitm" in name:
        return "MITM"
    elif "mqtt" in name:
        return "MQTT"
    else:
        return None


def merge_folder(folder_path, output_path):

    all_data = []
    files = [f for f in os.listdir(folder_path) if f.endswith(".csv")]

    # -------- FIRST PASS: COLUMNS --------
    all_columns = set()
    for file in files:
        try:
            df = pd.read_csv(os.path.join(folder_path, file), low_memory=False)
            all_columns.update(df.columns)
        except:
            continue

    all_columns = list(all_columns)

    # -------- SECOND PASS --------
    for file in files:

        label = map_label(file)
        if label is None:
            continue

        try:
            df = pd.read_csv(os.path.join(folder_path, file), low_memory=False)
        except:
            continue

        df["Label"] = label

        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df.fillna(0, inplace=True)

        # align columns
        for col in all_columns:
            if col not in df.columns:
                df[col] = 0

        df = df[all_columns + ["Label"]]
        all_data.append(df)

    final_df = pd.concat(all_data, ignore_index=True)
    final_df.to_csv(output_path, index=False)

    print("Saved:", output_path)
    print(final_df["Label"].value_counts())


merge_folder("data/train", "data/train_merged.csv")
merge_folder("data/test", "data/test_merged.csv")