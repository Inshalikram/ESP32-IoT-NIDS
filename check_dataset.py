import os
import pandas as pd

def analyze_folder(folder):
    total, attack, benign = 0, 0, 0

    print(f"\nAnalyzing {folder}")

    for file in os.listdir(folder):
        if file.endswith(".csv"):
            path = os.path.join(folder, file)
            df = pd.read_csv(path)
            rows = len(df)

            total += rows
            if "benign" in file.lower():
                benign += rows
            else:
                attack += rows

            print(f"{file} → {rows}")

    print("\nSUMMARY")
    print("Total:", total)
    print("Attack:", attack)
    print("Benign:", benign)

analyze_folder("data/train")
analyze_folder("data/test")