import numpy as np

def feature_engineering(df):
    def safe_div(a, b):
        return a / (b + 1)

    if "Tot_size" in df.columns and "Duration" in df.columns:
        df["size_rate"] = safe_div(df["Tot_size"], df["Duration"])

    if "Number" in df.columns and "Duration" in df.columns:
        df["packet_density"] = safe_div(df["Number"], df["Duration"])

    if "Rate" in df.columns and "IAT" in df.columns:
        df["interaction"] = df["Rate"] * df["IAT"]

    df.replace([np.inf, -np.inf], 0, inplace=True)
    df.fillna(0, inplace=True)

    return df