import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
import joblib


def feature_engineering(X_train, X_test, y_train):

    # -----------------------------
    # 0. SAFETY CLEANING (CRITICAL)
    # -----------------------------

    # Replace inf/-inf
    X_train = X_train.replace([np.inf, -np.inf], np.nan)
    X_test = X_test.replace([np.inf, -np.inf], np.nan)

    # Fill missing values using TRAIN statistics only
    train_mean = X_train.mean()
    X_train = X_train.fillna(train_mean)
    X_test = X_test.fillna(train_mean)

    # Remove constant columns (VERY IMPORTANT for NIDS)
    nunique = X_train.nunique()
    constant_cols = nunique[nunique <= 1].index

    X_train = X_train.drop(columns=constant_cols)
    X_test = X_test.drop(columns=constant_cols)

    print(f"Removed constant features: {len(constant_cols)}")

    # -----------------------------
    # 1. SCALING (NO LEAKAGE)
    # -----------------------------
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    joblib.dump(scaler, "models/scaler.pkl")

    # -----------------------------
    # 2. FEATURE SELECTION (SAFE)
    # -----------------------------
    k = min(25, X_train_scaled.shape[1])

    selector = SelectKBest(score_func=f_classif, k=k)

    X_train_selected = selector.fit_transform(X_train_scaled, y_train)
    X_test_selected = selector.transform(X_test_scaled)

    joblib.dump(selector, "models/selector.pkl")

    print(f"Selected features: {k}")

    return X_train_selected, X_test_selected