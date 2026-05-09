"""
preprocessing.py
-----------------
Functions for loading, cleaning, and encoding the NSL-KDD dataset.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler


def load_data(filepath: str, columns: list) -> pd.DataFrame:
    """Load NSL-KDD dataset from a CSV file."""
    df = pd.read_csv(filepath, header=None, names=columns)
    return df


def drop_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Remove duplicate rows."""
    return df.drop_duplicates()


def encode_categoricals(df: pd.DataFrame, cat_cols: list) -> pd.DataFrame:
    """Label-encode categorical columns."""
    le = LabelEncoder()
    for col in cat_cols:
        df[col] = le.fit_transform(df[col])
    return df


def scale_features(X_train, X_test):
    """Standardize numerical features using training statistics."""
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, scaler
