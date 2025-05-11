#!/usr/bin/env python3
"""
Robust data_processing.py
- Auto-detects categorical and numeric columns
- Drops ID and free-text address columns automatically
- Splits encoding into OHE and scaling
- Uses feature_selection.mutual_info_classif (no clustering warnings)
"""
import argparse
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.feature_selection import mutual_info_classif
from scipy.sparse import hstack, csr_matrix


def load_data(path: str) -> pd.DataFrame:
    """Load CSV into DataFrame"""
    return pd.read_csv(path)


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """Parse dates and extract components if present"""
    if 'Transaction Date' in df.columns:
        df['Transaction Date'] = pd.to_datetime(
            df['Transaction Date'], errors='coerce')
        df['Month'] = df['Transaction Date'].dt.month
        df['Weekday'] = df['Transaction Date'].dt.weekday
        df = df.drop(columns=['Transaction Date'])
    return df


def encode_features(df: pd.DataFrame,
                    cat_cols: list[str],
                    num_cols: list[str]) -> tuple[csr_matrix, np.ndarray]:
    """
    One-hot encode categorical cols and scale numeric cols separately,
    then combine into a sparse matrix and return feature names.
    """
    # One-hot encode categoricals
    ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=True)
    X_cat = ohe.fit_transform(df[cat_cols])
    cat_feature_names = ohe.get_feature_names_out(cat_cols)

    # Scale numerics
    scaler = StandardScaler()
    X_num = scaler.fit_transform(df[num_cols])
    X_num_sparse = csr_matrix(X_num)
    num_feature_names = np.array(num_cols)

    # Combine into one sparse matrix
    X_combined = hstack([X_cat, X_num_sparse], format='csr')
    feature_names = np.concatenate([cat_feature_names, num_feature_names])

    return X_combined, feature_names


def compute_mutual_info(X: csr_matrix,
                        y: np.ndarray) -> np.ndarray:
    """
    Compute mutual information between each column of X and target y
    using feature_selection.mutual_info_classif.
    """
    return mutual_info_classif(
        X,
        y,
        discrete_features='auto',
        random_state=42
    )


def main():
    p = argparse.ArgumentParser(description="Data processing + MI scoring")
    p.add_argument('input_csv', nargs='?', default='data/Fraudulent_E-Commerce_Transaction_Data_2.csv',
                   help="Path to input CSV file")
    p.add_argument('--target', default='Is Fraudulent',
                   help="Name of the target column (default: Is Fraudulent)")
    args = p.parse_args()

    # Load and preprocess
    df = load_data(args.input_csv)
    df = preprocess(df)

    # Split into features and target
    target = args.target
    if target not in df.columns:
        raise KeyError(f"Target column '{target}' not found in data")
    y = df[target].values
    X_df = df.drop(columns=[target])

    # Drop ID and free-text columns by pattern
    drop_patterns = ['ID', 'Address', 'IP']
    to_drop = [c for c in X_df.columns if any(
        pat in c for pat in drop_patterns)]
    X_df = X_df.drop(columns=to_drop, errors='ignore')

    # Auto-detect categorical vs. numeric columns
    cat_cols = X_df.select_dtypes(
        include=['object', 'category']).columns.tolist()
    num_cols = X_df.select_dtypes(include=['number']).columns.tolist()

    if not cat_cols and not num_cols:
        raise ValueError("No features detected after preprocessing.")

    # Encode and combine
    X_proc, feature_names = encode_features(X_df, cat_cols, num_cols)

    # Compute MI scores
    mi_scores = compute_mutual_info(X_proc, y)

    # Display sorted MI
    mi_series = pd.Series(
        mi_scores, index=feature_names).sort_values(ascending=False)

    threshold = 1e-3

    keep_feats = mi_series[mi_series >= threshold].index.tolist()

    df_proc = pd.DataFrame.sparse.from_spmatrix(
        X_proc,
        columns=feature_names
    )
    df_reduced = df_proc[keep_feats].copy()
    df_reduced['Is Fraudulent'] = y

    df_reduced.to_csv('data/reduced_fraud_data.csv', index=False)

    # 1) load your reduced data
    df = pd.read_csv('data/reduced_fraud_data.csv')

    # 2) split positives and negatives
    df_pos = df[df['Is Fraudulent'] == 1]
    df_neg = df[df['Is Fraudulent'] == 0]

    # 3) decide how many negatives you want (3× the positives)
    n_keep_neg = len(df_pos) * 3

    # 4) randomly sample that many negatives
    df_neg_under = df_neg.sample(n=n_keep_neg, random_state=42)

    # 5) recombine & shuffle
    df_balanced = (
        pd.concat([df_pos, df_neg_under])
        .sample(frac=1, random_state=42)
        .reset_index(drop=True)
    )

    # 6) save
    df_balanced.to_csv('data/balanced_1to3_fraud_data.csv', index=False)

    df = pd.read_csv("data/balanced_1to3_fraud_data.csv")

    # Split features and labels
    X = df.drop(columns=['Is Fraudulent'])
    y = df['Is Fraudulent']

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    X_train.to_csv("data/ecommerceX_train.csv", index=False)
    X_test.to_csv("data/ecommerceX_test.csv", index=False)
    y_train.to_csv("data/ecommerceY_train.csv", index=False)
    y_test.to_csv("data/ecommerceY_test.csv", index=False)


if __name__ == '__main__':
    main()
