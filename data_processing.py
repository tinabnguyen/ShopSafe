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

from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.feature_selection import mutual_info_classif
from scipy.sparse import hstack, csr_matrix


def load_data(path: str) -> pd.DataFrame:
    """Load CSV into DataFrame"""
    return pd.read_csv(path)


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """Parse dates into components"""
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
    One hot encode categorical columns and scale numeric columns.
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
    """
    load, preprocess, and encode the data -------------------------------------------------------------
    """
    p = argparse.ArgumentParser(description="Data processing + MI scoring")
    p.add_argument('input_csv', nargs='?', default='data/Fraudulent_E-Commerce_Transaction_Data_2.csv',
                   help="Path to input CSV file")
    p.add_argument('--target', default='Is Fraudulent',
                   help="Name of the target column (default: Is Fraudulent)")
    args = p.parse_args()

    # Load and preprocess
    df = load_data(args.input_csv)
    df = preprocess(df)

    target = args.target
    if target not in df.columns:
        raise KeyError(f"Target column '{target}' not found in data")
    y = df[target].values
    X_df = df.drop(columns=[target])

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

    X_proc, feature_names = encode_features(X_df, cat_cols, num_cols)

    mi_scores = compute_mutual_info(X_proc, y)

    mi_series = pd.Series(
        mi_scores, index=feature_names).sort_values(ascending=False)

    threshold = 1e-3
    # based on the Mutual information, only keep features above the threshold
    keep_feats = mi_series[mi_series >= threshold].index.tolist()

    df_proc = pd.DataFrame.sparse.from_spmatrix(
        X_proc,
        columns=feature_names
    )
    df_reduced = df_proc[keep_feats].copy()
    df_reduced['Is Fraudulent'] = y

    df_reduced.to_csv('data/reduced_fraud_data.csv', index=False)

    """
    balancing the dataset to a 1:1 ratio -----------------------------------------------------------------
    """
    df = pd.read_csv('data/reduced_fraud_data.csv')

    df_pos = df[df['Is Fraudulent'] == 1]
    df_neg = df[df['Is Fraudulent'] == 0]

    n_keep_neg = len(df_pos) * 1

    df_neg_under = df_neg.sample(n=n_keep_neg, random_state=42)

    df_balanced = (
        pd.concat([df_pos, df_neg_under])
        .sample(frac=1, random_state=42)
        .reset_index(drop=True)
    )

    df_balanced.to_csv('data/balanced_1to1_fraud_data.csv', index=False)


if __name__ == '__main__':
    main()
