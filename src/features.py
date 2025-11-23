import pandas as pd

def get_feature_columns(df: pd.DataFrame, target_col: str):
    return [col for col in df.columns if col != target_col]

def split_features_labels(df: pd.DataFrame, target_col: str):
    X = df.drop(columns=[target_col])
    y = df[target_col]
    return X, y
