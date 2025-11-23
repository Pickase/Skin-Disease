import pandas as pd
import numpy as np

def clean_dataframe(df: pd.DataFrame):
    # Replace '?' with NaN
    df = df.replace("?", np.nan)

    # Convert all columns to numeric if possible
    df = df.apply(pd.to_numeric, errors='ignore')

    # Fill NaN (previously '?' values)
    df = df.fillna(0)

    # Remove duplicates
    df = df.drop_duplicates()

    return df
