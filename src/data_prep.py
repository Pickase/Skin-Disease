import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import joblib

from .config import (
    RAW_DATA_PATH,
    PROCESSED_TRAIN_PATH,
    PROCESSED_TEST_PATH,
    TARGET_COL,
    TEST_SIZE,
    RANDOM_STATE,
)
from .utils import clean_dataframe

def prepare_data():
    print("Loading raw dataset...")
    df = pd.read_csv(RAW_DATA_PATH)

    df = clean_dataframe(df)

    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]

    # Encode labels
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    joblib.dump(le, "models/label_encoder.pkl")

    # Split using encoded y
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y_encoded
    )

    # Save CLEAN, ENCODED target values
    train_df = pd.concat([X_train.reset_index(drop=True), 
                          pd.Series(y_train, name=TARGET_COL)], axis=1)

    test_df = pd.concat([X_test.reset_index(drop=True), 
                         pd.Series(y_test, name=TARGET_COL)], axis=1)

    train_df.to_csv(PROCESSED_TRAIN_PATH, index=False)
    test_df.to_csv(PROCESSED_TEST_PATH, index=False)

    print("Data prepared successfully!")


if __name__ == "__main__":
    prepare_data()
