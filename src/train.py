import pandas as pd
import joblib
from imblearn.combine import SMOTEENN
from .config import PROCESSED_TRAIN_PATH, MODEL_PATH, TARGET_COL
from .features import split_features_labels
from .pipelines import build_xgb_pipeline

def train_model():
    print("\n=== TRAINING STARTED ===")

    print(f"Loading: {PROCESSED_TRAIN_PATH}")
    df = pd.read_csv(PROCESSED_TRAIN_PATH)
    print("Training dataset shape:", df.shape)

    X, y = split_features_labels(df, TARGET_COL)
    print("Splitted features:", X.shape, "labels:", y.shape)

    print("Applying SMOTEENN...")
    sme = SMOTEENN(random_state=42)
    X_resampled, y_resampled = sme.fit_resample(X, y)
    print("After SMOTEENN:", X_resampled.shape, y_resampled.shape)

    print("Building XGBoost model...")
    model = build_xgb_pipeline()

    print("Training XGBoost...")
    model.fit(X_resampled, y_resampled)
    print("Model training complete!")

    joblib.dump(model, MODEL_PATH)
    print(f"Saved trained model to {MODEL_PATH}")

    print("=== TRAINING FINISHED ===\n")


if __name__ == "__main__":
    train_model()
    print("Done!")