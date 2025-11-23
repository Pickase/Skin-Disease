import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, classification_report
from .config import PROCESSED_TEST_PATH, MODEL_PATH, TARGET_COL
from .features import split_features_labels

def evaluate_model():
    print("\n=== EVALUATION STARTED ===")

    df = pd.read_csv(PROCESSED_TEST_PATH)
    print("Test dataset shape:", df.shape)

    X_test, y_test = split_features_labels(df, TARGET_COL)
    print("Feature shape:", X_test.shape)

    model = joblib.load(MODEL_PATH)
    print("Model loaded.")

    preds = model.predict(X_test)

    acc = accuracy_score(y_test, preds)
    print(f"Accuracy: {acc}")

    print("\nClassification Report:")
    print(classification_report(y_test, preds))

    print("=== EVALUATION FINISHED ===\n")


if __name__ == "__main__":
    evaluate_model()
