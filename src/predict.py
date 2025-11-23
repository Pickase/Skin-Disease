import pandas as pd
import joblib
from .config import MODEL_PATH, LABEL_ENCODER_PATH

def predict_single(sample_dict):
    """
    Predicts disease name from a dictionary of input features.
    """
    model = joblib.load(MODEL_PATH)
    le = joblib.load(LABEL_ENCODER_PATH)

    df = pd.DataFrame([sample_dict])
    pred = model.predict(df)[0]
    return le.inverse_transform([pred])[0]
