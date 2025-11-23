import os

BASE_DIR = os.path.dirname(os.path.dirname(__file__))

# DATA PATHS
RAW_DATA_PATH = os.path.join(BASE_DIR, "data", "raw", "dataset_dermatology.csv")
PROCESSED_TRAIN_PATH = os.path.join(BASE_DIR, "data", "processed", "train.csv")
PROCESSED_TEST_PATH = os.path.join(BASE_DIR, "data", "processed", "test.csv")

# MODELS
MODEL_PATH = os.path.join(BASE_DIR, "models", "xgb_model.pkl")
LABEL_ENCODER_PATH = os.path.join(BASE_DIR, "models", "label_encoder.pkl")

# TARGET COLUMN
TARGET_COL = "class"

# TRAIN PARAMS
TEST_SIZE = 0.2
RANDOM_STATE = 42
