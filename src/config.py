from pathlib import Path


# Project Root

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Data Directories

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODEL_DIR = DATA_DIR / "save model"

# Raw Data

TRAIN_RAW_PATH = RAW_DATA_DIR / "fraudTrain.csv"
TEST_RAW_PATH = RAW_DATA_DIR / "fraudTest.csv"

# Processed Data
FEATURED_DATA_PATH = (PROCESSED_DATA_DIR /"creditcard_featured_data.csv")

PREPROCESSED_DATA_PATH = (PROCESSED_DATA_DIR /"creditcard_preprocessed.csv")

TRAIN_FEATURES_PATH = (PROCESSED_DATA_DIR /"train_features.csv")

VALIDATION_FEATURES_PATH = (PROCESSED_DATA_DIR /"validation_features.csv")

TEST_FEATURES_PATH = (PROCESSED_DATA_DIR /"test_features.csv")

# Saved Model

MODEL_PATH = (MODEL_DIR /"cardguard_xgboost.pkl")

# Model Settings

DEFAULT_THRESHOLD = 0.65