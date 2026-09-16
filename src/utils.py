# CardGuard - Utility Functions

from pathlib import Path
import joblib

def save_model(model, path):
    """
    Save a trained model or pipeline.
    """
    path = Path(path)
    joblib.dump(model,path)

    print("Model saved successfully.")
    print(f"Location: {path}")

def load_model(path):
    """
    Load a saved model or pipeline.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Model file not found: {path}")

    model = joblib.load(path)

    print("Model loaded successfully.")
    print(f"Location: {path}")
    return model