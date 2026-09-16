
# CardGuard - Prediction
import pandas as pd
from src.config import MODEL_PATH, DEFAULT_THRESHOLD
from src.utils import load_model

# Load Trained Model


def load_cardguard_model(model_path=MODEL_PATH):
    """
    Load the trained CardGuard XGBoost pipeline.
    """
    model = load_model(model_path)

    return model


# Predict Transactions

def predict(model, X, threshold=DEFAULT_THRESHOLD):
    """
    Predict fraud probability and fraud/legitimate class.

    Parameters
    ----------
    model : trained pipeline
    X : pandas DataFrame
        Feature-engineered transaction data.
    threshold : float
        Probability threshold for fraud.

    Returns
    -------
    pandas.DataFrame
    """

    # Fraud probability
    fraud_probability = model.predict_proba(X)[:, 1]

    # Apply threshold
    prediction = (fraud_probability >= threshold).astype(int)

    # Results
    results = pd.DataFrame({
        "fraud_probability": fraud_probability,
        "prediction": prediction})

    # Human-readable result
    results["prediction_label"] = results["prediction"].map({0: "LEGITIMATE",1: "FRAUD"})

    return results


# Predict One Transaction

def predict_single(model,transaction,
    threshold=DEFAULT_THRESHOLD):
    """
    Predict a single transaction.
    """
    # Dictionary
    if isinstance(transaction, dict):
        transaction = pd.DataFrame([transaction])

    # Series
    elif isinstance(transaction, pd.Series):
        transaction = transaction.to_frame().T

    # DataFrame
    elif isinstance(transaction, pd.DataFrame):
        transaction = transaction.copy()

    else:
        raise TypeError(
            "Transaction must be a dictionary, "
            "pandas Series, or pandas DataFrame."
        )

    # Make prediction
    result = predict(model=model,
        X=transaction,
        threshold=threshold)

    return result.iloc[0].to_dict()