import sys
from pathlib import Path

import pandas as pd

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    average_precision_score
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))


from src.config import MODEL_PATH, VALIDATION_FEATURES_PATH
from src.prediction import load_cardguard_model


# ============================================================
# Thresholds to Test
# ============================================================

THRESHOLDS = [
    0.10,
    0.20,
    0.30,
    0.40,
    0.45,
    0.50,
    0.55,
    0.60,
    0.61,
    0.62,
    0.63,
    0.64,
    0.65,
    0.66,
    0.67,
    0.68,
    0.69,
    0.70,
    0.71,
    0.75,
    0.80,
    0.85,
    0.90,
    0.95
]


# ============================================================
# Main Function
# ============================================================

def main():

    print("=" * 70)
    print("CARDGUARD - THRESHOLD ANALYSIS")
    print("=" * 70)


 
    # 1. Load Validation Data


    print("\n[1] Loading validation data...")

    validation_df = pd.read_csv(VALIDATION_FEATURES_PATH)

    print(f"Validation dataset shape: {validation_df.shape}")
    
    # 2. Check Target


    print("\n[2] Checking target column...")

    if "is_fraud" not in validation_df.columns:

        raise ValueError(
            "Target column 'is_fraud' was not found "
            "in validation data."
        )

    # 3. Separate Features and Target
    
    X_validation = validation_df.drop(columns=["is_fraud"])

    y_validation = validation_df["is_fraud"]

    print(f"X_validation shape: {X_validation.shape}")

    print(f"y_validation shape: {y_validation.shape}")

    # 4. Load Trained Model


    print("\n[3] Loading trained model...")

    model = load_cardguard_model(MODEL_PATH)

    print("Model loaded successfully.")

    # 5. Generate Probabilities

    print("\n[4] Generating validation probabilities...")

    y_probability = model.predict_proba(X_validation)[:, 1]

    print("Probabilities generated successfully.")

    # 6. Calculate PR-AUC

    pr_auc = average_precision_score(y_validation,y_probability)

    print(f"\nValidation PR-AUC: {pr_auc:.6f}")

    # 7. Threshold Analysis

    print("\n" + "=" * 70)
    print("THRESHOLD COMPARISON")
    print("=" * 70)

    results = []


    for threshold in THRESHOLDS:
        # Convert probability to prediction   
        y_prediction = (y_probability >= threshold).astype(int)

        # Metrics
        precision = precision_score(y_validation,y_prediction,zero_division=0)
        recall = recall_score(y_validation,y_prediction,zero_division=0)
        f1 = f1_score(y_validation,y_prediction,zero_division=0)

        # Store results

        results.append({
            "threshold": threshold,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "predicted_fraud": y_prediction.sum()
        })

    # 8. Create Results DataFrame

    results_df = pd.DataFrame(results)

    # 9. Display Results
    print(results_df.to_string(index=False,
            formatters={
                "threshold": "{:.2f}".format,
                "precision": "{:.4f}".format,
                "recall": "{:.4f}".format,
                "f1_score": "{:.4f}".format}))

    # 10. Best F1 Threshold
    
    best_f1_row = results_df.loc[results_df["f1_score"].idxmax()]

    print("\n" + "=" * 70)
    print("BEST F1 THRESHOLD")
    print("=" * 70)
    print(
        f"Threshold       : "f"{best_f1_row['threshold']:.2f}")

    print(f"Precision       : "f"{best_f1_row['precision']:.4f}")

    print(f"Recall          : "f"{best_f1_row['recall']:.4f}")

    print(
        f"F1 Score        : "
        f"{best_f1_row['f1_score']:.4f}")

    print(
        f"Predicted fraud : "
        f"{int(best_f1_row['predicted_fraud'])}")


    # 11. Best Precision with Recall >= 90%

    minimum_recall = 0.90
    eligible = results_df[
        results_df["recall"] >= minimum_recall
    ]

    if not eligible.empty:

        best_precision_row = eligible.loc[
            eligible["precision"].idxmax()
        ]

        print("\n" + "=" * 70)
        print(
            "BEST PRECISION WITH RECALL >= 90%"
        )
        print("=" * 70)

        print(
            f"Threshold       : "
            f"{best_precision_row['threshold']:.2f}"
        )

        print(
            f"Precision       : "
            f"{best_precision_row['precision']:.4f}"
        )

        print(
            f"Recall          : "
            f"{best_precision_row['recall']:.4f}"
        )

        print(
            f"F1 Score        : "
            f"{best_precision_row['f1_score']:.4f}"
        )

        print(
            f"Predicted fraud : "
            f"{int(best_precision_row['predicted_fraud'])}"
        )

    # 12. Save Threshold Results

    output_path = (
        PROJECT_ROOT
        / "data"
        / "processed"
        / "threshold_analysis.csv"
    )

    results_df.to_csv(
        output_path,
        index=False
    )

    print("\n" + "=" * 70)
    print("RESULTS SAVED")
    print("=" * 70)

    print(
        f"Saved to: {output_path}"
    )




    print("\nThreshold analysis completed successfully.")


if __name__ == "__main__":
    main()