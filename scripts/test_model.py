# CardGuard - Raw Data Model Testing

import sys
from pathlib import Path

import pandas as pd
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    average_precision_score,
    roc_auc_score
)


# Project Root
PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))



# CardGuard Imports

from src.config import (TRAIN_RAW_PATH,TEST_RAW_PATH,MODEL_PATH)

from src.feature_engineering import create_features
from src.prediction import load_cardguard_model


# Configuration

THRESHOLD = 0.65


# Main Function

def main():

    print("=" * 70)
    print("CARDGUARD - RAW DATA MODEL TESTING")
    print("=" * 70)

    # 1. Load Raw Training Data
  
    print("\n[1] Loading raw training data...")

    train_df = pd.read_csv(TRAIN_RAW_PATH)

    print(f"Raw training shape: {train_df.shape}")

    # 2. Load Raw Testing Data

    print("\n[2] Loading raw testing data...")

    test_df = pd.read_csv(TEST_RAW_PATH)

    print(f"Raw testing shape: {test_df.shape}")

    # 3. Mark Dataset

    print("\n[3] Marking train and test rows...")

    train_df["_dataset"] = "train"
    test_df["_dataset"] = "test"

    # 4. Combine Data

    print("\n[4] Combining train and test data...")

    combined_df = pd.concat([train_df, test_df],ignore_index=True)

    print(f"Combined shape: {combined_df.shape}")

    # 5. Feature Engineering
  
    print("\n[5] Creating features...")

    featured_df = create_features(combined_df)

    print(f"Feature-engineered shape: {featured_df.shape}")

    # 6. Extract Test Data
    print("\n[6] Extracting test data...")

    if "_dataset" not in featured_df.columns:

        raise ValueError("_dataset column was removed during "
                         "feature engineering.")

    test_featured = featured_df[featured_df["_dataset"] == "test"].copy()

    # 7. Remove Dataset Marker

    test_featured = test_featured.drop(columns=["_dataset"],errors="ignore")

    # 8. Check Target

    print("\n[7] Checking target column...")

    if "is_fraud" not in test_featured.columns:

        raise ValueError("Target column 'is_fraud' was not found.")

    # 9. Separate X and y

    X_test = test_featured.drop(columns=["is_fraud"])

    y_test = test_featured["is_fraud"]


    print(f"X_test shape: {X_test.shape}")

    print(f"y_test shape: {y_test.shape}")

    # 10. Load Saved Model

    print("\n[8] Loading trained CardGuard model...")

    model = load_cardguard_model(MODEL_PATH)

    print("Model loaded successfully.")
    
    # 11. Check Model Feature Compatibility


    print("\n[9] Checking feature compatibility...")

    if hasattr(model, "feature_names_in_"):

        model_features = list(model.feature_names_in_)

        test_features = list(X_test.columns)

        missing_features = [
            col
            for col in model_features
            if col not in test_features
        ]

        extra_features = [
            col
            for col in test_features
            if col not in model_features
        ]

        print(f"Model expects {len(model_features)} features.")

        print(f"Raw test produced {len(test_features)} features.")

        # Missing Features

        if missing_features:

            print("\nMissing features:")

            for col in missing_features:
                print(f"  - {col}")

            raise ValueError(
                "Feature mismatch: test data is missing "
                "features required by the model."
            )



        # Extra Features


        if extra_features:

            print("\nExtra features detected:")

            for col in extra_features:
                print(f"  - {col}")

        # Put Features in Model Order

        X_test = X_test[model_features]

        print("\nFeature compatibility check passed.")



    # 12. Generate Fraud Probability

    print("\n[10] Generating fraud probabilities...")

    y_probability = model.predict_proba(X_test)[:, 1]


    # 13. Apply Threshold
    
    print(f"\n[11] Applying threshold: {THRESHOLD}")

    y_prediction = (y_probability >= THRESHOLD).astype(int)


    # 14. Classification Report

    print("\n" + "=" * 70)
    print("CLASSIFICATION REPORT")
    print("=" * 70)

    print(classification_report( y_test,y_prediction,digits=4))


    # 15. Confusion Matrix


    print("=" * 70)
    print("CONFUSION MATRIX")
    print("=" * 70)

    cm = confusion_matrix( y_test,y_prediction)

    print(cm)


    # 16. TN / FP / FN / TP


    if cm.shape == (2, 2):

        tn, fp, fn, tp = cm.ravel()

        print("\nTrue Negatives :", tn)
        print("False Positives:", fp)
        print("False Negatives:", fn)
        print("True Positives :", tp)



    # 17. PR-AUC

    pr_auc = average_precision_score( y_test,y_probability)

    print("\n" + "=" * 70)
    print("PR-AUC")
    print("=" * 70)

    print(f"{pr_auc:.6f}")


    # 18. ROC-AUC

    roc_auc = roc_auc_score(y_test, y_probability)

    print("\n" + "=" * 70)
    print("ROC-AUC")
    print("=" * 70)

    print( f"{roc_auc:.6f}")

    # 19. Actual Class Distribution

    print("\n" + "=" * 70)
    print("ACTUAL CLASS DISTRIBUTION")
    print("=" * 70)

    print(y_test.value_counts())


    # 20. Prediction Distribution

    print("\n" + "=" * 70)
    print("PREDICTION DISTRIBUTION")
    print("=" * 70)

    print(pd.Series(y_prediction).value_counts())


    # 21. Final Summary

    print("\n" + "=" * 70)
    print("RAW DATA TESTING SUMMARY")
    print("=" * 70)

    print(f"Test samples      : {len(y_test)}")

    print(f"Actual fraud      : {y_test.sum()}")

    print(f"Predicted fraud   : {y_prediction.sum()}")

    print(f"Threshold         : {THRESHOLD}")

    print(f"PR-AUC            : {pr_auc:.6f}")

    print(f"ROC-AUC            : {roc_auc:.6f}")

    print( "\nRaw data testing completed successfully.")


# Run Script

if __name__ == "__main__":
    main()