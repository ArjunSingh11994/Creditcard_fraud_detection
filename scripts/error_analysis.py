import sys
from pathlib import Path
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))


from src.config import TRAIN_RAW_PATH,TEST_RAW_PATH,MODEL_PATH,DEFAULT_THRESHOLD
from src.feature_engineering import create_features
from src.prediction import load_cardguard_model

# Main Function
def main():

    print("=" * 70)
    print("CARDGUARD - ERROR ANALYSIS")
    print("=" * 70)

    print(f"\nThreshold: {DEFAULT_THRESHOLD}")

    # 1. Load Raw Training Data

    print("\n[1] Loading raw training data...")

    train_df = pd.read_csv(TRAIN_RAW_PATH)

    print(f"Training shape: {train_df.shape}")

    # 2. Load Raw Test Data

    print("\n[2] Loading raw test data...")

    test_df = pd.read_csv(TEST_RAW_PATH)

    print(f"Test shape: {test_df.shape}")

    # 3. Preserve Original Test Row ID

    test_df["_raw_test_id"] = range(len(test_df))

    train_df["_dataset"] = "train"
    test_df["_dataset"] = "test"

    # 4. Combine Data

    print("\n[3] Combining datasets...")

    combined_df = pd.concat([train_df, test_df],ignore_index=True)

    # 5. Feature Engineering

    print("\n[4] Creating features...")

    featured_df = create_features( combined_df )

    print(
        f"Feature-engineered shape: "
        f"{featured_df.shape}"
    )

    # 6. Extract Test Rows

    print("\n[5] Extracting test transactions...")

    test_featured = featured_df[ featured_df["_dataset"] == "test"].copy()
    
    # 7. Sort Back to Original Test Order

    test_featured = test_featured.sort_values( "_raw_test_id").reset_index(drop=True)


    # 8. Separate Target
 

    if "is_fraud" not in test_featured.columns:

        raise ValueError("Target column 'is_fraud' not found.")

    y_test = test_featured["is_fraud"].astype(int)


    X_test = test_featured.drop(columns=["is_fraud"], errors="ignore")

    # 9. Load Model

    print("\n[6] Loading trained model...")

    model = load_cardguard_model(MODEL_PATH)

    print("Model loaded successfully.")

    # 10. Match Model Features

    print("\n[7] Checking model features...")

    if hasattr(model, "feature_names_in_"):

        model_features = list(model.feature_names_in_)

        missing_features = [
            col
            for col in model_features
            if col not in X_test.columns
        ]

        if missing_features:

            print("\nMissing features:")

            for col in missing_features:
                print(f"  - {col}")

            raise ValueError(
                "Required model features are missing."
            )

        X_model = X_test[model_features]

    else:
        X_model = X_test

    # 11. Generate Probabilities

    print("\n[8] Generating predictions...")

    probabilities = model.predict_proba(X_model)[:, 1]

    # 12. Apply Threshold

    predictions = (probabilities >= DEFAULT_THRESHOLD).astype(int)


    # 13. Create Error Analysis Dataset

    print("\n[9] Creating error analysis dataset...")

    analysis_df = test_featured.copy()

    analysis_df["fraud_probability"] = ( probabilities)

    analysis_df["prediction"] = (predictions)


    # 14. Error Type

    analysis_df["error_type"] = "CORRECT"

    analysis_df.loc[(analysis_df["is_fraud"] == 0) & (analysis_df["prediction"] == 1),
        "error_type"
    ] = "FALSE_POSITIVE"

    analysis_df.loc[
        (analysis_df["is_fraud"] == 1)
        & (analysis_df["prediction"] == 0),
        "error_type"
    ] = "FALSE_NEGATIVE"

    analysis_df.loc[
        (analysis_df["is_fraud"] == 0)
        & (analysis_df["prediction"] == 0),
        "error_type"
    ] = "TRUE_NEGATIVE"

    analysis_df.loc[
        (analysis_df["is_fraud"] == 1)
        & (analysis_df["prediction"] == 1),
        "error_type"
    ] = "TRUE_POSITIVE"


    # 15. Error Counts

    print("\n" + "=" * 70)
    print("ERROR COUNTS")
    print("=" * 70)

    print(analysis_df["error_type"].value_counts())


    # 16. False Positives

    false_positives = analysis_df[analysis_df["error_type"] == "FALSE_POSITIVE"].copy()

    print("\n" + "=" * 70)
    print("FALSE POSITIVES")
    print("=" * 70)

    print(f"Total false positives: "f"{len(false_positives)}")


    # 17. False Negatives

    false_negatives = analysis_df[analysis_df["error_type"] == "FALSE_NEGATIVE"].copy()

    print("\n" + "=" * 70)
    print("FALSE NEGATIVES")
    print("=" * 70)

    print(f"Total false negatives: "
        f"{len(false_negatives)}")


    # 18. Most Confident False Positives

    print("\n" + "=" * 70)
    print("TOP FALSE POSITIVES")
    print("=" * 70)

    fp_display_columns = [
        "_raw_test_id",
        "amt",
        "merchant",
        "category",
        "city",
        "state",
        "gender",
        "age",
        "fraud_probability"
    ]

    fp_display_columns = [
        col
        for col in fp_display_columns
        if col in false_positives.columns]

    if len(false_positives) > 0:
        print( false_positives[fp_display_columns].sort_values("fraud_probability",ascending=False)
              .head(20).to_string(index=False))

    # 19. Most Confident False Negatives
    print("\n" + "=" * 70)
    print("TOP FALSE NEGATIVES")
    print("=" * 70)

    fn_display_columns = [
        "_raw_test_id",
        "amt",
        "merchant",
        "category",
        "city",
        "state",
        "gender",
        "age",
        "fraud_probability"
    ]

    fn_display_columns = [
        col
        for col in fn_display_columns
        if col in false_negatives.columns
    ]

    if len(false_negatives) > 0:
        print(false_negatives[ fn_display_columns ].sort_values(
                "fraud_probability",
                ascending=True).head(20).to_string(index=False))


    # 20. Amount Analysis

    print("\n" + "=" * 70)
    print("AVERAGE TRANSACTION AMOUNT")
    print("=" * 70)

    print(analysis_df.groupby("error_type")["amt"].mean().round(2))

    # 21. Category Analysis

    if "category" in analysis_df.columns:
        print("\n" + "=" * 70)
        print("ERRORS BY CATEGORY")
        print("=" * 70)
        category_errors = pd.crosstab(analysis_df["category"],analysis_df["error_type"])

        print(category_errors.to_string())

    # 22. Merchant Analysis

    if "merchant" in analysis_df.columns:

        print("\n" + "=" * 70)
        print("FALSE POSITIVES BY MERCHANT")
        print("=" * 70)

        fp_merchants = (false_positives["merchant"].value_counts().head(20))

        print(fp_merchants.to_string())



    # 23. Save All Errors

    errors_only = analysis_df[
        analysis_df["error_type"].isin([
            "FALSE_POSITIVE",
            "FALSE_NEGATIVE"
        ])
    ].copy()


    errors_path = (
        PROJECT_ROOT
        / "data"
        / "processed"
        / "model_errors.csv"
    )

    errors_only.to_csv(
        errors_path,
        index=False
    )


    # ========================================================
    # 24. Save False Positives
    # ========================================================

    fp_path = (
        PROJECT_ROOT
        / "data"
        / "processed"
        / "false_positives.csv"
    )

    false_positives.to_csv(
        fp_path,
        index=False
    )


    # ========================================================
    # 25. Save False Negatives
    # ========================================================

    fn_path = (
        PROJECT_ROOT
        / "data"
        / "processed"
        / "false_negatives.csv"
    )

    false_negatives.to_csv(
        fn_path,
        index=False
    )

    # 26. Final Summary


    print("\n" + "=" * 70)
    print("ERROR ANALYSIS SUMMARY")
    print("=" * 70)

    print(
        f"Threshold       : {DEFAULT_THRESHOLD}"
    )

    print(
        f"False positives : {len(false_positives)}"
    )

    print(
        f"False negatives : {len(false_negatives)}"
    )

    print(
        f"\nAll errors saved to:"
        f"\n{errors_path}"
    )

    print(
        f"\nFalse positives saved to:"
        f"\n{fp_path}"
    )

    print(
        f"\nFalse negatives saved to:"
        f"\n{fn_path}"
    )

    print(
        "\nError analysis completed successfully."
    )



if __name__ == "__main__":
    main()