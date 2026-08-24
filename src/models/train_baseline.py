import pandas as pd
import numpy as np

from pathlib import Path

from lightgbm import LGBMClassifier
from sklearn.metrics import (
    precision_score,
    recall_score,
    average_precision_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)


# ============================================================
# CONFIGURATION
# ============================================================

DATA_DIR = Path("data/processed")

TRAIN_FILE = DATA_DIR / "train.csv"
VALIDATION_FILE = DATA_DIR / "validation.csv"
TEST_FILE = DATA_DIR / "test.csv"


# ============================================================
# LOAD DATA
# ============================================================

def load_datasets():

    print("Loading datasets...")

    train = pd.read_csv(TRAIN_FILE)
    validation = pd.read_csv(VALIDATION_FILE)
    test = pd.read_csv(TEST_FILE)

    return train, validation, test


# ============================================================
# PREPARE FEATURES
# ============================================================

def prepare_features(
    train,
    validation,
    test
):

    print("Preparing model features...")

    train = train.copy()
    validation = validation.copy()
    test = test.copy()

    # --------------------------------------------------------
    # Convert categorical columns
    # --------------------------------------------------------

    categorical_columns = [
        "payment_method",
        "category",
        "size_variant"
    ]

    for column in categorical_columns:

        # Same category definition across all datasets
        combined = pd.concat(
            [
                train[column],
                validation[column],
                test[column]
            ]
        ).astype("category")

        categories = combined.cat.categories

        train[column] = pd.Categorical(
            train[column],
            categories=categories
        )

        validation[column] = pd.Categorical(
            validation[column],
            categories=categories
        )

        test[column] = pd.Categorical(
            test[column],
            categories=categories
        )

    # --------------------------------------------------------
    # Convert timestamps into useful numeric features
    # --------------------------------------------------------

    for df in [
        train,
        validation,
        test
    ]:

        df["order_hour"] = pd.to_datetime(
            df["order_ts"]
        ).dt.hour

        df["order_day_of_week"] = pd.to_datetime(
            df["order_ts"]
        ).dt.dayofweek

        df["order_month"] = pd.to_datetime(
            df["order_ts"]
        ).dt.month

        df["order_day"] = pd.to_datetime(
            df["order_ts"]
        ).dt.day

    # --------------------------------------------------------
    # Remove columns that should NOT enter the model
    # --------------------------------------------------------

    columns_to_drop = [
        "order_id",
        "customer_id",
        "product_id",
        "order_ts",
        "signup_date",
        "returned"
    ]

    X_train = train.drop(
        columns=columns_to_drop
    )

    X_validation = validation.drop(
        columns=columns_to_drop
    )

    X_test = test.drop(
        columns=columns_to_drop
    )

    y_train = train["returned"]
    y_validation = validation["returned"]
    y_test = test["returned"]

    return (
        X_train,
        X_validation,
        X_test,
        y_train,
        y_validation,
        y_test
    )


# ============================================================
# TRAIN MODEL
# ============================================================

def train_model(
    X_train,
    y_train
):

    print("\nTraining LightGBM baseline...")

    # Calculate imbalance ratio
    negative = (y_train == 0).sum()
    positive = (y_train == 1).sum()

    scale_pos_weight = (
        negative / positive
    )

    print(
        f"Negative examples: {negative:,}"
    )

    print(
        f"Positive examples: {positive:,}"
    )

    print(
        f"Scale positive weight: "
        f"{scale_pos_weight:.2f}"
    )

    model = LGBMClassifier(

        objective="binary",

        n_estimators=400,

        learning_rate=0.05,

        num_leaves=31,

        max_depth=-1,

        subsample=0.8,

        colsample_bytree=0.8,

        random_state=42,

        scale_pos_weight=scale_pos_weight,

        verbosity=-1
    )

    model.fit(
        X_train,
        y_train,

        categorical_feature=[
            "payment_method",
            "category",
            "size_variant"
        ]
    )

    return model


# ============================================================
# EVALUATION
# ============================================================

def evaluate_model(
    model,
    X,
    y,
    dataset_name,
    threshold=0.5
):

    print("\n" + "=" * 70)
    print(
        f"{dataset_name.upper()} EVALUATION"
    )
    print("=" * 70)

    probabilities = model.predict_proba(
        X
    )[:, 1]

    predictions = (
        probabilities >= threshold
    ).astype(int)

    precision = precision_score(
        y,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        y,
        predictions,
        zero_division=0
    )

    pr_auc = average_precision_score(
        y,
        probabilities
    )

    roc_auc = roc_auc_score(
        y,
        probabilities
    )

    cm = confusion_matrix(
        y,
        predictions
    )

    print(
        f"Threshold : {threshold:.2f}"
    )

    print(
        f"Precision : {precision:.4f}"
    )

    print(
        f"Recall    : {recall:.4f}"
    )

    print(
        f"PR-AUC    : {pr_auc:.4f}"
    )

    print(
        f"ROC-AUC   : {roc_auc:.4f}"
    )

    print("\nConfusion Matrix:")

    print(cm)

    print("\nClassification Report:")

    print(
        classification_report(
            y,
            predictions,
            zero_division=0
        )
    )

    return {
        "precision": precision,
        "recall": recall,
        "pr_auc": pr_auc,
        "roc_auc": roc_auc
    }


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("RETURN-RISK INTELLIGENCE")
    print("BASELINE LIGHTGBM MODEL")
    print("=" * 70)

    # --------------------------------------------------------
    # Load
    # --------------------------------------------------------

    train, validation, test = (
        load_datasets()
    )

    print(
        f"\nTrain rows      : {len(train):,}"
    )

    print(
        f"Validation rows : {len(validation):,}"
    )

    print(
        f"Test rows       : {len(test):,}"
    )

    # --------------------------------------------------------
    # Prepare
    # --------------------------------------------------------

    (
        X_train,
        X_validation,
        X_test,
        y_train,
        y_validation,
        y_test
    ) = prepare_features(
        train,
        validation,
        test
    )

    print(
        f"\nTraining features: "
        f"{X_train.shape[1]}"
    )

    # --------------------------------------------------------
    # Train
    # --------------------------------------------------------

    model = train_model(
        X_train,
        y_train
    )

    # --------------------------------------------------------
    # Validation
    # --------------------------------------------------------

    validation_results = evaluate_model(
        model,
        X_validation,
        y_validation,
        "validation",
        threshold=0.5
    )

    # --------------------------------------------------------
    # Test
    # --------------------------------------------------------

    test_results = evaluate_model(
        model,
        X_test,
        y_test,
        "test",
        threshold=0.5
    )

    # --------------------------------------------------------
    # Feature importance
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("TOP FEATURE IMPORTANCE")
    print("=" * 70)

    importance = pd.DataFrame({
        "feature": X_train.columns,
        "importance": model.feature_importances_
    })

    importance = importance.sort_values(
        "importance",
        ascending=False
    )

    print(
        importance.head(15).to_string(
            index=False
        )
    )

    print("\n" + "=" * 70)
    print("BASELINE MODEL COMPLETE 🚀")
    print("=" * 70)


if __name__ == "__main__":
    main()