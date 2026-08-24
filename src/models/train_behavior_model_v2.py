import pandas as pd
import numpy as np
import lightgbm as lgb

from pathlib import Path

from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    precision_score,
    recall_score,
    confusion_matrix,
    classification_report
)


# ============================================================
# CONFIGURATION
# ============================================================

DATA_DIR = Path("data/processed")
MODEL_DIR = Path("models")

TRAIN_FILE = DATA_DIR / "train.csv"
VALIDATION_FILE = DATA_DIR / "validation.csv"
TEST_FILE = DATA_DIR / "test.csv"

MODEL_FILE = MODEL_DIR / "behaviour_model_v2.txt"


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

def prepare_features(train, validation, test):

    print("Preparing behaviour-model features...")

    target = "returned"

    excluded_columns = [
        "returned",
        "order_id",
        "customer_id",
        "product_id",
        "order_ts",
        "signup_date",
        "behavior_profile"
    ]

    feature_columns = [
        column
        for column in train.columns
        if column not in excluded_columns
    ]

    feature_columns = [
        column
        for column in feature_columns
        if (
            column in validation.columns
            and
            column in test.columns
        )
    ]

    # ========================================================
    # Detect categorical columns
    # ========================================================

    categorical_columns = []

    for column in feature_columns:

        if (
            train[column].dtype == "object"
            or
            validation[column].dtype == "object"
            or
            test[column].dtype == "object"
        ):

            categorical_columns.append(column)

    print("\nCategorical columns detected:")

    if categorical_columns:

        for column in categorical_columns:
            print(f"  ✓ {column}")

    else:

        print("  None")

    # ========================================================
    # Encode categorical columns
    # ========================================================

    for column in categorical_columns:

        combined = pd.concat(
            [
                train[column],
                validation[column],
                test[column]
            ],
            ignore_index=True
        )

        combined = (
            combined
            .fillna("__MISSING__")
            .astype(str)
        )

        categories = pd.Index(
            combined.unique()
        )

        mapping = {
            value: index
            for index, value
            in enumerate(categories)
        }

        train[column] = (
            train[column]
            .fillna("__MISSING__")
            .astype(str)
            .map(mapping)
            .fillna(-1)
            .astype("int32")
        )

        validation[column] = (
            validation[column]
            .fillna("__MISSING__")
            .astype(str)
            .map(mapping)
            .fillna(-1)
            .astype("int32")
        )

        test[column] = (
            test[column]
            .fillna("__MISSING__")
            .astype(str)
            .map(mapping)
            .fillna(-1)
            .astype("int32")
        )

    # ========================================================
    # Convert all model features to numeric
    # ========================================================

    for dataframe in [
        train,
        validation,
        test
    ]:

        dataframe[feature_columns] = (
            dataframe[feature_columns]
            .replace(
                [np.inf, -np.inf],
                np.nan
            )
        )

        for column in feature_columns:

            dataframe[column] = pd.to_numeric(
                dataframe[column],
                errors="coerce"
            )

        dataframe[feature_columns] = (
            dataframe[feature_columns]
            .fillna(0)
        )

    # ========================================================
    # Force numpy arrays
    #
    # This avoids the LightGBM validation-data error.
    # ========================================================

    X_train = (
        train[feature_columns]
        .to_numpy(dtype=np.float32)
    )

    y_train = (
        train[target]
        .to_numpy(dtype=np.int32)
    )

    X_validation = (
        validation[feature_columns]
        .to_numpy(dtype=np.float32)
    )

    y_validation = (
        validation[target]
        .to_numpy(dtype=np.int32)
    )

    X_test = (
        test[feature_columns]
        .to_numpy(dtype=np.float32)
    )

    y_test = (
        test[target]
        .to_numpy(dtype=np.int32)
    )

    return (
        X_train,
        y_train,
        X_validation,
        y_validation,
        X_test,
        y_test,
        feature_columns,
        categorical_columns
    )


# ============================================================
# TRAIN MODEL
# ============================================================

def train_model(
    X_train,
    y_train
):

    print("\nTraining Behaviour-Aware LightGBM v2...")

    negative_count = int(
        (y_train == 0).sum()
    )

    positive_count = int(
        (y_train == 1).sum()
    )

    scale_positive_weight = (
        negative_count
        /
        positive_count
    )

    print(
        f"Negative examples: "
        f"{negative_count:,}"
    )

    print(
        f"Positive examples: "
        f"{positive_count:,}"
    )

    print(
        f"Scale positive weight: "
        f"{scale_positive_weight:.2f}"
    )

    # ========================================================
    # LightGBM
    # ========================================================

    model = lgb.LGBMClassifier(

        objective="binary",

        n_estimators=400,

        learning_rate=0.03,

        num_leaves=31,

        max_depth=-1,

        min_child_samples=40,

        subsample=0.8,

        colsample_bytree=0.8,

        reg_alpha=0.1,

        reg_lambda=1.0,

        scale_pos_weight=scale_positive_weight,

        random_state=42,

        n_jobs=-1,

        verbosity=-1
    )

    # ========================================================
    # IMPORTANT
    #
    # No eval_set here.
    #
    # Your installed LightGBM version is rejecting the
    # validation-list format. We will evaluate validation
    # manually after training.
    # ========================================================

    model.fit(
        X_train,
        y_train
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
    threshold=0.50
):

    probabilities = (
        model.predict_proba(X)[:, 1]
    )

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

    matrix = confusion_matrix(
        y,
        predictions
    )

    print("\n" + "=" * 70)
    print(
        f"{dataset_name.upper()} EVALUATION"
    )
    print("=" * 70)

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

    print(matrix)

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
# FEATURE IMPORTANCE
# ============================================================

def show_feature_importance(
    model,
    feature_columns
):

    print("\n" + "=" * 70)
    print("TOP FEATURE IMPORTANCE")
    print("=" * 70)

    importance = pd.DataFrame({

        "feature": feature_columns,

        "importance": model.feature_importances_

    })

    importance = importance.sort_values(
        "importance",
        ascending=False
    )

    print(
        importance.head(25)
        .to_string(index=False)
    )

    return importance


# ============================================================
# BEHAVIOUR FEATURE IMPORTANCE
# ============================================================

def show_behaviour_importance(
    importance
):

    behaviour_keywords = [
        "return_rate_shift",
        "recent_return_rate",
        "previous_return_rate",
        "return_rate_ratio",
        "recent_avg_value",
        "previous_avg_value",
        "order_value_shift",
        "recent_order_frequency"
    ]

    mask = importance["feature"].apply(
        lambda feature:
        any(
            keyword in feature
            for keyword in behaviour_keywords
        )
    )

    behaviour = importance[mask]

    print("\n" + "=" * 70)
    print("BEHAVIOUR FEATURE IMPORTANCE")
    print("=" * 70)

    if len(behaviour) == 0:

        print(
            "No behaviour features found."
        )

    else:

        print(
            behaviour.to_string(
                index=False
            )
        )


# ============================================================
# SAVE MODEL
# ============================================================

def save_model(model):

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    model.booster_.save_model(
        str(MODEL_FILE)
    )

    print(
        f"\nModel saved: {MODEL_FILE}"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("RETURN-RISK INTELLIGENCE")
    print("BEHAVIOUR-AWARE LIGHTGBM MODEL V2")
    print("=" * 70)

    # ========================================================
    # Load
    # ========================================================

    train, validation, test = (
        load_datasets()
    )

    print(
        f"\nTrain rows      : "
        f"{len(train):,}"
    )

    print(
        f"Validation rows : "
        f"{len(validation):,}"
    )

    print(
        f"Test rows       : "
        f"{len(test):,}"
    )

    # ========================================================
    # Prepare
    # ========================================================

    (
        X_train,
        y_train,
        X_validation,
        y_validation,
        X_test,
        y_test,
        feature_columns,
        categorical_columns
    ) = prepare_features(
        train,
        validation,
        test
    )

    print(
        f"\nTraining features: "
        f"{len(feature_columns)}"
    )

    print(
        f"Categorical features: "
        f"{len(categorical_columns)}"
    )

    # ========================================================
    # Behaviour features
    # ========================================================

    behaviour_features = [
        feature
        for feature in feature_columns
        if (
            "return_rate_shift" in feature
            or
            "recent_return_rate" in feature
            or
            "previous_return_rate" in feature
            or
            "return_rate_ratio" in feature
            or
            "order_value_shift" in feature
            or
            "recent_avg_value" in feature
            or
            "previous_avg_value" in feature
            or
            "recent_order_frequency" in feature
        )
    ]

    print(
        "\nBehaviour features included:"
    )

    for feature in behaviour_features:

        print(
            f"  ✓ {feature}"
        )

    # ========================================================
    # Train
    # ========================================================

    model = train_model(
        X_train,
        y_train
    )

    # ========================================================
    # Validation
    # ========================================================

    validation_results = evaluate_model(
        model,
        X_validation,
        y_validation,
        "Validation"
    )

    # ========================================================
    # Test
    # ========================================================

    test_results = evaluate_model(
        model,
        X_test,
        y_test,
        "Test"
    )

    # ========================================================
    # Feature importance
    # ========================================================

    importance = show_feature_importance(
        model,
        feature_columns
    )

    show_behaviour_importance(
        importance
    )

    # ========================================================
    # Baseline comparison
    # ========================================================

    baseline_pr_auc = 0.1479
    baseline_roc_auc = 0.6011

    print("\n" + "=" * 70)
    print("BASELINE vs BEHAVIOUR MODEL V2")
    print("=" * 70)

    print(
        f"\nBaseline PR-AUC  : "
        f"{baseline_pr_auc:.4f}"
    )

    print(
        f"Model V2 PR-AUC  : "
        f"{test_results['pr_auc']:.4f}"
    )

    print(
        f"PR-AUC change    : "
        f"{test_results['pr_auc'] - baseline_pr_auc:+.4f}"
    )

    print(
        f"\nBaseline ROC-AUC : "
        f"{baseline_roc_auc:.4f}"
    )

    print(
        f"Model V2 ROC-AUC : "
        f"{test_results['roc_auc']:.4f}"
    )

    print(
        f"ROC-AUC change   : "
        f"{test_results['roc_auc'] - baseline_roc_auc:+.4f}"
    )

    # ========================================================
    # Save
    # ========================================================

    save_model(model)

    print("\n" + "=" * 70)
    print("BEHAVIOUR MODEL V2 COMPLETE 🚀")
    print("=" * 70)


if __name__ == "__main__":
    main()