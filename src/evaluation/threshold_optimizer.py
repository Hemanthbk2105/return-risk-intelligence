import pandas as pd
import numpy as np
import lightgbm as lgb

from pathlib import Path

from sklearn.metrics import (
    precision_score,
    recall_score,
    average_precision_score,
    roc_auc_score
)


# ============================================================
# CONFIGURATION
# ============================================================

DATA_DIR = Path("data/processed")
MODEL_DIR = Path("models")

TEST_FILE = DATA_DIR / "test.csv"
MODEL_FILE = MODEL_DIR / "behaviour_model_v2.txt"


# ============================================================
# EXCLUDED COLUMNS
# ============================================================

EXCLUDED_COLUMNS = [
    "returned",
    "order_id",
    "customer_id",
    "product_id",
    "order_ts",
    "signup_date",
    "behavior_profile"
]


# ============================================================
# LOAD TEST DATA
# ============================================================

def load_test_data():

    print("=" * 70)
    print("RETURN-RISK INTELLIGENCE")
    print("THRESHOLD & INTERVENTION ANALYSIS")
    print("=" * 70)

    print("\nLoading test dataset...")

    test = pd.read_csv(
        TEST_FILE
    )

    print(
        f"Test rows: {len(test):,}"
    )

    return test


# ============================================================
# PREPARE FEATURES
# ============================================================

def prepare_features(test):

    print("\nPreparing model features...")

    feature_columns = [
        column
        for column in test.columns
        if column not in EXCLUDED_COLUMNS
    ]

    # --------------------------------------------------------
    # Convert categorical columns
    # --------------------------------------------------------

    categorical_columns = []

    for column in feature_columns:

        if test[column].dtype == "object":

            categorical_columns.append(
                column
            )

    print(
        f"Categorical columns: "
        f"{len(categorical_columns)}"
    )

    # --------------------------------------------------------
    # Encode categories
    # --------------------------------------------------------

    for column in categorical_columns:

        test[column] = (
            test[column]
            .fillna("__MISSING__")
            .astype("category")
            .cat.codes
        )

    # --------------------------------------------------------
    # Clean numerical columns
    # --------------------------------------------------------

    test[feature_columns] = (
        test[feature_columns]
        .replace(
            [np.inf, -np.inf],
            np.nan
        )
    )

    for column in feature_columns:

        test[column] = pd.to_numeric(
            test[column],
            errors="coerce"
        )

    test[feature_columns] = (
        test[feature_columns]
        .fillna(0)
    )

    X_test = test[
        feature_columns
    ].to_numpy(
        dtype=np.float32
    )

    y_test = test[
        "returned"
    ].to_numpy(
        dtype=np.int32
    )

    return (
        test,
        X_test,
        y_test,
        feature_columns
    )


# ============================================================
# LOAD MODEL
# ============================================================

def load_model():

    print("\nLoading trained model...")

    model = lgb.Booster(
        model_file=str(
            MODEL_FILE
        )
    )

    print("Model loaded successfully ✅")

    return model


# ============================================================
# GENERATE RISK SCORES
# ============================================================

def generate_predictions(
    model,
    X_test
):

    print("\nGenerating return-risk scores...")

    probabilities = model.predict(
        X_test
    )

    print(
        f"Minimum risk : "
        f"{probabilities.min():.4f}"
    )

    print(
        f"Maximum risk : "
        f"{probabilities.max():.4f}"
    )

    print(
        f"Average risk : "
        f"{probabilities.mean():.4f}"
    )

    return probabilities


# ============================================================
# THRESHOLD ANALYSIS
# ============================================================

def threshold_analysis(
    test,
    probabilities,
    y_test
):

    print("\n" + "=" * 70)
    print("THRESHOLD ANALYSIS")
    print("=" * 70)

    thresholds = [
        0.10,
        0.15,
        0.20,
        0.25,
        0.30,
        0.35,
        0.40,
        0.45,
        0.50,
        0.55,
        0.60,
        0.65,
        0.70,
        0.75,
        0.80
    ]

    rows = []

    for threshold in thresholds:

        predictions = (
            probabilities >= threshold
        ).astype(int)

        selected = predictions.sum()

        if selected == 0:

            precision = 0
            recall = 0

        else:

            precision = precision_score(
                y_test,
                predictions,
                zero_division=0
            )

            recall = recall_score(
                y_test,
                predictions,
                zero_division=0
            )

        selection_rate = (
            selected
            /
            len(y_test)
        )

        rows.append({

            "threshold": threshold,

            "orders_selected": int(
                selected
            ),

            "selection_rate": (
                selection_rate
            ),

            "precision": precision,

            "recall": recall
        })

    result = pd.DataFrame(
        rows
    )

    print(
        result.to_string(
            index=False
        )
    )

    return result


# ============================================================
# TOP-K ANALYSIS
# ============================================================

def top_k_analysis(
    test,
    probabilities,
    y_test
):

    print("\n" + "=" * 70)
    print("TOP-K INTERVENTION ANALYSIS")
    print("=" * 70)

    percentages = [
        0.05,
        0.10,
        0.15,
        0.20,
        0.25,
        0.30
    ]

    ranking = np.argsort(
        probabilities
    )[::-1]

    rows = []

    total_returns = int(
        y_test.sum()
    )

    for percentage in percentages:

        k = max(
            1,
            int(
                len(y_test)
                *
                percentage
            )
        )

        selected_indices = (
            ranking[:k]
        )

        selected_returns = int(
            y_test[
                selected_indices
            ].sum()
        )

        precision = (
            selected_returns
            /
            k
        )

        recall = (
            selected_returns
            /
            total_returns
            if total_returns > 0
            else 0
        )

        rows.append({

            "top_percent": (
                percentage * 100
            ),

            "orders_selected": k,

            "returns_captured": (
                selected_returns
            ),

            "precision": precision,

            "recall": recall
        })

    result = pd.DataFrame(
        rows
    )

    print(
        result.to_string(
            index=False
        )
    )

    return result


# ============================================================
# EXPECTED RETURN COST
# ============================================================

def expected_cost_analysis(
    test,
    probabilities
):

    print("\n" + "=" * 70)
    print("EXPECTED RETURN COST ANALYSIS")
    print("=" * 70)

    # --------------------------------------------------------
    # Synthetic business assumption
    #
    # This is NOT Razorpay's real cost.
    #
    # We use a transparent modelling assumption:
    #
    # return cost = 8% of order value
    #
    # This represents logistics / processing / operational
    # cost in our synthetic environment.
    # --------------------------------------------------------

    RETURN_COST_RATE = 0.08

    order_values = (
        test["order_value"]
        .to_numpy(
            dtype=float
        )
    )

    expected_cost = (
        probabilities
        *
        order_values
        *
        RETURN_COST_RATE
    )

    analysis = pd.DataFrame({

        "order_id":
            test["order_id"].values,

        "order_value":
            order_values,

        "return_probability":
            probabilities,

        "expected_return_cost":
            expected_cost
    })

    analysis = analysis.sort_values(
        "expected_return_cost",
        ascending=False
    )

    print(
        "\nTop 20 orders by expected return cost:"
    )

    print(
        analysis.head(20)
        .to_string(
            index=False
        )
    )

    return analysis


# ============================================================
# RISK TIERS
# ============================================================

def create_risk_tiers(
    test,
    probabilities
):

    print("\n" + "=" * 70)
    print("RISK TIER DISTRIBUTION")
    print("=" * 70)

    result = test[
        [
            "order_id",
            "order_value"
        ]
    ].copy()

    result[
        "return_probability"
    ] = probabilities

    # --------------------------------------------------------
    # Risk tiers
    # --------------------------------------------------------

    result["risk_tier"] = pd.cut(

        result[
            "return_probability"
        ],

        bins=[
            -np.inf,
            0.20,
            0.40,
            0.60,
            0.80,
            np.inf
        ],

        labels=[
            "LOW",
            "MEDIUM",
            "HIGH",
            "VERY_HIGH",
            "CRITICAL"
        ]
    )

    distribution = (
        result["risk_tier"]
        .value_counts(
            sort=False
        )
    )

    print(
        distribution
    )

    print("\nRisk percentages:")

    percentages = (
        distribution
        /
        len(result)
        *
        100
    )

    print(
        percentages.round(2)
    )

    return result


# ============================================================
# SAVE RESULTS
# ============================================================

def save_results(
    threshold_results,
    top_k_results,
    cost_results,
    risk_results
):

    output_dir = Path(
        "evaluation"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    threshold_results.to_csv(
        output_dir
        /
        "threshold_analysis.csv",
        index=False
    )

    top_k_results.to_csv(
        output_dir
        /
        "top_k_analysis.csv",
        index=False
    )

    cost_results.to_csv(
        output_dir
        /
        "expected_return_cost.csv",
        index=False
    )

    risk_results.to_csv(
        output_dir
        /
        "risk_scores.csv",
        index=False
    )

    print("\n" + "=" * 70)
    print("FILES SAVED")
    print("=" * 70)

    print(
        "✓ evaluation/threshold_analysis.csv"
    )

    print(
        "✓ evaluation/top_k_analysis.csv"
    )

    print(
        "✓ evaluation/expected_return_cost.csv"
    )

    print(
        "✓ evaluation/risk_scores.csv"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    # --------------------------------------------------------
    # Load
    # --------------------------------------------------------

    test = load_test_data()

    # --------------------------------------------------------
    # Prepare
    # --------------------------------------------------------

    (
        test,
        X_test,
        y_test,
        feature_columns
    ) = prepare_features(
        test
    )

    # --------------------------------------------------------
    # Load model
    # --------------------------------------------------------

    model = load_model()

    # --------------------------------------------------------
    # Predict
    # --------------------------------------------------------

    probabilities = generate_predictions(
        model,
        X_test
    )

    # --------------------------------------------------------
    # Overall metrics
    # --------------------------------------------------------

    pr_auc = (
        average_precision_score(
            y_test,
            probabilities
        )
    )

    roc_auc = (
        roc_auc_score(
            y_test,
            probabilities
        )
    )

    print("\n" + "=" * 70)
    print("MODEL QUALITY")
    print("=" * 70)

    print(
        f"PR-AUC  : {pr_auc:.4f}"
    )

    print(
        f"ROC-AUC : {roc_auc:.4f}"
    )

    # --------------------------------------------------------
    # Threshold analysis
    # --------------------------------------------------------

    threshold_results = (
        threshold_analysis(
            test,
            probabilities,
            y_test
        )
    )

    # --------------------------------------------------------
    # Top-K analysis
    # --------------------------------------------------------

    top_k_results = (
        top_k_analysis(
            test,
            probabilities,
            y_test
        )
    )

    # --------------------------------------------------------
    # Expected cost
    # --------------------------------------------------------

    cost_results = (
        expected_cost_analysis(
            test,
            probabilities
        )
    )

    # --------------------------------------------------------
    # Risk tiers
    # --------------------------------------------------------

    risk_results = (
        create_risk_tiers(
            test,
            probabilities
        )
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    save_results(
        threshold_results,
        top_k_results,
        cost_results,
        risk_results
    )

    print("\n" + "=" * 70)
    print("THRESHOLD ANALYSIS COMPLETE 🚀")
    print("=" * 70)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()