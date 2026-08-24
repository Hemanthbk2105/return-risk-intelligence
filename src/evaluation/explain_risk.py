import pandas as pd
import numpy as np
import lightgbm as lgb
import shap

from pathlib import Path


# ============================================================
# CONFIGURATION
# ============================================================

TEST_FILE = Path(
    "data/processed/test.csv"
)

MODEL_FILE = Path(
    "models/behaviour_model_v2.txt"
)

OUTPUT_DIR = Path(
    "evaluation"
)

OUTPUT_FILE = (
    OUTPUT_DIR /
    "risk_explanations.csv"
)


# ============================================================
# COLUMNS NOT USED BY THE MODEL
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
# IMPORTANT MODEL FEATURES
# ============================================================

BEHAVIOUR_FEATURES = [
    "return_rate_shift_5",
    "return_rate_shift_10",
    "return_rate_ratio_5",
    "return_rate_ratio_10",
    "order_value_shift_5",
    "recent_order_frequency",
    "recent_return_rate_3",
    "previous_return_rate_3",
    "return_rate_shift_3",
    "recent_return_rate_5",
    "previous_return_rate_5",
    "return_rate_shift_window5",
    "recent_avg_value_3",
    "previous_avg_value_3",
    "order_value_shift_3"
]


# ============================================================
# LOAD DATA
# ============================================================

def load_data():

    print("=" * 70)
    print("RETURN-RISK INTELLIGENCE")
    print("EXPLAINABLE RISK ENGINE")
    print("=" * 70)

    print("\nLoading test dataset...")

    data = pd.read_csv(
        TEST_FILE
    )

    print(
        f"Test orders: {len(data):,}"
    )

    return data


# ============================================================
# PREPARE FEATURES
# ============================================================

def prepare_features(data):

    print(
        "\nPreparing model features..."
    )

    feature_columns = [
        column
        for column in data.columns
        if column not in EXCLUDED_COLUMNS
    ]

    categorical_columns = []

    for column in feature_columns:

        if data[column].dtype == "object":

            categorical_columns.append(
                column
            )

    print(
        f"Categorical columns detected: "
        f"{categorical_columns}"
    )

    # --------------------------------------------------------
    # Encode categorical columns
    # --------------------------------------------------------

    for column in categorical_columns:

        data[column] = (
            data[column]
            .fillna("__MISSING__")
            .astype("category")
            .cat.codes
        )

    # --------------------------------------------------------
    # Clean numerical columns
    # --------------------------------------------------------

    data[feature_columns] = (
        data[feature_columns]
        .replace(
            [np.inf, -np.inf],
            np.nan
        )
    )

    for column in feature_columns:

        data[column] = pd.to_numeric(
            data[column],
            errors="coerce"
        )

    data[feature_columns] = (
        data[feature_columns]
        .fillna(0)
    )

    X = data[
        feature_columns
    ].to_numpy(
        dtype=np.float32
    )

    return (
        data,
        X,
        feature_columns
    )


# ============================================================
# LOAD MODEL
# ============================================================

def load_model():

    print(
        "\nLoading LightGBM model..."
    )

    model = lgb.Booster(
        model_file=str(
            MODEL_FILE
        )
    )

    print(
        "Model loaded successfully ✅"
    )

    return model


# ============================================================
# CREATE SHAP EXPLAINER
# ============================================================

def create_explainer(
    model
):

    print(
        "\nCreating SHAP explainer..."
    )

    explainer = shap.TreeExplainer(
        model
    )

    print(
        "SHAP explainer ready ✅"
    )

    return explainer


# ============================================================
# CALCULATE SHAP VALUES
# ============================================================

def calculate_shap_values(
    explainer,
    X
):

    print(
        "\nCalculating SHAP values..."
    )

    shap_values = (
        explainer.shap_values(
            X
        )
    )

    # --------------------------------------------------------
    # Binary LightGBM sometimes returns:
    #
    # [negative_class, positive_class]
    #
    # We want positive class.
    # --------------------------------------------------------

    if isinstance(
        shap_values,
        list
    ):

        shap_values = (
            shap_values[1]
        )

    shap_values = np.asarray(
        shap_values
    )

    print(
        f"SHAP matrix shape: "
        f"{shap_values.shape}"
    )

    return shap_values


# ============================================================
# GENERATE MODEL PREDICTIONS
# ============================================================

def generate_predictions(
    model,
    X
):

    print(
        "\nGenerating model predictions..."
    )

    probabilities = (
        model.predict(
            X
        )
    )

    return probabilities


# ============================================================
# GET FEATURE REASONS
# ============================================================

def get_top_reasons(
    shap_row,
    feature_columns,
    row,
    top_n=5
):

    # --------------------------------------------------------
    # Sort by absolute SHAP impact
    # --------------------------------------------------------

    ranked_indices = np.argsort(
        np.abs(shap_row)
    )[::-1]

    positive_reasons = []

    negative_reasons = []

    # --------------------------------------------------------
    # Inspect important features
    # --------------------------------------------------------

    for index in ranked_indices:

        feature = (
            feature_columns[index]
        )

        impact = (
            shap_row[index]
        )

        # Ignore tiny contributions
        if abs(impact) < 0.001:

            continue

        value = row[
            feature
        ]

        # ----------------------------------------------------
        # Positive SHAP = pushes risk upward
        # Negative SHAP = pushes risk downward
        # ----------------------------------------------------

        if impact > 0:

            positive_reasons.append({

                "feature": feature,

                "impact": float(
                    impact
                ),

                "value": value
            })

        else:

            negative_reasons.append({

                "feature": feature,

                "impact": float(
                    impact
                ),

                "value": value
            })

        if (
            len(positive_reasons)
            >= top_n
            and
            len(negative_reasons)
            >= top_n
        ):

            break

    return (
        positive_reasons[:top_n],
        negative_reasons[:top_n]
    )


# ============================================================
# HUMAN-READABLE FEATURE NAMES
# ============================================================

def feature_name(
    feature
):

    names = {

        "historical_return_rate":
            "Historical product return rate",

        "historical_return_rate_customer":
            "Customer historical return rate",

        "return_rate_last_5":
            "Recent return rate",

        "return_rate_last_10":
            "10-order return rate",

        "recent_return_rate_3":
            "Recent 3-order return rate",

        "previous_return_rate_3":
            "Previous 3-order return rate",

        "recent_return_rate_5":
            "Recent 5-order return rate",

        "previous_return_rate_5":
            "Previous 5-order return rate",

        "return_rate_shift_3":
            "3-order return-rate change",

        "return_rate_shift_5":
            "5-order return-rate change",

        "return_rate_shift_10":
            "10-order return-rate change",

        "return_rate_shift_window5":
            "Recent vs previous return behaviour",

        "return_rate_ratio_5":
            "5-order return-rate ratio",

        "return_rate_ratio_10":
            "10-order return-rate ratio",

        "order_value_ratio":
            "Order value vs customer average",

        "order_value_shift_3":
            "Recent order-value change",

        "order_value_shift_5":
            "5-order value change",

        "recent_avg_value_3":
            "Recent average order value",

        "previous_avg_value_3":
            "Previous average order value",

        "recent_avg_order_value_5":
            "Recent average order value",

        "previous_avg_order_value":
            "Historical average order value",

        "recent_order_frequency":
            "Recent purchase frequency",

        "days_since_last_order":
            "Days since previous order",

        "discount_change":
            "Discount behaviour change",

        "discount_pct":
            "Discount percentage",

        "order_value":
            "Order value",

        "price":
            "Product price",

        "account_age_at_order":
            "Account age",

        "city_tier":
            "City tier",

        "previous_orders":
            "Previous order count",

        "previous_returns":
            "Previous return count",

        "category_switch":
            "Category switching behaviour"
    }

    return names.get(
        feature,
        feature.replace(
            "_",
            " "
        ).title()
    )


# ============================================================
# CREATE EXPLANATION TEXT
# ============================================================

def create_reason_text(
    reasons
):

    output = []

    for reason in reasons:

        feature = feature_name(
            reason["feature"]
        )

        impact = reason[
            "impact"
        ]

        if impact > 0:

            output.append(
                f"{feature} increased predicted risk"
            )

        else:

            output.append(
                f"{feature} reduced predicted risk"
            )

    return " | ".join(
        output
    )


# ============================================================
# BUILD EXPLANATION DATASET
# ============================================================

def build_explanations(
    data,
    X,
    shap_values,
    feature_columns,
    probabilities
):

    print(
        "\nBuilding explanations..."
    )

    results = []

    for i in range(
        len(data)
    ):

        row = data.iloc[i]

        positive, negative = (
            get_top_reasons(
                shap_values[i],
                feature_columns,
                row
            )
        )

        # ----------------------------------------------------
        # Main risk drivers
        # ----------------------------------------------------

        positive_text = (
            create_reason_text(
                positive
            )
        )

        negative_text = (
            create_reason_text(
                negative
            )
        )

        results.append({

            "order_id":
                row["order_id"],

            "return_probability":
                round(
                    probabilities[i],
                    6
                ),

            "risk_score":
                round(
                    probabilities[i] * 100,
                    2
                ),

            "top_risk_drivers":
                positive_text,

            "risk_reducing_factors":
                negative_text
        })

    return pd.DataFrame(
        results
    )


# ============================================================
# SAVE
# ============================================================

def save_results(
    explanations
):

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    explanations.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print(
        "\nSaved:"
    )

    print(
        f"✓ {OUTPUT_FILE}"
    )


# ============================================================
# SHOW EXAMPLES
# ============================================================

def show_examples(
    explanations
):

    print(
        "\n" + "=" * 70
    )

    print(
        "EXAMPLE EXPLANATIONS"
    )

    print(
        "=" * 70
    )

    examples = (
        explanations
        .sort_values(
            "risk_score",
            ascending=False
        )
        .head(10)
    )

    for _, row in examples.iterrows():

        print(
            "\n" + "-" * 70
        )

        print(
            f"Order: "
            f"{row['order_id']}"
        )

        print(
            f"Risk Score: "
            f"{row['risk_score']:.2f}/100"
        )

        print(
            "\nWhy risk increased:"
        )

        print(
            row[
                "top_risk_drivers"
            ]
            or
            "No strong positive drivers"
        )

        print(
            "\nRisk reducing factors:"
        )

        print(
            row[
                "risk_reducing_factors"
            ]
            or
            "None identified"
        )


# ============================================================
# MAIN
# ============================================================

def main():

    # --------------------------------------------------------
    # Load
    # --------------------------------------------------------

    data = load_data()

    # --------------------------------------------------------
    # Prepare
    # --------------------------------------------------------

    (
        data,
        X,
        feature_columns
    ) = prepare_features(
        data
    )

    print(
        f"\nModel features: "
        f"{len(feature_columns)}"
    )

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    model = load_model()

    # --------------------------------------------------------
    # Explainer
    # --------------------------------------------------------

    explainer = create_explainer(
        model
    )

    # --------------------------------------------------------
    # SHAP
    # --------------------------------------------------------

    shap_values = (
        calculate_shap_values(
            explainer,
            X
        )
    )

    # --------------------------------------------------------
    # Predictions
    # --------------------------------------------------------

    probabilities = (
        generate_predictions(
            model,
            X
        )
    )

    # --------------------------------------------------------
    # Explanations
    # --------------------------------------------------------

    explanations = (
        build_explanations(
            data,
            X,
            shap_values,
            feature_columns,
            probabilities
        )
    )

    # --------------------------------------------------------
    # Examples
    # --------------------------------------------------------

    show_examples(
        explanations
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    save_results(
        explanations
    )

    print(
        "\n" + "=" * 70
    )

    print(
        "EXPLAINABLE RISK ENGINE COMPLETE 🚀"
    )

    print(
        "=" * 70
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()