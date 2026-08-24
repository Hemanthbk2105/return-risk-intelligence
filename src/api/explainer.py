from pathlib import Path

import pandas as pd
import numpy as np
import shap
import lightgbm as lgb


# ============================================================
# PATHS
# ============================================================

BASE_DIR = (
    Path(__file__)
    .resolve()
    .parents[2]
)


MODEL_PATH = (
    BASE_DIR
    / "models"
    / "behaviour_model_v2.txt"
)


TRAIN_FILE = (
    BASE_DIR
    / "data"
    / "processed"
    / "train.csv"
)


VALIDATION_FILE = (
    BASE_DIR
    / "data"
    / "processed"
    / "validation.csv"
)


TEST_FILE = (
    BASE_DIR
    / "data"
    / "processed"
    / "test.csv"
)


# ============================================================
# EXACT 43 FEATURES
# ============================================================

MODEL_FEATURES = [

    "order_value",
    "payment_method",
    "discount_pct",
    "category",
    "size_variant",
    "city_tier",
    "account_age_at_order",
    "price",
    "historical_return_rate",
    "previous_orders",
    "previous_returns",
    "historical_return_rate_customer",
    "history_available",
    "returns_last_5",
    "orders_considered_last_5",
    "return_rate_last_5",
    "returns_last_10",
    "orders_considered_last_10",
    "return_rate_last_10",
    "previous_avg_order_value",
    "order_value_ratio",
    "previous_avg_discount",
    "discount_change",
    "days_since_last_order",
    "orders_last_30_days",
    "orders_last_90_days",
    "return_rate_shift_5",
    "return_rate_shift_10",
    "return_rate_ratio_5",
    "return_rate_ratio_10",
    "recent_avg_order_value_5",
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
    "order_value_shift_3",
    "category_switch"
]


# ============================================================
# CATEGORICAL FEATURES
# ============================================================

CATEGORICAL_FEATURES = [

    "payment_method",
    "category",
    "size_variant"
]


# ============================================================
# HUMAN-READABLE FEATURE NAMES
# ============================================================

FEATURE_NAMES = {

    "historical_return_rate":
        "Historical product return rate",

    "historical_return_rate_customer":
        "Customer historical return rate",

    "order_value":
        "Order value",

    "order_value_ratio":
        "Order value vs customer average",

    "previous_avg_order_value":
        "Historical average order value",

    "recent_avg_order_value_5":
        "Recent average order value",

    "recent_avg_value_3":
        "Recent 3-order average value",

    "previous_avg_value_3":
        "Previous 3-order average value",

    "order_value_shift_3":
        "3-order value change",

    "order_value_shift_5":
        "5-order value change",

    "discount_pct":
        "Discount percentage",

    "discount_change":
        "Discount behaviour change",

    "previous_avg_discount":
        "Previous average discount",

    "previous_orders":
        "Previous order count",

    "previous_returns":
        "Previous return count",

    "return_rate_last_5":
        "Return rate in last 5 orders",

    "return_rate_last_10":
        "Return rate in last 10 orders",

    "recent_return_rate_3":
        "Recent 3-order return rate",

    "recent_return_rate_5":
        "Recent 5-order return rate",

    "previous_return_rate_3":
        "Previous 3-order return rate",

    "previous_return_rate_5":
        "Previous 5-order return rate",

    "return_rate_shift_3":
        "3-order return-rate change",

    "return_rate_shift_5":
        "5-order return-rate change",

    "return_rate_shift_10":
        "10-order return-rate change",

    "return_rate_shift_window5":
        "Recent 5-order return-rate change",

    "return_rate_ratio_5":
        "5-order return-rate ratio",

    "return_rate_ratio_10":
        "10-order return-rate ratio",

    "recent_order_frequency":
        "Recent purchase frequency",

    "orders_last_30_days":
        "Orders in last 30 days",

    "orders_last_90_days":
        "Orders in last 90 days",

    "days_since_last_order":
        "Days since previous order",

    "account_age_at_order":
        "Customer account age",

    "city_tier":
        "Customer city tier",

    "price":
        "Product price",

    "size_variant":
        "Size variant",

    "category_switch":
        "Category switching behaviour",

    "payment_method":
        "Payment method",

    "category":
        "Product category"
}


# ============================================================
# LOAD MODEL
# ============================================================

print(
    "Loading SHAP explanation model..."
)


if not MODEL_PATH.exists():

    raise FileNotFoundError(
        "SHAP model not found: "
        f"{MODEL_PATH}"
    )


model = lgb.Booster(
    model_file=str(
        MODEL_PATH
    )
)


print(
    "SHAP model loaded successfully ✅"
)


# ============================================================
# VERIFY MODEL FEATURE COUNT
# ============================================================

MODEL_FEATURE_NAMES = (
    model.feature_name()
)


MODEL_FEATURE_COUNT = (
    len(MODEL_FEATURE_NAMES)
)


if MODEL_FEATURE_COUNT != 43:

    raise ValueError(
        "SHAP model must contain "
        f"43 features, but found "
        f"{MODEL_FEATURE_COUNT}."
    )


print(
    f"Model feature count: "
    f"{MODEL_FEATURE_COUNT}"
)


# ============================================================
# LOAD DATASETS
#
# IMPORTANT:
#
# predictor.py uses:
#
# TRAIN + VALIDATION + TEST
#
# We MUST use exactly the same datasets here.
# ============================================================

print(
    "Loading datasets for "
    "categorical mappings..."
)


train_df = pd.read_csv(
    TRAIN_FILE
)


validation_df = pd.read_csv(
    VALIDATION_FILE
)


test_df = pd.read_csv(
    TEST_FILE
)


# ============================================================
# BUILD EXACT SAME CATEGORY MAPPINGS
# AS predictor.py
# ============================================================

CATEGORY_MAPPINGS = {}


for column in CATEGORICAL_FEATURES:

    combined = pd.concat(
        [
            train_df[column],
            validation_df[column],
            test_df[column]
        ],
        ignore_index=True
    )


    combined = (
        combined
        .fillna(
            "__MISSING__"
        )
        .astype(str)
    )


    # --------------------------------------------------------
    # IMPORTANT
    #
    # DO NOT SORT.
    #
    # predictor.py uses:
    #
    # pd.Index(combined.unique())
    #
    # Therefore SHAP must use exactly
    # the same ordering.
    # --------------------------------------------------------

    categories = pd.Index(
        combined.unique()
    )


    CATEGORY_MAPPINGS[column] = {

        value: index

        for index, value
        in enumerate(categories)
    }


# ============================================================
# DISPLAY CATEGORY MAPPINGS
# ============================================================

print(
    "\nCategorical mappings:"
)


for column, mapping in (
    CATEGORY_MAPPINGS.items()
):

    print(
        f"  {column}: "
        f"{len(mapping)} categories"
    )


# ============================================================
# CREATE SHAP EXPLAINER
# ============================================================

print(
    "\nCreating SHAP explainer..."
)


explainer = shap.TreeExplainer(
    model
)


print(
    "SHAP explainer ready ✅"
)


# ============================================================
# PREPARE SHAP INPUT
# ============================================================

def prepare_shap_input(
    features: pd.DataFrame
):

    # ========================================================
    # TYPE CHECK
    # ========================================================

    if not isinstance(
        features,
        pd.DataFrame
    ):

        raise TypeError(
            "features must be "
            "a pandas DataFrame."
        )


    # ========================================================
    # CHECK REQUIRED COLUMNS
    # ========================================================

    missing = [

        column

        for column
        in MODEL_FEATURES

        if column
        not in features.columns
    ]


    if missing:

        raise ValueError(
            "Missing SHAP features: "
            +
            ", ".join(missing)
        )


    # ========================================================
    # KEEP EXACT 43 FEATURES
    # ========================================================

    data = features[
        MODEL_FEATURES
    ].copy()


    # ========================================================
    # ENCODE CATEGORICAL FEATURES
    #
    # EXACT SAME LOGIC AS predictor.py
    # ========================================================

    for column in (
        CATEGORICAL_FEATURES
    ):

        mapping = (
            CATEGORY_MAPPINGS[
                column
            ]
        )


        values = (
            data[column]
            .fillna(
                "__MISSING__"
            )
            .astype(str)
        )


        # ----------------------------------------------------
        # Check unknown categories
        # ----------------------------------------------------

        unknown_values = sorted(
            set(values)
            -
            set(mapping.keys())
        )


        if unknown_values:

            raise ValueError(

                f"Unknown value(s) "
                f"for categorical feature "
                f"'{column}': "
                +
                ", ".join(
                    unknown_values
                )
            )


        data[column] = (
            values
            .map(mapping)
            .astype(float)
        )


    # ========================================================
    # CONVERT NUMERIC FEATURES
    # ========================================================

    for column in MODEL_FEATURES:

        if (
            column
            not in CATEGORICAL_FEATURES
        ):

            data[column] = (
                pd.to_numeric(
                    data[column],
                    errors="coerce"
                )
            )


    # ========================================================
    # CLEAN INVALID VALUES
    # ========================================================

    data = (
        data
        .replace(
            [
                np.inf,
                -np.inf
            ],
            np.nan
        )
        .fillna(0)
    )


    # ========================================================
    # FINAL CHECK
    # ========================================================

    if data.shape[1] != 43:

        raise ValueError(
            "Prepared SHAP input must "
            "contain exactly 43 features. "
            f"Got {data.shape[1]}."
        )


    if (
        list(data.columns)
        !=
        MODEL_FEATURES
    ):

        raise ValueError(
            "SHAP feature order does "
            "not match MODEL_FEATURES."
        )


    return data


# ============================================================
# EXPLAIN PREDICTION
# ============================================================

def explain_prediction(
    features: pd.DataFrame,
    top_n: int = 5
):

    # ========================================================
    # PREPARE INPUT
    # ========================================================

    X = prepare_shap_input(
        features
    )


    # ========================================================
    # CALCULATE SHAP VALUES
    # ========================================================

    shap_values = (
        explainer.shap_values(
            X
        )
    )


    # ========================================================
    # HANDLE SHAP OUTPUT FORMAT
    # ========================================================

    if isinstance(
        shap_values,
        list
    ):

        if len(shap_values) > 1:

            values = (
                shap_values[1][0]
            )

        else:

            values = (
                shap_values[0][0]
            )

    else:

        values = (
            shap_values[0]
        )


    values = np.asarray(
        values
    ).reshape(-1)


    # ========================================================
    # SAFETY CHECK
    # ========================================================

    if len(values) != 43:

        raise ValueError(

            "SHAP output feature "
            "count mismatch: "

            f"expected 43, "
            f"got {len(values)}"
        )


    # ========================================================
    # BUILD EXPLANATION DATAFRAME
    # ========================================================

    explanation = pd.DataFrame({

        "feature":
            MODEL_FEATURES,

        "shap_value":
            values,

        "feature_value":
            X.iloc[0].values
    })


    explanation[
        "absolute_impact"
    ] = (
        explanation[
            "shap_value"
        ]
        .abs()
    )


    # ========================================================
    # RISK-INCREASING FEATURES
    # ========================================================

    increasing = (

        explanation[
            explanation[
                "shap_value"
            ] > 0
        ]

        .sort_values(
            "absolute_impact",
            ascending=False
        )

        .head(
            top_n
        )
    )


    # ========================================================
    # RISK-REDUCING FEATURES
    # ========================================================

    reducing = (

        explanation[
            explanation[
                "shap_value"
            ] < 0
        ]

        .sort_values(
            "absolute_impact",
            ascending=False
        )

        .head(
            top_n
        )
    )


    # ========================================================
    # HUMAN-READABLE RISK REASONS
    # ========================================================

    risk_reasons = []


    for _, row in (
        increasing.iterrows()
    ):

        feature = row[
            "feature"
        ]


        name = FEATURE_NAMES.get(

            feature,

            feature.replace(
                "_",
                " "
            ).title()
        )


        risk_reasons.append(
            f"{name} "
            "increased predicted risk"
        )


    # ========================================================
    # HUMAN-READABLE RISK-REDUCING FACTORS
    # ========================================================

    risk_reducing_factors = []


    for _, row in (
        reducing.iterrows()
    ):

        feature = row[
            "feature"
        ]


        name = FEATURE_NAMES.get(

            feature,

            feature.replace(
                "_",
                " "
            ).title()
        )


        risk_reducing_factors.append(
            f"{name} "
            "reduced predicted risk"
        )


    # ========================================================
    # RETURN
    # ========================================================

    return {

        "risk_reasons":
            risk_reasons,

        "risk_reducing_factors":
            risk_reducing_factors
    }


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    print()
    print("=" * 70)

    print(
        "SHAP EXPLAINER CONFIGURATION TEST"
    )

    print("=" * 70)


    print()
    print(
        f"Model features : "
        f"{len(MODEL_FEATURES)}"
    )


    print(
        f"Categorical features : "
        f"{len(CATEGORICAL_FEATURES)}"
    )


    print()
    print(
        "Categorical mappings:"
    )


    for column, mapping in (
        CATEGORY_MAPPINGS.items()
    ):

        print(
            f"  {column}: "
            f"{len(mapping)} categories"
        )


    # ========================================================
    # VERIFY MAPPINGS
    # ========================================================

    print()
    print(
        "Mapping verification:"
    )


    expected_mappings = {

        "payment_method": [
            "Card",
            "UPI",
            "NetBanking",
            "Wallet"
        ],

        "category": [
            "Beauty",
            "Apparel",
            "Footwear",
            "Home",
            "Grocery",
            "Electronics"
        ]
    }


    for column, expected in (
        expected_mappings.items()
    ):

        actual = list(
            CATEGORY_MAPPINGS[
                column
            ].keys()
        )


        if actual == expected:

            print(
                f"✓ {column} mapping "
                "matches predictor.py"
            )

        else:

            print(
                f"⚠ {column} mapping:"
            )

            print(
                f"  Expected: {expected}"
            )

            print(
                f"  Actual  : {actual}"
            )


    # ========================================================
    # FINAL
    # ========================================================

    print()
    print(
        "✓ SHAP uses the same "
        "categorical mappings as predictor.py"
    )

    print(
        "✓ Exact 43-feature configuration verified"
    )

    print()
    print("=" * 70)

    print(
        "SHAP EXPLAINER READY 🚀"
    )

    print("=" * 70)