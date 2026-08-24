from pathlib import Path

import lightgbm as lgb
import pandas as pd
import numpy as np


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
#
# IMPORTANT:
# The saved LightGBM model uses Column_0 ... Column_42.
# Therefore feature POSITION is what matters.
# This list defines that exact position.
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
# LOAD MODEL
# ============================================================

print(
    "Loading real-time LightGBM model..."
)


if not MODEL_PATH.exists():

    raise FileNotFoundError(
        "Model file not found: "
        f"{MODEL_PATH}"
    )


model = lgb.Booster(
    model_file=str(
        MODEL_PATH
    )
)


print(
    "Real-time model loaded successfully ✅"
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


EXPECTED_FEATURE_COUNT = (
    len(MODEL_FEATURES)
)


print(
    f"Loaded model feature count: "
    f"{MODEL_FEATURE_COUNT}"
)


if (
    MODEL_FEATURE_COUNT
    !=
    EXPECTED_FEATURE_COUNT
):

    raise ValueError(

        "Model feature count mismatch. "

        f"Model has "
        f"{MODEL_FEATURE_COUNT} features, "

        f"but predictor expects "
        f"{EXPECTED_FEATURE_COUNT}."
    )


# ============================================================
# IMPORTANT
#
# The model was saved with generic names:
#
# Column_0
# Column_1
# ...
# Column_42
#
# Therefore we DO NOT compare the saved names with the
# human-readable feature names.
#
# We verify the feature count and preserve the exact
# MODEL_FEATURES order when constructing model input.
# ============================================================

print(
    "✓ Model contains exactly 43 features"
)


if all(
    str(name).startswith("Column_")
    for name in MODEL_FEATURE_NAMES
):

    print(
        "✓ Model uses generic LightGBM "
        "feature names (Column_0 ... Column_42)"
    )

else:

    print(
        "✓ Model feature names detected"
    )


# ============================================================
# LOAD DATASETS FOR CATEGORY MAPPING
#
# train_behavior_model_v2.py creates categorical mappings
# using TRAIN + VALIDATION + TEST together.
#
# We reproduce that same process.
# ============================================================

print(
    "Loading datasets for categorical mappings..."
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
# BUILD CATEGORY MAPPINGS
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
    # Do NOT sort.
    #
    # Training uses:
    #
    # pd.Index(combined.unique())
    #
    # Therefore the same order is reproduced.
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
# DISPLAY CONFIGURATION
# ============================================================

print(
    f"Model features: "
    f"{len(MODEL_FEATURES)}"
)


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
# DISPLAY MODEL POSITION MAPPING
# ============================================================

print(
    "\nModel feature position mapping:"
)


for index, feature in enumerate(
    MODEL_FEATURES
):

    model_name = (
        MODEL_FEATURE_NAMES[index]
    )

    print(
        f"{index:02d}  "
        f"{model_name:<12} "
        f"← {feature}"
    )


# ============================================================
# PREPARE MODEL INPUT
# ============================================================

def prepare_model_input(
    feature_data
):

    data = feature_data.copy()


    # ========================================================
    # INPUT TYPE CHECK
    # ========================================================

    if not isinstance(
        data,
        pd.DataFrame
    ):

        raise TypeError(
            "feature_data must be "
            "a pandas DataFrame."
        )


    # ========================================================
    # REQUIRED FEATURE CHECK
    # ========================================================

    missing = [

        feature

        for feature
        in MODEL_FEATURES

        if feature
        not in data.columns
    ]


    if missing:

        raise ValueError(

            "Missing required model "
            "features: "
            +
            ", ".join(missing)
        )


    # ========================================================
    # KEEP EXACT 43 FEATURES
    # AND EXACT TRAINING ORDER
    # ========================================================

    data = data[
        MODEL_FEATURES
    ].copy()


    # ========================================================
    # ENCODE CATEGORICAL FEATURES
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
        # CHECK UNKNOWN VALUES
        # ----------------------------------------------------

        unknown_values = sorted(
            set(values)
            -
            set(mapping.keys())
        )


        if unknown_values:

            raise ValueError(

                f"Unknown value(s) "
                f"for '{column}': "
                +
                ", ".join(
                    unknown_values
                )
                +
                ". Valid values are: "
                +
                ", ".join(
                    mapping.keys()
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
    # FINAL FEATURE COUNT
    # ========================================================

    if data.shape[1] != 43:

        raise ValueError(

            "Expected exactly 43 "
            "model features, "

            f"got {data.shape[1]}."
        )


    # ========================================================
    # FINAL FEATURE ORDER
    # ========================================================

    if (
        list(data.columns)
        !=
        MODEL_FEATURES
    ):

        raise ValueError(
            "Feature order does not "
            "match MODEL_FEATURES."
        )


    return data


# ============================================================
# PREDICT RISK
# ============================================================

def predict_risk(
    feature_data
):

    # --------------------------------------------------------
    # Prepare input
    # --------------------------------------------------------

    model_input = (
        prepare_model_input(
            feature_data
        )
    )


    # --------------------------------------------------------
    # Final safety check
    # --------------------------------------------------------

    if (
        model_input.shape[1]
        !=
        MODEL_FEATURE_COUNT
    ):

        raise ValueError(

            "Prepared model input "
            "does not match the "
            "loaded model feature count."
        )


    # --------------------------------------------------------
    # Predict
    # --------------------------------------------------------

    probability = model.predict(
        model_input
    )


    probability = float(
        probability[0]
    )


    # --------------------------------------------------------
    # Safety boundary
    # --------------------------------------------------------

    probability = max(
        0.0,
        min(
            1.0,
            probability
        )
    )


    # --------------------------------------------------------
    # Risk score
    # --------------------------------------------------------

    risk_score = (
        probability
        *
        100
    )


    return {

        "return_probability":
            round(
                probability,
                6
            ),

        "risk_score":
            round(
                risk_score,
                2
            )
    }


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    print(
        "\n"
        +
        "=" * 70
    )


    print(
        "REAL-TIME PREDICTOR"
    )


    print(
        "=" * 70
    )


    # ========================================================
    # BASIC VALIDATION
    # ========================================================

    print(
        "\nExpected model features: "
        f"{EXPECTED_FEATURE_COUNT}"
    )


    print(
        "Loaded model features: "
        f"{MODEL_FEATURE_COUNT}"
    )


    if (
        MODEL_FEATURE_COUNT
        ==
        EXPECTED_FEATURE_COUNT
    ):

        print(
            "✓ Feature count verified"
        )


    # ========================================================
    # FEATURE ORDER
    # ========================================================

    print(
        "\nFeature order:"
    )


    for number, feature in enumerate(
        MODEL_FEATURES,
        start=1
    ):

        print(
            f"{number:02d}. {feature}"
        )


    # ========================================================
    # CATEGORICAL MAPPINGS
    # ========================================================

    print(
        "\nCategorical mapping details:"
    )


    for column, mapping in (
        CATEGORY_MAPPINGS.items()
    ):

        print(
            f"\n{column}:"
        )


        for value, code in (
            mapping.items()
        ):

            print(
                f"  {value} → {code}"
            )


    # ========================================================
    # CONFIGURATION COMPLETE
    # ========================================================

    print(
        "\n"
        +
        "=" * 70
    )


    print(
        "✓ Model loaded"
    )


    print(
        "✓ Exactly 43 model positions detected"
    )


    print(
        "✓ Categorical mappings loaded"
    )


    print(
        "✓ Exact 43-feature input order configured"
    )


    print(
        "✓ Predictor configuration complete ✅"
    )


    print(
        "=" * 70
    )