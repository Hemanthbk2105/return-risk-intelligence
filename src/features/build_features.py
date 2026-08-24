import pandas as pd
import numpy as np
from pathlib import Path


# ============================================================
# CONFIGURATION
# ============================================================

RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")

OUTPUT_FILE = PROCESSED_DIR / "model_dataset.csv"


# ============================================================
# LOAD RAW DATA
# ============================================================

def load_data():

    print("Loading raw datasets...")

    customers = pd.read_csv(
        RAW_DIR / "customers.csv"
    )

    products = pd.read_csv(
        RAW_DIR / "products.csv"
    )

    orders = pd.read_csv(
        RAW_DIR / "orders.csv"
    )

    returns = pd.read_csv(
        RAW_DIR / "returns.csv"
    )

    return customers, products, orders, returns


# ============================================================
# PREPARE DATA
# ============================================================

def prepare_data(
    customers,
    products,
    orders,
    returns
):

    print("Preparing data...")

    orders["order_ts"] = pd.to_datetime(
        orders["order_ts"]
    )

    customers["signup_date"] = pd.to_datetime(
        customers["signup_date"]
    )

    # IMPORTANT:
    # behavior_profile is intentionally NOT included.
    # It is hidden ground truth and must never enter the model.

    customers_for_model = customers[
        [
            "customer_id",
            "signup_date",
            "city_tier"
        ]
    ].copy()

    # --------------------------------------------------------
    # Join customer information
    # --------------------------------------------------------

    data = orders.merge(
        customers_for_model,
        on="customer_id",
        how="left"
    )

    # --------------------------------------------------------
    # Point-in-time account age
    # --------------------------------------------------------

    data["account_age_at_order"] = (
        data["order_ts"]
        - data["signup_date"]
    ).dt.days

    # --------------------------------------------------------
    # Join product information
    # --------------------------------------------------------

    data = data.merge(
        products[
            [
                "product_id",
                "price",
                "historical_return_rate"
            ]
        ],
        on="product_id",
        how="left"
    )

    # --------------------------------------------------------
    # Join target
    # --------------------------------------------------------

    data = data.merge(
        returns[
            [
                "order_id",
                "returned"
            ]
        ],
        on="order_id",
        how="left"
    )

    # --------------------------------------------------------
    # Sort chronologically
    # --------------------------------------------------------

    data = data.sort_values(
        [
            "customer_id",
            "order_ts"
        ]
    ).reset_index(drop=True)

    return data


# ============================================================
# BASIC CUSTOMER HISTORY FEATURES
# ============================================================

def create_history_features(data):

    print("Creating historical customer features...")

    # Number of previous orders

    data["previous_orders"] = (
        data.groupby("customer_id")
        .cumcount()
    )

    # Number of previous returns

    data["previous_returns"] = (
        data.groupby("customer_id")["returned"]
        .transform(
            lambda x:
            x.shift(1)
            .fillna(0)
            .cumsum()
        )
    )

    # Historical return rate

    data["historical_return_rate_customer"] = (
        data["previous_returns"]
        /
        data["previous_orders"].replace(
            0,
            np.nan
        )
    )

    data[
        "historical_return_rate_customer"
    ] = (
        data[
            "historical_return_rate_customer"
        ]
        .fillna(0)
    )

    # Whether history exists

    data["history_available"] = (
        data["previous_orders"] > 0
    ).astype(int)

    return data


# ============================================================
# RECENT RETURN FEATURES
# ============================================================

def create_recent_return_features(data):

    print("Creating recent return features...")

    grouped = data.groupby(
        "customer_id"
    )["returned"]

    # --------------------------------------------------------
    # Last 5 orders
    # --------------------------------------------------------

    data["returns_last_5"] = (
        grouped
        .transform(
            lambda x:
            x.shift(1)
            .rolling(
                window=5,
                min_periods=1
            )
            .sum()
        )
        .fillna(0)
    )

    data["orders_considered_last_5"] = (
        grouped
        .transform(
            lambda x:
            x.shift(1)
            .rolling(
                window=5,
                min_periods=1
            )
            .count()
        )
        .fillna(0)
    )

    data["return_rate_last_5"] = (
        data["returns_last_5"]
        /
        data[
            "orders_considered_last_5"
        ].replace(
            0,
            np.nan
        )
    ).fillna(0)

    # --------------------------------------------------------
    # Last 10 orders
    # --------------------------------------------------------

    data["returns_last_10"] = (
        grouped
        .transform(
            lambda x:
            x.shift(1)
            .rolling(
                window=10,
                min_periods=1
            )
            .sum()
        )
        .fillna(0)
    )

    data["orders_considered_last_10"] = (
        grouped
        .transform(
            lambda x:
            x.shift(1)
            .rolling(
                window=10,
                min_periods=1
            )
            .count()
        )
        .fillna(0)
    )

    data["return_rate_last_10"] = (
        data["returns_last_10"]
        /
        data[
            "orders_considered_last_10"
        ].replace(
            0,
            np.nan
        )
    ).fillna(0)

    return data


# ============================================================
# ORDER VALUE FEATURES
# ============================================================

def create_order_value_features(data):

    print("Creating order value features...")

    # --------------------------------------------------------
    # Previous average order value
    # --------------------------------------------------------

    data["previous_avg_order_value"] = (
        data.groupby("customer_id")[
            "order_value"
        ]
        .transform(
            lambda x:
            x.shift(1)
            .expanding()
            .mean()
        )
    )

    data[
        "previous_avg_order_value"
    ] = (
        data[
            "previous_avg_order_value"
        ].fillna(
            data["order_value"]
        )
    )

    # --------------------------------------------------------
    # Current order / previous average
    # --------------------------------------------------------

    data["order_value_ratio"] = (
        data["order_value"]
        /
        data[
            "previous_avg_order_value"
        ].replace(
            0,
            np.nan
        )
    ).fillna(1.0)

    # --------------------------------------------------------
    # Previous average discount
    # --------------------------------------------------------

    data["previous_avg_discount"] = (
        data.groupby("customer_id")[
            "discount_pct"
        ]
        .transform(
            lambda x:
            x.shift(1)
            .expanding()
            .mean()
        )
    )

    data[
        "previous_avg_discount"
    ] = (
        data[
            "previous_avg_discount"
        ].fillna(
            data["discount_pct"]
        )
    )

    # --------------------------------------------------------
    # Discount change
    # --------------------------------------------------------

    data["discount_change"] = (
        data["discount_pct"]
        -
        data["previous_avg_discount"]
    )

    return data


# ============================================================
# TIME FEATURES
# ============================================================

def create_time_features(data):

    print("Creating temporal features...")

    # --------------------------------------------------------
    # Previous order timestamp
    # --------------------------------------------------------

    data["previous_order_ts"] = (
        data.groupby("customer_id")[
            "order_ts"
        ].shift(1)
    )

    # --------------------------------------------------------
    # Days since previous order
    # --------------------------------------------------------

    data["days_since_last_order"] = (
        (
            data["order_ts"]
            -
            data["previous_order_ts"]
        )
        .dt.total_seconds()
        / 86400
    )

    data[
        "days_since_last_order"
    ] = (
        data[
            "days_since_last_order"
        ].fillna(-1)
    )

    # --------------------------------------------------------
    # Rolling order counts
    # --------------------------------------------------------

    data["orders_last_30_days"] = 0
    data["orders_last_90_days"] = 0

    for customer_id, group in data.groupby(
        "customer_id"
    ):

        timestamps = (
            group["order_ts"].values
        )

        counts_30 = []
        counts_90 = []

        for i, current_time in enumerate(
            timestamps
        ):

            previous_times = timestamps[:i]

            current_time = pd.Timestamp(
                current_time
            )

            count_30 = sum(
                (
                    current_time
                    -
                    pd.Timestamp(t)
                ).days <= 30
                for t in previous_times
            )

            count_90 = sum(
                (
                    current_time
                    -
                    pd.Timestamp(t)
                ).days <= 90
                for t in previous_times
            )

            counts_30.append(count_30)
            counts_90.append(count_90)

        data.loc[
            group.index,
            "orders_last_30_days"
        ] = counts_30

        data.loc[
            group.index,
            "orders_last_90_days"
        ] = counts_90

    return data


# ============================================================
# BEHAVIOUR SHIFT FEATURES
# ============================================================

def create_behavior_shift_features(data):

    print("Creating behaviour shift features...")

    # --------------------------------------------------------
    # Return-rate shift
    # --------------------------------------------------------

    data["return_rate_shift_5"] = (
        data["return_rate_last_5"]
        -
        data[
            "historical_return_rate_customer"
        ]
    )

    data["return_rate_shift_10"] = (
        data["return_rate_last_10"]
        -
        data[
            "historical_return_rate_customer"
        ]
    )

    # --------------------------------------------------------
    # Return-rate ratio
    # --------------------------------------------------------

    historical_rate = (
        data[
            "historical_return_rate_customer"
        ]
    )

    data["return_rate_ratio_5"] = (
        data["return_rate_last_5"]
        /
        historical_rate.replace(
            0,
            np.nan
        )
    )

    data["return_rate_ratio_10"] = (
        data["return_rate_last_10"]
        /
        historical_rate.replace(
            0,
            np.nan
        )
    )

    data["return_rate_ratio_5"] = (
        data["return_rate_ratio_5"]
        .replace(
            [np.inf, -np.inf],
            np.nan
        )
        .fillna(1.0)
    )

    data["return_rate_ratio_10"] = (
        data["return_rate_ratio_10"]
        .replace(
            [np.inf, -np.inf],
            np.nan
        )
        .fillna(1.0)
    )

    # --------------------------------------------------------
    # Recent average order value
    # --------------------------------------------------------

    grouped_values = data.groupby(
        "customer_id"
    )["order_value"]

    data["recent_avg_order_value_5"] = (
        grouped_values
        .transform(
            lambda x:
            x.shift(1)
            .rolling(
                window=5,
                min_periods=1
            )
            .mean()
        )
    )

    data[
        "recent_avg_order_value_5"
    ] = (
        data[
            "recent_avg_order_value_5"
        ].fillna(
            data["order_value"]
        )
    )

    # --------------------------------------------------------
    # Order-value shift
    # --------------------------------------------------------

    data["order_value_shift_5"] = (
        data[
            "recent_avg_order_value_5"
        ]
        /
        data[
            "previous_avg_order_value"
        ].replace(
            0,
            np.nan
        )
    )

    data["order_value_shift_5"] = (
        data["order_value_shift_5"]
        .replace(
            [np.inf, -np.inf],
            np.nan
        )
        .fillna(1.0)
    )

    # --------------------------------------------------------
    # Recent order frequency
    # --------------------------------------------------------

    data["historical_order_rate"] = (
        data["previous_orders"]
        /
        (
            data["account_age_at_order"]
            .clip(lower=1)
            / 30
        )
    )

    data["recent_order_frequency"] = (
        data["orders_last_30_days"]
        /
        data[
            "historical_order_rate"
        ].replace(
            0,
            np.nan
        )
    )

    data["recent_order_frequency"] = (
        data[
            "recent_order_frequency"
        ]
        .replace(
            [np.inf, -np.inf],
            np.nan
        )
        .fillna(0)
    )

    return data


# ============================================================
# WINDOW-TO-WINDOW BEHAVIOUR FEATURES
# ============================================================

def create_window_shift_features(data):

    print(
        "Creating window-to-window "
        "behaviour features..."
    )

    # --------------------------------------------------------
    # IMPORTANT
    #
    # For every current order:
    #
    # Recent window:
    #   immediately previous 3 orders
    #
    # Previous window:
    #   3 orders before that
    #
    # Example:
    #
    # Orders: 1 2 3 4 5 6 7
    #
    # Current order = 7
    #
    # Previous window = 1 2 3
    # Recent window  = 4 5 6
    #
    # Current order 7 itself is NEVER included.
    # --------------------------------------------------------

    data = data.sort_values(
        [
            "customer_id",
            "order_ts"
        ]
    ).copy()

    # --------------------------------------------------------
    # Create order position within each customer
    # --------------------------------------------------------

    data["_customer_order_index"] = (
        data.groupby("customer_id")
        .cumcount()
    )

    # --------------------------------------------------------
    # Previous completed return values
    # --------------------------------------------------------

    return_series = (
        data.groupby("customer_id")[
            "returned"
        ]
    )

    # --------------------------------------------------------
    # Recent 3 completed orders
    #
    # Orders i-3, i-2, i-1
    # --------------------------------------------------------

    recent_3 = (
        return_series
        .transform(
            lambda x:
            x.shift(1)
            .rolling(
                window=3,
                min_periods=3
            )
            .mean()
        )
    )

    # --------------------------------------------------------
    # Previous 3 completed orders
    #
    # Orders i-6, i-5, i-4
    # --------------------------------------------------------

    previous_3 = (
        return_series
        .transform(
            lambda x:
            x.shift(4)
            .rolling(
                window=3,
                min_periods=3
            )
            .mean()
        )
    )

    data["recent_return_rate_3"] = (
        recent_3.fillna(0)
    )

    data["previous_return_rate_3"] = (
        previous_3.fillna(0)
    )

    # --------------------------------------------------------
    # Behaviour shift
    # --------------------------------------------------------

    data["return_rate_shift_3"] = (
        data["recent_return_rate_3"]
        -
        data["previous_return_rate_3"]
    )

    # --------------------------------------------------------
    # Recent 5 completed orders
    #
    # Orders i-5 ... i-1
    # --------------------------------------------------------

    recent_5 = (
        return_series
        .transform(
            lambda x:
            x.shift(1)
            .rolling(
                window=5,
                min_periods=5
            )
            .mean()
        )
    )

    # --------------------------------------------------------
    # Previous 5 completed orders
    #
    # Orders i-10 ... i-6
    # --------------------------------------------------------

    previous_5 = (
        return_series
        .transform(
            lambda x:
            x.shift(6)
            .rolling(
                window=5,
                min_periods=5
            )
            .mean()
        )
    )

    data["recent_return_rate_5"] = (
        recent_5.fillna(0)
    )

    data["previous_return_rate_5"] = (
        previous_5.fillna(0)
    )

    # --------------------------------------------------------
    # 5-order behaviour shift
    # --------------------------------------------------------

    data["return_rate_shift_window5"] = (
        data["recent_return_rate_5"]
        -
        data["previous_return_rate_5"]
    )

    # ========================================================
    # ORDER VALUE WINDOW SHIFT
    # ========================================================

    value_series = (
        data.groupby("customer_id")[
            "order_value"
        ]
    )

    # --------------------------------------------------------
    # Recent 3-order average value
    # --------------------------------------------------------

    recent_value_3 = (
        value_series
        .transform(
            lambda x:
            x.shift(1)
            .rolling(
                window=3,
                min_periods=3
            )
            .mean()
        )
    )

    # --------------------------------------------------------
    # Previous 3-order average value
    # --------------------------------------------------------

    previous_value_3 = (
        value_series
        .transform(
            lambda x:
            x.shift(4)
            .rolling(
                window=3,
                min_periods=3
            )
            .mean()
        )
    )

    data["recent_avg_value_3"] = (
        recent_value_3.fillna(0)
    )

    data["previous_avg_value_3"] = (
        previous_value_3.fillna(0)
    )

    # --------------------------------------------------------
    # Order value shift
    #
    # Ratio:
    #
    # recent average
    # ----------------
    # previous average
    # --------------------------------------------------------

    data["order_value_shift_3"] = (
        data["recent_avg_value_3"]
        /
        data[
            "previous_avg_value_3"
        ].replace(
            0,
            np.nan
        )
    )

    data["order_value_shift_3"] = (
        data["order_value_shift_3"]
        .replace(
            [np.inf, -np.inf],
            np.nan
        )
        .fillna(1.0)
    )

    # --------------------------------------------------------
    # Remove helper column
    # --------------------------------------------------------

    data = data.drop(
        columns=[
            "_customer_order_index"
        ]
    )

    return data


# ============================================================
# CATEGORY FEATURES
# ============================================================

def create_category_features(data):

    print("Creating category behaviour features...")

    data["previous_category"] = (
        data.groupby("customer_id")[
            "category"
        ].shift(1)
    )

    data["category_switch"] = (
        (
            data["previous_category"].notna()
        )
        &
        (
            data["category"]
            !=
            data["previous_category"]
        )
    ).astype(int)

    return data


# ============================================================
# CLEAN FINAL DATASET
# ============================================================

def clean_dataset(data):

    print("Cleaning final dataset...")

    columns_to_remove = [
        "previous_order_ts",
        "previous_category",
        "historical_order_rate"
    ]

    data = data.drop(
        columns=[
            column
            for column in columns_to_remove
            if column in data.columns
        ]
    )

    # Replace infinity

    data = data.replace(
        [np.inf, -np.inf],
        np.nan
    )

    # Fill numerical missing values

    numeric_columns = (
        data
        .select_dtypes(
            include=[np.number]
        )
        .columns
    )

    data[numeric_columns] = (
        data[numeric_columns]
        .fillna(0)
    )

    return data


# ============================================================
# VALIDATION
# ============================================================

def validate_features(data):

    print("\n" + "=" * 70)
    print("FEATURE DATASET VALIDATION")
    print("=" * 70)

    print(
        f"Rows: {len(data):,}"
    )

    print(
        f"Columns: {len(data.columns)}"
    )

    print("\nColumns:")

    for column in data.columns:
        print(
            f"  ✓ {column}"
        )

    print("\nMissing values:")

    missing = data.isnull().sum()

    remaining_missing = (
        missing[
            missing > 0
        ]
    )

    if len(remaining_missing) == 0:

        print("None ✅")

    else:

        print(
            remaining_missing
        )

    print("\nTarget distribution:")

    print(
        data["returned"]
        .value_counts()
    )

    print(
        "\nTarget return rate:",
        f"{data['returned'].mean():.2%}"
    )

    # --------------------------------------------------------
    # Point-in-time account age check
    # --------------------------------------------------------

    print(
        "\nAccount age at order:"
    )

    print(
        f"Minimum: "
        f"{data['account_age_at_order'].min()} days"
    )

    print(
        f"Maximum: "
        f"{data['account_age_at_order'].max()} days"
    )

    # --------------------------------------------------------
    # Leakage check
    # --------------------------------------------------------

    if "account_age_days" in data.columns:

        print(
            "WARNING: account_age_days "
            "still exists ❌"
        )

    else:

        print(
            "account_age_days removed ✅"
        )

    print(
        "account_age_at_order exists:",
        "account_age_at_order"
        in data.columns
    )

    # --------------------------------------------------------
    # Behaviour feature check
    # --------------------------------------------------------

    behaviour_features = [
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
        "order_value_shift_3"
    ]

    print(
        "\nBehaviour features:"
    )

    for feature in behaviour_features:

        if feature in data.columns:

            print(
                f"  ✓ {feature}"
            )

        else:

            print(
                f"  ✗ {feature} MISSING"
            )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("RETURN-RISK INTELLIGENCE")
    print("FEATURE ENGINEERING PIPELINE")
    print("=" * 70)

    # --------------------------------------------------------
    # Load
    # --------------------------------------------------------

    customers, products, orders, returns = (
        load_data()
    )

    # --------------------------------------------------------
    # Prepare
    # --------------------------------------------------------

    data = prepare_data(
        customers,
        products,
        orders,
        returns
    )

    # --------------------------------------------------------
    # Feature engineering
    # --------------------------------------------------------

    data = create_history_features(
        data
    )

    data = create_recent_return_features(
        data
    )

    data = create_order_value_features(
        data
    )

    data = create_time_features(
        data
    )

    data = create_behavior_shift_features(
        data
    )

    data = create_window_shift_features(
        data
    )

    data = create_category_features(
        data
    )

    # --------------------------------------------------------
    # Clean
    # --------------------------------------------------------

    data = clean_dataset(
        data
    )

    # --------------------------------------------------------
    # Validate
    # --------------------------------------------------------

    validate_features(
        data
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    PROCESSED_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    data.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print(
        "\nSaved:",
        OUTPUT_FILE
    )

    print(
        "\nFEATURE ENGINEERING COMPLETE 🚀"
    )


if __name__ == "__main__":
    main()