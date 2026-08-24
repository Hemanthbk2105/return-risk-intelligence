import pandas as pd
from pathlib import Path


RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")


def main():

    print("=" * 70)
    print("RETURN-RISK INTELLIGENCE")
    print("BEHAVIOUR TRANSITION DIAGNOSTIC")
    print("=" * 70)

    # --------------------------------------------------------
    # Load data
    # --------------------------------------------------------

    customers = pd.read_csv(
        RAW_DIR / "customers.csv"
    )

    data = pd.read_csv(
        PROCESSED_DIR / "model_dataset.csv"
    )

    data["order_ts"] = pd.to_datetime(
        data["order_ts"]
    )

    # --------------------------------------------------------
    # Add hidden behaviour profile
    #
    # ONLY for diagnostic validation.
    # NEVER used by the ML model.
    # --------------------------------------------------------

    data = data.merge(
        customers[
            [
                "customer_id",
                "behavior_profile"
            ]
        ],
        on="customer_id",
        how="left"
    )

    drift = data[
        data["behavior_profile"] == "drift"
    ].copy()

    print(
        f"\nDrift customers: "
        f"{drift['customer_id'].nunique()}"
    )

    # --------------------------------------------------------
    # Sort
    # --------------------------------------------------------

    drift = drift.sort_values(
        [
            "customer_id",
            "order_ts"
        ]
    )

    # --------------------------------------------------------
    # We know our synthetic generator introduces
    # drift after order number 5.
    #
    # Therefore:
    #
    # BEFORE = orders 3,4,5
    # AFTER  = orders 6,7,8
    #
    # We examine the actual target outcomes here.
    # --------------------------------------------------------

    before_rates = []
    after_rates = []

    for customer_id, group in drift.groupby(
        "customer_id"
    ):

        group = group.sort_values(
            "order_ts"
        )

        # Need at least 8 orders
        if len(group) < 8:
            continue

        before = group.iloc[
            2:5
        ]

        after = group.iloc[
            5:8
        ]

        before_rates.append(
            before["returned"].mean()
        )

        after_rates.append(
            after["returned"].mean()
        )

    print("\n" + "=" * 70)
    print("ACTUAL TRANSITION BEHAVIOUR")
    print("=" * 70)

    before_mean = (
        sum(before_rates)
        /
        len(before_rates)
    )

    after_mean = (
        sum(after_rates)
        /
        len(after_rates)
    )

    print(
        f"Customers analysed: "
        f"{len(before_rates)}"
    )

    print(
        f"\nBefore transition: "
        f"{before_mean:.2%}"
    )

    print(
        f"After transition : "
        f"{after_mean:.2%}"
    )

    print(
        f"Difference        : "
        f"{after_mean - before_mean:+.2%}"
    )

    # --------------------------------------------------------
    # Feature diagnostic
    #
    # We look at the rows around the transition.
    # --------------------------------------------------------

    transition_before = drift[
        drift["previous_orders"].between(
            3,
            5
        )
    ]

    transition_after = drift[
        drift["previous_orders"].between(
            5,
            7
        )
    ]

    features = [
        "recent_return_rate_3",
        "previous_return_rate_3",
        "return_rate_shift_3",

        "recent_return_rate_5",
        "previous_return_rate_5",
        "return_rate_shift_window5",

        "recent_avg_value_3",
        "previous_avg_value_3",
        "order_value_shift_3",

        "return_rate_last_5",
        "return_rate_last_10"
    ]

    print("\n" + "=" * 70)
    print("FEATURE VALUES AROUND TRANSITION")
    print("=" * 70)

    rows = []

    for feature in features:

        before_value = (
            transition_before[
                feature
            ].mean()
        )

        after_value = (
            transition_after[
                feature
            ].mean()
        )

        difference = (
            after_value
            -
            before_value
        )

        rows.append({

            "feature": feature,

            "before": before_value,

            "after": after_value,

            "difference": difference
        })

    result = pd.DataFrame(
        rows
    )

    print(
        result.to_string(
            index=False
        )
    )

    # --------------------------------------------------------
    # Correlation with target
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("FEATURE CORRELATION WITH RETURN")
    print("=" * 70)

    numeric_features = [
        feature
        for feature in features
        if feature in data.columns
    ]

    correlations = (
        data[
            numeric_features
            +
            ["returned"]
        ]
        .corr()["returned"]
        .drop("returned")
        .sort_values(
            ascending=False
        )
    )

    print(
        correlations
    )

    print("\n" + "=" * 70)
    print("TRANSITION DIAGNOSTIC COMPLETE 🔎")
    print("=" * 70)


if __name__ == "__main__":
    main()