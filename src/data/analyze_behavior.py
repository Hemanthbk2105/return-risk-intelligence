import pandas as pd
from pathlib import Path


DATA_DIR = Path("data/raw")


def load_data():
    customers = pd.read_csv(DATA_DIR / "customers.csv")
    orders = pd.read_csv(DATA_DIR / "orders.csv")
    returns = pd.read_csv(DATA_DIR / "returns.csv")

    orders["order_ts"] = pd.to_datetime(
        orders["order_ts"]
    )

    data = orders.merge(
        returns,
        on="order_id",
        how="left"
    )

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

    data = data.sort_values(
        ["customer_id", "order_ts"]
    )

    data["order_number"] = (
        data.groupby("customer_id")
        .cumcount() + 1
    )

    return data


def analyze_drift_customers(data):

    print("\n" + "=" * 70)
    print("BEHAVIOURAL DRIFT ANALYSIS")
    print("=" * 70)

    drift = data[
        data["behavior_profile"] == "drift"
    ].copy()

    print(
        f"\nDrift customers found: "
        f"{drift['customer_id'].nunique()}"
    )

    # --------------------------------------------------------
    # Before vs after order 5
    # --------------------------------------------------------

    before = drift[
        drift["order_number"] <= 5
    ]

    after = drift[
        drift["order_number"] > 5
    ]

    before_rate = before["returned"].mean()
    after_rate = after["returned"].mean()

    print("\nRETURN RATE")

    print(
        f"Before behaviour change: "
        f"{before_rate:.2%}"
    )

    print(
        f"After behaviour change : "
        f"{after_rate:.2%}"
    )

    print(
        f"Difference              : "
        f"{(after_rate - before_rate):.2%}"
    )


def show_example_customer(data):

    drift = data[
        data["behavior_profile"] == "drift"
    ]

    if drift.empty:
        print("No drift customers found.")
        return

    customer_id = (
        drift["customer_id"]
        .value_counts()
        .index[0]
    )

    customer = drift[
        drift["customer_id"] == customer_id
    ].sort_values("order_ts")

    print("\n" + "=" * 70)
    print("EXAMPLE DRIFT CUSTOMER")
    print("=" * 70)

    print(
        customer[
            [
                "customer_id",
                "order_number",
                "order_ts",
                "order_value",
                "category",
                "returned"
            ]
        ].to_string(index=False)
    )


def compare_profiles(data):

    print("\n" + "=" * 70)
    print("RETURN RATE BY BEHAVIOUR PROFILE")
    print("=" * 70)

    profile_stats = (
        data.groupby("behavior_profile")
        .agg(
            orders=("order_id", "count"),
            returns=("returned", "sum"),
            return_rate=("returned", "mean")
        )
    )

    profile_stats["return_rate"] *= 100

    profile_stats["return_rate"] = (
        profile_stats["return_rate"]
        .round(2)
    )

    print(profile_stats)


def main():

    data = load_data()

    analyze_drift_customers(data)

    show_example_customer(data)

    compare_profiles(data)

    print("\n" + "=" * 70)
    print("BEHAVIOURAL ANALYSIS COMPLETE ✅")
    print("=" * 70)


if __name__ == "__main__":
    main()