import pandas as pd
from pathlib import Path


DATA_DIR = Path("data/raw")


def load_data():
    customers = pd.read_csv(DATA_DIR / "customers.csv")
    products = pd.read_csv(DATA_DIR / "products.csv")
    orders = pd.read_csv(DATA_DIR / "orders.csv")
    returns = pd.read_csv(DATA_DIR / "returns.csv")

    return customers, products, orders, returns


def validate_row_counts(customers, products, orders, returns):

    print("\n" + "=" * 60)
    print("1. ROW COUNT CHECK")
    print("=" * 60)

    print(f"Customers : {len(customers):,}")
    print(f"Products  : {len(products):,}")
    print(f"Orders    : {len(orders):,}")
    print(f"Returns   : {len(returns):,}")


def validate_duplicates(customers, products, orders, returns):

    print("\n" + "=" * 60)
    print("2. DUPLICATE CHECK")
    print("=" * 60)

    print(
        "Duplicate customer IDs:",
        customers["customer_id"].duplicated().sum()
    )

    print(
        "Duplicate product IDs:",
        products["product_id"].duplicated().sum()
    )

    print(
        "Duplicate order IDs:",
        orders["order_id"].duplicated().sum()
    )

    print(
        "Duplicate return order IDs:",
        returns["order_id"].duplicated().sum()
    )


def validate_missing_values(
    customers,
    products,
    orders,
    returns
):

    print("\n" + "=" * 60)
    print("3. MISSING VALUE CHECK")
    print("=" * 60)

    print("\nCustomers:")
    print(customers.isnull().sum())

    print("\nProducts:")
    print(products.isnull().sum())

    print("\nOrders:")
    print(orders.isnull().sum())

    print("\nReturns:")
    print(returns.isnull().sum())


def validate_relationships(
    customers,
    products,
    orders,
    returns
):

    print("\n" + "=" * 60)
    print("4. RELATIONSHIP CHECK")
    print("=" * 60)

    valid_customers = set(
        customers["customer_id"]
    )

    valid_products = set(
        products["product_id"]
    )

    order_customer_errors = (
        ~orders["customer_id"].isin(valid_customers)
    ).sum()

    order_product_errors = (
        ~orders["product_id"].isin(valid_products)
    ).sum()

    return_order_errors = (
        ~returns["order_id"].isin(
            orders["order_id"]
        )
    ).sum()

    print(
        "Orders with invalid customer:",
        order_customer_errors
    )

    print(
        "Orders with invalid product:",
        order_product_errors
    )

    print(
        "Returns with invalid order:",
        return_order_errors
    )


def validate_return_rate(returns):

    print("\n" + "=" * 60)
    print("5. RETURN RATE")
    print("=" * 60)

    return_rate = returns["returned"].mean()

    print(
        f"Overall return rate: {return_rate:.2%}"
    )

    print("\nReturn distribution:")

    print(
        returns["returned"]
        .value_counts()
        .rename({
            0: "Not Returned",
            1: "Returned"
        })
    )


def category_analysis(orders, returns):

    print("\n" + "=" * 60)
    print("6. RETURN RATE BY CATEGORY")
    print("=" * 60)

    data = orders.merge(
        returns,
        on="order_id",
        how="left"
    )

    category_stats = (
        data.groupby("category")
        .agg(
            orders=("order_id", "count"),
            returned=("returned", "sum"),
            return_rate=("returned", "mean")
        )
        .sort_values(
            "return_rate",
            ascending=False
        )
    )

    category_stats["return_rate"] = (
        category_stats["return_rate"] * 100
    ).round(2)

    print(category_stats)


def customer_order_distribution(orders):

    print("\n" + "=" * 60)
    print("7. ORDERS PER CUSTOMER")
    print("=" * 60)

    orders_per_customer = (
        orders.groupby("customer_id")
        .size()
    )

    print(
        f"Average orders/customer: "
        f"{orders_per_customer.mean():.2f}"
    )

    print(
        f"Minimum orders/customer: "
        f"{orders_per_customer.min()}"
    )

    print(
        f"Maximum orders/customer: "
        f"{orders_per_customer.max()}"
    )

    print("\nOrder count distribution:")

    print(
        orders_per_customer.describe()
    )


def main():

    print("=" * 60)
    print("RETURN-RISK DATA VALIDATION")
    print("=" * 60)

    customers, products, orders, returns = (
        load_data()
    )

    validate_row_counts(
        customers,
        products,
        orders,
        returns
    )

    validate_duplicates(
        customers,
        products,
        orders,
        returns
    )

    validate_missing_values(
        customers,
        products,
        orders,
        returns
    )

    validate_relationships(
        customers,
        products,
        orders,
        returns
    )

    validate_return_rate(returns)

    category_analysis(
        orders,
        returns
    )

    customer_order_distribution(
        orders
    )

    print("\n" + "=" * 60)
    print("VALIDATION COMPLETE ✅")
    print("=" * 60)


if __name__ == "__main__":
    main()