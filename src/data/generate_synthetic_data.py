import os
import numpy as np
import pandas as pd

# ============================================================
# CONFIGURATION
# ============================================================

RANDOM_SEED = 42

NUM_CUSTOMERS = 2000
NUM_PRODUCTS = 500
NUM_ORDERS = 20000

OUTPUT_DIR = "data/raw"

np.random.seed(RANDOM_SEED)


# ============================================================
# 1. GENERATE CUSTOMERS
# ============================================================

def generate_customers():
    print("Generating customers...")

    customer_ids = [
        f"C{i:04d}"
        for i in range(1, NUM_CUSTOMERS + 1)
    ]

    signup_dates = pd.to_datetime(
        np.random.choice(
            pd.date_range("2022-01-01", "2025-12-31"),
            size=NUM_CUSTOMERS
        )
    )

    city_tiers = np.random.choice(
        [1, 2, 3],
        size=NUM_CUSTOMERS,
        p=[0.40, 0.35, 0.25]
    )

    today = pd.Timestamp("2026-08-23")

    account_age_days = (
        today - signup_dates
    ).days

    customers = pd.DataFrame({
        "customer_id": customer_ids,
        "signup_date": signup_dates,
        "city_tier": city_tiers,
        "account_age_days": account_age_days
    })

    return customers


# ============================================================
# 2. GENERATE PRODUCTS
# ============================================================

def generate_products():
    print("Generating products...")

    product_ids = [
        f"P{i:04d}"
        for i in range(1, NUM_PRODUCTS + 1)
    ]

    categories = np.random.choice(
        [
            "Electronics",
            "Apparel",
            "Footwear",
            "Home",
            "Beauty",
            "Grocery"
        ],
        size=NUM_PRODUCTS,
        p=[
            0.20,
            0.25,
            0.15,
            0.15,
            0.15,
            0.10
        ]
    )

    prices = []

    historical_return_rates = []

    for category in categories:

        if category == "Electronics":
            price = np.random.uniform(1000, 100000)
            return_rate = np.random.uniform(0.02, 0.04)

        elif category == "Apparel":
            price = np.random.uniform(500, 10000)
            return_rate = np.random.uniform(0.08, 0.15)

        elif category == "Footwear":
            price = np.random.uniform(800, 15000)
            return_rate = np.random.uniform(0.10, 0.16)

        elif category == "Home":
            price = np.random.uniform(500, 30000)
            return_rate = np.random.uniform(0.05, 0.09)

        elif category == "Beauty":
            price = np.random.uniform(300, 8000)
            return_rate = np.random.uniform(0.04, 0.08)

        else:
            price = np.random.uniform(100, 5000)
            return_rate = np.random.uniform(0.02, 0.05)

        prices.append(round(price, 2))
        historical_return_rates.append(round(return_rate, 4))

    products = pd.DataFrame({
        "product_id": product_ids,
        "category": categories,
        "price": prices,
        "historical_return_rate": historical_return_rates
    })

    return products


# ============================================================
# 3. GENERATE ORDERS
# ============================================================

def generate_orders(customers, products):
    print("Generating orders...")

    customer_ids = customers["customer_id"].values
    product_ids = products["product_id"].values

    orders = []

    for i in range(1, NUM_ORDERS + 1):

        customer_id = np.random.choice(customer_ids)
        product_id = np.random.choice(product_ids)

        product = products[
            products["product_id"] == product_id
        ].iloc[0]

        order_ts = pd.Timestamp(
            np.random.choice(
                pd.date_range(
                    "2025-01-01",
                    "2026-08-20",
                    freq="h"
                )
            )
        )

        discount_pct = np.random.choice(
            [0, 5, 10, 15, 20, 30, 40, 50],
            p=[0.15, 0.10, 0.15, 0.15,
               0.15, 0.15, 0.10, 0.05]
        )

        order_value = (
            product["price"] *
            (1 - discount_pct / 100)
        )

        payment_method = np.random.choice(
            [
                "UPI",
                "Card",
                "NetBanking",
                "Wallet"
            ],
            p=[0.45, 0.35, 0.15, 0.05]
        )

        size_variant = None

        if product["category"] in ["Apparel", "Footwear"]:

            if product["category"] == "Apparel":
                size_variant = np.random.choice(
                    ["S", "M", "L", "XL"]
                )

            else:
                size_variant = np.random.choice(
                    ["7", "8", "9", "10", "11"]
                )

        orders.append({
            "order_id": f"O{i:05d}",
            "customer_id": customer_id,
            "product_id": product_id,
            "order_ts": order_ts,
            "order_value": round(order_value, 2),
            "payment_method": payment_method,
            "discount_pct": discount_pct,
            "category": product["category"],
            "size_variant": size_variant
        })

    return pd.DataFrame(orders)


# ============================================================
# 4. GENERATE RETURN OUTCOMES
# ============================================================

def generate_returns(orders, products):
    print("Generating return outcomes...")

    returns = []

    product_rates = products.set_index(
        "product_id"
    )["historical_return_rate"].to_dict()

    for _, order in orders.iterrows():

        base_probability = product_rates[
            order["product_id"]
        ]

        probability = base_probability

        # Higher discount can slightly increase return probability
        if order["discount_pct"] >= 30:
            probability += 0.03

        # Apparel and footwear have size/fit uncertainty
        if order["category"] in ["Apparel", "Footwear"]:
            probability += 0.02

        probability = min(probability, 0.50)

        returned = np.random.random() < probability

        if returned:

            if order["category"] in ["Apparel", "Footwear"]:
                reason = np.random.choice(
                    [
                        "Size issue",
                        "Changed mind",
                        "Product mismatch"
                    ]
                )
            else:
                reason = np.random.choice(
                    [
                        "Changed mind",
                        "Product mismatch",
                        "Damaged product"
                    ]
                )

            days_to_return = np.random.randint(2, 15)

        else:
            reason = None
            days_to_return = 0

        returns.append({
            "order_id": order["order_id"],
            "returned": int(returned),
            "return_reason": reason,
            "days_to_return": days_to_return
        })

    return pd.DataFrame(returns)


# ============================================================
# 5. SAVE DATA
# ============================================================

def save_data(customers, products, orders, returns):

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    customers.to_csv(
        f"{OUTPUT_DIR}/customers.csv",
        index=False
    )

    products.to_csv(
        f"{OUTPUT_DIR}/products.csv",
        index=False
    )

    orders.to_csv(
        f"{OUTPUT_DIR}/orders.csv",
        index=False
    )

    returns.to_csv(
        f"{OUTPUT_DIR}/returns.csv",
        index=False
    )

    print("\nData saved successfully.")


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 60)
    print("RETURN-RISK INTELLIGENCE")
    print("Synthetic Dataset Generator")
    print("=" * 60)

    customers = generate_customers()

    products = generate_products()

    orders = generate_orders(
        customers,
        products
    )

    returns = generate_returns(
        orders,
        products
    )

    save_data(
        customers,
        products,
        orders,
        returns
    )

    print("\n" + "=" * 60)
    print("DATASET GENERATION COMPLETE 🚀")
    print("=" * 60)

    print(f"\nCustomers : {len(customers):,}")
    print(f"Products  : {len(products):,}")
    print(f"Orders    : {len(orders):,}")
    print(f"Returns   : {len(returns):,}")

    return_rate = returns["returned"].mean()

    print(
        f"\nOverall return rate: "
        f"{return_rate:.2%}"
    )


if __name__ == "__main__":
    main()