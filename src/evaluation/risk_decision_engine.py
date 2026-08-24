import pandas as pd
import numpy as np
from pathlib import Path


# ============================================================
# CONFIGURATION
# ============================================================

INPUT_FILE = Path(
    "evaluation/risk_scores.csv"
)

OUTPUT_DIR = Path(
    "evaluation"
)

OUTPUT_FILE = (
    OUTPUT_DIR /
    "risk_decisions.csv"
)

DASHBOARD_FILE = (
    OUTPUT_DIR /
    "dashboard_orders.csv"
)


# ============================================================
# BUSINESS ASSUMPTION
# ============================================================

# Synthetic assumption only.
# This is NOT Razorpay's actual return cost.

RETURN_COST_RATE = 0.08


# ============================================================
# RISK TIER
# ============================================================

def assign_risk_tier(probability):

    if probability < 0.20:
        return "LOW"

    elif probability < 0.40:
        return "MEDIUM"

    elif probability < 0.60:
        return "HIGH"

    elif probability < 0.80:
        return "VERY_HIGH"

    else:
        return "CRITICAL"


# ============================================================
# PRIORITY
# ============================================================

def assign_priority(
    risk_tier,
    expected_exposure
):

    if (
        risk_tier == "CRITICAL"
        or
        (
            risk_tier == "VERY_HIGH"
            and expected_exposure >= 1000
        )
    ):
        return "P1"

    elif (
        risk_tier == "VERY_HIGH"
        or
        (
            risk_tier == "HIGH"
            and expected_exposure >= 1000
        )
    ):
        return "P2"

    elif risk_tier == "HIGH":
        return "P3"

    else:
        return "P4"


# ============================================================
# ACTION
# ============================================================

def assign_action(priority):

    if priority == "P1":
        return "INTERVENE"

    elif priority == "P2":
        return "REVIEW"

    elif priority == "P3":
        return "MONITOR"

    else:
        return "NORMAL_PROCESSING"


# ============================================================
# REASONS
# ============================================================

def generate_reasons(row):

    reasons = []

    probability = row[
        "return_probability"
    ]

    order_value = row[
        "order_value"
    ]

    expected_exposure = row[
        "expected_return_exposure"
    ]

    # --------------------------------------------------------
    # Return probability
    # --------------------------------------------------------

    if probability >= 0.60:

        reasons.append(
            "High predicted return probability"
        )

    elif probability >= 0.40:

        reasons.append(
            "Elevated predicted return probability"
        )

    # --------------------------------------------------------
    # Financial exposure
    # --------------------------------------------------------

    if expected_exposure >= 2000:

        reasons.append(
            "Very high estimated financial exposure"
        )

    elif expected_exposure >= 1000:

        reasons.append(
            "High estimated financial exposure"
        )

    # --------------------------------------------------------
    # Order value
    # --------------------------------------------------------

    if order_value >= 50000:

        reasons.append(
            "High-value order"
        )

    elif order_value >= 25000:

        reasons.append(
            "Above-average order value"
        )

    # --------------------------------------------------------
    # Fallback
    # --------------------------------------------------------

    if not reasons:

        reasons.append(
            "No major risk signal detected"
        )

    return " | ".join(reasons)


# ============================================================
# LOAD DATA
# ============================================================

def load_data():

    print("=" * 70)
    print("RETURN-RISK INTELLIGENCE")
    print("RISK DECISION ENGINE")
    print("=" * 70)

    print(
        "\nLoading risk scores..."
    )

    data = pd.read_csv(
        INPUT_FILE
    )

    print(
        f"Orders loaded: "
        f"{len(data):,}"
    )

    return data


# ============================================================
# CREATE DECISIONS
# ============================================================

def create_decisions(data):

    print(
        "\nCreating business decisions..."
    )

    result = data.copy()

    # ========================================================
    # Validate basic columns
    # ========================================================

    required_columns = [
        "order_id",
        "order_value",
        "return_probability"
    ]

    missing = [
        column
        for column in required_columns
        if column not in result.columns
    ]

    if missing:

        raise ValueError(
            "Missing required columns: "
            + str(missing)
        )

    # ========================================================
    # Ensure numeric values
    # ========================================================

    result[
        "order_value"
    ] = pd.to_numeric(
        result[
            "order_value"
        ],
        errors="coerce"
    ).fillna(0)

    result[
        "return_probability"
    ] = pd.to_numeric(
        result[
            "return_probability"
        ],
        errors="coerce"
    ).fillna(0)

    # ========================================================
    # Expected return exposure
    #
    # probability × order value × cost rate
    # ========================================================

    result[
        "expected_return_exposure"
    ] = (
        result[
            "return_probability"
        ]
        *
        result[
            "order_value"
        ]
        *
        RETURN_COST_RATE
    )

    result[
        "expected_return_exposure"
    ] = result[
        "expected_return_exposure"
    ].round(2)

    # ========================================================
    # Risk score
    # ========================================================

    result[
        "risk_score"
    ] = (
        result[
            "return_probability"
        ]
        *
        100
    ).round(2)

    # ========================================================
    # Risk tier
    # ========================================================

    result[
        "risk_tier"
    ] = result[
        "return_probability"
    ].apply(
        assign_risk_tier
    )

    # ========================================================
    # Priority
    # ========================================================

    result[
        "priority"
    ] = result.apply(

        lambda row:
        assign_priority(
            row["risk_tier"],
            row[
                "expected_return_exposure"
            ]
        ),

        axis=1
    )

    # ========================================================
    # Recommended action
    # ========================================================

    result[
        "recommended_action"
    ] = result[
        "priority"
    ].apply(
        assign_action
    )

    # ========================================================
    # Explanation
    # ========================================================

    result[
        "risk_reasons"
    ] = result.apply(
        generate_reasons,
        axis=1
    )

    return result


# ============================================================
# SUMMARY
# ============================================================

def show_summary(data):

    print(
        "\n" + "=" * 70
    )

    print(
        "RISK DECISION SUMMARY"
    )

    print(
        "=" * 70
    )

    print(
        "\nRisk tier distribution:"
    )

    print(
        data[
            "risk_tier"
        ].value_counts(
            sort=False
        )
    )

    print(
        "\nRecommended actions:"
    )

    print(
        data[
            "recommended_action"
        ].value_counts()
    )

    print(
        "\nPriority distribution:"
    )

    print(
        data[
            "priority"
        ].value_counts(
            sort=False
        )
    )


# ============================================================
# TOP PRIORITY ORDERS
# ============================================================

def show_top_orders(data):

    print(
        "\n" + "=" * 70
    )

    print(
        "TOP 20 PRIORITY ORDERS"
    )

    print(
        "=" * 70
    )

    columns = [
        "order_id",
        "order_value",
        "risk_score",
        "risk_tier",
        "expected_return_exposure",
        "priority",
        "recommended_action",
        "risk_reasons"
    ]

    top = (
        data
        .sort_values(
            [
                "priority",
                "expected_return_exposure"
            ],
            ascending=[
                True,
                False
            ]
        )
        .head(20)
    )

    print(
        top[
            columns
        ].to_string(
            index=False
        )
    )


# ============================================================
# SAVE RESULTS
# ============================================================

def save_results(data):

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Full risk decisions
    # --------------------------------------------------------

    data.to_csv(
        OUTPUT_FILE,
        index=False
    )

    # --------------------------------------------------------
    # Dashboard file
    # --------------------------------------------------------

    dashboard_columns = [
        "order_id",
        "order_value",
        "risk_score",
        "return_probability",
        "risk_tier",
        "expected_return_exposure",
        "priority",
        "recommended_action",
        "risk_reasons"
    ]

    data[
        dashboard_columns
    ].to_csv(
        DASHBOARD_FILE,
        index=False
    )

    print(
        "\n" + "=" * 70
    )

    print(
        "FILES SAVED"
    )

    print(
        "=" * 70
    )

    print(
        f"✓ {OUTPUT_FILE}"
    )

    print(
        f"✓ {DASHBOARD_FILE}"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    data = load_data()

    result = create_decisions(
        data
    )

    show_summary(
        result
    )

    show_top_orders(
        result
    )

    save_results(
        result
    )

    print(
        "\n" + "=" * 70
    )

    print(
        "RISK DECISION ENGINE COMPLETE 🚀"
    )

    print(
        "=" * 70
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()