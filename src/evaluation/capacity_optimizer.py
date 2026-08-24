import os
import pandas as pd


# ============================================================
# RETURN-RISK INTELLIGENCE
# CAPACITY-AWARE INTERVENTION OPTIMIZER
# ============================================================

INPUT_PATH = "evaluation/risk_scores.csv"

OUTPUT_PATH = (
    "evaluation/capacity_optimization.csv"
)


# ============================================================
# BUSINESS ASSUMPTIONS
# ============================================================

# Same financial assumption used by the API
RETURN_COST_RATE = 0.08

# Percentage of expected financial exposure
# that can potentially be avoided.
INTERVENTION_EFFECTIVENESS = 0.40

# Cost of one intervention
INTERVENTION_COST = 150.00


# ============================================================
# OPERATIONAL CAPACITIES
# ============================================================

CAPACITIES = [
    0.05,
    0.10,
    0.15,
    0.20,
    0.25,
    0.30
]


# ============================================================
# LOAD DATA
# ============================================================

def load_data():

    print(
        "Loading risk scores..."
    )

    data = pd.read_csv(
        INPUT_PATH
    )

    print(
        f"Orders loaded: {len(data):,}"
    )

    required = [
        "order_id",
        "order_value",
        "return_probability"
    ]

    missing = [
        column
        for column in required
        if column not in data.columns
    ]

    if missing:

        raise ValueError(
            "Missing required columns: "
            + str(missing)
        )

    return data


# ============================================================
# PREPARE DATA
# ============================================================

def prepare_data(data):

    data = data.copy()

    # --------------------------------------------------------
    # Expected financial exposure
    #
    # Order Value
    # × Return Probability
    # × Return Cost Rate
    # --------------------------------------------------------

    data[
        "baseline_expected_loss"
    ] = (
        data["order_value"]
        *
        data["return_probability"]
        *
        RETURN_COST_RATE
    )

    return data


# ============================================================
# CAPACITY ANALYSIS
# ============================================================

def analyze_capacity(
    data,
    capacity
):

    # --------------------------------------------------------
    # Number of orders allowed
    # --------------------------------------------------------

    number_of_orders = int(
        len(data) * capacity
    )

    number_of_orders = max(
        1,
        number_of_orders
    )

    # --------------------------------------------------------
    # Rank orders by expected financial exposure
    # --------------------------------------------------------

    ranked = (
        data
        .sort_values(
            "baseline_expected_loss",
            ascending=False
        )
    )

    # --------------------------------------------------------
    # Select highest exposure orders
    # --------------------------------------------------------

    selected = (
        ranked
        .head(
            number_of_orders
        )
        .copy()
    )

    # --------------------------------------------------------
    # Baseline expected loss
    # --------------------------------------------------------

    baseline_loss = (
        selected[
            "baseline_expected_loss"
        ]
        .sum()
    )

    # --------------------------------------------------------
    # Potential loss avoided
    # --------------------------------------------------------

    potential_loss_avoided = (
        baseline_loss
        *
        INTERVENTION_EFFECTIVENESS
    )

    # --------------------------------------------------------
    # Intervention cost
    # --------------------------------------------------------

    intervention_cost = (
        number_of_orders
        *
        INTERVENTION_COST
    )

    # --------------------------------------------------------
    # Net benefit
    # --------------------------------------------------------

    net_benefit = (
        potential_loss_avoided
        -
        intervention_cost
    )

    # --------------------------------------------------------
    # ROI
    # --------------------------------------------------------

    if intervention_cost > 0:

        roi = (
            net_benefit
            /
            intervention_cost
        )

    else:

        roi = 0.0

    # --------------------------------------------------------
    # Expected return capture
    # --------------------------------------------------------

    selected_expected_returns = (
        selected[
            "return_probability"
        ]
        .sum()
    )

    total_expected_returns = (
        data[
            "return_probability"
        ]
        .sum()
    )

    if total_expected_returns > 0:

        capture_rate = (
            selected_expected_returns
            /
            total_expected_returns
        )

    else:

        capture_rate = 0.0

    # --------------------------------------------------------
    # Average risk
    # --------------------------------------------------------

    average_probability = (
        selected[
            "return_probability"
        ]
        .mean()
    )

    # --------------------------------------------------------
    # Average order value
    # --------------------------------------------------------

    average_order_value = (
        selected[
            "order_value"
        ]
        .mean()
    )

    return {

        "capacity":
            capacity,

        "orders_selected":
            number_of_orders,

        "selection_rate":
            number_of_orders / len(data),

        "average_return_probability":
            average_probability,

        "average_order_value":
            average_order_value,

        "baseline_expected_loss":
            baseline_loss,

        "potential_loss_avoided":
            potential_loss_avoided,

        "intervention_cost":
            intervention_cost,

        "net_benefit":
            net_benefit,

        "roi":
            roi,

        "expected_return_capture_rate":
            capture_rate
    }


# ============================================================
# RUN OPTIMIZATION
# ============================================================

def optimize(data):

    results = []

    for capacity in CAPACITIES:

        result = analyze_capacity(
            data,
            capacity
        )

        results.append(
            result
        )

    return pd.DataFrame(
        results
    )


# ============================================================
# PRINT RESULTS
# ============================================================

def print_results(results):

    print()

    print(
        "=" * 70
    )

    print(
        "CAPACITY-AWARE INTERVENTION ANALYSIS"
    )

    print(
        "=" * 70
    )

    display = results.copy()

    display[
        "capacity"
    ] *= 100

    display[
        "selection_rate"
    ] *= 100

    display[
        "expected_return_capture_rate"
    ] *= 100

    print(
        display.to_string(
            index=False
        )
    )


# ============================================================
# BEST CAPACITY
# ============================================================

def print_best(results):

    # --------------------------------------------------------
    # Best by net benefit
    # --------------------------------------------------------

    best = results.loc[
        results[
            "net_benefit"
        ].idxmax()
    ]

    # --------------------------------------------------------
    # Best by ROI
    # --------------------------------------------------------

    best_roi = results.loc[
        results[
            "roi"
        ].idxmax()
    ]

    print()

    print(
        "=" * 70
    )

    print(
        "BEST CAPACITY"
    )

    print(
        "=" * 70
    )

    print(
        f"Best capacity by net benefit : "
        f"{best['capacity'] * 100:.0f}%"
    )

    print(
        f"Orders selected              : "
        f"{int(best['orders_selected']):,}"
    )

    print(
        f"Average return probability   : "
        f"{best['average_return_probability'] * 100:.2f}%"
    )

    print(
        f"Potential loss avoided       : "
        f"₹{best['potential_loss_avoided']:,.2f}"
    )

    print(
        f"Intervention cost            : "
        f"₹{best['intervention_cost']:,.2f}"
    )

    print(
        f"Net benefit                  : "
        f"₹{best['net_benefit']:,.2f}"
    )

    print(
        f"ROI                          : "
        f"{best['roi']:.2f}x"
    )

    print(
        f"Expected return capture     : "
        f"{best['expected_return_capture_rate'] * 100:.2f}%"
    )

    print()

    print(
        "-" * 70
    )

    print(
        f"Best ROI capacity            : "
        f"{best_roi['capacity'] * 100:.0f}%"
    )

    print(
        f"ROI                          : "
        f"{best_roi['roi']:.2f}x"
    )


# ============================================================
# SAVE RESULTS
# ============================================================

def save_results(results):

    os.makedirs(
        "evaluation",
        exist_ok=True
    )

    results.to_csv(
        OUTPUT_PATH,
        index=False
    )

    print()

    print(
        "=" * 70
    )

    print(
        f"✓ Saved: {OUTPUT_PATH}"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print(
        "=" * 70
    )

    print(
        "RETURN-RISK INTELLIGENCE"
    )

    print(
        "CAPACITY-AWARE INTERVENTION OPTIMIZER"
    )

    print(
        "=" * 70
    )

    # --------------------------------------------------------
    # Load
    # --------------------------------------------------------

    data = load_data()

    # --------------------------------------------------------
    # Prepare
    # --------------------------------------------------------

    data = prepare_data(
        data
    )

    print(
        "\nOptimizing intervention capacity..."
    )

    # --------------------------------------------------------
    # Optimize
    # --------------------------------------------------------

    results = optimize(
        data
    )

    # --------------------------------------------------------
    # Display
    # --------------------------------------------------------

    print_results(
        results
    )

    # --------------------------------------------------------
    # Best capacity
    # --------------------------------------------------------

    print_best(
        results
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    save_results(
        results
    )

    print()

    print(
        "=" * 70
    )

    print(
        "CAPACITY OPTIMIZATION COMPLETE 🚀"
    )

    print(
        "=" * 70
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()