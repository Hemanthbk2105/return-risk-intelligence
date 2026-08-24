import os
import pandas as pd


# ============================================================
# RETURN-RISK INTELLIGENCE
# RANKING STRATEGY COMPARISON
# ============================================================

INPUT_PATH = "evaluation/risk_scores.csv"

OUTPUT_PATH = (
    "evaluation/ranking_strategy_comparison.csv"
)


# ============================================================
# BUSINESS ASSUMPTIONS
# ============================================================

INTERVENTION_EFFECTIVENESS = 0.40

INTERVENTION_COST = 150.00


# Capacities to compare

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

    return data


# ============================================================
# VALIDATE
# ============================================================

def validate_data(data):

    required_columns = [

        "order_id",

        "order_value",

        "return_probability"
    ]


    missing = [

        column

        for column in required_columns

        if column not in data.columns
    ]


    if missing:

        raise ValueError(
            "Missing required columns: "
            + str(missing)
        )


# ============================================================
# PREPARE FEATURES
# ============================================================

def prepare_data(data):

    data = data.copy()


    # Expected financial exposure

    data[
        "expected_return_exposure"
    ] = (

        data[
            "order_value"
        ]

        *

        data[
            "return_probability"
        ]
    )


    return data


# ============================================================
# ANALYZE ONE STRATEGY
# ============================================================

def analyze_strategy(
    data,
    capacity,
    strategy
):

    number_of_orders = int(
        len(data) * capacity
    )


    number_of_orders = max(
        1,
        number_of_orders
    )


    # --------------------------------------------------------
    # Probability strategy
    # --------------------------------------------------------

    if strategy == "RETURN_PROBABILITY":

        ranked = data.sort_values(
            "return_probability",
            ascending=False
        )


    # --------------------------------------------------------
    # Financial exposure strategy
    # --------------------------------------------------------

    elif strategy == "EXPECTED_EXPOSURE":

        ranked = data.sort_values(
            "expected_return_exposure",
            ascending=False
        )


    else:

        raise ValueError(
            "Unknown strategy: "
            + str(strategy)
        )


    selected = ranked.head(
        number_of_orders
    ).copy()


    # --------------------------------------------------------
    # Baseline expected loss
    # --------------------------------------------------------

    baseline_loss = selected[
        "expected_return_exposure"
    ].sum()


    # --------------------------------------------------------
    # Potential avoided loss
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
        ].sum()
    )


    total_expected_returns = (
        data[
            "return_probability"
        ].sum()
    )


    if total_expected_returns > 0:

        return_capture = (

            selected_expected_returns

            /

            total_expected_returns
        )

    else:

        return_capture = 0.0


    # --------------------------------------------------------
    # Average values
    # --------------------------------------------------------

    average_order_value = (
        selected[
            "order_value"
        ].mean()
    )


    average_probability = (
        selected[
            "return_probability"
        ].mean()
    )


    return {

        "strategy":
            strategy,

        "capacity":
            capacity,

        "orders_selected":
            number_of_orders,

        "selection_rate":
            number_of_orders
            /
            len(data),

        "average_order_value":
            average_order_value,

        "average_return_probability":
            average_probability,

        "baseline_expected_exposure":
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
            return_capture
    }


# ============================================================
# RUN COMPARISON
# ============================================================

def run_comparison(data):

    results = []


    strategies = [

        "RETURN_PROBABILITY",

        "EXPECTED_EXPOSURE"
    ]


    for capacity in CAPACITIES:

        for strategy in strategies:

            result = analyze_strategy(

                data,

                capacity,

                strategy
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
    print("=" * 70)
    print(
        "RANKING STRATEGY COMPARISON"
    )
    print("=" * 70)


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


    display[
        "baseline_expected_exposure"
    ] = display[
        "baseline_expected_exposure"
    ].round(2)


    display[
        "potential_loss_avoided"
    ] = display[
        "potential_loss_avoided"
    ].round(2)


    display[
        "intervention_cost"
    ] = display[
        "intervention_cost"
    ].round(2)


    display[
        "net_benefit"
    ] = display[
        "net_benefit"
    ].round(2)


    display[
        "roi"
    ] = display[
        "roi"
    ].round(2)


    print(
        display.to_string(
            index=False
        )
    )


# ============================================================
# COMPARE STRATEGIES DIRECTLY
# ============================================================

def print_strategy_winners(results):

    print()
    print("=" * 70)
    print(
        "STRATEGY WINNERS"
    )
    print("=" * 70)


    for capacity in CAPACITIES:

        subset = results[
            results[
                "capacity"
            ] == capacity
        ]


        best = subset.loc[
            subset[
                "net_benefit"
            ].idxmax()
        ]


        probability = subset[
            subset[
                "strategy"
            ]
            ==
            "RETURN_PROBABILITY"
        ].iloc[0]


        exposure = subset[
            subset[
                "strategy"
            ]
            ==
            "EXPECTED_EXPOSURE"
        ].iloc[0]


        print()
        print(
            f"Capacity: {capacity * 100:.0f}%"
        )


        print(
            f"Probability ranking "
            f"net benefit: "
            f"₹{probability['net_benefit']:,.2f}"
        )


        print(
            f"Financial ranking "
            f"net benefit: "
            f"₹{exposure['net_benefit']:,.2f}"
        )


        difference = (

            exposure[
                "net_benefit"
            ]

            -

            probability[
                "net_benefit"
            ]
        )


        print(
            f"Financial ranking advantage: "
            f"₹{difference:,.2f}"
        )


        print(
            f"Winner: "
            f"{best['strategy']}"
        )


# ============================================================
# BEST OVERALL STRATEGY
# ============================================================

def print_best_overall(results):

    best_net = results.loc[
        results[
            "net_benefit"
        ].idxmax()
    ]


    best_roi = results.loc[
        results[
            "roi"
        ].idxmax()
    ]


    print()
    print("=" * 70)
    print(
        "BEST OVERALL STRATEGY"
    )
    print("=" * 70)


    print(
        "Maximum net benefit:"
    )


    print(
        f"Strategy       : "
        f"{best_net['strategy']}"
    )


    print(
        f"Capacity       : "
        f"{best_net['capacity'] * 100:.0f}%"
    )


    print(
        f"Net benefit    : "
        f"₹{best_net['net_benefit']:,.2f}"
    )


    print(
        f"ROI            : "
        f"{best_net['roi']:.2f}x"
    )


    print()
    print(
        "Maximum ROI:"
    )


    print(
        f"Strategy       : "
        f"{best_roi['strategy']}"
    )


    print(
        f"Capacity       : "
        f"{best_roi['capacity'] * 100:.0f}%"
    )


    print(
        f"ROI            : "
        f"{best_roi['roi']:.2f}x"
    )


# ============================================================
# SAVE
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
    print("=" * 70)
    print(
        "FILE SAVED"
    )
    print("=" * 70)


    print(
        f"✓ {OUTPUT_PATH}"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print(
        "RETURN-RISK INTELLIGENCE"
    )

    print(
        "RANKING STRATEGY COMPARISON"
    )

    print("=" * 70)


    data = load_data()


    validate_data(
        data
    )


    data = prepare_data(
        data
    )


    print(
        "\nComparing ranking strategies..."
    )


    results = run_comparison(
        data
    )


    print_results(
        results
    )


    print_strategy_winners(
        results
    )


    print_best_overall(
        results
    )


    save_results(
        results
    )


    print()
    print("=" * 70)
    print(
        "RANKING COMPARISON COMPLETE 🚀"
    )
    print("=" * 70)


if __name__ == "__main__":

    main()