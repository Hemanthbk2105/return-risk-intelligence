import os
import pandas as pd


# ============================================================
# RETURN-RISK INTELLIGENCE
# COST-SENSITIVE THRESHOLD OPTIMIZER
# ============================================================

RISK_SCORES_PATH = (
    "evaluation/risk_scores.csv"
)

OUTPUT_PATH = (
    "evaluation/optimal_threshold_analysis.csv"
)

SUMMARY_PATH = (
    "evaluation/optimal_threshold_summary.csv"
)


# ============================================================
# BUSINESS ASSUMPTIONS
# ============================================================

# Same financial assumption used by the API
RETURN_COST_RATE = 0.08


# Percentage of expected financial exposure
# avoided when an intervention is applied.
INTERVENTION_EFFECTIVENESS = 0.40


# Cost of one intervention
INTERVENTION_COST = 150.00


# ============================================================
# THRESHOLDS
# ============================================================

# Test every threshold from 0.10 to 0.90
# in increments of 0.01.

THRESHOLDS = [
    round(
        value / 100,
        2
    )
    for value in range(
        10,
        91
    )
]


# ============================================================
# LOAD DATA
# ============================================================

def load_data():

    print(
        "Loading risk scores..."
    )

    data = pd.read_csv(
        RISK_SCORES_PATH
    )

    print(
        f"Orders loaded: {len(data):,}"
    )

    return data


# ============================================================
# VALIDATE DATA
# ============================================================

def validate_data(
    data
):

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
# PREPARE DATA
# ============================================================

def prepare_data(
    data
):

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
# ANALYZE ONE THRESHOLD
# ============================================================

def analyze_threshold(
    data,
    threshold
):

    # --------------------------------------------------------
    # Select orders above threshold
    # --------------------------------------------------------

    selected = (
        data["return_probability"]
        >= threshold
    )

    selected_orders = (
        data.loc[
            selected
        ]
        .copy()
    )


    # --------------------------------------------------------
    # Number of selected orders
    # --------------------------------------------------------

    orders_selected = (
        len(selected_orders)
    )


    # --------------------------------------------------------
    # Baseline expected loss
    # --------------------------------------------------------

    baseline_loss = (
        selected_orders[
            "baseline_expected_loss"
        ]
        .sum()
    )


    # --------------------------------------------------------
    # Potential loss avoided
    # --------------------------------------------------------

    loss_avoided = (
        baseline_loss
        *
        INTERVENTION_EFFECTIVENESS
    )


    # --------------------------------------------------------
    # Intervention cost
    # --------------------------------------------------------

    intervention_cost = (
        orders_selected
        *
        INTERVENTION_COST
    )


    # --------------------------------------------------------
    # Net benefit
    # --------------------------------------------------------

    net_benefit = (
        loss_avoided
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
    # Selection rate
    # --------------------------------------------------------

    if len(data) > 0:

        selection_rate = (
            orders_selected
            /
            len(data)
        )

    else:

        selection_rate = 0.0


    # --------------------------------------------------------
    # Expected return capture
    # --------------------------------------------------------

    expected_returns_captured = (
        selected_orders[
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

        expected_return_capture_rate = (
            expected_returns_captured
            /
            total_expected_returns
        )

    else:

        expected_return_capture_rate = 0.0


    # --------------------------------------------------------
    # Average selected risk
    # --------------------------------------------------------

    if orders_selected > 0:

        average_return_probability = (
            selected_orders[
                "return_probability"
            ]
            .mean()
        )

    else:

        average_return_probability = 0.0


    # --------------------------------------------------------
    # Return result
    # --------------------------------------------------------

    return {

        "threshold":
            threshold,

        "orders_selected":
            orders_selected,

        "selection_rate":
            selection_rate,

        "average_return_probability":
            average_return_probability,

        "baseline_expected_loss":
            baseline_loss,

        "potential_loss_avoided":
            loss_avoided,

        "intervention_cost":
            intervention_cost,

        "net_benefit":
            net_benefit,

        "roi":
            roi,

        "expected_returns_captured":
            expected_returns_captured,

        "expected_return_capture_rate":
            expected_return_capture_rate
    }


# ============================================================
# RUN ALL THRESHOLDS
# ============================================================

def run_analysis(
    data
):

    results = []

    for threshold in THRESHOLDS:

        result = analyze_threshold(
            data,
            threshold
        )

        results.append(
            result
        )

    return pd.DataFrame(
        results
    )


# ============================================================
# FIND BEST THRESHOLDS
# ============================================================

def find_best_threshold(
    results
):

    # --------------------------------------------------------
    # Best by net benefit
    # --------------------------------------------------------

    best_net = results.loc[
        results[
            "net_benefit"
        ]
        .idxmax()
    ]


    # --------------------------------------------------------
    # Best by ROI
    # --------------------------------------------------------

    best_roi = results.loc[
        results[
            "roi"
        ]
        .idxmax()
    ]


    return (
        best_net,
        best_roi
    )


# ============================================================
# PRINT ANALYSIS
# ============================================================

def print_analysis(
    results
):

    print()

    print(
        "=" * 70
    )

    print(
        "THRESHOLD ANALYSIS"
    )

    print(
        "=" * 70
    )


    display_columns = [

        "threshold",

        "orders_selected",

        "selection_rate",

        "average_return_probability",

        "potential_loss_avoided",

        "intervention_cost",

        "net_benefit",

        "roi",

        "expected_return_capture_rate"
    ]


    display = results[
        display_columns
    ].copy()


    # Convert rates to percentages
    display[
        "selection_rate"
    ] *= 100


    display[
        "average_return_probability"
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
# PRINT BEST RESULTS
# ============================================================

def print_best_results(
    best_net,
    best_roi
):

    print()

    print(
        "=" * 70
    )

    print(
        "OPTIMAL BUSINESS THRESHOLD"
    )

    print(
        "=" * 70
    )


    # --------------------------------------------------------
    # Best net benefit
    # --------------------------------------------------------

    print(
        f"Best threshold by net benefit : "
        f"{best_net['threshold']:.2f}"
    )


    print(
        f"Orders selected               : "
        f"{int(best_net['orders_selected']):,}"
    )


    print(
        f"Selection rate                : "
        f"{best_net['selection_rate'] * 100:.2f}%"
    )


    print(
        f"Average return probability    : "
        f"{best_net['average_return_probability'] * 100:.2f}%"
    )


    print(
        f"Potential loss avoided        : "
        f"₹{best_net['potential_loss_avoided']:,.2f}"
    )


    print(
        f"Intervention cost             : "
        f"₹{best_net['intervention_cost']:,.2f}"
    )


    print(
        f"Net benefit                   : "
        f"₹{best_net['net_benefit']:,.2f}"
    )


    print(
        f"ROI                           : "
        f"{best_net['roi']:.2f}x"
    )


    print(
        f"Expected return capture      : "
        f"{best_net['expected_return_capture_rate'] * 100:.2f}%"
    )


    # --------------------------------------------------------
    # Best ROI
    # --------------------------------------------------------

    print()

    print(
        "-" * 70
    )


    print(
        f"Best threshold by ROI         : "
        f"{best_roi['threshold']:.2f}"
    )


    print(
        f"Orders selected               : "
        f"{int(best_roi['orders_selected']):,}"
    )


    print(
        f"ROI                           : "
        f"{best_roi['roi']:.2f}x"
    )


# ============================================================
# SAVE RESULTS
# ============================================================

def save_results(
    results,
    best_net,
    best_roi
):

    os.makedirs(
        "evaluation",
        exist_ok=True
    )


    # --------------------------------------------------------
    # Save complete threshold analysis
    # --------------------------------------------------------

    results.to_csv(
        OUTPUT_PATH,
        index=False
    )


    # --------------------------------------------------------
    # Save summary
    # --------------------------------------------------------

    summary = pd.DataFrame([

        {
            "optimization_metric":
                "net_benefit",

            "optimal_threshold":
                best_net[
                    "threshold"
                ],

            "orders_selected":
                best_net[
                    "orders_selected"
                ],

            "selection_rate":
                best_net[
                    "selection_rate"
                ],

            "average_return_probability":
                best_net[
                    "average_return_probability"
                ],

            "potential_loss_avoided":
                best_net[
                    "potential_loss_avoided"
                ],

            "intervention_cost":
                best_net[
                    "intervention_cost"
                ],

            "net_benefit":
                best_net[
                    "net_benefit"
                ],

            "roi":
                best_net[
                    "roi"
                ],

            "expected_return_capture_rate":
                best_net[
                    "expected_return_capture_rate"
                ]
        },

        {
            "optimization_metric":
                "roi",

            "optimal_threshold":
                best_roi[
                    "threshold"
                ],

            "orders_selected":
                best_roi[
                    "orders_selected"
                ],

            "selection_rate":
                best_roi[
                    "selection_rate"
                ],

            "average_return_probability":
                best_roi[
                    "average_return_probability"
                ],

            "potential_loss_avoided":
                best_roi[
                    "potential_loss_avoided"
                ],

            "intervention_cost":
                best_roi[
                    "intervention_cost"
                ],

            "net_benefit":
                best_roi[
                    "net_benefit"
                ],

            "roi":
                best_roi[
                    "roi"
                ],

            "expected_return_capture_rate":
                best_roi[
                    "expected_return_capture_rate"
                ]
        }
    ])


    summary.to_csv(
        SUMMARY_PATH,
        index=False
    )


    print()

    print(
        "=" * 70
    )

    print(
        "FILES SAVED"
    )

    print(
        "=" * 70
    )


    print(
        f"✓ {OUTPUT_PATH}"
    )


    print(
        f"✓ {SUMMARY_PATH}"
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
        "COST-SENSITIVE THRESHOLD OPTIMIZER"
    )

    print(
        "=" * 70
    )


    # --------------------------------------------------------
    # Load
    # --------------------------------------------------------

    data = load_data()


    # --------------------------------------------------------
    # Validate
    # --------------------------------------------------------

    validate_data(
        data
    )


    # --------------------------------------------------------
    # Prepare
    # --------------------------------------------------------

    data = prepare_data(
        data
    )


    print()

    print(
        "Testing business thresholds..."
    )

    print(
        f"Testing {len(THRESHOLDS)} thresholds "
        f"from {THRESHOLDS[0]:.2f} "
        f"to {THRESHOLDS[-1]:.2f}"
    )


    # --------------------------------------------------------
    # Run optimization
    # --------------------------------------------------------

    results = run_analysis(
        data
    )


    # --------------------------------------------------------
    # Display all results
    # --------------------------------------------------------

    print_analysis(
        results
    )


    # --------------------------------------------------------
    # Find optimal thresholds
    # --------------------------------------------------------

    best_net, best_roi = (
        find_best_threshold(
            results
        )
    )


    # --------------------------------------------------------
    # Display best results
    # --------------------------------------------------------

    print_best_results(
        best_net,
        best_roi
    )


    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    save_results(
        results,
        best_net,
        best_roi
    )


    print()

    print(
        "=" * 70
    )

    print(
        "THRESHOLD OPTIMIZATION COMPLETE 🚀"
    )

    print(
        "=" * 70
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()