import os
import pandas as pd


# ============================================================
# RETURN-RISK INTELLIGENCE
# INTERVENTION SIMULATOR
# ============================================================

RISK_SCORES_PATH = (
    "evaluation/risk_scores.csv"
)

RISK_DECISIONS_PATH = (
    "evaluation/risk_decisions.csv"
)

OUTPUT_PATH = (
    "evaluation/intervention_simulation.csv"
)


# ============================================================
# BUSINESS ASSUMPTIONS
# ============================================================

# Same financial assumption used by the API
RETURN_COST_RATE = 0.08


# Expected reduction in financial exposure
# for each intervention type
ACTION_EFFECTIVENESS = {

    "NORMAL_PROCESSING": 0.00,

    "MONITOR": 0.10,

    "REVIEW": 0.25,

    "INTERVENE": 0.40
}


# Cost of each intervention
ACTION_COST = {

    "NORMAL_PROCESSING": 0.00,

    "MONITOR": 20.00,

    "REVIEW": 75.00,

    "INTERVENE": 150.00
}


# ============================================================
# LOAD DATA
# ============================================================

def load_data():

    print(
        "Loading evaluation datasets..."
    )

    risk_scores = pd.read_csv(
        RISK_SCORES_PATH
    )

    risk_decisions = pd.read_csv(
        RISK_DECISIONS_PATH
    )

    print(
        f"Risk scores : "
        f"{len(risk_scores):,}"
    )

    print(
        f"Decisions   : "
        f"{len(risk_decisions):,}"
    )

    return (
        risk_scores,
        risk_decisions
    )


# ============================================================
# PREPARE DATA
# ============================================================

def prepare_data(
    risk_scores,
    risk_decisions
):

    data = risk_scores.merge(
        risk_decisions[
            [
                "order_id",
                "recommended_action"
            ]
        ],
        on="order_id",
        how="left"
    )

    # --------------------------------------------------------
    # BASELINE EXPECTED FINANCIAL LOSS
    #
    # Same business assumption as API:
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
# SIMULATE INTERVENTIONS
# ============================================================

def simulate_interventions(
    data
):

    results = []

    for _, row in data.iterrows():

        action = row[
            "recommended_action"
        ]

        baseline_loss = row[
            "baseline_expected_loss"
        ]

        # ----------------------------------------------------
        # Intervention effectiveness
        # ----------------------------------------------------

        effectiveness = (
            ACTION_EFFECTIVENESS.get(
                action,
                0.0
            )
        )

        # ----------------------------------------------------
        # Potential financial loss avoided
        # ----------------------------------------------------

        potential_loss_avoided = (
            baseline_loss
            *
            effectiveness
        )

        # ----------------------------------------------------
        # Action-specific intervention cost
        # ----------------------------------------------------

        intervention_cost = (
            ACTION_COST.get(
                action,
                0.0
            )
        )

        # ----------------------------------------------------
        # Net benefit
        # ----------------------------------------------------

        net_benefit = (
            potential_loss_avoided
            -
            intervention_cost
        )

        # ----------------------------------------------------
        # ROI
        # ----------------------------------------------------

        if intervention_cost > 0:

            roi = (
                net_benefit
                /
                intervention_cost
            )

        else:

            roi = 0.0

        # ----------------------------------------------------
        # Expected loss after intervention
        # ----------------------------------------------------

        expected_loss_after = (
            baseline_loss
            -
            potential_loss_avoided
        )

        results.append({

            "order_id":
                row["order_id"],

            "order_value":
                row["order_value"],

            "return_probability":
                row["return_probability"],

            "recommended_action":
                action,

            "baseline_expected_loss":
                baseline_loss,

            "potential_loss_avoided":
                potential_loss_avoided,

            "intervention_cost":
                intervention_cost,

            "net_benefit":
                net_benefit,

            "intervention_roi":
                roi,

            "expected_loss_after_intervention":
                expected_loss_after
        })

    return pd.DataFrame(
        results
    )


# ============================================================
# BUSINESS VALUE
# ============================================================

def calculate_business_value(
    results
):

    baseline_loss = (
        results[
            "baseline_expected_loss"
        ].sum()
    )

    expected_loss_after = (
        results[
            "expected_loss_after_intervention"
        ].sum()
    )

    potential_loss_avoided = (
        results[
            "potential_loss_avoided"
        ].sum()
    )

    intervention_cost = (
        results[
            "intervention_cost"
        ].sum()
    )

    net_benefit = (
        potential_loss_avoided
        -
        intervention_cost
    )

    if intervention_cost > 0:

        overall_roi = (
            net_benefit
            /
            intervention_cost
        )

    else:

        overall_roi = 0.0

    return {

        "baseline_loss":
            baseline_loss,

        "expected_loss_after":
            expected_loss_after,

        "potential_loss_avoided":
            potential_loss_avoided,

        "intervention_cost":
            intervention_cost,

        "net_benefit":
            net_benefit,

        "overall_roi":
            overall_roi
    }


# ============================================================
# ACTION-WISE ANALYSIS
# ============================================================

def action_wise_analysis(
    results
):

    summary = (
        results
        .groupby(
            "recommended_action"
        )
        .agg(

            orders=(
                "order_id",
                "count"
            ),

            baseline_loss=(
                "baseline_expected_loss",
                "sum"
            ),

            avoided_loss=(
                "potential_loss_avoided",
                "sum"
            ),

            intervention_cost=(
                "intervention_cost",
                "sum"
            ),

            net_benefit=(
                "net_benefit",
                "sum"
            )
        )
        .reset_index()
    )

    summary["roi"] = 0.0

    mask = (
        summary[
            "intervention_cost"
        ]
        > 0
    )

    summary.loc[
        mask,
        "roi"
    ] = (
        summary.loc[
            mask,
            "net_benefit"
        ]
        /
        summary.loc[
            mask,
            "intervention_cost"
        ]
    )

    return summary


# ============================================================
# TOP ORDERS
# ============================================================

def top_orders(
    results,
    n=20
):

    return (
        results
        .sort_values(
            "net_benefit",
            ascending=False
        )
        .head(n)
    )


# ============================================================
# SAVE RESULTS
# ============================================================

def save_results(
    results
):

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
        "FILE SAVED"
    )

    print(
        "=" * 70
    )

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
        "INTERVENTION SIMULATOR"
    )

    print("=" * 70)


    # --------------------------------------------------------
    # Load
    # --------------------------------------------------------

    risk_scores, risk_decisions = (
        load_data()
    )


    # --------------------------------------------------------
    # Prepare
    # --------------------------------------------------------

    data = prepare_data(
        risk_scores,
        risk_decisions
    )


    print()

    print(
        "Calculating baseline expected loss..."
    )

    print(
        "Simulating interventions..."
    )


    # --------------------------------------------------------
    # Simulation
    # --------------------------------------------------------

    results = (
        simulate_interventions(
            data
        )
    )


    # --------------------------------------------------------
    # Business value
    # --------------------------------------------------------

    business = (
        calculate_business_value(
            results
        )
    )


    # --------------------------------------------------------
    # Action-wise analysis
    # --------------------------------------------------------

    action_summary = (
        action_wise_analysis(
            results
        )
    )


    # --------------------------------------------------------
    # Top orders
    # --------------------------------------------------------

    top = top_orders(
        results
    )


    # ========================================================
    # BUSINESS VALUE OUTPUT
    # ========================================================

    print()

    print(
        "=" * 70
    )

    print(
        "INTERVENTION BUSINESS VALUE"
    )

    print(
        "=" * 70
    )


    print(
        f"Baseline expected loss : "
        f"₹{business['baseline_loss']:,.2f}"
    )

    print(
        f"Expected loss after    : "
        f"₹{business['expected_loss_after']:,.2f}"
    )

    print(
        f"Potential loss avoided : "
        f"₹{business['potential_loss_avoided']:,.2f}"
    )

    print(
        f"Intervention cost      : "
        f"₹{business['intervention_cost']:,.2f}"
    )

    print(
        f"Net benefit            : "
        f"₹{business['net_benefit']:,.2f}"
    )

    print(
        f"Overall intervention ROI : "
        f"{business['overall_roi']:.2f}x"
    )


    # ========================================================
    # ACTION-WISE OUTPUT
    # ========================================================

    print()

    print(
        "=" * 70
    )

    print(
        "ACTION-WISE ANALYSIS"
    )

    print(
        "=" * 70
    )

    print(
        action_summary.to_string(
            index=False
        )
    )


    # ========================================================
    # TOP 20 OUTPUT
    # ========================================================

    print()

    print(
        "=" * 70
    )

    print(
        "TOP 20 ORDERS BY POTENTIAL NET BENEFIT"
    )

    print(
        "=" * 70
    )

    print(
        top.to_string(
            index=False
        )
    )


    # ========================================================
    # SAVE
    # ========================================================

    save_results(
        results
    )


    print()

    print(
        "=" * 70
    )

    print(
        "INTERVENTION SIMULATION COMPLETE 🚀"
    )

    print(
        "=" * 70
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()