import pandas as pd
from pathlib import Path


# ============================================================
# FILES
# ============================================================

DECISIONS_FILE = Path(
    "evaluation/risk_decisions.csv"
)

EXPLANATIONS_FILE = Path(
    "evaluation/risk_explanations.csv"
)

OUTPUT_DIR = Path(
    "evaluation"
)

OUTPUT_FILE = (
    OUTPUT_DIR /
    "unified_risk_dataset.csv"
)


# ============================================================
# LOAD
# ============================================================

def load_files():

    print("=" * 70)
    print("RETURN-RISK INTELLIGENCE")
    print("UNIFIED RISK DATASET BUILDER")
    print("=" * 70)

    print(
        "\nLoading risk decisions..."
    )

    decisions = pd.read_csv(
        DECISIONS_FILE
    )

    print(
        f"Risk decisions: "
        f"{len(decisions):,}"
    )

    print(
        "\nLoading SHAP explanations..."
    )

    explanations = pd.read_csv(
        EXPLANATIONS_FILE
    )

    print(
        f"Explanations: "
        f"{len(explanations):,}"
    )

    return (
        decisions,
        explanations
    )


# ============================================================
# VALIDATE
# ============================================================

def validate_data(
    decisions,
    explanations
):

    print(
        "\nValidating datasets..."
    )

    if (
        decisions["order_id"]
        .duplicated()
        .any()
    ):

        raise ValueError(
            "Duplicate order IDs found "
            "in risk decisions."
        )

    if (
        explanations["order_id"]
        .duplicated()
        .any()
    ):

        raise ValueError(
            "Duplicate order IDs found "
            "in explanations."
        )

    decision_ids = set(
        decisions["order_id"]
    )

    explanation_ids = set(
        explanations["order_id"]
    )

    missing_explanations = (
        decision_ids
        -
        explanation_ids
    )

    if missing_explanations:

        print(
            f"Warning: "
            f"{len(missing_explanations)} "
            "orders have no explanation."
        )

    print(
        "Validation passed ✅"
    )


# ============================================================
# MERGE
# ============================================================

def build_dataset(
    decisions,
    explanations
):

    print(
        "\nCombining risk decisions "
        "with explanations..."
    )

    explanation_columns = [
        "order_id",
        "top_risk_drivers",
        "risk_reducing_factors"
    ]

    explanations = explanations[
        explanation_columns
    ]

    unified = decisions.merge(
        explanations,
        on="order_id",
        how="left"
    )

    return unified


# ============================================================
# ORGANIZE COLUMNS
# ============================================================

def organize_columns(
    data
):

    preferred_columns = [

        # Order
        "order_id",
        "order_value",

        # Model
        "return_probability",
        "risk_score",

        # Decision
        "risk_tier",
        "expected_return_exposure",
        "priority",
        "recommended_action",

        # Explainability
        "risk_reasons",
        "top_risk_drivers",
        "risk_reducing_factors"
    ]

    existing_columns = [
        column
        for column in preferred_columns
        if column in data.columns
    ]

    remaining_columns = [
        column
        for column in data.columns
        if column not in existing_columns
    ]

    return data[
        existing_columns
        +
        remaining_columns
    ]


# ============================================================
# QUALITY CHECK
# ============================================================

def quality_check(
    data
):

    print(
        "\n" + "=" * 70
    )

    print(
        "UNIFIED DATASET VALIDATION"
    )

    print(
        "=" * 70
    )

    print(
        f"\nRows    : {len(data):,}"
    )

    print(
        f"Columns : {len(data.columns)}"
    )

    print(
        "\nImportant columns:"
    )

    required = [
        "order_id",
        "return_probability",
        "risk_score",
        "risk_tier",
        "expected_return_exposure",
        "priority",
        "recommended_action",
        "top_risk_drivers",
        "risk_reducing_factors"
    ]

    for column in required:

        if column in data.columns:

            print(
                f"  ✓ {column}"
            )

        else:

            print(
                f"  ✗ {column}"
            )

    print(
        "\nRisk distribution:"
    )

    print(
        data[
            "risk_tier"
        ].value_counts(
            sort=False
        )
    )

    print(
        "\nAction distribution:"
    )

    print(
        data[
            "recommended_action"
        ].value_counts()
    )


# ============================================================
# SHOW SAMPLE
# ============================================================

def show_sample(
    data
):

    print(
        "\n" + "=" * 70
    )

    print(
        "SAMPLE UNIFIED RECORDS"
    )

    print(
        "=" * 70
    )

    columns = [
        "order_id",
        "risk_score",
        "risk_tier",
        "expected_return_exposure",
        "priority",
        "recommended_action"
    ]

    print(
        data[
            columns
        ]
        .sort_values(
            "risk_score",
            ascending=False
        )
        .head(10)
        .to_string(
            index=False
        )
    )


# ============================================================
# SAVE
# ============================================================

def save_dataset(
    data
):

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    data.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print(
        "\nSaved:"
    )

    print(
        f"✓ {OUTPUT_FILE}"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    decisions, explanations = (
        load_files()
    )

    validate_data(
        decisions,
        explanations
    )

    unified = build_dataset(
        decisions,
        explanations
    )

    unified = organize_columns(
        unified
    )

    quality_check(
        unified
    )

    show_sample(
        unified
    )

    save_dataset(
        unified
    )

    print(
        "\n" + "=" * 70
    )

    print(
        "UNIFIED RISK DATASET COMPLETE 🚀"
    )

    print(
        "=" * 70
    )


if __name__ == "__main__":

    main()