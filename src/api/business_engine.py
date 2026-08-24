# ============================================================
# RETURN-RISK INTELLIGENCE
# BUSINESS DECISION ENGINE
# ============================================================


# ============================================================
# DEFAULT COST ASSUMPTIONS
# ============================================================

# Synthetic business assumption:
# 8% of the order value is treated as the financial
# cost/exposure when a return occurs.

DEFAULT_RETURN_COST_RATE = 0.08


# ============================================================
# EXPECTED RETURN EXPOSURE
# ============================================================

def calculate_expected_return_exposure(
    order_value: float,
    return_probability: float,
    return_cost_rate: float = DEFAULT_RETURN_COST_RATE
):
    """
    Calculate expected financial exposure.

    Formula:

        Expected Exposure =
            Order Value
            × Return Probability
            × Return Cost Rate
    """

    if order_value < 0:
        raise ValueError(
            "Order value cannot be negative."
        )

    if not 0 <= return_probability <= 1:
        raise ValueError(
            "Return probability must be between 0 and 1."
        )

    if not 0 <= return_cost_rate <= 1:
        raise ValueError(
            "Return cost rate must be between 0 and 1."
        )

    expected_exposure = (
        order_value
        * return_probability
        * return_cost_rate
    )

    return round(
        expected_exposure,
        2
    )


# ============================================================
# FINANCIAL RISK LEVEL
# ============================================================

def get_financial_risk(
    expected_exposure: float
):
    """
    Convert expected financial exposure into
    a business-friendly financial risk level.
    """

    if expected_exposure < 250:

        return "LOW"

    elif expected_exposure < 1000:

        return "MEDIUM"

    elif expected_exposure < 2500:

        return "HIGH"

    elif expected_exposure < 5000:

        return "VERY_HIGH"

    else:

        return "CRITICAL"


# ============================================================
# COST-AWARE BUSINESS ACTION
# ============================================================

def get_cost_aware_action(
    expected_exposure: float,
    risk_score: float
):
    """
    Determine the recommended business action.

    Decision hierarchy:

    1. Critical financial exposure
       -> INTERVENE

    2. Very high financial exposure
       -> REVIEW

    3. High financial exposure + high ML risk
       -> REVIEW

    4. High financial exposure
       -> MONITOR

    5. Very high ML risk
       -> REVIEW

    6. Medium/high ML risk
       -> MONITOR

    7. Otherwise
       -> NORMAL_PROCESSING
    """

    # --------------------------------------------------------
    # CRITICAL FINANCIAL EXPOSURE
    # --------------------------------------------------------

    if expected_exposure >= 5000:

        return "INTERVENE"


    # --------------------------------------------------------
    # VERY HIGH FINANCIAL EXPOSURE
    # --------------------------------------------------------

    if expected_exposure >= 2500:

        return "REVIEW"


    # --------------------------------------------------------
    # HIGH FINANCIAL EXPOSURE
    #
    # If both financial exposure and ML risk are high,
    # human review is more appropriate.
    # --------------------------------------------------------

    if expected_exposure >= 1000:

        if risk_score >= 50:

            return "REVIEW"

        return "MONITOR"


    # --------------------------------------------------------
    # VERY HIGH ML RISK
    # --------------------------------------------------------

    if risk_score >= 70:

        return "REVIEW"


    # --------------------------------------------------------
    # MEDIUM / HIGH ML RISK
    # --------------------------------------------------------

    if risk_score >= 50:

        return "MONITOR"


    # --------------------------------------------------------
    # DEFAULT
    # --------------------------------------------------------

    return "NORMAL_PROCESSING"


# ============================================================
# COMPLETE BUSINESS DECISION
# ============================================================

def build_business_decision(
    order_value: float,
    return_probability: float,
    risk_score: float
):
    """
    Build the complete business decision.

    Returns:

        expected_return_exposure
        financial_risk
        cost_aware_action
    """

    # --------------------------------------------------------
    # Calculate expected financial exposure
    # --------------------------------------------------------

    expected_exposure = (
        calculate_expected_return_exposure(

            order_value=order_value,

            return_probability=return_probability
        )
    )


    # --------------------------------------------------------
    # Determine financial risk
    # --------------------------------------------------------

    financial_risk = (
        get_financial_risk(
            expected_exposure
        )
    )


    # --------------------------------------------------------
    # Determine recommended action
    # --------------------------------------------------------

    recommended_action = (
        get_cost_aware_action(

            expected_exposure=
                expected_exposure,

            risk_score=
                risk_score
        )
    )


    # --------------------------------------------------------
    # Return complete decision
    # --------------------------------------------------------

    return {

        "expected_return_exposure":
            expected_exposure,

        "financial_risk":
            financial_risk,

        "cost_aware_action":
            recommended_action
    }


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    print("=" * 70)
    print("RETURN-RISK INTELLIGENCE")
    print("BUSINESS DECISION ENGINE TEST")
    print("=" * 70)


    # --------------------------------------------------------
    # Test scenarios
    # --------------------------------------------------------

    test_cases = [

        {
            "name": "Low-value order",
            "order_value": 1000,
            "probability": 0.40,
            "risk_score": 40
        },

        {
            "name": "Medium-value order",
            "order_value": 10000,
            "probability": 0.40,
            "risk_score": 40
        },

        {
            "name": "High-value order",
            "order_value": 50000,
            "probability": 0.60,
            "risk_score": 60
        },

        {
            "name": "Critical exposure",
            "order_value": 100000,
            "probability": 0.80,
            "risk_score": 80
        }
    ]


    # --------------------------------------------------------
    # Run tests
    # --------------------------------------------------------

    for case in test_cases:

        decision = (
            build_business_decision(

                order_value=
                    case["order_value"],

                return_probability=
                    case["probability"],

                risk_score=
                    case["risk_score"]
            )
        )


        print("\n" + "-" * 70)

        print(
            f"Scenario              : "
            f"{case['name']}"
        )

        print(
            f"Order value           : "
            f"₹{case['order_value']:,.2f}"
        )

        print(
            f"Return probability    : "
            f"{case['probability']:.2%}"
        )

        print(
            f"Risk score            : "
            f"{case['risk_score']:.2f}"
        )

        print(
            f"Expected exposure     : "
            f"₹{decision['expected_return_exposure']:,.2f}"
        )

        print(
            f"Financial risk        : "
            f"{decision['financial_risk']}"
        )

        print(
            f"Recommended action    : "
            f"{decision['cost_aware_action']}"
        )


    print("\n" + "=" * 70)
    print("BUSINESS DECISION ENGINE TEST COMPLETE 🚀")
    print("=" * 70)