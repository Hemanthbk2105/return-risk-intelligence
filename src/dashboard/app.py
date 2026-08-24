import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import ast
import sys


# ============================================================
# PROJECT PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[2]

EVALUATION_DIR = BASE_DIR / "evaluation"

RISK_DECISIONS_FILE = (
    EVALUATION_DIR / "risk_decisions.csv"
)

RISK_SCORES_FILE = (
    EVALUATION_DIR / "risk_scores.csv"
)

RISK_EXPLANATIONS_FILE = (
    EVALUATION_DIR / "risk_explanations.csv"
)

INTERVENTION_FILE = (
    EVALUATION_DIR / "intervention_simulation.csv"
)

CAPACITY_FILE = (
    EVALUATION_DIR / "capacity_optimization.csv"
)

RANKING_FILE = (
    EVALUATION_DIR / "ranking_strategy_comparison.csv"
)

THRESHOLD_FILE = (
    EVALUATION_DIR / "optimal_threshold_analysis.csv"
)

OPTIMAL_THRESHOLD_SUMMARY_FILE = (
    EVALUATION_DIR / "optimal_threshold_summary.csv"
)


# ============================================================
# API IMPORT PATH
# ============================================================

if str(BASE_DIR) not in sys.path:

    sys.path.insert(
        0,
        str(BASE_DIR)
    )


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(

    page_title="Return-Risk Intelligence",

    page_icon="📦",

    layout="wide",

    initial_sidebar_state="expanded",
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    .main {
        background-color: #f7f8fa;
    }

    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    .hero {
        background: white;
        padding: 28px;
        border-radius: 18px;
        border: 1px solid #e5e7eb;
        margin-bottom: 24px;
    }

    .hero h1 {
        margin-bottom: 5px;
        color: #111827 !important;
    }

    .hero p {
        color: #374151 !important;
        margin-bottom: 10px;
    }

    .hero b {
        color: #1f2937 !important;
    }

    [data-testid="stMetricLabel"] {
        color: #d1d5db;
    }

    [data-testid="stMetricValue"] {
        color: #f9fafb;
    }

    .section-note {
        color: #cbd5e1;
        font-size: 0.95rem;
        margin-top: -8px;
        margin-bottom: 12px;
    }

    .table-wrap {
        overflow-x: auto;
        width: 100%;
    }

    .reason-box-red {
        background: #fff7f7;
        color: #7f1d1d;
        padding: 14px 18px;
        border-radius: 12px;
        border: 1px solid #fecaca;
        margin-bottom: 10px;
        line-height: 1.5;
    }

    .reason-box-green {
        background: #f0fdf4;
        color: #14532d;
        padding: 14px 18px;
        border-radius: 12px;
        border: 1px solid #bbf7d0;
        margin-bottom: 10px;
        line-height: 1.5;
    }

    .decision-box {
    background: #ffffff;
    color: #111827 !important;
    padding: 20px;
    border-radius: 15px;
    border: 1px solid #e5e7eb;
    margin-top: 15px;
    margin-bottom: 15px;
    }

    .decision-box b {
        color: #111827 !important;
        font-weight: 700;
    }

    .decision-box h3 {
        color: #111827 !important;
        margin-top: 10px;
    }

    .risk-low {
        color: #15803d;
        font-weight: 700;
        font-size: 20px;
    }

    .risk-medium {
        color: #ca8a04;
        font-weight: 700;
        font-size: 20px;
    }

    .risk-high {
        color: #ea580c;
        font-weight: 700;
        font-size: 20px;
    }

    .risk-very-high {
        color: #dc2626;
        font-weight: 700;
        font-size: 20px;
    }

    .risk-critical {
        color: #991b1b;
        font-weight: 800;
        font-size: 20px;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# CONSTANTS
# ============================================================

RETURN_COST_RATE = 0.08

MODEL_FEATURE_COUNT = 43


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def load_csv(path):

    if not path.exists():

        return pd.DataFrame()

    try:

        return pd.read_csv(path)

    except Exception as error:

        st.error(
            f"Could not load {path.name}: {error}"
        )

        return pd.DataFrame()


def safe_float(
    value,
    default=0.0
):

    try:

        if value is None:
            return default

        if pd.isna(value):
            return default

        return float(value)

    except Exception:

        return default


def money(value):

    return (
        f"₹{safe_float(value):,.2f}"
    )


def probability_percent(value):

    value = safe_float(value)

    return (
        f"{value * 100:.2f}%"
    )


def parse_reasons(value):

    if value is None:

        return []

    if isinstance(
        value,
        float
    ) and np.isnan(value):

        return []

    if isinstance(value, list):

        return [
            str(item).strip()
            for item in value
            if str(item).strip()
        ]

    text = str(value).strip()

    if not text:

        return []

    if text.lower() in {
        "nan",
        "none",
        "null",
        "[]",
    }:

        return []

    try:

        parsed = ast.literal_eval(text)

        if isinstance(parsed, list):

            return [
                str(item).strip()
                for item in parsed
                if str(item).strip()
            ]

    except Exception:

        pass

    if "|" in text:

        parts = text.split("|")

    elif ";" in text:

        parts = text.split(";")

    else:

        parts = [text]

    return [
        part.strip()
        for part in parts
        if part.strip()
    ]


def calculate_tier(score):

    score = safe_float(score)

    if score >= 80:

        return "CRITICAL"

    if score >= 60:

        return "VERY_HIGH"

    if score >= 40:

        return "HIGH"

    if score >= 20:

        return "MEDIUM"

    return "LOW"


def risk_class(tier):

    mapping = {

        "LOW":
            "risk-low",

        "MEDIUM":
            "risk-medium",

        "HIGH":
            "risk-high",

        "VERY_HIGH":
            "risk-very-high",

        "CRITICAL":
            "risk-critical",
    }

    return mapping.get(
        str(tier).upper(),
        "risk-low"
    )


# ============================================================
# LOAD EVALUATION DATA
# ============================================================

decisions = load_csv(
    RISK_DECISIONS_FILE
)

risk_scores = load_csv(
    RISK_SCORES_FILE
)

explanations = load_csv(
    RISK_EXPLANATIONS_FILE
)

intervention = load_csv(
    INTERVENTION_FILE
)

capacity = load_csv(
    CAPACITY_FILE
)

ranking = load_csv(
    RANKING_FILE
)

thresholds = load_csv(
    THRESHOLD_FILE
)

optimal_threshold_summary = load_csv(
    OPTIMAL_THRESHOLD_SUMMARY_FILE
)


# ============================================================
# PREPARE MAIN ORDER DATA
# ============================================================

if not decisions.empty:

    orders = decisions.copy()

elif not risk_scores.empty:

    orders = risk_scores.copy()

else:

    orders = pd.DataFrame()


if orders.empty:

    st.error(
        "No evaluation data found. "
        "Please run the evaluation pipeline first."
    )

    st.stop()


# ============================================================
# MERGE RISK SCORES
# ============================================================

if (
    not risk_scores.empty
    and "order_id" in risk_scores.columns
    and "order_id" in orders.columns
):

    additional_columns = [

        column

        for column in risk_scores.columns

        if (
            column not in orders.columns
            and column != "order_id"
        )
    ]

    if additional_columns:

        orders = orders.merge(

            risk_scores[
                ["order_id"]
                + additional_columns
            ],

            on="order_id",

            how="left",
        )


# ============================================================
# MERGE EXPLANATIONS
# ============================================================

if (
    not explanations.empty
    and "order_id" in explanations.columns
    and "order_id" in orders.columns
):

    explanation_columns = [

        column

        for column in explanations.columns

        if column != "order_id"
    ]

    for column in explanation_columns:

        if column in orders.columns:

            orders.drop(
                columns=[column],
                inplace=True
            )

    orders = orders.merge(

        explanations[
            ["order_id"]
            + explanation_columns
        ],

        on="order_id",

        how="left",
    )


# ============================================================
# NORMALIZE IMPORTANT COLUMNS
# ============================================================

if "return_probability" not in orders.columns:

    orders["return_probability"] = 0.0


orders["return_probability"] = pd.to_numeric(

    orders["return_probability"],

    errors="coerce"

).fillna(0)


if "risk_score" not in orders.columns:

    orders["risk_score"] = (

        orders["return_probability"]
        * 100
    )


orders["risk_score"] = pd.to_numeric(

    orders["risk_score"],

    errors="coerce"

).fillna(

    orders["return_probability"] * 100
)


if "risk_tier" not in orders.columns:

    orders["risk_tier"] = (

        orders["risk_score"]
        .apply(calculate_tier)
    )


if "order_value" not in orders.columns:

    orders["order_value"] = 0.0


orders["order_value"] = pd.to_numeric(

    orders["order_value"],

    errors="coerce"

).fillna(0)


# ============================================================
# IMPORTANT:
# USE THE SAME 8% BUSINESS COST ASSUMPTION
# AS business_engine.py
# ============================================================

orders["expected_return_exposure"] = (

    orders["order_value"]

    * orders["return_probability"]

    * RETURN_COST_RATE
)


orders["expected_return_exposure"] = (

    orders["expected_return_exposure"]
    .round(2)
)


# ============================================================
# PRIORITY
# ============================================================

if "priority" not in orders.columns:

    orders["priority"] = "P4"

    orders.loc[
        orders["risk_score"] >= 80,
        "priority"
    ] = "P1"

    orders.loc[
        (
            orders["risk_score"] >= 40
        )
        &
        (
            orders["risk_score"] < 80
        ),
        "priority"
    ] = "P2"

    orders.loc[
        (
            orders["risk_score"] >= 20
        )
        &
        (
            orders["risk_score"] < 40
        ),
        "priority"
    ] = "P3"


# ============================================================
# ACTION
#
# Keep dashboard consistent with business_engine.py
# ============================================================

def dashboard_action(
    exposure,
    risk_score
):
    # Keep this exactly aligned with src/api/business_engine.py.
    if exposure >= 5000:
        return "INTERVENE"

    if exposure >= 2500:
        return "REVIEW"

    if risk_score >= 70:
        return "REVIEW"

    if risk_score >= 50:
        return "MONITOR"

    return "NORMAL_PROCESSING"


orders["recommended_action"] = [

    dashboard_action(
        exposure,
        score
    )

    for exposure, score

    in zip(
        orders["expected_return_exposure"],
        orders["risk_score"]
    )
]


# ============================================================
# PAGE HEADER
# ============================================================

st.markdown(
    """
    <div class="hero">

    <h1>📦 Return-Risk Intelligence</h1>

    <p>
    AI-powered return prediction, explainability,
    financial exposure and cost-aware intervention.
    </p>

    <b>LightGBM V2 • 43 Features • SHAP • Business Decision Engine</b>

    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title(
    "🧭 Navigation"
)

page = st.sidebar.radio(

    "Select module",

    [
        "📊 Executive Overview",
        "⚡ Real-Time Prediction",
        "🚨 Risk Monitoring",
        "🔎 Order Investigation",
        "💰 Business Impact",
    ]
)


# ============================================================
# EXECUTIVE OVERVIEW
# ============================================================

if page == "📊 Executive Overview":

    st.header(
        "📊 Executive Overview"
    )

    total_orders = len(orders)

    high_risk_orders = orders[
        orders["risk_score"] >= 40
    ].shape[0]

    very_high_orders = orders[
        orders["risk_score"] >= 60
    ].shape[0]

    critical_orders = orders[
        orders["risk_score"] >= 80
    ].shape[0]

    total_exposure = safe_float(
        orders[
            "expected_return_exposure"
        ].sum()
    )

    average_probability = safe_float(
        orders[
            "return_probability"
        ].mean()
    )


    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Total Orders",
        f"{total_orders:,}"
    )

    c2.metric(
        "High+ Risk Orders",
        f"{high_risk_orders:,}"
    )

    c3.metric(
        "Very High+ Orders",
        f"{very_high_orders:,}"
    )

    c4.metric(
        "Critical Orders",
        f"{critical_orders:,}"
    )


    st.divider()


    c1, c2, c3 = st.columns(3)

    c1.metric(
        "Expected Return Exposure",
        money(total_exposure)
    )

    c2.metric(
        "Average Return Probability",
        probability_percent(
            average_probability
        )
    )

    c3.metric(
        "Model Features",
        f"{MODEL_FEATURE_COUNT}"
    )


    st.divider()
    
    # ========================================================
    # MODEL PERFORMANCE
    # ========================================================

    st.subheader(
        "🧠 Model Performance"
    )

    st.caption(
        "Held-out test-set evaluation for Behaviour-Aware LightGBM V2"
    )

    m1, m2, m3, m4, m5, m6 = st.columns(6)

    m1.metric(
        "Test Orders",
        "3,965"
    )

    m2.metric(
        "Precision",
        "17.25%"
    )

    m3.metric(
        "Recall",
        "22.99%"
    )

    m4.metric(
        "PR-AUC",
        "0.1650"
    )

    m5.metric(
        "ROC-AUC",
        "0.6036"
    )

    m6.metric(
        "Accuracy",
        "79.00%"
    )

    st.divider()


    # ========================================================
    # ERROR COST ANALYSIS
    # ========================================================

    st.subheader(
        "💰 Error Cost Analysis"
    )

    st.caption(
        "Cost analysis on the held-out test set at a 0.50 prediction threshold"
    )

    e1, e2, e3, e4, e5 = st.columns(5)

    e1.metric(
        "False Positives",
        "494"
    )

    e2.metric(
        "False Negatives",
        "345"
    )

    e3.metric(
        "FP Intervention Cost",
        "₹74,100"
    )

    e4.metric(
        "FN Missed-Return Cost",
        "₹2,95,793.99"
    )

    e5.metric(
        "Total Error Cost",
        "₹3,69,893.99"
    )

    st.info(
        "Missed returns create substantially more financial exposure "
        "than unnecessary interventions. This is why the system uses "
        "cost-aware risk ranking and business intervention rather "
        "than relying on accuracy alone."
    )

    st.divider()


    st.subheader(
        "Risk Distribution"
    )

    distribution = (

        orders["risk_tier"]
        .value_counts()
        .reindex(
            [
                "LOW",
                "MEDIUM",
                "HIGH",
                "VERY_HIGH",
                "CRITICAL"
            ],
            fill_value=0
        )
    )

    # Horizontal chart keeps long risk-tier labels readable.
    st.bar_chart(
        distribution,
        horizontal=True,
        height=320
    )


    distribution_table = pd.DataFrame({

        "Risk Tier":
            distribution.index,

        "Orders":
            distribution.values,

        "Percentage":
            (
                distribution.values
                /
                max(total_orders, 1)
                * 100
            ).round(2)
    })


    st.dataframe(

        distribution_table,

        use_container_width=True,

        hide_index=True
    )


    st.subheader(
        "Recommended Actions"
    )


    action_distribution = (

        orders[
            "recommended_action"
        ]
        .value_counts()
    )


    # Horizontal chart prevents NORMAL_PROCESSING from being truncated.
    st.bar_chart(
        action_distribution,
        horizontal=True,
        height=280
    )


# ============================================================
# REAL-TIME PREDICTION
# ============================================================

elif page == "⚡ Real-Time Prediction":

    st.header(
        "⚡ Real-Time Risk Prediction"
    )

    st.info(
        "Enter an order below. The system will generate "
        "43 point-in-time features, run LightGBM, "
        "generate SHAP explanations and calculate "
        "the cost-aware business decision."
    )


    # --------------------------------------------------------
    # INPUTS
    # --------------------------------------------------------

    col1, col2 = st.columns(2)


    with col1:

        customer_id = st.text_input(
            "Customer ID",
            value="C1558"
        )

        product_id = st.text_input(
            "Product ID",
            value="P0234"
        )

        order_value = st.number_input(
            "Order Value (₹)",
            min_value=1.0,
            value=12500.0,
            step=500.0
        )

        discount_pct = st.number_input(
            "Discount (%)",
            min_value=0.0,
            max_value=100.0,
            value=10.0,
            step=5.0
        )


    with col2:

        payment_method = st.selectbox(

            "Payment Method",

            [
                "Card",
                "UPI",
                "NetBanking",
                "Wallet"
            ],

            index=0
        )


        category = st.selectbox(

            "Category",

            [
                "Electronics",
                "Apparel",
                "Footwear",
                "Home",
                "Beauty",
                "Grocery"
            ],

            index=4
        )


        size_option = st.selectbox(

            "Size Variant",

            [
                "None",
                "S",
                "M",
                "L",
                "XL",
                "7",
                "8",
                "9",
                "10",
                "11"
            ]
        )


        if size_option == "None":

            size_variant = None

        else:

            size_variant = size_option


    st.divider()


    predict_button = st.button(

        "🔍 ANALYZE RETURN RISK",

        type="primary",

        use_container_width=True
    )


    if predict_button:

        try:

            from src.api.realtime_features import (
                build_features
            )

            from src.api.predictor import (
                predict_risk
            )

            from src.api.explainer import (
                explain_prediction
            )

            from src.api.business_engine import (
                build_business_decision
            )


            # ------------------------------------------------
            # BUILD FEATURES
            # ------------------------------------------------

            features = build_features(

                customer_id=
                    customer_id,

                product_id=
                    product_id,

                order_value=
                    order_value,

                discount_pct=
                    discount_pct,

                payment_method=
                    payment_method,

                category=
                    category,

                size_variant=
                    size_variant
            )


            if features.shape[1] != 43:

                st.error(
                    f"Expected 43 features, "
                    f"received {features.shape[1]}."
                )

                st.stop()


            # ------------------------------------------------
            # MODEL
            # ------------------------------------------------

            prediction = predict_risk(
                features
            )


            probability = prediction[
                "return_probability"
            ]

            risk_score = prediction[
                "risk_score"
            ]


            # ------------------------------------------------
            # RISK TIER
            # ------------------------------------------------

            risk_tier = (
                calculate_tier(
                    risk_score
                )
            )


            # ------------------------------------------------
            # SHAP
            # ------------------------------------------------

            explanation = explain_prediction(

                features,

                top_n=5
            )


            # ------------------------------------------------
            # BUSINESS DECISION
            # ------------------------------------------------

            business = (
                build_business_decision(

                    order_value=
                        order_value,

                    return_probability=
                        probability,

                    risk_score=
                        risk_score
                )
            )


            exposure = business[
                "expected_return_exposure"
            ]

            financial_risk = business[
                "financial_risk"
            ]

            action = business[
                "cost_aware_action"
            ]


            st.success(
                "Risk analysis completed successfully ✅"
            )


            # ------------------------------------------------
            # RESULT METRICS
            # ------------------------------------------------

            st.subheader(
                "Risk Result"
            )


            c1, c2, c3, c4 = st.columns(4)


            c1.metric(

                "Return Probability",

                f"{probability * 100:.2f}%"
            )


            c2.metric(

                "Risk Score",

                f"{risk_score:.2f}"
            )


            c3.metric(

                "Expected Exposure",

                money(exposure)
            )


            c4.metric(

                "Financial Risk",

                financial_risk
            )


            c1, c2 = st.columns(2)


            with c1:

                st.markdown(

                    f"""
                    <div class="decision-box">

                    <b>Risk Tier</b>

                    <div class="{risk_class(risk_tier)}">
                    {risk_tier}
                    </div>

                    </div>
                    """,

                    unsafe_allow_html=True
                )


            with c2:

                st.markdown(

                    f"""
                    <div class="decision-box">

                    <b>Recommended Action</b>

                    <h3>{action}</h3>

                    </div>
                    """,

                    unsafe_allow_html=True
                )


            # ------------------------------------------------
            # SHAP REASONS
            # ------------------------------------------------

            st.divider()

            left, right = st.columns(2)


            with left:

                st.subheader(
                    "🔴 Why Risk Increased"
                )


                reasons = explanation.get(
                    "risk_reasons",
                    []
                )


                for reason in reasons:

                    st.markdown(

                        f"""
                        <div class="reason-box-red">
                        🔴 {reason}
                        </div>
                        """,

                        unsafe_allow_html=True
                    )


            with right:

                st.subheader(
                    "🟢 Risk-Reducing Factors"
                )


                reducing = explanation.get(
                    "risk_reducing_factors",
                    []
                )


                for factor in reducing:

                    st.markdown(

                        f"""
                        <div class="reason-box-green">
                        🟢 {factor}
                        </div>
                        """,

                        unsafe_allow_html=True
                    )


            # ------------------------------------------------
            # FEATURES
            # ------------------------------------------------

            with st.expander(
                "🔬 View generated 43 features"
            ):

                st.dataframe(

                    features.T,

                    use_container_width=True
                )


        except Exception as error:

            st.error(
                f"Prediction failed: {error}"
            )


# ============================================================
# RISK MONITORING
# ============================================================

elif page == "🚨 Risk Monitoring":

    st.header(
        "🚨 Risk Monitoring"
    )


    col1, col2, col3 = st.columns(3)


    with col1:

        tiers = st.multiselect(

            "Risk Tier",

            [
                "LOW",
                "MEDIUM",
                "HIGH",
                "VERY_HIGH",
                "CRITICAL"
            ],

            default=[
                "HIGH",
                "VERY_HIGH",
                "CRITICAL"
            ]
        )


    with col2:

        minimum_score = st.slider(

            "Minimum Risk Score",

            0.0,

            100.0,

            0.0,

            1.0
        )


    with col3:

        search = st.text_input(
            "Search Order ID"
        )


    filtered = orders.copy()


    if tiers:

        filtered = filtered[
            filtered["risk_tier"]
            .isin(tiers)
        ]


    filtered = filtered[
        filtered["risk_score"]
        >= minimum_score
    ]


    if search:

        filtered = filtered[

            filtered["order_id"]
            .astype(str)
            .str.contains(
                search,
                case=False,
                na=False
            )
        ]


    st.metric(
        "Matching Orders",
        f"{len(filtered):,}"
    )


    display_columns = [

        "order_id",

        "customer_id",

        "product_id",

        "order_value",

        "return_probability",

        "risk_score",

        "risk_tier",

        "expected_return_exposure",

        "priority",

        "recommended_action"
    ]


    display_columns = [

        column

        for column in display_columns

        if column in filtered.columns
    ]


    table = filtered[
        display_columns
    ].copy()


    table = table.sort_values(

        "risk_score",

        ascending=False
    )


    if "return_probability" in table.columns:

        table["return_probability"] = (

            table[
                "return_probability"
            ]
            .apply(
                probability_percent
            )
        )


    if "order_value" in table.columns:

        table["order_value"] = (

            table["order_value"]
            .apply(money)
        )


    if "expected_return_exposure" in table.columns:

        table[
            "expected_return_exposure"
        ] = (

            table[
                "expected_return_exposure"
            ]
            .apply(money)
        )


    st.dataframe(

        table.head(200),

        use_container_width=True,

        hide_index=True
    )


# ============================================================
# ORDER INVESTIGATION
# ============================================================

elif page == "🔎 Order Investigation":

    st.header(
        "🔎 Order Investigation"
    )


    order_ids = (

        orders[
            "order_id"
        ]
        .astype(str)
        .tolist()
    )


    selected_order = st.selectbox(

        "Select Order",

        order_ids
    )


    row = orders[

        orders[
            "order_id"
        ]
        .astype(str)

        ==

        str(selected_order)

    ]


    if row.empty:

        st.error(
            "Order not found."
        )

        st.stop()


    record = row.iloc[0]


    probability = safe_float(
        record[
            "return_probability"
        ]
    )

    risk_score = safe_float(
        record[
            "risk_score"
        ]
    )

    order_value = safe_float(
        record[
            "order_value"
        ]
    )

    exposure = safe_float(
        record[
            "expected_return_exposure"
        ]
    )

    risk_tier = str(
        record[
            "risk_tier"
        ]
    )

    action = str(
        record[
            "recommended_action"
        ]
    )

    priority = str(
        record[
            "priority"
        ]
    )


    c1, c2, c3, c4 = st.columns(4)


    c1.metric(
        "Return Probability",
        probability_percent(
            probability
        )
    )


    c2.metric(
        "Risk Score",
        f"{risk_score:.2f}"
    )


    c3.metric(
        "Order Value",
        money(order_value)
    )


    c4.metric(
        "Expected Exposure",
        money(exposure)
    )


    c1, c2, c3 = st.columns(3)


    c1.metric(
        "Risk Tier",
        risk_tier
    )

    c2.metric(
        "Financial Risk",

        (
            "CRITICAL"
            if exposure >= 5000

            else
            "VERY_HIGH"
            if exposure >= 2500

            else
            "HIGH"
            if exposure >= 1000

            else
            "MEDIUM"
            if exposure >= 250

            else
            "LOW"
        )
    )

    c3.metric(
        "Recommended Action",
        action
    )


    st.divider()


    left, right = st.columns(2)


    risk_reasons = parse_reasons(

        record.get(
            "risk_reasons",
            None
        )
    )


    if not risk_reasons:

        risk_reasons = parse_reasons(

            record.get(
                "top_risk_drivers",
                None
            )
        )


    reducing = parse_reasons(

        record.get(
            "risk_reducing_factors",
            None
        )
    )


    with left:

        st.subheader(
            "🔴 Risk Increasing Factors"
        )


        if risk_reasons:

            for reason in risk_reasons:

                st.markdown(

                    f"""
                    <div class="reason-box-red">
                    🔴 {reason}
                    </div>
                    """,

                    unsafe_allow_html=True
                )

        else:

            st.info(
                "No risk reasons available."
            )


    with right:

        st.subheader(
            "🟢 Risk Reducing Factors"
        )


        if reducing:

            for factor in reducing:

                st.markdown(

                    f"""
                    <div class="reason-box-green">
                    🟢 {factor}
                    </div>
                    """,

                    unsafe_allow_html=True
                )

        else:

            st.info(
                "No risk-reducing factors available."
            )


    st.divider()


    with st.expander(
        "🔬 View complete order record"
    ):

        st.dataframe(

            record.to_frame(
                name="Value"
            ),

            use_container_width=True
        )


# ============================================================
# BUSINESS IMPACT
# ============================================================

elif page == "💰 Business Impact":

    st.header(
        "💰 Business Impact"
    )


    # ========================================================
    # INTERVENTION
    # ========================================================

    if not intervention.empty:

        st.subheader(
            "Intervention Simulation"
        )


        numeric_columns = [

            "baseline_expected_loss",

            "expected_loss_after",

            "potential_loss_avoided",

            "intervention_cost",

            "net_benefit"
        ]


        values = {}


        # The simulator stores the post-intervention value as
        # `expected_loss_after_intervention`.
        # Keep the dashboard label business-friendly.
        intervention_column_map = {
            "baseline_expected_loss": "baseline_expected_loss",
            "expected_loss_after": "expected_loss_after_intervention",
            "potential_loss_avoided": "potential_loss_avoided",
            "intervention_cost": "intervention_cost",
            "net_benefit": "net_benefit",
        }

        for display_column, source_column in intervention_column_map.items():

            if source_column in intervention.columns:

                values[display_column] = safe_float(
                    pd.to_numeric(
                        intervention[source_column],
                        errors="coerce"
                    ).sum()
                )

            else:

                values[display_column] = 0.0


        baseline = values[
            "baseline_expected_loss"
        ]

        after = values[
            "expected_loss_after"
        ]

        avoided = values[
            "potential_loss_avoided"
        ]

        cost = values[
            "intervention_cost"
        ]

        net = values[
            "net_benefit"
        ]


        roi = (

            net / cost

            if cost > 0

            else 0
        )


        c1, c2, c3 = st.columns(3)


        c1.metric(
            "Baseline Expected Loss",
            money(baseline)
        )

        c2.metric(
            "Expected Loss After",
            money(after)
        )

        c3.metric(
            "Potential Loss Avoided",
            money(avoided)
        )


        c1, c2, c3 = st.columns(3)


        c1.metric(
            "Intervention Cost",
            money(cost)
        )

        c2.metric(
            "Net Benefit",
            money(net)
        )

        c3.metric(
            "ROI",
            f"{roi:.2f}x"
        )


    # ========================================================
    # CAPACITY
    # ========================================================

    if not capacity.empty:

        st.divider()

        st.subheader(
            "🎯 Capacity Optimization"
        )


        if "net_benefit" in capacity.columns:

            benefits = pd.to_numeric(

                capacity[
                    "net_benefit"
                ],

                errors="coerce"
            )


            best_index = benefits.idxmax()

            best = capacity.loc[
                best_index
            ]


            c1, c2, c3, c4 = st.columns(4)


            raw_capacity = safe_float(
                best.get("capacity", 0)
            )

            # Evaluation files store capacity as a fraction
            # (0.05 = 5%, 0.15 = 15%, etc.).
            best_capacity_pct = (
                raw_capacity * 100
                if raw_capacity <= 1
                else raw_capacity
            )

            best_roi = safe_float(
                best.get("roi", 0)
            )

            c1.metric(
                "Best Capacity",
                f"{best_capacity_pct:.0f}%"
            )

            c2.metric(
                "Net Benefit",
                money(
                    best.get(
                        "net_benefit",
                        0
                    )
                )
            )

            c3.metric(
                "ROI",
                f"{best_roi:.2f}x"
            )

            c4.metric(
                "Loss Avoided",
                money(
                    best.get(
                        "potential_loss_avoided",
                        0
                    )
                )
            )

            st.caption(
                "Best capacity is selected using the highest net benefit "
                "among the evaluated capacity levels."
            )

        capacity_display = capacity.copy()

        if "capacity" in capacity_display.columns:
            capacity_display["capacity"] = (
                pd.to_numeric(
                    capacity_display["capacity"],
                    errors="coerce"
                )
                .apply(
                    lambda x:
                    f"{x * 100:.0f}%"
                    if pd.notna(x) and x <= 1
                    else (
                        f"{x:.0f}%"
                        if pd.notna(x)
                        else "-"
                    )
                )
            )

        money_columns = [
            "average_order_value",
            "baseline_expected_loss",
            "potential_loss_avoided",
            "intervention_cost",
            "net_benefit",
        ]

        for column in money_columns:
            if column in capacity_display.columns:
                capacity_display[column] = (
                    pd.to_numeric(
                        capacity_display[column],
                        errors="coerce"
                    )
                    .apply(money)
                )

        if "average_return_probability" in capacity_display.columns:
            capacity_display["average_return_probability"] = (
                pd.to_numeric(
                    capacity_display["average_return_probability"],
                    errors="coerce"
                )
                .apply(probability_percent)
            )

        if "selection_rate" in capacity_display.columns:
            capacity_display["selection_rate"] = (
                pd.to_numeric(
                    capacity_display["selection_rate"],
                    errors="coerce"
                )
                .apply(probability_percent)
            )

        st.dataframe(
            capacity_display,
            use_container_width=True,
            hide_index=True
        )


    # ========================================================
    # RANKING
    # ========================================================

    if not ranking.empty:

        st.divider()

        st.subheader(
            "🏆 Ranking Strategy Comparison"
        )

        ranking_display = ranking.copy()

        for column in [
            "average_order_value",
            "baseline_expected_exposure",
            "baseline_expected_loss",
            "potential_loss_avoided",
            "intervention_cost",
            "net_benefit",
        ]:
            if column in ranking_display.columns:
                ranking_display[column] = (
                    pd.to_numeric(
                        ranking_display[column],
                        errors="coerce"
                    )
                    .apply(money)
                )

        for column in [
            "selection_rate",
            "average_return_probability",
        ]:
            if column in ranking_display.columns:
                ranking_display[column] = (
                    pd.to_numeric(
                        ranking_display[column],
                        errors="coerce"
                    )
                    .apply(probability_percent)
                )

        if "capacity" in ranking_display.columns:
            ranking_display["capacity"] = (
                pd.to_numeric(
                    ranking_display["capacity"],
                    errors="coerce"
                )
                .apply(
                    lambda x:
                    f"{x * 100:.0f}%"
                    if pd.notna(x) and x <= 1
                    else (
                        f"{x:.0f}%"
                        if pd.notna(x)
                        else "-"
                    )
                )
            )

        st.dataframe(
            ranking_display,
            use_container_width=True,
            hide_index=True
        )


    # ========================================================
    # THRESHOLD
    # ========================================================

    if not thresholds.empty:

        st.divider()

        st.subheader(
            "📈 Threshold Analysis"
        )

        st.caption(
            "The threshold controls which orders are selected for intervention. "
            "The recommended threshold maximizes net business benefit under "
            "the current 40% intervention-effectiveness and ₹150 intervention-cost assumptions."
        )

        # ----------------------------------------------------
        # Optimal threshold summary
        # ----------------------------------------------------

        if not optimal_threshold_summary.empty:

            net_rows = optimal_threshold_summary[
                optimal_threshold_summary["optimization_metric"] == "net_benefit"
            ]

            roi_rows = optimal_threshold_summary[
                optimal_threshold_summary["optimization_metric"] == "roi"
            ]

            if not net_rows.empty:

                best_threshold = net_rows.iloc[0]

                threshold_value = safe_float(
                    best_threshold.get("optimal_threshold", 0)
                )

                orders_selected = safe_float(
                    best_threshold.get("orders_selected", 0)
                )

                selection_rate = safe_float(
                    best_threshold.get("selection_rate", 0)
                )

                loss_avoided = safe_float(
                    best_threshold.get("potential_loss_avoided", 0)
                )

                intervention_cost = safe_float(
                    best_threshold.get("intervention_cost", 0)
                )

                net_benefit = safe_float(
                    best_threshold.get("net_benefit", 0)
                )

                roi = safe_float(
                    best_threshold.get("roi", 0)
                )

                st.markdown("### 🎯 Recommended Business Threshold")

                c1, c2, c3, c4, c5 = st.columns(5)

                c1.metric(
                    "Threshold",
                    f"{threshold_value * 100:.0f}%"
                )

                c2.metric(
                    "Orders Selected",
                    f"{orders_selected:,.0f}"
                )

                c3.metric(
                    "Loss Avoided",
                    money(loss_avoided)
                )

                c4.metric(
                    "Net Benefit",
                    money(net_benefit)
                )

                c5.metric(
                    "ROI",
                    f"{roi:.2f}x"
                )

                st.success(
                    f"Recommended threshold: {threshold_value * 100:.0f}% "
                    f"({orders_selected:,.0f} orders, {selection_rate * 100:.2f}% of orders). "
                    f"This produces the highest net benefit among the evaluated thresholds."
                )

                if not roi_rows.empty:
                    best_roi = roi_rows.iloc[0]
                    st.caption(
                        f"Highest ROI occurs at {safe_float(best_roi.get('optimal_threshold', 0)) * 100:.0f}% "
                        f"with {safe_float(best_roi.get('roi', 0)):.2f}x ROI. "
                        "The dashboard uses maximum net benefit as the primary business objective."
                    )

        # ----------------------------------------------------
        # Threshold trade-off chart
        # ----------------------------------------------------

        chart_data = thresholds.copy()

        if "threshold" in chart_data.columns and "net_benefit" in chart_data.columns:

            chart_data["Threshold"] = (
                pd.to_numeric(chart_data["threshold"], errors="coerce") * 100
            )

            chart_data["Net Benefit (₹)"] = pd.to_numeric(
                chart_data["net_benefit"],
                errors="coerce"
            )

            chart_data = chart_data.dropna(
                subset=["Threshold", "Net Benefit (₹)"]
            )

            st.subheader("Net Benefit by Threshold")

            st.line_chart(
                chart_data.set_index("Threshold")["Net Benefit (₹)"],
                height=320
            )

        # ----------------------------------------------------
        # Detailed table
        # ----------------------------------------------------

        threshold_display = thresholds.copy()

        if "threshold" in threshold_display.columns:
            threshold_display["threshold"] = (
                pd.to_numeric(
                    threshold_display["threshold"],
                    errors="coerce"
                )
                .apply(probability_percent)
            )

        for column in [
            "baseline_expected_loss",
            "potential_loss_avoided",
            "intervention_cost",
            "net_benefit",
        ]:
            if column in threshold_display.columns:
                threshold_display[column] = (
                    pd.to_numeric(
                        threshold_display[column],
                        errors="coerce"
                    )
                    .apply(money)
                )

        for column in [
            "selection_rate",
            "average_return_probability",
            "expected_return_capture_rate",
        ]:
            if column in threshold_display.columns:
                threshold_display[column] = (
                    pd.to_numeric(
                        threshold_display[column],
                        errors="coerce"
                    )
                    .apply(probability_percent)
                )

        if "roi" in threshold_display.columns:
            threshold_display["roi"] = (
                pd.to_numeric(
                    threshold_display["roi"],
                    errors="coerce"
                )
                .apply(lambda x: f"{x:.2f}x" if pd.notna(x) else "-")
            )

        # Human-readable column names
        threshold_display = threshold_display.rename(columns={
            "threshold": "Risk Threshold",
            "orders_selected": "Orders Selected",
            "selection_rate": "Selection Rate",
            "average_return_probability": "Avg. Return Probability",
            "baseline_expected_loss": "Baseline Expected Loss",
            "potential_loss_avoided": "Loss Avoided",
            "intervention_cost": "Intervention Cost",
            "net_benefit": "Net Benefit",
            "roi": "ROI",
            "expected_returns_captured": "Expected Returns Captured",
            "expected_return_capture_rate": "Expected Return Capture",
        })

        st.subheader("Detailed Threshold Results")

        st.dataframe(
            threshold_display,
            use_container_width=True,
            hide_index=True
        )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(

    "Return-Risk Intelligence 🚀 | "
    "LightGBM V2 + Behaviour Analytics + SHAP + "
    "Cost-Aware Decision Engine"
)