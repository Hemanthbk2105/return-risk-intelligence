from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.api.realtime_features import build_features
from src.api.predictor import predict_risk
from src.api.explainer import explain_prediction
from src.api.business_engine import build_business_decision


# ============================================================
# FASTAPI APPLICATION
# ============================================================

app = FastAPI(

    title="Return-Risk Intelligence API",

    description=(
        "Real-time ML-powered return risk prediction "
        "with explainability and cost-aware decisions."
    ),

    version="1.3.0"
)


# ============================================================
# REQUEST MODEL
# ============================================================

class PredictionRequest(BaseModel):

    customer_id: str = Field(
        ...,
        min_length=1
    )

    product_id: str = Field(
        ...,
        min_length=1
    )

    order_value: float = Field(
        ...,
        gt=0
    )

    discount_pct: float = Field(
        0,
        ge=0,
        le=100
    )

    payment_method: str = Field(
        ...,
        min_length=1
    )

    category: str = Field(
        ...,
        min_length=1
    )

    size_variant: Optional[str] = None


# ============================================================
# RISK TIER
# ============================================================

def get_risk_tier(
    risk_score: float
):

    if risk_score < 20:

        return "LOW"

    elif risk_score < 40:

        return "MEDIUM"

    elif risk_score < 60:

        return "HIGH"

    elif risk_score < 80:

        return "VERY_HIGH"

    else:

        return "CRITICAL"


# ============================================================
# ROOT
# ============================================================

@app.get("/")
def root():

    return {

        "service":
            "Return-Risk Intelligence",

        "status":
            "online",

        "model":
            "Behaviour-Aware LightGBM V2",

        "explainable":
            True,

        "cost_aware":
            True,

        "features":
            43
    }


# ============================================================
# HEALTH
# ============================================================

@app.get("/health")
def health():

    return {

        "status":
            "healthy",

        "model":
            "loaded",

        "explainability":
            "enabled",

        "cost_engine":
            "enabled",

        "feature_count":
            43
    }


# ============================================================
# PREDICT
# ============================================================

@app.post("/predict")
def predict(
    request: PredictionRequest
):

    try:

        # ====================================================
        # 1. REAL-TIME FEATURE GENERATION
        # ====================================================

        features = build_features(

            customer_id=
                request.customer_id,

            product_id=
                request.product_id,

            order_value=
                request.order_value,

            discount_pct=
                request.discount_pct,

            payment_method=
                request.payment_method,

            category=
                request.category,

            size_variant=
                request.size_variant
        )


        # ====================================================
        # 2. VERIFY 43 FEATURES
        # ====================================================

        if features.shape[1] != 43:

            raise ValueError(

                "Real-time feature "
                "generation returned "

                f"{features.shape[1]} "
                "features instead of 43."
            )


        # ====================================================
        # 3. ML PREDICTION
        # ====================================================

        prediction = predict_risk(
            features
        )


        probability = prediction[
            "return_probability"
        ]


        risk_score = prediction[
            "risk_score"
        ]


        # ====================================================
        # 4. RISK TIER
        # ====================================================

        risk_tier = get_risk_tier(
            risk_score
        )


        # ====================================================
        # 5. SHAP EXPLANATION
        # ====================================================

        explanation = explain_prediction(

            features,

            top_n=5
        )


        # ====================================================
        # 6. BUSINESS DECISION
        # ====================================================

        business = (
            build_business_decision(

                order_value=
                    request.order_value,

                return_probability=
                    probability,

                risk_score=
                    risk_score
            )
        )


        # ====================================================
        # 7. FINAL RESPONSE
        # ====================================================

        return {

            "status":
                "success",

            "customer_id":
                request.customer_id,

            "product_id":
                request.product_id,

            "order_value":
                request.order_value,


            # ------------------------------------------------
            # MACHINE LEARNING
            # ------------------------------------------------

            "return_probability":
                probability,

            "risk_score":
                risk_score,

            "risk_tier":
                risk_tier,


            # ------------------------------------------------
            # BUSINESS INTELLIGENCE
            # ------------------------------------------------

            "expected_return_exposure":
                business[
                    "expected_return_exposure"
                ],

            "financial_risk":
                business[
                    "financial_risk"
                ],

            "recommended_action":
                business[
                    "cost_aware_action"
                ],


            # ------------------------------------------------
            # SHAP EXPLAINABILITY
            # ------------------------------------------------

            "risk_reasons":
                explanation[
                    "risk_reasons"
                ],

            "risk_reducing_factors":
                explanation[
                    "risk_reducing_factors"
                ]
        }


    # ========================================================
    # EXPECTED USER / DATA ERRORS
    # ========================================================

    except ValueError as error:

        raise HTTPException(

            status_code=400,

            detail=str(error)
        )


    # ========================================================
    # UNEXPECTED SERVER ERROR
    # ========================================================

    except Exception as error:

        raise HTTPException(

            status_code=500,

            detail=(
                "Prediction failed: "
                +
                str(error)
            )
        )