from realtime_features import build_features
from predictor import predict_risk


print("=" * 70)
print("RETURN-RISK INTELLIGENCE")
print("REAL-TIME PREDICTION TEST")
print("=" * 70)


# ============================================================
# SAMPLE NEW ORDER
# ============================================================

customer_id = "C1558"
product_id = "P0234"

order_value = 12500
discount_pct = 20

payment_method = "COD"
category = "Apparel"
size_variant = None


# ============================================================
# BUILD REAL-TIME FEATURES
# ============================================================

print("\nBuilding real-time customer behaviour features...")

features = build_features(
    customer_id=customer_id,
    product_id=product_id,
    order_value=order_value,
    discount_pct=discount_pct,
    payment_method=payment_method,
    category=category,
    size_variant=size_variant
)


print(
    f"Features generated: {features.shape[1]}"
)


# ============================================================
# PREDICT
# ============================================================

print("\nGenerating return-risk prediction...")

result = predict_risk(
    features
)


# ============================================================
# DISPLAY RESULT
# ============================================================

probability = result[
    "return_probability"
]

risk_score = result[
    "risk_score"
]


print("\n" + "=" * 70)
print("PREDICTION RESULT")
print("=" * 70)

print(
    f"Customer       : {customer_id}"
)

print(
    f"Product        : {product_id}"
)

print(
    f"Order value    : ₹{order_value:,.2f}"
)

print(
    f"Payment        : {payment_method}"
)

print(
    f"Category       : {category}"
)

print(
    f"Return risk    : {probability * 100:.2f}%"
)

print(
    f"Risk score     : {risk_score:.2f}/100"
)


# ============================================================
# RISK TIER
# ============================================================

if risk_score < 30:

    risk_tier = "LOW"

elif risk_score < 50:

    risk_tier = "MEDIUM"

elif risk_score < 70:

    risk_tier = "HIGH"

elif risk_score < 85:

    risk_tier = "VERY_HIGH"

else:

    risk_tier = "CRITICAL"


print(
    f"Risk tier      : {risk_tier}"
)


# ============================================================
# RECOMMENDED ACTION
# ============================================================

if risk_tier == "LOW":

    action = "NORMAL_PROCESSING"

elif risk_tier == "MEDIUM":

    action = "MONITOR"

elif risk_tier == "HIGH":

    action = "REVIEW"

else:

    action = "INTERVENE"


print(
    f"Recommended    : {action}"
)

print("=" * 70)

print(
    "\nREAL-TIME PREDICTION TEST COMPLETE 🚀"
)