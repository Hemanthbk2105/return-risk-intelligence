# Return-Risk Intelligence 🚀

AI-powered return-risk prediction and business decision system for e-commerce orders.

## 📌 Project Overview

Return-Risk Intelligence predicts the probability that an order will be returned and converts that prediction into an actionable business decision.

The system combines:

- LightGBM machine learning
- 43 point-in-time features
- Behaviour analytics
- SHAP explainability
- Risk scoring
- Financial exposure calculation
- Risk monitoring
- Order-level investigation
- Cost-aware intervention
- Business impact analysis
- Capacity optimization
- Ranking strategy comparison
- Threshold optimization

The goal is not only to predict whether an order may be returned, but also to help the business decide **which orders require attention and whether intervention is financially worthwhile**.

---

## 🎯 Main Features

### 1. 📊 Executive Overview

Provides a high-level view of the return-risk system:

- Total orders
- High-risk orders
- Very-high-risk orders
- Critical orders
- Expected return exposure
- Average return probability
- Risk distribution
- Recommended actions

### 2. ⚡ Real-Time Risk Prediction

Users can enter order information and generate a real-time risk assessment.

The system provides:

- Return probability
- Risk score
- Risk tier
- Expected return exposure
- Financial risk
- Recommended business action
- Risk-increasing factors
- Risk-reducing factors
- SHAP-based explanations

The prediction pipeline uses the same 43-feature configuration used during model training.

### 3. 🚨 Risk Monitoring

Allows users to identify and monitor high-risk orders.

Users can filter orders based on:

- Risk tier
- Minimum risk score
- Order ID

The monitoring view displays:

- Order value
- Return probability
- Risk score
- Risk tier
- Expected return exposure
- Priority
- Recommended action

### 4. 🔎 Order Investigation

Provides detailed analysis for an individual order.

The investigation view includes:

- Return probability
- Risk score
- Risk tier
- Order value
- Expected return exposure
- Financial risk
- Recommended action
- Risk-increasing factors
- Risk-reducing factors
- Complete order information

This allows users to understand not only the prediction, but also **why the model produced that prediction**.

### 5. 💰 Business Impact

Evaluates the financial impact of risk-based intervention.

The system analyzes:

- Baseline expected loss
- Expected loss after intervention
- Potential loss avoided
- Intervention cost
- Net benefit
- ROI
- Capacity optimization
- Ranking strategies
- Intervention thresholds

---

## 🧠 Machine Learning

The prediction system uses **LightGBM** for return-risk classification.

The model uses **43 point-in-time features** generated from:

- Order information
- Customer information
- Product information
- Historical return behaviour
- Recent customer behaviour
- Order-value changes
- Return-rate changes
- Category behaviour

### Model Features

The real-time predictor uses the same feature order as the trained model.

The 43 features include:

- Order value
- Payment method
- Discount percentage
- Product category
- Size variant
- City tier
- Account age
- Product price
- Historical product return rate
- Previous order behaviour
- Historical customer return rate
- Recent return behaviour
- Recent order frequency
- Order-value shifts
- Return-rate shifts
- Category switching behaviour

---

## 🔎 Explainability

The project uses **SHAP** to explain individual predictions.

For each prediction, the system identifies:

### Risk-increasing factors

Features that contributed towards increasing the predicted return risk.

### Risk-reducing factors

Features that contributed towards reducing the predicted return risk.

This makes the ML system easier for business users to understand.

---

## 📊 Risk Scoring

The predicted return probability is converted into a risk score.

```text
Risk Score = Return Probability × 100
```

The system categorizes orders into different risk tiers:

| Risk Score | Risk Tier |
|------------|-----------|
| 0–19.99 | LOW |
| 20–39.99 | MEDIUM |
| 40–59.99 | HIGH |
| 60–79.99 | VERY_HIGH |
| 80–100 | CRITICAL |

---

## 💼 Business Decision Engine

The machine-learning prediction is converted into a business action using both:

- ML risk score
- Expected financial exposure

Possible actions include:

- **Normal Processing**
- **Monitor**
- **Review**
- **Intervene**

### Expected Return Exposure

The system uses:

```text
Expected Exposure =
Order Value × Return Probability × Return Cost Rate
```

The default synthetic return cost assumption is:

```text
Return Cost Rate = 8%
```

---

## 📈 Business Optimization

The project evaluates different intervention strategies.

### Capacity Optimization

Tests different intervention capacities to determine the capacity that produces the highest net benefit.

### Ranking Strategy Comparison

Compares different ways of selecting orders for intervention, including:

- Return probability
- Expected financial exposure

### Threshold Optimization

Evaluates different prediction thresholds to determine how selective the intervention should be.

The objective is to balance:

```text
Loss Avoided
      ↓
Intervention Cost
      ↓
Net Business Benefit
```

---

## 🛠️ Technology Stack

### Programming

- Python

### Machine Learning

- LightGBM
- Scikit-learn
- NumPy
- Pandas

### Explainable AI

- SHAP

### Dashboard

- Streamlit

### API

- FastAPI
- Uvicorn
- Pydantic

---

## 📁 Project Structure

```text
return-risk-intelligence/
│
├── data/
│   ├── raw/
│   └── processed/
│
├── evaluation/
│
├── models/
│   └── behaviour_model_v2.txt
│
├── notebooks/
│
├── src/
│   ├── api/
│   │   ├── business_engine.py
│   │   ├── explainer.py
│   │   ├── main.py
│   │   ├── predictor.py
│   │   ├── realtime_features.py
│   │   └── test_prediction.py
│   │
│   ├── dashboard/
│   │   └── app.py
│   │
│   ├── data/
│   ├── evaluation/
│   ├── features/
│   └── models/
│
├── tests/
│
├── config.yaml
├── requirements.txt
├── README.md
└── .gitignore
```

---

## ▶️ How to Run

### 1. Create a Python virtual environment

Open a terminal in the project root:

```bash
python -m venv .venv
```

### 2. Activate the virtual environment

On Windows:

```powershell
.venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Start the Streamlit dashboard

The main dashboard is located at:

```text
src/dashboard/app.py
```

Run:

```bash
streamlit run src/dashboard/app.py
```

The dashboard will open in your browser.

---

## ⚡ Real-Time Prediction API

The project also contains a FastAPI application.

The API entry point is:

```text
src/api/main.py
```

Start the API with:

```bash
uvicorn src.api.main:app --reload
```

The API provides endpoints for:

- Health checking
- Real-time risk prediction
- Risk scoring
- SHAP explanations
- Business decisions

---

## 🧪 Model and Data Pipeline

The overall pipeline is:

```text
Raw Customer Data
        +
Raw Product Data
        +
Raw Order Data
        +
Return Outcomes
        ↓
Feature Engineering
        ↓
43 Point-in-Time Features
        ↓
Time-Based Dataset Split
        ↓
LightGBM Model
        ↓
Return Probability
        ↓
Risk Score
        ↓
SHAP Explanation
        ↓
Financial Exposure
        ↓
Business Decision
        ↓
Business Impact Analysis
```

---

## 📊 Business Decision Flow

```text
Customer + Product + Order
            ↓
      Feature Builder
            ↓
       LightGBM Model
            ↓
    Return Probability
            ↓
        Risk Score
            ↓
      ┌─────┴─────┐
      ↓           ↓
    SHAP      Financial
 Explanation   Exposure
      ↓           ↓
      └─────┬─────┘
            ↓
    Business Decision
            ↓
Normal / Monitor / Review / Intervene
```

---

## 🎯 Project Objective

The objective of Return-Risk Intelligence is to move beyond simple return prediction.

The system answers three important questions:

1. **How likely is this order to be returned?**
2. **Why does the model think the order is risky?**
3. **What should the business do about it?**

This creates an end-to-end system connecting:

**Machine Learning + Explainable AI + Business Intelligence**

---

## 🚀 Project Highlights

- 19,821 synthetic orders
- 2,000 customers
- 500 products
- 43 real-time model features
- Behaviour-aware feature engineering
- Time-based train/validation/test split
- LightGBM return-risk model
- SHAP explanations
- Real-time prediction API
- Interactive Streamlit dashboard
- Risk monitoring
- Order investigation
- Cost-aware business decisions
- Capacity optimization
- Ranking strategy comparison
- Threshold optimization

---

## 👨‍💻 Project

**Return-Risk Intelligence**

AI + Behaviour Analytics + Explainable ML + Cost-Aware Decision Intelligence