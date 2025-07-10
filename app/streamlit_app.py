import sys
import os
import pickle

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st
import pandas as pd
import plotly.express as px
import shap
from src.utils import load_and_prepare_data
from src.rule_engine import score_application
from src.pdf_generator import generate_pdf
from sklearn.preprocessing import LabelEncoder

st.set_page_config(page_title="Credit Risk Decision Engine", layout="wide")
st.title("🏦 Smart Credit Risk Decision Engine")

# Load and prepare the data
df = load_and_prepare_data("data/loan_applications.csv")

@st.cache_resource
def load_model():
    with open("src/model.pkl", "rb") as f:
        return pickle.load(f)

ml_model = load_model()

# Apply rule-based decision
df["SystemDecision"] = df.apply(score_application, axis=1)

# Prepare features for model
# Encode EmploymentStatus exactly like training
le = LabelEncoder()
df["EmploymentStatus"] = le.fit_transform(df["EmploymentStatus"])

ml_features = df[[
    "CreditScore", "AnnualIncome", "LoanAmount", "OutstandingDebt",
    "DebtToIncome", "TenureMonths", "EmploymentStatus"
]]

# Use TreeExplainer for Random Forest
explainer = shap.TreeExplainer(ml_model)
shap_values = explainer.shap_values(ml_features)

# Predict
df["MLPrediction"] = ml_model.predict(ml_features)
df["MLPredictionLabel"] = df["MLPrediction"].map({1: "Approved", 0: "Denied"})

# Show results
st.subheader("📊 Application Decisions")
st.dataframe(df[[
    "CreditScore", "AnnualIncome", "LoanAmount", "EmploymentStatus",
    "OutstandingDebt", "TenureMonths", "DebtToIncome", "SystemDecision", "LoanOutcome"
]])

df["RiskFlag"] = df["MLPredictionLabel"].apply(lambda x: "🚨" if x == "Denied" else "✅")

# 🔍 Feature Importance Visualization
st.subheader("🔍 ML Model Feature Importance (Interactive)")

# Feature labels (used during training)
feature_names = [
    "CreditScore", "AnnualIncome", "LoanAmount", "OutstandingDebt",
    "DebtToIncome", "TenureMonths", "EmploymentStatus"
]

importances = ml_model.feature_importances_

importance_df = pd.DataFrame({
    "Feature": feature_names,
    "Importance": importances
}).sort_values("Importance", ascending=True)

# Plotly bar chart
fig = px.bar(
    importance_df,
    x="Importance",
    y="Feature",
    orientation="h",
    title="Feature Importance in Loan Decision Model",
    color="Importance",
    color_continuous_scale="Blues"
)

st.plotly_chart(fig, use_container_width=True)

st.subheader("🔍 SHAP Explanation for Individual Predictions")

selected_index = st.number_input("Select Applicant Index", min_value=0, max_value=len(df) - 1, value=0, step=1)

selected_features = ml_features.iloc[selected_index]
selected_shap_values = shap_values[1][selected_index]  # class 1 = Approved

st.markdown(f"### Prediction for Applicant #{selected_index}")
st.markdown(f"**ML Prediction**: {df.loc[selected_index, 'MLPredictionLabel']}  \n"
            f"**Actual Outcome**: {df.loc[selected_index, 'LoanOutcome']}")

# Force plot (text explanation)
st.markdown("#### 🔎 Feature impact (positive/negative):")

# Generate a text summary of important features
top_features = sorted(zip(selected_features.index, selected_shap_values), key=lambda x: abs(x[1]), reverse=True)[:3]
for feature, value in top_features:
    direction = "⬆️ Increases approval" if value > 0 else "⬇️ Decreases approval"
    st.write(f"- **{feature}** ({value:.3f}): {direction}")

# Compare system decision with actual outcome
accuracy = (df["SystemDecision"] == df["LoanOutcome"]).mean()
st.success(f"🤖 Rule Engine Accuracy Compared to Real Outcome: **{accuracy:.2%}**")

# 📄 Generate PDF reports
st.subheader("📄 Downloadable Loan Decision Reports")

if st.button("Generate PDF Reports for All Applicants"):
    with st.spinner("Generating PDF reports..."):
        for idx, row in df.iterrows():
            generate_pdf(row)
    st.success("✅ PDF reports saved in the `outputs/reports/` folder.")

st.subheader("🤖 ML Model Predictions vs Rule-Based vs Actual")
st.dataframe(
    df[[
        "CreditScore", "LoanAmount", "EmploymentStatus", 
        "SystemDecision", "MLPredictionLabel", "LoanOutcome", "RiskFlag"
    ]].style.applymap(lambda x: "color: red;" if x == "🚨 Denied" else "color: green;", subset=["RiskFlag"])
)

ml_accuracy = (df["MLPredictionLabel"] == df["LoanOutcome"]).mean()
st.info(f"🎯 ML Model Accuracy Compared to Real Outcome: **{ml_accuracy:.2%}**")

st.subheader("📊 Model Comparison Summary")

col1, col2 = st.columns(2)
col1.metric("🧠 ML Accuracy", f"{ml_accuracy:.2%}")
col2.metric("📝 Rule-Based Accuracy", f"{accuracy:.2%}")

if ml_accuracy > accuracy:
    st.success("✅ ML Model is significantly more accurate than the Rule Engine.")
else:
    st.warning("📌 Rule Engine still outperforms the ML model.")

# 🖨️ Generate PDFs for high-risk cases
st.subheader("🖨️ Generate PDFs for High-Risk Applicants")

if st.button("Generate PDF Reports for 🚨 Denied Applicants (ML Prediction)"):
    risky_df = df[df["MLPredictionLabel"] == "Denied"]
    with st.spinner(f"Generating {len(risky_df)} PDFs..."):
        for _, row in risky_df.iterrows():
            generate_pdf(row)
    st.success(f"✅ PDF reports saved for {len(risky_df)} high-risk applicants in `outputs/reports/`")