# app.py

import streamlit as st
import pandas as pd
import joblib

# Load trained XGBoost model and scaler
model = joblib.load("xgb_mismatch_model.pkl")
scaler = joblib.load("xgb_scaler.pkl")

# Page config
st.set_page_config(page_title="Operator-Job Mismatch Predictor", layout="centered")
st.title("👷 Operator-Job Mismatch Predictor")

st.markdown("This smart tool predicts whether assigning an operator to a job may lead to a **mismatch**, combining machine learning with performance logic.")

# Input Form
with st.form("predict_form"):
    st.subheader("🔧 Enter Operator and Job Details")

    col1, col2 = st.columns(2)

    with col1:
        skill_level = st.slider("Operator Skill Level", 1, 5, 3)
        job_skill_required = st.slider("Job Skill Required", 1, 5, 3)
        safety_incidents = st.slider("Past Safety Incidents", 0, 10, 0)

    with col2:
        product_quality_score = st.number_input("Product Quality Score", 0.0, 100.0, 50.0, step=0.1)
        rework_cost = st.number_input("Rework Cost (₹)", 0.0, 10000.0, 2500.0, step=100.0)
        operational_efficiency = st.slider("Operational Efficiency", 0.0, 1.0, 0.75, step=0.01)

    submitted = st.form_submit_button("🎯 Predict Mismatch")

# Prediction Logic
if submitted:
    skill_gap = job_skill_required - skill_level
    is_overqualified = int(skill_gap < 0)
    is_underqualified = int(skill_gap > 0)

    # Prepare input
    input_dict = {
        "skill_level": skill_level,
        "job_skill_required": job_skill_required,
        "skill_gap": skill_gap,
        "is_overqualified": is_overqualified,
        "is_underqualified": is_underqualified,
        "safety_incidents": safety_incidents,
        "product_quality_score": product_quality_score,
        "rework_cost": rework_cost,
        "operational_efficiency": operational_efficiency
    }

    input_df = pd.DataFrame([input_dict])

    # Scale performance features
    input_df[['product_quality_score', 'rework_cost', 'operational_efficiency']] = scaler.transform(
        input_df[['product_quality_score', 'rework_cost', 'operational_efficiency']]
    )

    # ML prediction
    prediction = model.predict(input_df)[0]

    # Hybrid logic override
    st.markdown("---")
    st.subheader("📊 Prediction Result")

    if safety_incidents > 5 or operational_efficiency < 0.3 or rework_cost > 3000:
        st.error("❌ Mismatch Likely — Operator performance is not suitable.")
    else:
        if prediction == 1:
            st.error("❌ Mismatch Likely — Avoid assigning this operator.")
        else:
            st.success("✅ Match Likely — Operator is suitable for this job.")

    # Skill gap explanation
    st.markdown(f"**Skill Gap:** `{job_skill_required} - {skill_level} = {skill_gap}`")
    if skill_gap > 1:
        st.warning("⚠️ Skill gap is large. Operator may struggle.")
    elif skill_gap < 0:
        st.info("ℹ️ Operator is overqualified for this job.")
