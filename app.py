import streamlit as st
import numpy as np
import pandas as pd

import joblib

# Load the model and scaler using joblib
model = joblib.load("rf_model.pkl")     # Make sure filename is correct
scaler = joblib.load("scaler.pkl")

# Feature list
feature_names = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
                 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']

# Page config
st.set_page_config(page_title="Diabetes Risk Predictor", page_icon="💖", layout="centered")

# Title section
st.markdown("<h1 style='text-align: center; color: #de3163;'>💖 Diabetes Risk Predictor</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Advanced ML model for early diabetes risk assessment</p>", unsafe_allow_html=True)
st.markdown("---")

# Sidebar form inputs
with st.form("predict_form"):
    st.markdown("### <span style='color:#256D85;'>🔷 Patient Information</span>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        pregnancies = st.number_input("Pregnancies", 0, 20, 1)
        glucose = st.slider("Glucose (mg/dL)", 0, 300, 120)
        bp = st.slider("Blood Pressure (mmHg)", 0, 250, 80)
        skin = st.slider("Skin Thickness (mm)", 0, 99, 20)

    with col2:
        insulin = st.slider("Insulin (μU/mL)", 0, 100, 80)
        bmi = st.slider("BMI", 0.0, 67.1, 25.0, step=0.1)
        dpf = st.slider("Diabetes Pedigree Function", 0.0, 2.5, 0.5, step=0.01)
        age = st.slider("Age", 21, 100, 30)

    submit = st.form_submit_button("🔍 Predict Risk")

# Prediction
if submit:
    features = [pregnancies, glucose, bp, skin, insulin, bmi, dpf, age]
    input_df = pd.DataFrame([features], columns=feature_names)
    scaled_input = scaler.transform(input_df)

    prediction = model.predict(scaled_input)[0]
    prob = model.predict_proba(scaled_input)[0][1]
    confidence = model.predict_proba(scaled_input)[0][0]

    # Result display
    st.markdown("### 📊 Prediction Results")
    
    if prediction == 1:
        st.markdown("<div style='background-color:#ffe6e6;padding:15px;border-radius:10px'><b>Risk Level:</b> <span style='color:red;'>High ⚠️</span></div>", unsafe_allow_html=True)
    else:
        st.markdown("<div style='background-color:#e6ffe6;padding:15px;border-radius:10px'><b>Risk Level:</b> <span style='color:green;'>Low ✅</span></div>", unsafe_allow_html=True)

    st.markdown(f"<div style='background-color:#e6f0ff;padding:10px;border-radius:10px'><b>Probability:</b> {prob*100:.1f}%</div>", unsafe_allow_html=True)
    st.markdown(f"<div style='background-color:#f3e6ff;padding:10px;border-radius:10px'><b>Confidence:</b> {confidence*100:.1f}%</div>", unsafe_allow_html=True)

    # Risk Progress Bar
    st.markdown("**Risk Probability**")
    st.progress(prob)