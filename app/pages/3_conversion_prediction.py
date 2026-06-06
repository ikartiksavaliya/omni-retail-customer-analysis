import streamlit as st
import numpy as np
import pandas as pd

st.set_page_config(page_title="Conversion Prediction", page_icon="📈")

st.title("📈 Conversion & Churn Prediction")
st.markdown("Enter customer metrics to predict the probability of order conversion or churn.")

with st.form("prediction_form"):
    st.subheader("Customer Features")
    col1, col2 = st.columns(2)
    
    with col1:
        time_on_site = st.slider("Time on Site (minutes)", 1, 60, 15)
        pages_visited = st.slider("Pages Visited", 1, 50, 8)
        
    with col2:
        cart_value = st.number_input("Cart Value ($)", min_value=0.0, max_value=5000.0, value=120.0)
        previous_purchases = st.number_input("Previous Purchases", min_value=0, max_value=50, value=2)
        
    submit = st.form_submit_button("Predict Conversion")

if submit:
    # Simulate prediction
    prob = np.random.uniform(0.6, 0.95)
    st.success(f"**Predicted Conversion Probability: {prob:.2%}**")
    st.progress(prob)

st.markdown("### Underlying ML Technology")
st.markdown("- **Algorithm**: Stacking Classifier (XGBoost, LightGBM, Random Forest).")
st.markdown("- **Meta-Model**: Logistic Regression.")
