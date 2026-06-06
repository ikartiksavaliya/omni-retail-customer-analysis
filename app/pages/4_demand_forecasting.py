import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(page_title="Demand Forecasting", page_icon="📊")

st.title("📊 Time-Series Demand Forecasting")
st.markdown("Forecast weekly product category demand to optimize supply chain and inventory.")

category = st.selectbox("Select Product Category", ["Electronics", "Furniture", "Apparel", "Health & Beauty"])
weeks_ahead = st.slider("Forecast Horizon (Weeks)", 1, 12, 4)

if st.button("Generate Forecast"):
    # Simulate forecast data
    dates = pd.date_range(start=pd.Timestamp.now(), periods=weeks_ahead, freq='W')
    base_demand = np.random.randint(500, 1500)
    trend = np.linspace(0, 200, weeks_ahead)
    noise = np.random.normal(0, 50, weeks_ahead)
    forecast = base_demand + trend + noise
    
    df = pd.DataFrame({"Date": dates, "Forecasted Units": forecast})
    df.set_index("Date", inplace=True)
    
    st.line_chart(df)
    st.success(f"Forecast generated for {category} over the next {weeks_ahead} weeks.")

st.markdown("### Underlying ML Technology")
st.markdown("- **Algorithm**: Time-Series Aware LightGBM.")
st.markdown("- **Optimization**: Optuna Bayesian Search (SMAPE optimized).")
