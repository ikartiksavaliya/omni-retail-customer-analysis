import streamlit as st

st.set_page_config(
    page_title="E-Commerce Intelligence Platform",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🛍️ Omni-Retail Customer Intelligence Platform")
st.markdown("---")

st.markdown("""
### Welcome to the Production Inference Server

This platform unifies multiple machine learning models developed during the analytical phase of the project.
Use the sidebar on the left to navigate between different intelligence modules:

1. **💳 Fraud Detection Demo**: Real-time streaming anomaly detection on financial transactions.
2. **👥 Customer Segmentation**: Cluster exploration using K-Means and GMM.
3. **📈 Conversion Prediction**: Stacked ensemble prediction of user checkout probability.
4. **📊 Demand Forecasting**: Time-series demand prediction using LightGBM.
5. **💬 Review Sentiment Analysis**: NLP inference for Portuguese text urgency and sentiment.
6. **🎯 Recommendation Engine**: Personalized product suggestions using Matrix Factorization (SVD).

**Technical Stack Behind This App:**
- UI built with `Streamlit`
- GPU-accelerated Data Preprocessing via `RAPIDS cuML`
- Complex feature extraction pipelines (`scikit-learn`)
- Stacking Classifiers (`XGBoost`, `LightGBM`)
- Online Streaming Learning (`SGDClassifier`)
""")

st.info("Select a module from the sidebar to begin.")
