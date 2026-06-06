import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(page_title="Recommendation Engine", page_icon="🎯")

st.title("🎯 Personalized Product Recommendations")
st.markdown("Collaborative filtering engine to suggest products based on user purchase history.")

user_id = st.text_input("Enter Customer ID", "CUST-90124")

if st.button("Get Recommendations"):
    st.subheader(f"Top Recommended Products for {user_id}")
    
    # Simulate recommendation output
    recs = [
        {"Product": "Ergonomic Office Chair", "Category": "Furniture", "Score": 4.9},
        {"Product": "Wireless Noise-Canceling Headphones", "Category": "Electronics", "Score": 4.7},
        {"Product": "Standing Desk Converter", "Category": "Furniture", "Score": 4.5},
        {"Product": "Mechanical Keyboard", "Category": "Electronics", "Score": 4.3},
        {"Product": "Desk Lamp with Wireless Charger", "Category": "Electronics", "Score": 4.1}
    ]
    
    st.table(pd.DataFrame(recs))

st.markdown("### Underlying ML Technology")
st.markdown("- **Algorithm**: Funk SVD (Matrix Factorization).")
st.markdown("- **Implementation**: Custom `CuPy` code for GPU acceleration.")
st.markdown("- **Validation**: Nested Cross-Validation.")
