import streamlit as st
import pandas as pd
import numpy as np
import time

st.set_page_config(page_title="Fraud Detection", page_icon="💳")

st.title("💳 Financial Fraud Detection Simulator")
st.markdown("This module simulates a streaming data environment where transactions are evaluated in real-time.")

st.sidebar.header("Simulator Controls")
batch_size = st.sidebar.slider("Batch Size", 10, 500, 100)
stream_speed = st.sidebar.slider("Stream Speed (ms)", 100, 2000, 500)

if st.button("Start Streaming Simulation"):
    placeholder = st.empty()
    chart_placeholder = st.empty()
    
    anomalies = []
    for i in range(1, 21):
        with placeholder.container():
            st.metric("Batch Processed", f"{i * batch_size} Transactions")
            # Simulate processing delay
            time.sleep(stream_speed / 1000.0)
            
            # Simulate anomaly rate
            anomaly_rate = np.random.uniform(0.01, 0.05)
            anomalies.append(anomaly_rate * 100)
            st.warning(f"Detected Anomaly Rate: {anomaly_rate:.2%}")
            
        with chart_placeholder.container():
            st.line_chart(anomalies)

st.markdown("### Underlying ML Technology")
st.markdown("- **Algorithm**: Streaming `SGDClassifier` and `Isolation Forest`.")
st.markdown("- **Feature Extraction**: Online standard scaling and categorical encoding.")
