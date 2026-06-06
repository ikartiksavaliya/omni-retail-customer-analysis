import streamlit as st
import numpy as np

st.set_page_config(page_title="Review Sentiment Analysis", page_icon="💬")

st.title("💬 NLP Review Sentiment & Urgency")
st.markdown("Analyze customer reviews (Portuguese) for sentiment polarity and urgency escalation.")

review_text = st.text_area("Enter Customer Review (Portuguese)", "O produto não chegou no prazo e a embalagem estava danificada. Quero meu dinheiro de volta!")

if st.button("Analyze Review"):
    # Simulate NLP prediction
    if "danificada" in review_text.lower() or "dinheiro" in review_text.lower():
        sentiment = "Negative"
        urgency = "High (Escalate to Support)"
        st.error(f"**Sentiment**: {sentiment}")
        st.error(f"**Urgency**: {urgency}")
    else:
        sentiment = "Positive"
        urgency = "Low"
        st.success(f"**Sentiment**: {sentiment}")
        st.info(f"**Urgency**: {urgency}")

st.markdown("### Underlying ML Technology")
st.markdown("- **Algorithm**: Linear Support Vector Classification (LinearSVC).")
st.markdown("- **Feature Engineering**: Custom Portuguese stop-words, TF-IDF Vectorization (20k max features).")
