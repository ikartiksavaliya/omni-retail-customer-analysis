# Resume Bullets Generator

Select the version that best fits the role you are applying for.

## 1-Line Version (For summary sections)
- Architected an End-to-End ML E-Commerce Platform featuring GPU-accelerated segmentation, time-series forecasting, and streaming fraud detection deployed via Streamlit.

## 3-Line Version (Standard)
- **E-Commerce ML Platform**: Engineered an end-to-end data platform unifying 10+ relational databases to drive insights via 5+ ML pipelines (Churn, Demand, Segmentation, NLP, Fraud).
- **GPU Acceleration & Optimization**: Leveraged RAPIDS cuML and CuPy for 10x faster clustering and matrix factorization, optimizing hyperparameters using Nested Cross-Validation and Optuna.
- **Production Deployment**: Designed a streaming `SGDClassifier` for real-time anomaly detection and deployed all models into an interactive Streamlit application.

## ATS Optimized Version (Keyword dense)
- Developed an **End-to-End Machine Learning** platform using **Python**, **Scikit-Learn**, **XGBoost**, and **LightGBM** to predict customer **churn**, optimize **time-series demand forecasting**, and perform **NLP sentiment analysis**.
- Implemented **GPU-accelerated** clustering (**PCA**, **K-Means**, **RAPIDS**) and recommendation systems (**Collaborative Filtering**, **Funk SVD**, **CuPy**) to enhance marketing personalization.
- Engineered a **streaming data pipeline** for **fraud detection** using **Isolation Forests** and **SGDClassifier**, ensuring **target leakage** prevention and explaining models with **SHAP** and **LIME**.

## Data Scientist Version (Focus on algorithms & rigor)
- Engineered a stacking classifier (XGBoost, LightGBM, Logistic Regression) to predict conversion, strictly preventing target leakage and handling class imbalance via intra-CV SMOTE.
- Wrote custom Funk SVD matrix factorization from scratch using CuPy to leverage GPU compute for massive collaborative filtering datasets.
- Optimized time-series LightGBM models via 30-trial Optuna Bayesian Search within a Nested Cross-Validation loop to ensure robust generalization.

## Data Analyst Version (Focus on insights & business value)
- Designed an automated RFM (Recency, Frequency, Monetary) segmentation pipeline, identifying 4 distinct customer personas to drive targeted marketing campaigns.
- Extracted actionable insights from Portuguese customer reviews using TF-IDF and LinearSVC, enabling automated triage of urgent support tickets.
- Consolidated disparate operational data into a unified SQLite database, conducting leakage-free EDA and building interactive Streamlit dashboards for stakeholders.

## Internship / Entry-Level Version (Focus on learning & initiative)
- Built a comprehensive Machine Learning portfolio project demonstrating the full data science lifecycle, from SQL data ingestion to model deployment.
- Proactively learned and implemented advanced techniques typically reserved for senior roles, including Streaming Learning and Model Explainability (SHAP/LIME).
- Independently researched and applied GPU-acceleration libraries to overcome computational bottlenecks in standard scikit-learn models.
