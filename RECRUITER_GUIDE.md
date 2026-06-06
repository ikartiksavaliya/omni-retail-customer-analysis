# Recruiter's Guide: E-Commerce Intelligence Platform

Welcome to my portfolio! If you are a technical recruiter or hiring manager, this document is designed to give you a quick overview of why this project matters, the skills it demonstrates, and how it translates to real-world business value.

## 🎯 Business Impact (Why this matters)
This is not a toy project. It is a simulated production environment that solves real business problems:
1. **Revenue Retention**: Prevents revenue loss by predicting customer churn *before* it happens.
2. **Supply Chain Efficiency**: Reduces inventory holding costs through accurate time-series demand forecasting.
3. **Personalization & Upselling**: Increases Average Order Value (AOV) by serving targeted product recommendations.
4. **Risk Management**: Detects fraudulent financial transactions in real-time, reducing chargeback losses.
5. **Customer Experience**: Automatically triages urgent negative reviews using NLP, reducing support resolution time.

## 💻 Technical Skills Demonstrated

| Skill Category | Tools / Techniques Demonstrated |
|----------------|---------------------------------|
| **Data Engineering** | SQL (SQLite), Pandas, Leakage-free Group Splitting, Feature Engineering (Cyclical mapping, Lag features) |
| **Machine Learning** | Scikit-Learn, XGBoost, LightGBM, Custom Matrix Factorization, Streaming `SGDClassifier` |
| **GPU Acceleration** | RAPIDS cuML (GPU PCA/Clustering), CuPy (GPU Math) |
| **MLOps / Evaluation** | Nested Cross-Validation, Optuna (Bayesian Search), SHAP & LIME for explainability |
| **Software Engineering**| Object-Oriented Pipelines, Modular Architecture, Streamlit Web App Deployment |

## 🎤 Interview Talking Points

If we speak in an interview, I would love to discuss:
1. **Target Leakage**: Ask me how I prevented target leakage when predicting conversion by stripping post-event features (like delivery dates).
2. **Imbalanced Data**: Ask me why I applied SMOTE *only* to the training split inside the Cross-Validation loop instead of the whole dataset.
3. **GPU Compute**: Ask me how writing my own Funk SVD algorithm in `CuPy` resulted in a massive speedup compared to CPU-based matrix factorization.
4. **Streaming ML**: Ask me how the Fraud Detection module updates its weights in real-time using mini-batches without needing to be retrained from scratch.
