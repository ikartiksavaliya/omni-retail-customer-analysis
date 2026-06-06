# Model Pipeline Architecture

This document breaks down the individual machine learning pipelines implemented in the platform.

## 1. Classification Engine (Churn & Conversion)
- **Goal**: Predict whether a user will convert (complete an order) or churn.
- **Handling Imbalance**: SMOTE applied *only* to the training set to prevent validation leakage.
- **Model Architecture**:
  - A Stacking Classifier using a Meta-Model.
  - Base Estimators: LightGBM, XGBoost, Random Forest.
  - Meta-Estimator: Logistic Regression.
- **Explainability**: SHAP values are extracted from the tree-based estimators to determine global feature importance.

## 2. Demand Forecasting
- **Goal**: Predict weekly order volume for inventory optimization.
- **Model Architecture**:
  - Time-series aware LightGBM optimized via Optuna (30 trials).
  - Stacking Regressor combining LightGBM and XGBoost via a Linear Regression meta-model.
- **Metrics**: SMAPE (Symmetric Mean Absolute Percentage Error).

## 3. Segmentation Engine
- **Goal**: Group users into actionable marketing personas.
- **Architecture**:
  - Dimensionality Reduction: RAPIDS cuML PCA (down to 10 components) followed by t-SNE for 2D visualization.
  - Clustering: K-Means, DBSCAN, Agglomerative, and Gaussian Mixture Models.
  - Evaluation: Linear Discriminant Analysis (LDA) for cluster separability and Silhouette Scores.

## 4. NLP Sentiment & Urgency Analysis
- **Goal**: Read Portuguese reviews, score sentiment, and classify urgency.
- **Architecture**:
  - Linear Support Vector Classification (LinearSVC) for high-dimensional text data.
  - TF-IDF Vectorizer with `max_features=20000`.

## 5. Recommendation Engine
- **Goal**: Suggest products based on collaborative filtering.
- **Architecture**:
  - Custom matrix factorization: Funk SVD implemented from scratch using CuPy for GPU memory efficiency.
  - Advanced Tuning: 3x3 Nested Cross-Validation.

## 6. Financial Fraud Detection (Streaming)
- **Goal**: Identify anomalous transactions in a continuous stream.
- **Architecture**:
  - Anomaly detection via Isolation Forest.
  - Online Learning via `SGDClassifier` using mini-batches (simulating a stream) to continually update weights without retraining from scratch.
