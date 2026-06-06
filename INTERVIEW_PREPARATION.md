# Interview Preparation & Technical Explanations

This document provides technical deep-dives for every notebook in the repository, serving as a study guide for data science and ML engineering interviews.

## Notebook 01: Data Ingestion & Engineering

**Q: How did you handle cyclical features like "Hour of Day" or "Month"?**
* **Answer**: Linear representations of time (e.g., Month 1 to 12) confuse tree-based algorithms because December (12) and January (1) seem far apart, despite being adjacent. I transformed these using trigonometric sine and cosine functions.
* **Tradeoff**: It doubles the feature count for time variables but strictly preserves temporal proximity.

**Q: Explain how you prevented Target Leakage.**
* **Answer**: When predicting if an order would be successfully delivered (conversion), I dropped features like `actual_delivery_date` and `review_score`. In a real-world scenario, at the time of prediction (when the user clicks checkout), these variables do not exist yet. Including them would yield artificially high validation scores.

## Notebook 02: Classification Engine

**Q: Why use a Stacking Classifier instead of just Random Forest?**
* **Answer**: Stacking allows me to combine the strengths of different architectures. XGBoost is great with complex nonlinearities, and LightGBM is highly efficient with large datasets. The meta-estimator (Logistic Regression) learns which base model to trust for specific data distributions, reducing overall variance.
* **Tradeoff**: Slower inference time and higher compute cost.

**Q: How did you handle class imbalance for Churn?**
* **Answer**: I used SMOTE (Synthetic Minority Over-sampling Technique). Crucially, I only applied SMOTE to the training splits *during* cross-validation. Applying it before splitting causes synthetic data leakage into the validation set.

## Notebook 03: Demand Forecasting

**Q: Why use LightGBM for Time-Series instead of ARIMA?**
* **Answer**: While ARIMA is statistically robust for univariate series, retail demand depends heavily on exogenous variables (holidays, pricing, promotions). LightGBM handles high-dimensional, non-linear feature spaces easily and can predict across multiple categories simultaneously.

## Notebook 04: Segmentation

**Q: Why use both PCA and t-SNE?**
* **Answer**: PCA was used for initial dimensionality reduction to capture global variance and reduce the feature space from ~30 dimensions down to 10. t-SNE was then used specifically for 2D visualization because it preserves local neighborhood structures (clusters) much better than PCA.

## Notebook 05: NLP Sentiment

**Q: Why a LinearSVC over a Neural Network for text?**
* **Answer**: For TF-IDF vectorized text with 20,000 features, LinearSVC provides excellent performance with extremely fast training times. Given the dataset size, an LLM or deep Neural Network would have been overkill and computationally expensive to serve.

## Notebook 06: Recommendations & Fraud

**Q: What is Streaming Learning?**
* **Answer**: For fraud detection, transaction patterns change rapidly. Instead of retraining a model from scratch every week, I used an `SGDClassifier` using the `partial_fit` method. This allows the model to process mini-batches of new data, updating its weights in real-time, simulating a true streaming environment.
