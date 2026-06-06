# Project 1: E-Commerce Intelligence — Comprehensive Audit & Review

This document provides a complete summary of the progress on **Project 1: E-Commerce Intelligence**, evaluating your pipeline, checking for correct implementation, and breaking down all six notebooks.

---

## 1. Executive Summary & Progress Evaluation
**How are you doing?**
You are doing **exceptionally well**. The pipeline is constructed with high academic rigor and production-grade software engineering principles. 

### What You Did Correctly:
* **Leakage-Free Validation Splitting**: Splitting train/validation/test sets using group-based customer IDs rather than random row-based shuffling. This prevents models from memorizing specific customers.
* **Leakage-Free EDA**: Exploring and visualizing data patterns *only* on the training set to prevent cognitive leakage.
* **Strict Target Leakage Mitigation**: In the ingestion notebook (`01`), you correctly identified and removed post-event features (e.g. actual delivery durations, review scores) that would artificially inflate classification metrics.
* **High-Performance Acceleration**: Leveraged GPU-accelerated RAPIDS `cuml` and CuPy libraries to train clustering, SVD, and SGD algorithms, significantly reducing training overhead.
* **Advanced Architecture**: Used Optuna Bayesian Search inside Nested Cross-Validation to optimize classifiers without introducing validation bias, and implemented model explainability (SHAP & LIME) to demystify black-box models.

### Areas for Refinement & Minor Corrections:
1. **Vocabulary Leakage in NLP (Notebook 05)**: The TF-IDF vectorizer fits on the *entire* text corpus before splitting. This leads to mild vocabulary leakage. In production, you should fit the vectorizer *only* on the training split and transform validation/test splits.
2. **Cold Start Validation Clipping (Notebook 06)**: When validating SVD recommender weights, unseen validation indices are clipped to the last known index. This forces the model to evaluate an unseen user as if they were a known user, distorting validation RMSE. Unseen users should be predicted using global/bias averages.
3. **Stop-Words Expansion (Notebook 05)**: The custom Portuguese stop-words list is quite small (~40 words). Utilizing a comprehensive library like NLTK or SpaCy's Portuguese stop-words would remove common noise more effectively.

---

## 2. Notebook-by-Notebook Summary

Below is a summary of the methodology, algorithms, and key outcomes for each of the six notebooks.

### Notebook 01: Data Ingestion & Feature Engineering
* **Purpose**: Load raw relational CSV files into a structured SQLite database, join them using leakage-free aggregation queries, execute train/test splitting on customer groups, engineer robust features, and build scaling pipelines.
* **Methodology**:
  * SQLite db connection and relational queries.
  * Group-based split (80% Train, 10% Val, 10% Test) on `customer_unique_id`.
  * Feature engineering: cyclical time representations (trigonometric sin/cos of purchase month/hour/day), delivery delays, product dimensions, and payment details.
  * Pipeline preprocessing: Scikit-learn `ColumnTransformer` with median imputation, outlier clipping, and standard scaling.
  * Collinearity (corr > 0.95) and near-constant variance (var < 0.01) pruning.
* **Outcome**: A clean, Leakage-Free dataset exported to `outputs/` ready for modeling.

### Notebook 02: Churn & Conversion Prediction
* **Purpose**: Predict customer conversion (completed delivery status) and customer churn using ensemble classification.
* **Methodology**:
  * Balanced class distribution using SMOTE *strictly on the training set*.
  * Benchmark baseline models (Logistic Regression, KNN, Naive Bayes, Random Forest).
  * Build a stacked ensemble (`StackingClassifier`) combining XGBoost, LightGBM, and Random Forest, using a Logistic Regression meta-classifier.
  * Evaluated performance using ROC-AUC curves, Precision-Recall curves, and Confusion Matrices.
* **Outcome**: A robust stacking model achieving high prediction reliability on unbalanced customer data.

### Notebook 03: Demand Forecasting & Price Regression
* **Purpose**: Predict weekly transaction orders and average product prices per category.
* **Methodology**:
  * Time-series feature engineering: lag variables, rolling averages, seasonal variables, and national holidays.
  * Baseline models (Lasso, Ridge, ElasticNet, SVR, Polynomial) compared against XGBoost and LightGBM.
  * GPU-accelerated LightGBM optimization via Optuna Bayesian Search (30 trials).
  * Stacking regressor combining LightGBM and XGBoost via a Linear Regression meta-regressor.
* **Outcome**: Accurate forecasts with SMAPE evaluation and weekly category-level predictions.

### Notebook 04: Customer Segmentation & Clustering
* **Purpose**: Segment customers based on purchase behavior, finance, and feedback to generate business personas.
* **Methodology**:
  * Feature aggregation into customer-level RFM (Recency, Frequency, Monetary) metrics.
  * Dimensionality reduction using GPU-accelerated PCA and t-SNE (cuml).
  * Evaluated and compared K-Means, DBSCAN, Agglomerative Clustering, and GMM models.
  * Used Linear Discriminant Analysis (LDA) to evaluate cluster separation.
  * Customer profiling to produce 4 distinct segments (e.g. "Loyal High-Satisfaction Shoppers").
* **Outcome**: Highly distinct segments saved to `customer_segments.csv.gz` to enable targeted marketing.

### Notebook 05: Customer Review Sentiment & Urgency NLP
* **Purpose**: Evaluate Portuguese review text to extract customer sentiments and flag urgent feedback.
* **Methodology**:
  * Merged review title and comment body into `full_text`.
  * Preprocessing: lowercase mapping, accent stripping, custom stop-words removal, and TF-IDF vectorization (20,000 feature limit).
  * Trained and compared LinearSVC, Multinomial Naive Bayes, and Bernoulli Naive Bayes.
  * Optimized LinearSVC via Grid Search.
  * Analyzed top feature importance (key words) per class.
* **Outcome**: Serialized text vectorizer and sentiment/urgency models exported to `models/` for downstream inference.

### Notebook 06: Recommendation Engine & Scaling
* **Purpose**: Build collaborative filtering models, optimize model hyperparameters, interpret predictions, and run streaming simulation.
* **Methodology**:
  * **Surprise/Funk SVD**: Custom Funk SVD matrix factorization written from scratch in CuPy to leverage GPU memory.
  * **Nested CV**: Tuned XGBoost on GPU using a 3x3 nested loop comparing Grid, Random, and Optuna searches.
  * **Explainability**: SHAP summary plots for global evaluation; LIME explanations for local individual predictions.
  * **Online Learning**: Custom online logistic regression `CuPySGDClassifier` trained incrementally in batches of 256 to simulate streaming data.
* **Outcome**: A comprehensive model serving pipeline ready for deployment.

---

## 3. Git Repository Architecture
We successfully consolidated the work into a single branch.

### Created Branch:
* **`project1-ecommerce-intelligence`**

### Files Tracked & Committed:
1. **Relational / Ingestion Pipeline**: `01_data_ingestion_features.ipynb` *(includes timeline and target leakage fixes from the forecasting branch)*
2. **Classification Pipeline**: `02_churn_conversion_classification.ipynb` *(staged stacking models)*
3. **Forecasting Pipeline**: `03_demand_forecasting_regression.ipynb` *(staged regression models)*
4. **Segmentation Pipeline**: `04_customer_segmentation_clustering.ipynb` *(cuml GPU clustering)*
5. **NLP Text Pipeline**: `05_customer_review_nlp.ipynb` *(review sentiment/urgency models)*
6. **Serving/Optimization Pipeline**: `06_recommendations_production.ipynb` *(restored from git stash; contains SVD recommender, online SGD learning, nested CV, and SHAP/LIME explainability)*
7. **Environment Dependencies**: `requirements.txt` *(synchronized with optuna and holidays package entries)*
8. **Models & Visualizations**: All serialized `.joblib` / `.json` model files and generated evaluation plots are fully indexed and committed.
