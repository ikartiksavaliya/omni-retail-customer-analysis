# High-Level Architecture

The **End-to-End E-Commerce Intelligence Platform** is a monolithic repository combining data engineering, multiple machine learning pipelines, and an interactive front-end application.

## System Components

```mermaid
graph TD
    subgraph Data Layer
        DB[(SQLite Database)]
        CSV[Raw CSV Extracts]
        FeatStore[(Feature Store / Outputs)]
    end

    subgraph ML Pipelines Layer
        Ingestion[Data Ingestion & Feature Engineering]
        Churn[Churn & Conversion Predictor]
        Demand[Demand Forecaster]
        Segment[Customer Segmenter]
        NLP[Review NLP Engine]
        Recommender[SVD Recommender]
        Fraud[Fraud Detection Engine]
    end

    subgraph Serving Layer
        App[Streamlit App]
        Models[(Serialized .joblib / .json)]
    end

    CSV -->|ETL| DB
    DB --> Ingestion
    Ingestion -->|Engineered Data| FeatStore
    FeatStore --> Churn
    FeatStore --> Demand
    FeatStore --> Segment
    FeatStore --> NLP
    FeatStore --> Recommender
    FeatStore --> Fraud
    
    Churn -->|Weights| Models
    Demand -->|Weights| Models
    Segment -->|Weights| Models
    NLP -->|Weights| Models
    Recommender -->|Weights| Models
    Fraud -->|Weights| Models

    Models --> App
    FeatStore --> App
```

## Technology Stack

- **Data Processing**: Pandas, NumPy, RAPIDS cuML (GPU-accelerated algorithms).
- **Database**: SQLite.
- **Machine Learning**: Scikit-Learn, XGBoost, LightGBM, CuPy (for custom GPU SVD).
- **Hyperparameter Optimization**: Optuna.
- **Explainability**: SHAP, LIME.
- **Serving / App**: Streamlit.
