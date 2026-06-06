# Data Flow Documentation

This document explains the lifecycle of data as it passes through the E-Commerce Intelligence Platform.

## 1. Ingestion Flow

The ingestion pipeline handles 10+ relational datasets (customers, orders, products, reviews, etc.) and transforms them into ML-ready matrices.

```mermaid
sequenceDiagram
    participant Raw as CSV Files
    participant SQLite as Database
    participant Ingestion as Ingestion Script
    participant Split as Group Splitter
    participant Pipeline as Sklearn Pipeline
    participant Output as Final Matrices

    Raw->>SQLite: Load into relational tables
    SQLite->>Ingestion: Complex JOINs & Aggregations
    Ingestion->>Ingestion: Feature Engineering (Trig cycles, Lag variables)
    Ingestion->>Split: Group by `customer_unique_id`
    Split->>Pipeline: Train Split (80%)
    Split->>Pipeline: Val Split (10%)
    Split->>Pipeline: Test Split (10%)
    Pipeline->>Pipeline: Median Imputation
    Pipeline->>Pipeline: Standard Scaling
    Pipeline->>Output: Export to `outputs/`
```

## 2. Feature Engineering Highlights

- **Leakage Prevention**: All features requiring timeline awareness (e.g., actual delivery time, review score) are stripped when predicting events that happen *prior* to those metrics being known (e.g., Conversion Prediction).
- **Cyclical Features**: Timestamp features (hour, day, month) are converted into `sin` and `cos` components to capture cyclicality.
- **RFM Extraction**: Recency, Frequency, and Monetary metrics are calculated for segmentation.
- **Text Vectorization**: Customer reviews are cleaned, normalized, stripped of custom Portuguese stop-words, and converted to TF-IDF vectors capped at 20,000 features.
