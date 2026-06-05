import json

notebook = {
 "cells": [
  {
   "cell_type": "markdown",
   "id": "45784f1e",
   "metadata": {},
   "source": [
    "# Project 2: Fraud Detection - Step 1: Preprocessing\n",
    "\n",
    "This notebook handles the data ingestion, train/val/test splitting, and scaling/outlier preparation for the financial transaction dataset.\n",
    "\n",
    "## Blueprint Actions Covered\n",
    "* **The Split**: Strict train/validation/test split performed immediately.\n",
    "* **Missing Values & Outliers**: Imputation (median/mode) and outlier capping/flooring.\n",
    "* **Scaling**: StandardScaler and MinMaxScaler execution."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "id": "a87eb1c4",
   "metadata": {},
   "outputs": [],
   "source": [
    "import os\n",
    "import numpy as np\n",
    "import pandas as pd\n",
    "from sklearn.model_selection import train_test_split\n",
    "from sklearn.preprocessing import StandardScaler, MinMaxScaler\n",
    "\n",
    "FRAUD_DATA = \"../datasets/fraud/credit_card_fraud_10k.csv\"\n",
    "print(f\"Fraud dataset exists: {os.path.exists(FRAUD_DATA)}\")"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "split_section_title",
   "metadata": {},
   "source": [
    "## 1. Strict Stratified Train/Val/Test Split\n",
    "\n",
    "To prevent any form of data leakage (cognitive or preprocessing), we immediately split our data into training (70%), validation (15%), and test (15%) partitions. Because financial fraud datasets are highly imbalanced, we use **stratified** splitting to maintain the proportion of fraud vs. non-fraud transactions across all splits."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "id": "split_execution",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Load raw dataset\n",
    "df = pd.read_csv(FRAUD_DATA)\n",
    "\n",
    "# Separate features and target\n",
    "X = df.drop(columns=['transaction_id', 'is_fraud'])\n",
    "y = df['is_fraud']\n",
    "\n",
    "# Perform Stratified Train (70%), Validation (15%), and Test (15%) splits\n",
    "X_train, X_temp, y_train, y_temp = train_test_split(\n",
    "    X, y, test_size=0.30, random_state=42, stratify=y\n",
    ")\n",
    "X_val, X_test, y_val, y_test = train_test_split(\n",
    "    X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp\n",
    ")\n",
    "\n",
    "print(f\"Train set shape:      {X_train.shape}, positive rate: {y_train.mean():.4%}\")\n",
    "print(f\"Validation set shape: {X_val.shape}, positive rate: {y_val.mean():.4%}\")\n",
    "print(f\"Test set shape:       {X_test.shape}, positive rate: {y_test.mean():.4%}\")"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "eda_section_title",
   "metadata": {},
   "source": [
    "## 2. Exploratory Data Analysis (EDA) on Training Split\n",
    "\n",
    "We perform EDA exclusively on the training set to prevent cognitive leakage. We analyze distribution curves for key numerical features and examine categorical frequencies."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "id": "eda_execution",
   "metadata": {},
   "outputs": [],
   "source": [
    "import matplotlib.pyplot as plt\n",
    "import seaborn as sns\n",
    "\n",
    "PLOT_DIR = \"plots\"\n",
    "os.makedirs(PLOT_DIR, exist_ok=True)\n",
    "\n",
    "# Set plotting style\n",
    "sns.set_theme(style=\"whitegrid\")\n",
    "plt.rcParams.update({'font.size': 12, 'axes.labelsize': 14, 'axes.titlesize': 16})\n",
    "\n",
    "# Create a visual grid of distributions\n",
    "fig, axes = plt.subplots(2, 2, figsize=(15, 12))\n",
    "\n",
    "# 1. Distribution of Transaction Amounts (Log-scaled)\n",
    "sns.histplot(X_train['amount'], bins=50, kde=True, ax=axes[0, 0], color='royalblue')\n",
    "axes[0, 0].set_title('Transaction Amount Distribution')\n",
    "axes[0, 0].set_xlabel('Amount (BRL)')\n",
    "axes[0, 0].set_yscale('log') # Log scale since amount is heavily skewed\n",
    "\n",
    "# 2. Distribution of Cardholder Age\n",
    "sns.histplot(X_train['cardholder_age'], bins=20, kde=True, ax=axes[0, 1], color='seagreen')\n",
    "axes[0, 1].set_title('Cardholder Age Distribution')\n",
    "axes[0, 1].set_xlabel('Age')\n",
    "\n",
    "# 3. Distribution of Device Trust Score\n",
    "sns.histplot(X_train['device_trust_score'], bins=20, kde=True, ax=axes[1, 0], color='darkorange')\n",
    "axes[1, 0].set_title('Device Trust Score Distribution')\n",
    "axes[1, 0].set_xlabel('Trust Score')\n",
    "\n",
    "# 4. Merchant Category counts\n",
    "sns.countplot(data=X_train, x='merchant_category', ax=axes[1, 1], palette='Set2')\n",
    "axes[1, 1].set_title('Transactions by Merchant Category')\n",
    "axes[1, 1].set_xlabel('Category')\n",
    "axes[1, 1].tick_params(axis='x', rotation=15)\n",
    "\n",
    "plt.tight_layout()\n",
    "plt.savefig(os.path.join(PLOT_DIR, \"numerical_categorical_distributions.png\"), dpi=150)\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "id": "eda_correlation",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Correlation heatmap\n",
    "numerical_cols = ['amount', 'transaction_hour', 'foreign_transaction', \n",
    "                    'location_mismatch', 'device_trust_score', 'velocity_last_24h', 'cardholder_age']\n",
    "plt.figure(figsize=(10, 8))\n",
    "sns.heatmap(X_train[numerical_cols].corr(), annot=True, cmap='coolwarm', fmt=\".2f\", linewidths=0.5)\n",
    "plt.title('Feature Correlation Heatmap (Training Split)')\n",
    "plt.tight_layout()\n",
    "plt.savefig(os.path.join(PLOT_DIR, \"features_correlation_heatmap.png\"), dpi=150)\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "capper_desc",
   "metadata": {},
   "source": [
    "## 3. Custom Outlier Capper Transformer\n",
    "\n",
    "To handle extreme outliers robustly without dropping records (which is important for distance-based/anomaly-detection models), we write a custom Scikit-Learn transformer to clip values outside the 1st and 99th percentiles. The boundaries are calculated strictly on the training partition and applied uniformly to validation/test splits."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "id": "capper_implementation",
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.base import BaseEstimator, TransformerMixin\n",
    "\n",
    "class OutlierCapper(BaseEstimator, TransformerMixin):\n",
    "    \"\"\"\n",
    "    Custom transformer to cap extreme outliers using percentiles (1st and 99th)\n",
    "    fitted strictly on the training partition.\n",
    "    \"\"\"\n",
    "    def __init__(self, lower_quantile=0.01, upper_quantile=0.99):\n",
    "        self.lower_quantile = lower_quantile\n",
    "        self.upper_quantile = upper_quantile\n",
    "        self.lower_bounds_ = {}\n",
    "        self.upper_bounds_ = {}\n",
    "        \n",
    "    def fit(self, X, y=None):\n",
    "        X_df = pd.DataFrame(X)\n",
    "        for col in X_df.columns:\n",
    "            self.lower_bounds_[col] = X_df[col].quantile(self.lower_quantile)\n",
    "            self.upper_bounds_[col] = X_df[col].quantile(self.upper_quantile)\n",
    "        return self\n",
    "        \n",
    "    def transform(self, X):\n",
    "        X_df = pd.DataFrame(X).copy()\n",
    "        for col in X_df.columns:\n",
    "            X_df[col] = X_df[col].clip(lower=self.lower_bounds_[col], upper=self.upper_bounds_[col])\n",
    "        return X_df.values"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "pipeline_desc",
   "metadata": {},
   "source": [
    "## 4. Pipeline Construction & Fitting\n",
    "\n",
    "We build two variants of preprocessing pipelines:\n",
    "1. **Standard Scaled Pipeline**: Scaled to zero mean and unit variance. Suitable for Isolation Forest, One-Class SVM, and LOF.\n",
    "2. **Min-Max Scaled Pipeline**: Scaled between 0 and 1. Suitable for distance-sensitive neural network autoencoders."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "id": "pipeline_execution",
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.compose import ColumnTransformer\n",
    "from sklearn.pipeline import Pipeline\n",
    "from sklearn.preprocessing import OneHotEncoder\n",
    "from sklearn.impute import SimpleImputer\n",
    "\n",
    "# Define feature groups\n",
    "num_features = ['amount', 'transaction_hour', 'device_trust_score', 'velocity_last_24h', 'cardholder_age']\n",
    "bin_features = ['foreign_transaction', 'location_mismatch']\n",
    "cat_features = ['merchant_category']\n",
    "\n",
    "# 1. Pipeline for Standard Scaling\n",
    "num_pipeline_std = Pipeline([\n",
    "    ('imputer', SimpleImputer(strategy='median')),\n",
    "    ('capper', OutlierCapper(lower_quantile=0.01, upper_quantile=0.99)),\n",
    "    ('scaler', StandardScaler())\n",
    "])\n",
    "\n",
    "preprocessor_std = ColumnTransformer([\n",
    "    ('num', num_pipeline_std, num_features),\n",
    "    ('bin', SimpleImputer(strategy='most_frequent'), bin_features),\n",
    "    ('cat', Pipeline([\n",
    "        ('imputer', SimpleImputer(strategy='most_frequent')),\n",
    "        ('ohe', OneHotEncoder(drop='first', sparse_output=False))\n",
    "    ]), cat_features)\n",
    "])\n",
    "\n",
    "# 2. Pipeline for Min-Max Scaling\n",
    "num_pipeline_mm = Pipeline([\n",
    "    ('imputer', SimpleImputer(strategy='median')),\n",
    "    ('capper', OutlierCapper(lower_quantile=0.01, upper_quantile=0.99)),\n",
    "    ('scaler', MinMaxScaler())\n",
    "])\n",
    "\n",
    "preprocessor_mm = ColumnTransformer([\n",
    "    ('num', num_pipeline_mm, num_features),\n",
    "    ('bin', SimpleImputer(strategy='most_frequent'), bin_features),\n",
    "    ('cat', Pipeline([\n",
    "        ('imputer', SimpleImputer(strategy='most_frequent')),\n",
    "        ('ohe', OneHotEncoder(drop='first', sparse_output=False))\n",
    "    ]), cat_features)\n",
    "])\n",
    "\n",
    "# Fit preprocessors strictly on training split\n",
    "preprocessor_std.fit(X_train)\n",
    "preprocessor_mm.fit(X_train)\n",
    "\n",
    "print(\"Preprocessing pipelines fitted successfully.\")"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "transform_desc",
   "metadata": {},
   "source": [
    "## 5. Splitting Transformation"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "id": "transform_execution",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Extract transformed feature names\n",
    "cat_encoder = preprocessor_std.named_transformers_['cat'].named_steps['ohe']\n",
    "cat_cols_transformed = cat_encoder.get_feature_names_out(cat_features).tolist()\n",
    "feature_names = num_features + bin_features + cat_cols_transformed\n",
    "\n",
    "print(f\"Total features after transformation: {len(feature_names)}\")\n",
    "print(\"Feature list:\", feature_names)\n",
    "\n",
    "# Transform splits\n",
    "X_train_std = preprocessor_std.transform(X_train)\n",
    "X_val_std = preprocessor_std.transform(X_val)\n",
    "X_test_std = preprocessor_std.transform(X_test)\n",
    "\n",
    "X_train_mm = preprocessor_mm.transform(X_train)\n",
    "X_val_mm = preprocessor_mm.transform(X_val)\n",
    "X_test_mm = preprocessor_mm.transform(X_test)\n",
    "\n",
    "print(f\"\\nStandard Scaled Train shape: {X_train_std.shape}\")\n",
    "print(f\"Standard Scaled Val shape:   {X_val_std.shape}\")\n",
    "print(f\"Standard Scaled Test shape:  {X_test_std.shape}\")"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "save_desc",
   "metadata": {},
   "source": [
    "## 6. Serialization & Target Persistence"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "id": "save_execution",
   "metadata": {},
   "outputs": [],
   "source": [
    "import joblib\n",
    "\n",
    "OUTPUT_DIR = \"outputs\"\n",
    "os.makedirs(OUTPUT_DIR, exist_ok=True)\n",
    "os.makedirs(\"models\", exist_ok=True)\n",
    "\n",
    "# Save scaled features to outputs/\n",
    "pd.DataFrame(X_train_std, columns=feature_names).to_csv(os.path.join(OUTPUT_DIR, \"train_scaled_std.csv\"), index=False)\n",
    "pd.DataFrame(X_val_std, columns=feature_names).to_csv(os.path.join(OUTPUT_DIR, \"val_scaled_std.csv\"), index=False)\n",
    "pd.DataFrame(X_test_std, columns=feature_names).to_csv(os.path.join(OUTPUT_DIR, \"test_scaled_std.csv\"), index=False)\n",
    "\n",
    "pd.DataFrame(X_train_mm, columns=feature_names).to_csv(os.path.join(OUTPUT_DIR, \"train_scaled_mm.csv\"), index=False)\n",
    "pd.DataFrame(X_val_mm, columns=feature_names).to_csv(os.path.join(OUTPUT_DIR, \"val_scaled_mm.csv\"), index=False)\n",
    "pd.DataFrame(X_test_mm, columns=feature_names).to_csv(os.path.join(OUTPUT_DIR, \"test_scaled_mm.csv\"), index=False)\n",
    "\n",
    "# Save targets to outputs/\n",
    "pd.DataFrame(y_train).to_csv(os.path.join(OUTPUT_DIR, \"y_train.csv\"), index=False)\n",
    "pd.DataFrame(y_val).to_csv(os.path.join(OUTPUT_DIR, \"y_val.csv\"), index=False)\n",
    "pd.DataFrame(y_test).to_csv(os.path.join(OUTPUT_DIR, \"y_test.csv\"), index=False)\n",
    "\n",
    "# Serialize fitted preprocessors\n",
    "preprocessors_params = {\n",
    "    'preprocessor_std': preprocessor_std,\n",
    "    'preprocessor_mm': preprocessor_mm,\n",
    "    'feature_names': feature_names\n",
    "}\n",
    "joblib.dump(preprocessors_params, \"models/preprocessor.joblib\")\n",
    "\n",
    "print(\"All preprocessed splits and pipelines serialized and exported successfully.\")"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "verify_desc",
   "metadata": {},
   "source": [
    "## 7. Pipeline Output Verification"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "id": "verify_execution",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Load standard scaled train set to verify columns and scaling distributions\n",
    "train_check = pd.read_csv(os.path.join(OUTPUT_DIR, \"train_scaled_std.csv\"))\n",
    "print(\"Loaded Standard Scaled Train shape:\", train_check.shape)\n",
    "print(\"\\nFirst 5 rows:\")\n",
    "print(train_check.head())\n",
    "print(\"\\nNumerical stats (confirming zero mean and capping bounds):\")\n",
    "print(train_check[num_features].describe())"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.12.3"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}

with open("project2_financial_fraud_detection/01_data_preprocessing.ipynb", "w") as f:
    json.dump(notebook, f, indent=1)
print("01_data_preprocessing.ipynb written successfully!")
