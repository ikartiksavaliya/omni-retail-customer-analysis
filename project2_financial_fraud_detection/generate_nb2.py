import json

notebook = {
 "cells": [
  {
   "cell_type": "markdown",
   "id": "intro",
   "metadata": {},
   "source": [
    "# Project 2: Fraud Detection - Step 2: Anomaly Detection Models\n",
    "\n",
    "This notebook trains unsupervised and semi-supervised anomaly detection algorithms to identify suspicious transactions.\n",
    "\n",
    "## Blueprint Actions Covered\n",
    "* **Algorithms**: Isolation Forest, One-Class SVM, and Local Outlier Factor (LOF)."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "id": "imports",
   "metadata": {},
   "outputs": [],
   "source": [
    "import os\n",
    "import numpy as np\n",
    "import pandas as pd\n",
    "from sklearn.ensemble import IsolationForest\n",
    "from sklearn.svm import OneClassSVM\n",
    "from sklearn.neighbors import LocalOutlierFactor"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "data_loading_title",
   "metadata": {},
   "source": [
    "## 1. Load Preprocessed Datasets\n",
    "\n",
    "We load the Standard Scaled features and target variables generated in Step 1."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "id": "data_loading_execution",
   "metadata": {},
   "outputs": [],
   "source": [
    "OUTPUT_DIR = \"outputs\"\n",
    "\n",
    "X_train = pd.read_csv(os.path.join(OUTPUT_DIR, \"train_scaled_std.csv\"))\n",
    "X_val = pd.read_csv(os.path.join(OUTPUT_DIR, \"val_scaled_std.csv\"))\n",
    "X_test = pd.read_csv(os.path.join(OUTPUT_DIR, \"test_scaled_std.csv\"))\n",
    "\n",
    "y_train = pd.read_csv(os.path.join(OUTPUT_DIR, \"y_train.csv\")).values.ravel()\n",
    "y_val = pd.read_csv(os.path.join(OUTPUT_DIR, \"y_val.csv\")).values.ravel()\n",
    "y_test = pd.read_csv(os.path.join(OUTPUT_DIR, \"y_test.csv\")).values.ravel()\n",
    "\n",
    "print(f\"Train shapes:      X = {X_train.shape}, y = {y_train.shape}\")\n",
    "print(f\"Validation shapes: X = {X_val.shape}, y = {y_val.shape}\")\n",
    "print(f\"Test shapes:       X = {X_test.shape}, y = {y_test.shape}\")"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "semi_supervised_title",
   "metadata": {},
   "source": [
    "## 2. Semi-Supervised Data Filtering\n",
    "\n",
    "In financial anomaly detection, we want our models to learn the clean baseline of normal (non-fraudulent) transactions. Therefore, we fit our estimators *only* on normal transactions (`is_fraud == 0`) in the training partition, while validating and testing on the full datasets."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "id": "semi_supervised_execution",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Filter training features to contain only normal transactions\n",
    "X_train_normal = X_train[y_train == 0]\n",
    "print(f\"Filtered normal train set shape: {X_train_normal.shape} (Original train set had {X_train.shape[0]} samples)\")"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "training_title",
   "metadata": {},
   "source": [
    "## 3. Model Training\n",
    "\n",
    "We train three classical anomaly detection models:\n",
    "1. **Isolation Forest**: Fits a set of isolation trees. Anomalies isolate faster and have shorter path lengths.\n",
    "2. **One-Class SVM**: Fits a hyperplane dividing normal transaction points from the origin in RBF latent space.\n",
    "3. **Local Outlier Factor (LOF)**: Computes local density deviations. We fit in `novelty=True` mode to allow testing on validation and test data."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "id": "training_execution",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Define the expected contamination rate based on empirical training class balance\n",
    "fraud_rate = np.mean(y_train)\n",
    "print(f\"Empirical fraud rate in training partition: {fraud_rate:.4%}\")\n",
    "\n",
    "# 1. Isolation Forest\n",
    "print(\"\\nTraining Isolation Forest...\")\n",
    "iforest = IsolationForest(contamination=fraud_rate, random_state=42, n_jobs=-1)\n",
    "iforest.fit(X_train_normal)\n",
    "\n",
    "# 2. One-Class SVM\n",
    "print(\"Training One-Class SVM...\")\n",
    "ocsvm = OneClassSVM(kernel='rbf', nu=fraud_rate, gamma='scale')\n",
    "ocsvm.fit(X_train_normal)\n",
    "\n",
    "# 3. Local Outlier Factor\n",
    "print(\"Training Local Outlier Factor...\")\n",
    "lof = LocalOutlierFactor(contamination=fraud_rate, novelty=True, n_jobs=-1)\n",
    "lof.fit(X_train_normal)\n",
    "\n",
    "print(\"\\nAll models trained successfully.\")"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "evaluation_title",
   "metadata": {},
   "source": [
    "## 4. Evaluation and Mapping Logic\n",
    "\n",
    "Standard Scikit-Learn anomaly detection models predict `1` for inliers and `-1` for outliers. We define a helper function to map these to standard binary outputs (`0` = normal, `1` = fraud) and compute anomaly decision scores (negating the decision function so that larger values indicate higher fraud probability)."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "id": "evaluation_helpers",
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.metrics import classification_report, f1_score, precision_score, recall_score, roc_auc_score, average_precision_score\n",
    "\n",
    "def evaluate_anomaly_model(model, X, y_true):\n",
    "    # Predict anomaly labels: 1 = inlier, -1 = outlier\n",
    "    preds_raw = model.predict(X)\n",
    "    # Map: 1 -> 0 (normal), -1 -> 1 (fraud)\n",
    "    y_pred = np.where(preds_raw == 1, 0, 1)\n",
    "    \n",
    "    # Anomaly scores: higher score means more anomalous\n",
    "    # decision_function returns negative anomaly scores (lower is more anomalous)\n",
    "    # Therefore, we negate it so that larger values indicate higher anomaly\n",
    "    anomaly_scores = -model.decision_function(X)\n",
    "    \n",
    "    metrics = {\n",
    "        'precision': precision_score(y_true, y_pred, zero_division=0),\n",
    "        'recall': recall_score(y_true, y_pred, zero_division=0),\n",
    "        'f1': f1_score(y_true, y_pred, zero_division=0),\n",
    "        'roc_auc': roc_auc_score(y_true, anomaly_scores),\n",
    "        'pr_auc': average_precision_score(y_true, anomaly_scores),\n",
    "        'y_pred': y_pred,\n",
    "        'anomaly_scores': anomaly_scores\n",
    "    }\n",
    "    return metrics"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "leaderboard_title",
   "metadata": {},
   "source": [
    "## 5. Model Validation & Leaderboard"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "id": "leaderboard_execution",
   "metadata": {},
   "outputs": [],
   "source": [
    "models = {\n",
    "    'Isolation Forest': iforest,\n",
    "    'One-Class SVM': ocsvm,\n",
    "    'Local Outlier Factor': lof\n",
    "}\n",
    "\n",
    "results = {}\n",
    "for name, model in models.items():\n",
    "    results[name] = evaluate_anomaly_model(model, X_val, y_val)\n",
    "    \n",
    "leaderboard_df = pd.DataFrame({\n",
    "    name: {\n",
    "        'F1-Score': results[name]['f1'],\n",
    "        'Precision': results[name]['precision'],\n",
    "        'Recall': results[name]['recall'],\n",
    "        'ROC-AUC': results[name]['roc_auc'],\n",
    "        'PR-AUC': results[name]['pr_auc']\n",
    "    } for name in models\n",
    "}).T\n",
    "\n",
    "print(\"Validation Leaderboard:\")\n",
    "print(leaderboard_df.round(4))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "id": "leaderboard_plotting",
   "metadata": {},
   "outputs": [],
   "source": [
    "import matplotlib.pyplot as plt\n",
    "import seaborn as sns\n",
    "\n",
    "PLOT_DIR = \"plots\"\n",
    "os.makedirs(PLOT_DIR, exist_ok=True)\n",
    "\n",
    "sns.set_theme(style=\"whitegrid\")\n",
    "plt.rcParams.update({'font.size': 12, 'axes.labelsize': 14, 'axes.titlesize': 16})\n",
    "\n",
    "# Melt data for plotting\n",
    "plot_df = leaderboard_df.reset_index().rename(columns={'index': 'Model'})\n",
    "plot_df = pd.melt(plot_df, id_vars=['Model'], value_vars=['F1-Score', 'Recall', 'PR-AUC'], \n",
    "                    var_name='Metric', value_name='Value')\n",
    "\n",
    "plt.figure(figsize=(12, 7))\n",
    "sns.barplot(data=plot_df, x='Model', y='Value', hue='Metric', palette='Set2')\n",
    "plt.title('Anomaly Detection Model Comparison (Validation Split)')\n",
    "plt.ylabel('Score')\n",
    "plt.xlabel('Model')\n",
    "plt.ylim(0, 1.05)\n",
    "plt.legend(loc='lower right')\n",
    "plt.tight_layout()\n",
    "plt.savefig(os.path.join(PLOT_DIR, \"02_anomaly_leaderboard.png\"), dpi=150)\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "best_model_title",
   "metadata": {},
   "source": [
    "## 6. Test Set Evaluation\n",
    "\n",
    "We select the best performing model on the validation split (based on F1-Score) and evaluate it on our unseen test dataset."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "id": "best_model_execution",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Determine best model\n",
    "best_model_name = leaderboard_df['F1-Score'].idxmax()\n",
    "print(f\"Best Model Selected: {best_model_name}\")\n",
    "\n",
    "best_model = models[best_model_name]\n",
    "test_metrics = evaluate_anomaly_model(best_model, X_test, y_test)\n",
    "\n",
    "print(f\"\\nTest Set Performance for {best_model_name}:\")\n",
    "print(f\"Precision: {test_metrics['precision']:.4f}\")\n",
    "print(f\"Recall:    {test_metrics['recall']:.4f}\")\n",
    "print(f\"F1-Score:  {test_metrics['f1']:.4f}\")\n",
    "print(f\"ROC-AUC:   {test_metrics['roc_auc']:.4f}\")\n",
    "print(f\"PR-AUC:    {test_metrics['pr_auc']:.4f}\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "id": "best_model_curves",
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve, precision_recall_curve\n",
    "\n",
    "fig, axes = plt.subplots(1, 3, figsize=(20, 6))\n",
    "\n",
    "# 1. Confusion Matrix\n",
    "cm = confusion_matrix(y_test, test_metrics['y_pred'])\n",
    "disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Normal', 'Fraud'])\n",
    "disp.plot(cmap='Blues', ax=axes[0], colorbar=False)\n",
    "axes[0].set_title(f'Confusion Matrix')\n",
    "\n",
    "# 2. ROC Curve\n",
    "fpr, tpr, _ = roc_curve(y_test, test_metrics['anomaly_scores'])\n",
    "axes[1].plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {test_metrics[\"roc_auc\"]:.4f})')\n",
    "axes[1].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')\n",
    "axes[1].set_xlabel('False Positive Rate')\n",
    "axes[1].set_ylabel('True Positive Rate')\n",
    "axes[1].set_title('Receiver Operating Characteristic')\n",
    "axes[1].legend(loc=\"lower right\")\n",
    "\n",
    "# 3. Precision-Recall Curve\n",
    "precision, recall, _ = precision_recall_curve(y_test, test_metrics['anomaly_scores'])\n",
    "axes[2].plot(recall, precision, color='blue', lw=2, label=f'PR curve (area = {test_metrics[\"pr_auc\"]:.4f})')\n",
    "axes[2].set_xlabel('Recall')\n",
    "axes[2].set_ylabel('Precision')\n",
    "axes[2].set_title('Precision-Recall Curve')\n",
    "axes[2].legend(loc=\"lower left\")\n",
    "\n",
    "plt.tight_layout()\n",
    "plt.savefig(os.path.join(PLOT_DIR, \"02_best_anomaly_evaluation.png\"), dpi=150)\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "save_title",
   "metadata": {},
   "source": [
    "## 7. Model Serialization\n",
    "\n",
    "We serialize the best performing anomaly detection model to disk."
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
    "MODELS_DIR = \"models\"\n",
    "os.makedirs(MODELS_DIR, exist_ok=True)\n",
    "\n",
    "model_payload = {\n",
    "    'model_name': best_model_name,\n",
    "    'model': best_model,\n",
    "    'test_metrics': {\n",
    "        'precision': test_metrics['precision'],\n",
    "        'recall': test_metrics['recall'],\n",
    "        'f1': test_metrics['f1'],\n",
    "        'roc_auc': test_metrics['roc_auc'],\n",
    "        'pr_auc': test_metrics['pr_auc']\n",
    "    }\n",
    "}\n",
    "joblib.dump(model_payload, os.path.join(MODELS_DIR, \"anomaly_detector.joblib\"))\n",
    "print(f\"Best model ({best_model_name}) serialized to models/anomaly_detector.joblib successfully.\")"
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
   "name": "python"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}

with open("project2_financial_fraud_detection/02_anomaly_detection_models.ipynb", "w") as f:
    json.dump(notebook, f, indent=1)
print("02_anomaly_detection_models.ipynb written successfully!")
