import json

notebook = {
 "cells": [
  {
   "cell_type": "markdown",
   "id": "intro",
   "metadata": {},
   "source": [
    "# Project 2: Fraud Detection - Step 3: Explainability & Online Learning\n",
    "\n",
    "This notebook implements explainability overlays and simulates an incremental learning streaming environment.\n",
    "\n",
    "## Blueprint Actions Covered\n",
    "* **Explainability**: Generating SHAP and LIME plots to detail exactly why specific transactions are flagged as fraudulent.\n",
    "* **Online Learning**: Simulating streaming environments using incremental `.partial_fit()` (like SGDClassifier)."
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
    "import joblib\n",
    "from sklearn.linear_model import SGDClassifier\n",
    "import warnings\n",
    "warnings.filterwarnings('ignore')"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "data_loading_title",
   "metadata": {},
   "source": [
    "## 1. Load Preprocessed Datasets & serialized best model\n",
    "\n",
    "We load the Standard Scaled features and target variables, alongside the best anomaly detection model (Local Outlier Factor) generated in Step 2."
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
    "MODELS_DIR = \"models\"\n",
    "\n",
    "# Load datasets\n",
    "X_train = pd.read_csv(os.path.join(OUTPUT_DIR, \"train_scaled_std.csv\"))\n",
    "X_val = pd.read_csv(os.path.join(OUTPUT_DIR, \"val_scaled_std.csv\"))\n",
    "X_test = pd.read_csv(os.path.join(OUTPUT_DIR, \"test_scaled_std.csv\"))\n",
    "\n",
    "y_train = pd.read_csv(os.path.join(OUTPUT_DIR, \"y_train.csv\")).values.ravel()\n",
    "y_val = pd.read_csv(os.path.join(OUTPUT_DIR, \"y_val.csv\")).values.ravel()\n",
    "y_test = pd.read_csv(os.path.join(OUTPUT_DIR, \"y_test.csv\")).values.ravel()\n",
    "\n",
    "# Load anomaly detector\n",
    "anomaly_payload = joblib.load(os.path.join(MODELS_DIR, \"anomaly_detector.joblib\"))\n",
    "best_model = anomaly_payload['model']\n",
    "best_model_name = anomaly_payload['model_name']\n",
    "feature_names = X_train.columns.tolist()\n",
    "\n",
    "print(f\"Loaded best anomaly model: {best_model_name}\")\n",
    "print(f\"Train features shape:      {X_train.shape}\")"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "wrapper_title",
   "metadata": {},
   "source": [
    "## 2. Model Explainability Probability-Mapping Wrapper\n",
    "\n",
    "Model-agnostic explainers like LIME and SHAP expect classification predictions to return probability arrays. Since Local Outlier Factor outputs decision scores (density metric), we map the negated decision function to $[0, 1]$ using a sigmoid function to represent fraud probability."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "id": "wrapper_execution",
   "metadata": {},
   "outputs": [],
   "source": [
    "def predict_proba_wrapper(X):\n",
    "    # Get anomaly scores (higher = more anomalous/fraudulent)\n",
    "    scores = -best_model.decision_function(X)\n",
    "    # Sigmoid function to map to [0, 1]\n",
    "    probs_fraud = 1 / (1 + np.exp(-scores))\n",
    "    probs_normal = 1 - probs_fraud\n",
    "    return np.column_stack([probs_normal, probs_fraud])"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "lime_title",
   "metadata": {},
   "source": [
    "## 3. LIME Explainer\n",
    "\n",
    "We use `lime` to generate a local feature attribution plot showing exactly why a specific transaction was flagged as fraudulent."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "id": "lime_execution",
   "metadata": {},
   "outputs": [],
   "source": [
    "import lime\n",
    "import lime.lime_tabular\n",
    "import matplotlib.pyplot as plt\n",
    "\n",
    "PLOT_DIR = \"plots\"\n",
    "os.makedirs(PLOT_DIR, exist_ok=True)\n",
    "\n",
    "# Setup LIME explainer\n",
    "lime_explainer = lime.lime_tabular.LimeTabularExplainer(\n",
    "    training_data=X_train.values,\n",
    "    feature_names=feature_names,\n",
    "    class_names=['Normal', 'Fraud'],\n",
    "    mode='classification',\n",
    "    random_state=42\n",
    ")\n",
    "\n",
    "# Find anomalous transactions in test split\n",
    "test_preds_raw = best_model.predict(X_test.values)\n",
    "anomaly_indices = np.where(test_preds_raw == -1)[0]\n",
    "\n",
    "if len(anomaly_indices) > 0:\n",
    "    target_idx = anomaly_indices[0]\n",
    "    sample_to_explain = X_test.iloc[target_idx]\n",
    "    \n",
    "    print(f\"Explaining transaction at test index {target_idx}...\")\n",
    "    \n",
    "    # Generate LIME explanation\n",
    "    exp = lime_explainer.explain_instance(\n",
    "        data_row=sample_to_explain.values,\n",
    "        predict_fn=predict_proba_wrapper,\n",
    "        num_features=len(feature_names)\n",
    "    )\n",
    "    \n",
    "    fig = exp.as_pyplot_figure()\n",
    "    plt.title(f\"LIME Local Explanation - Flagged Fraud (Test Index {target_idx})\")\n",
    "    plt.tight_layout()\n",
    "    plt.savefig(os.path.join(PLOT_DIR, \"03_lime_explanation.png\"), dpi=150)\n",
    "    plt.show()\n",
    "else:\n",
    "    print(\"No test set anomalies found to explain.\")"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "shap_title",
   "metadata": {},
   "source": [
    "## 4. SHAP Explainer\n",
    "\n",
    "We use SHAP KernelExplainer to analyze global feature importances across a subset of 30 test transactions."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "id": "shap_execution",
   "metadata": {},
   "outputs": [],
   "source": [
    "import shap\n",
    "\n",
    "print(\"Initializing SHAP KernelExplainer...\")\n",
    "# Use 50 training rows as background reference to speed up calculation\n",
    "background_data = X_train.values[:50]\n",
    "shap_explainer = shap.KernelExplainer(predict_proba_wrapper, background_data)\n",
    "\n",
    "# Create subset of test data (20 normal, 10 fraud)\n",
    "test_sample_indices = np.concatenate([\n",
    "    np.where(y_test == 0)[0][:20],\n",
    "    np.where(y_test == 1)[0][:10]\n",
    "])\n",
    "X_shap_eval = X_test.iloc[test_sample_indices].values\n",
    "\n",
    "print(f\"Computing SHAP values for {len(X_shap_eval)} observations...\")\n",
    "shap_values = shap_explainer.shap_values(X_shap_eval)\n",
    "\n",
    "# Extract SHAP values for positive fraud class\n",
    "if isinstance(shap_values, list):\n",
    "    shap_vals_fraud = shap_values[1]\n",
    "elif len(shap_values.shape) == 3:\n",
    "    shap_vals_fraud = shap_values[:, :, 1]\n",
    "else:\n",
    "    shap_vals_fraud = shap_values\n",
    "\n",
    "plt.figure(figsize=(10, 6))\n",
    "shap.summary_plot(shap_vals_fraud, X_shap_eval, feature_names=feature_names, show=False)\n",
    "plt.title(\"SHAP Global Feature Importances (Fraud Probability)\", fontsize=16)\n",
    "plt.tight_layout()\n",
    "plt.savefig(os.path.join(PLOT_DIR, \"03_shap_summary.png\"), dpi=150)\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "streaming_title",
   "metadata": {},
   "source": [
    "## 5. Simulated Online Learning Stream\n",
    "\n",
    "We simulate a streaming data environment by training an `SGDClassifier` incrementally in sequential batches of size **256**, mimicking real-time card transactions. We evaluate the F1-Score, Recall, and Accuracy of the model on the validation split as it updates."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "id": "streaming_execution",
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.metrics import accuracy_score, f1_score, recall_score\n",
    "\n",
    "print(\"Initializing SGDClassifier for incremental streaming simulation...\")\n",
    "sgd_clf = SGDClassifier(loss='log_loss', random_state=42, learning_rate='constant', eta0=0.01)\n",
    "\n",
    "batch_size = 256\n",
    "n_samples = X_train.shape[0]\n",
    "\n",
    "batches = []\n",
    "accuracies = []\n",
    "recalls = []\n",
    "f1_scores = []\n",
    "\n",
    "batch_count = 0\n",
    "for start_idx in range(0, n_samples, batch_size):\n",
    "    end_idx = min(start_idx + batch_size, n_samples)\n",
    "    X_batch = X_train.iloc[start_idx:end_idx].values\n",
    "    y_batch = y_train[start_idx:end_idx]\n",
    "    \n",
    "    # Incrementally train model parameters on current streaming batch\n",
    "    sgd_clf.partial_fit(X_batch, y_batch, classes=np.array([0, 1]))\n",
    "    \n",
    "    # Evaluate updated model on validation split\n",
    "    y_val_pred = sgd_clf.predict(X_val.values)\n",
    "    val_acc = accuracy_score(y_val, y_val_pred)\n",
    "    val_rec = recall_score(y_val, y_val_pred, zero_division=0)\n",
    "    val_f1 = f1_score(y_val, y_val_pred, zero_division=0)\n",
    "    \n",
    "    batch_count += 1\n",
    "    batches.append(batch_count)\n",
    "    accuracies.append(val_acc)\n",
    "    recalls.append(val_rec)\n",
    "    f1_scores.append(val_f1)\n",
    "    \n",
    "    if batch_count % 5 == 0 or end_idx == n_samples:\n",
    "        print(f\"Batch {batch_count:02d} processed: Samples seen = {end_idx:5d} | \"\n",
    "              f\"Val Accuracy = {val_acc:.4f} | Val F1 = {val_f1:.4f} | Val Recall = {val_rec:.4f}\")\n",
    "\n",
    "print(\"\\nOnline Learning Simulation completed successfully.\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "id": "streaming_plotting",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Plot online learning curves\n",
    "plt.figure(figsize=(12, 6))\n",
    "plt.plot(batches, accuracies, marker='o', label='Accuracy', color='seagreen', lw=2)\n",
    "plt.plot(batches, recalls, marker='s', label='Recall (Fraud)', color='darkorange', lw=2)\n",
    "plt.plot(batches, f1_scores, marker='^', label='F1-Score (Fraud)', color='crimson', lw=2)\n",
    "plt.title('SGDClassifier Online Learning Simulation Curves')\n",
    "plt.xlabel('Batch Number (256 Transactions / Batch)')\n",
    "plt.ylabel('Metric Score')\n",
    "plt.ylim(-0.05, 1.05)\n",
    "plt.xticks(batches)\n",
    "plt.legend(loc='lower right')\n",
    "plt.grid(True, linestyle='--', alpha=0.6)\n",
    "plt.tight_layout()\n",
    "plt.savefig(os.path.join(PLOT_DIR, \"03_streaming_learning_curve.png\"), dpi=150)\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "save_title",
   "metadata": {},
   "source": [
    "## 6. Model Serialization"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "id": "save_execution",
   "metadata": {},
   "outputs": [],
   "source": [
    "joblib.dump(sgd_clf, os.path.join(MODELS_DIR, \"online_sgd_clf.joblib\"))\n",
    "print(\"Online SGD Classifier successfully saved to models/online_sgd_clf.joblib\")"
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

with open("project2_financial_fraud_detection/03_explainability_streaming_learning.ipynb", "w") as f:
    json.dump(notebook, f, indent=1)
print("03_explainability_streaming_learning.ipynb written successfully!")
