💳 Financial Transaction Fraud Detection

This project focuses on detecting fraudulent financial transactions using Machine Learning.
It employs the Isolation Forest algorithm to identify anomalies in credit card transaction data.
A Streamlit-based dashboard enables real-time testing, batch analysis, and interpretability using SHAP values.

🚀 Project Overview

The system is designed to identify fraudulent transactions through a complete ML pipeline — from data preprocessing to model deployment.
It offers interactive modes for both users and developers, supporting manual entry, CSV uploads, and live transaction simulations.

📊 Dataset Description

Source: Public Credit Card Fraud Detection Dataset (contains anonymized transaction data).

Features: 30 columns including Time, Amount, and PCA-transformed variables V1–V28.

Target Variable: Class (0 = Legitimate, 1 = Fraud).

Size: ~285,000 transactions with only ~0.17% fraudulent cases.

🧠 Model Architecture

The fraud detection system is a four-layer pipeline integrating data preprocessing, anomaly detection, evaluation, and deployment.

Data Preprocessing

Removed duplicates and scaled Time and Amount using StandardScaler.

Prepared clean, normalized input data.

Anomaly Detection (Isolation Forest)

Isolation Forest isolates anomalies instead of modeling normal data.

Parameters:
n_estimators=200, contamination=0.0017, random_state=42.

Evaluation

Metrics used: Precision, Recall, F1-Score, ROC-AUC.

Ensured high accuracy with minimal false positives.

Deployment (Streamlit App)

Interactive web interface with:

User Mode (manual input)

Developer Mode (CSV upload)

Real-Time Mode (stream simulation)

Explainability Mode (SHAP analysis)

⚙️ Algorithm Explanation

Isolation Forest works by randomly selecting a feature and splitting it between minimum and maximum values.

Anomalies (frauds) are easier to isolate and thus have shorter path lengths in the tree.

The model is unsupervised and efficient for highly imbalanced datasets.

It identifies outliers without requiring extensive labeled data, making it ideal for real-world fraud detection.

📈 Performance Metrics

Accuracy: High performance with robust fraud detection.

Precision & Recall: Balanced trade-off to minimize false positives.

F1-Score: Ensures reliable fraud identification performance.

ROC-AUC Score: Demonstrates model’s ability to distinguish between legitimate and fraudulent transactions.

⚡ Challenges Faced & Solutions

Data Imbalance: Very few fraud samples.
→ Used contamination parameter tuning for better anomaly detection.

Noise & Missing Values: Inconsistent records.
→ Applied preprocessing and scaling for data uniformity.

Interpretability: Difficult to explain model decisions.
→ Added SHAP-based visual explanations.

Real-Time Simulation: Needed for deployment.
→ Integrated live transaction feed with progress visualization.

🔮 Future Improvements

Incorporate deep learning models like autoencoders for enhanced detection.

Implement adaptive learning to update models with new fraud patterns.

Add real-time alert systems and transaction dashboards.

Expand dataset with additional sources like IP, geolocation, and device info.

Deploy on cloud platforms for scalability and production readiness.

🖥️ Installation & Usage
# Clone repository
git clone(https://github.com/kunal30kunal/fraud_detection)
cd fraud-detection

# Install dependencies
pip install -r requirements.txt

# Run Streamlit app
streamlit run app.py

🧩 Tech Stack

Python, Pandas, NumPy

Scikit-learn (Isolation Forest)

Streamlit (Frontend & Deployment)

SHAP (Model Explainability)

Matplotlib, Seaborn (Visualization)

Joblib (Model Serialization)

📚 Acknowledgments

Dataset inspired by the Credit Card Fraud Detection dataset available on Kaggle
.
Developed as part of a project on AI-based Financial Security Systems.


