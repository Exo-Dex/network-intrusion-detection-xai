# Network Intrusion Detection System with Explainable AI (NIDS-XAI)

An end-to-end Network Intrusion Detection System evaluated on the benchmark **NSL-KDD** dataset, integrating Machine Learning, Deep Learning (MLP & Autoencoder), and Explainable AI (SHAP & LIME) served via an interactive **Streamlit** dashboard.

Developed under the **IEEE Computer Society Bangalore Chapter — Student Internship and Mentorship Program (SIMP 2026)**.

---

## 📌 Project Overview
- **Dataset**: NSL-KDD (125,973 train records, 22,544 test records across 41 raw traffic features).
- **Classification Tasks**:
  - **Binary**: Normal vs. Attack (Intrusion Detection).
  - **Multiclass**: Normal, DoS, Probe, R2L, U2R (5-Class Threat Categorization).
  - **Unsupervised Anomaly Detection**: Deep Symmetric Autoencoder trained on Normal traffic.
- **Explainable AI**:
  - **SHAP (TreeExplainer)**: Global feature importance (beeswarm, mean |SHAP|) and local waterfall attributions.
  - **LIME (LimeTabularExplainer)**: Local linear surrogate models for instance-level rule attribution.
- **Demonstration**: Streamlit analytics dashboard with Dark/Light mode toggle, dynamic prediction engine, and XAI Deep Dive.

---

## 📁 Repository Structure


---

## 🚀 Key Results Summary
- **Binary Classification (KDDTest+)**:
  - **MLP (Deep Net)**: **79.28% Accuracy, 79.08% F1-Score** (92.97% Precision, 68.81% Recall).
  - **Decision Tree**: 79.15% Accuracy, 78.21% F1-Score (96.56% Precision).
  - **XGBoost**: 78.79% Accuracy, 77.69% F1-Score (96.81% Precision).
  - **Random Forest**: 76.48% Accuracy, 74.62% F1-Score (96.70% Precision).
  - **Logistic Regression**: 75.35% Accuracy, 74.21% F1-Score.
  - **Autoencoder**: 74.33% Accuracy, 72.94% F1-Score (91.20% Precision).
- **Multiclass Threat Categorization**:
  - **XGBoost**: **77.70% Accuracy, 74.44% Weighted F1-Score** (Best Multiclass Performer).
  - **Decision Tree**: 76.85% Accuracy, 73.31% Weighted F1-Score.
  - **Logistic Regression**: 76.38% Accuracy, 72.27% Weighted F1-Score.
  - **MLP**: 76.12% Accuracy, 72.25% Weighted F1-Score.
  - **Random Forest**: 75.12% Accuracy, 70.47% Weighted F1-Score.
- **Top Discriminative Features**: , , , , .

---

## 👥 Authors
- **Piyush M. Borkar** (Team Lead) — Department of Artificial Intelligence & Data Science, MMIT Pune.
- **Varun Gada** — Department of Computer Engineering, MMIT Pune.
- **Mentor**: Boppuru Rudra Prathap (M26), MSR University.
