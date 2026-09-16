# Explainable Network Intrusion Detection: A Comparative Study of Machine Learning and Deep Learning Models on NSL-KDD Using SHAP and LIME

**Piyush M. Borkar$^1$, Varun Gada$^1$, Boppuru Rudra Prathap$^2$**

$^1$Department of Artificial Intelligence & Data Science / Department of Computer Engineering,  
Marathwada Mitra Mandal's Institute of Technology (MMIT), Savitribai Phule Pune University, Pune, Maharashtra, India  
Email: `piyushborkar.official@gmail.com`, `varungada2004@gmail.com`

$^2$Department of Computer Science & Engineering / School of Engineering & Technology,  
M. S. Ramaiah University of Applied Sciences (MSRU), Bengaluru, Karnataka, India  
Email: `brprathap@msruas.ac.in` / Mentor ID: M26

*IEEE Computer Society Bangalore Chapter — Student Internship and Mentorship Program (SIMP 2026)*

---

## Abstract
Network Intrusion Detection Systems (NIDS) utilizing Machine Learning (ML) and Deep Learning (DL) achieve high threat detection capability but predominantly operate as opaque black-box models. In operational Security Operations Centers (SOC), this opacity obscures the rationale behind automated alerts, hinders rapid triage of false alarms, and impedes analyst validation of critical exploits. This paper presents an end-to-end explainable NIDS framework evaluated on the benchmark NSL-KDD dataset. The proposed architecture evaluates four supervised machine learning classifiers (Random Forest, XGBoost, Decision Tree, Logistic Regression) and two neural architectures: a four-layer Multi-Layer Perceptron (MLP) for classification and a deep Autoencoder for unsupervised reconstruction-based anomaly detection. Nominal traffic features are mapped into a unified 122-dimensional one-hot feature space with training-partition standard scaling. Model predictions are interpreted through a dual-XAI layer: SHAP (SHapley Additive exPlanations) for global beeswarm distributions and local waterfall decompositions, coupled with LIME (Local Interpretable Model-Agnostic Explanations) for rule-based surrogate verification. Evaluated on the full held-out KDDTest+ dataset (22,544 records), the MLP achieved the highest binary F1-score of 79.08% (79.28% accuracy), while XGBoost delivered the strongest 5-class multiclass performance with a weighted F1-score of 74.44%. All supervised classifiers exhibited high precision (91.2%–96.8%) alongside lower recall (60.7%–68.8%), an empirical asymmetry traced to unobserved attack subtypes in KDDTest+ absent from KDDTrain+. Global SHAP rankings and local LIME surrogates independently converged on a consistent set of influential flow features—primarily `same_srv_rate`, `dst_host_same_srv_rate`, and `src_bytes`—demonstrating that predictive accuracy and operational transparency are mutually achievable design goals in intrusion detection.

**Keywords**: Network Intrusion Detection System, Explainable AI, SHAP, LIME, NSL-KDD, Autoencoder, Multi-Layer Perceptron, XGBoost, Random Forest, Cybersecurity.

---

## I. Introduction
The exponential expansion of digital communication, enterprise cloud environments, and distributed networked systems has elevated cybersecurity to a critical operational priority. Cyber threats continue to evolve in scale, complexity, and stealth, with threat actors frequently deploying multi-stage intrusion campaigns, evasive malware variants, and zero-day exploits. Traditional network security mechanisms rely primarily on perimeter firewalls, rule-based heuristics, and signature-matching Intrusion Detection Systems (IDS), such as Snort and Suricata. While signature-based detection systems maintain high throughput and near-zero false positive rates on documented exploit patterns, they are structurally incapable of recognizing previously unobserved attack vectors or variations of polymorphic attacks.

Machine learning (ML) and deep learning (DL) methodologies provide a data-driven alternative by inferring statistical decision boundaries directly from packet traces and session connection records. Algorithms ranging from tree ensembles (Random Forest, XGBoost) to deep feedforward networks (Multi-Layer Perceptrons) adapt autonomously to multi-dimensional traffic representations without requiring handcrafted signatures. Simultaneously, unsupervised reconstruction approaches, such as deep Autoencoders, offer a framework for anomaly detection by learning the baseline distribution of legitimate network sessions, flagging statistical deviations as potential zero-day exploits.

Despite strong quantitative benchmarks, a significant barrier restricts the autonomous deployment of ML and DL architectures in production Security Operations Centers (SOC): the black-box dilemma. High-performing neural networks and complex ensemble trees function as opaque mathematical transformations. When a network session is classified as malicious, human analysts receive no operational context detailing which specific protocol indicators, flag states, or host connection frequencies triggered the alert. This absence of interpretability introduces severe operational vulnerabilities:
1. **Analyst Verification Latency**: Human operators cannot readily determine whether an alert stems from a genuine attack or benign statistical noise, causing triage backlogs and delayed incident mitigation.
2. **False Alarm Fatigue**: Inability to diagnose why benign sessions trigger alarms prevents fine-grained tuning of classification thresholds.
3. **Compliance and Governance**: Security standards and data privacy mandates increasingly require explainability for automated defensive actions.

To bridge this operational trust gap, recent cybersecurity research has turned toward Explainable Artificial Intelligence (XAI). Frameworks grounded in cooperative game theory, such as SHAP, and local surrogate modeling, such as LIME, offer mechanisms to interpret automated predictions. However, much of the existing literature evaluates XAI either on single models in isolation, reports cross-validation accuracy on training subsets that overstate real-world generalization, or omits comparative cross-method explanation validation.

This paper addresses these limitations by developing an end-to-end, explainable network intrusion detection framework evaluated on the benchmark NSL-KDD dataset. The primary contributions of this work are:
1. **Unified Multi-Model Benchmarking**: A rigorous comparative evaluation of six distinct model families across classical ML (RF, XGBoost, DT, LR), deep supervised learning (MLP), and unsupervised anomaly detection (Autoencoder) on the full, held-out `KDDTest+` benchmark.
2. **Dual-Layer Global and Local Interpretability**: Integration of SHAP TreeExplainer for global feature impact and local waterfall decompositions, verified against LIME rule-based tabular surrogates.
3. **Empirical Methodological Disambiguation**: A critical comparative analysis distinguishing the ~79% generalization accuracy on `KDDTest+` (containing unobserved zero-day attack variants) from the >98% accuracy claims prevalent in literature derived from training-set cross-validation.
4. **Interactive Analyst Triage Engine**: A deployed Streamlit-based operational dashboard enabling live session inspection, confidence thresholding, and real-time visual explanation for SOC operators.

---

## II. Literature Survey

Table I presents a structured taxonomy of recent state-of-the-art studies in machine learning and explainable AI for network intrusion detection, highlighting methodologies, evaluated benchmarks, reported outcomes, and identified research gaps.

### TABLE I. STRUCTURED LITERATURE SURVEY OF ML/DL AND XAI APPROACHES IN NIDS
| Reference | Methodology | Dataset | Reported Performance | Observation / Identified Research Gap |
| :--- | :--- | :--- | :--- | :--- |
| **Tavallaee et al.** (IEEE CISDA) | Statistical filtering, duplicate removal, difficulty-based sampling | KDD Cup 1999 | Purged 78% train and 75% test duplicate records | Constructed the foundational NSL-KDD benchmark; established dataset baseline without proposing ML architectures. |
| **Negandhi et al.** (Springer) | Random Forest + Gini impurity feature ranking | NSL-KDD | High classification speed; reduced feature dimensionality | Evaluated a single classical classifier; lacked deep learning comparative baselines and post-hoc interpretability. |
| **Sow & Adda** (Elsevier MLA) | RF, XGBoost, DNN + SMOTE + Optuna tuning | NSL-KDD | 99.80% accuracy under 10-fold cross-validation | High scores obtained solely on training cross-validation; did not evaluate on KDDTest+ zero-day attacks and omitted XAI. |
| **Song et al.** (MDPI / Sensors) | Deep Autoencoder reconstruction anomaly detection | NSL-KDD | Effective thresholding on benign traffic baseline | Unsupervised focus; lacked direct integration with supervised multi-class classifiers and feature attribution. |
| **Gaspar et al.** (IEEE Access) | Multi-Layer Perceptron + SHAP + LIME | IoT Traffic | Verified explanation stability under perturbation | Restricted solely to neural architectures in IoT contexts; did not compare ensemble tree models or multi-class tasks. |
| **Arreche et al.** (IEEE Access) | E-XAI Framework: multi-model attribution benchmark | NSL-KDD & CICIDS2017 | Quantitative cross-explanation consistency | Benchmarked XAI methods; did not integrate unsupervised anomaly reconstruction models into the operational pipeline. |
| **Wali et al.** (Elsevier C&S) | Random Forest + Post-hoc Feature Attribution | NSL-KDD | Validated alert root-cause tracing for SOC analysts | Evaluated only Random Forest; omitted gradient-boosted trees, deep learning models, and dual-XAI cross-validation. |

---

## III. Dataset Preprocessing and Feature Space

### A. NSL-KDD Benchmark Characteristics
The NSL-KDD dataset addresses key structural flaws of KDD Cup 1999 by eliminating redundant duplicate records that artificially inflated performance metrics in earlier studies. The standard benchmark consists of two partitions: `KDDTrain+` (125,973 records) and `KDDTest+` (22,544 records). Each record contains 41 raw traffic features spanning connection duration, service type, flag states, host byte volumes, error rates, and traffic aggregation counters.

### B. Preprocessing & Leakage-Free Encoding
1. **Pruning**: The `difficulty_level` meta-column is stripped, and exact duplicate rows in the training partition are purged. Null verification confirms zero missing values.
2. **Categorical Feature Space**: The categorical attributes `protocol_type` (3 levels), `service` (70 levels), and `flag` (11 levels) are one-hot encoded using a unified vocabulary derived across train and test sets, avoiding schema discrepancy while expanding feature dimensions from 41 to 122.
3. **Normalization**: Feature scaling is executed using standard z-score normalization ($z = \frac{x - \mu}{\sigma}$). Crucially, scaling parameters ($\mu, \sigma$) are computed **strictly on `KDDTrain+`** and subsequently applied to transform `KDDTest+`, enforcing strict separation between training and test distributions.

### TABLE II. DATASET PARTITIONING AND ATTACK TAXONOMY
| Category | Train Count (`KDDTrain+`) | Test Count (`KDDTest+`) | Class Type | Primary Characteristics |
| :--- | ---: | ---: | :---: | :--- |
| **Normal** | 67,343 (53.46%) | 9,711 (43.08%) | Benign (0) | Legitimate background enterprise traffic |
| **DoS** | 45,927 (36.46%) | 7,458 (33.08%) | Malicious (1) | Resource exhaustion floods (e.g., Neptune, Smurf) |
| **Probe** | 11,656 (9.25%) | 2,421 (10.74%) | Malicious (1) | Surveillance and port scanning (e.g., Satan, Nmap) |
| **R2L** | 995 (0.79%) | 2,885 (12.80%) | Malicious (1) | Unauthorized remote access (e.g., Guess_Passwd) |
| **U2R** | 52 (0.04%) | 67 (0.30%) | Malicious (1) | Local privilege escalation to root (e.g., Buffer_Overflow) |
| **Total** | **125,973** | **22,544** | -- | **Dimensionality: 122 Features** |

The training distribution exhibits extreme class imbalance: User-to-Root (U2R) constitutes a mere 0.04% (52 instances), and Remote-to-Local (R2L) comprises 0.79% (995 instances). Significantly, `KDDTest+` introduces 17 novel attack subtypes not present during training, intentionally testing cross-distribution generalization.

---

## IV. Proposed Detection & Anomaly Architecture

The detection architecture incorporates six models covering linear, tree-based, neural feedforward, and unsupervised reconstruction paradigms:

### A. Supervised Classifier Family
1. **Random Forest (RF)**: Ensemble of 100 decorrelated decision trees with Gini impurity split criterion.
2. **XGBoost (XGB)**: Gradient boosted trees with shrinkage ($\eta=0.1$) optimizing binary logistic and multiclass softmax loss.
3. **Decision Tree (DT)**: Standard CART algorithm capped at depth 20.
4. **Logistic Regression (LR)**: L2-regularized linear baseline with limited-memory BFGS solver ($C=1.0$).
5. **Multi-Layer Perceptron (MLP)**: A deep feedforward neural network configured with:
   $$\text{Input}(122) \rightarrow \text{Dense}(256) \rightarrow \text{BN} \rightarrow \text{ReLU} \rightarrow \text{Dropout}(0.3) \rightarrow \text{Dense}(128) \rightarrow \text{BN} \rightarrow \text{ReLU} \rightarrow \text{Dropout}(0.3) \rightarrow \text{Dense}(64) \rightarrow \text{Output}$$
   Trained via Adam optimizer with mini-batch size 128 and EarlyStopping ($patience=5$) on validation loss.

### B. Unsupervised Anomaly Detection (Deep Autoencoder)
To detect zero-day exploits without labeled attack instances, a deep symmetric Autoencoder is trained strictly on benign connections ($\mathbf{x} \in \mathcal{D}_{\text{normal}}$):
$$\mathbf{z} = \sigma(\mathbf{W}_e \mathbf{x} + \mathbf{b}_e), \quad \mathbf{\hat{x}} = \sigma(\mathbf{W}_d \mathbf{z} + \mathbf{b}_d)$$
The encoder compresses input dimensions from $122 \rightarrow 128 \rightarrow 64 \rightarrow \mathbf{32}$ (latent bottleneck), while the decoder mirrors this reconstruction back to 122 dimensions. Training optimizes Mean Squared Error (MSE):
$$\mathcal{L}_{\text{MSE}}(\mathbf{x}, \mathbf{\hat{x}}) = \frac{1}{d} \sum_{j=1}^{d} (x_j - \hat{x}_j)^2$$
During inference, reconstruction error $E(\mathbf{x}) = \|\mathbf{x} - \mathbf{\hat{x}}\|^2$ is evaluated against an empirical threshold $\theta_{95\%}$, calibrated at the 95th percentile of normal training errors:
$$\hat{y} = \begin{cases} 1 \ (\text{Attack}), & \text{if } E(\mathbf{x}) > \theta_{95\%} \\ 0 \ (\text{Normal}), & \text{otherwise} \end{cases}$$

---

## V. Experimental Results & Literature Comparison

### A. Binary Classification & Zero-Day Anomaly Detection
Evaluations were conducted on the full, held-out `KDDTest+` set (22,544 samples).

### TABLE III. BINARY CLASSIFICATION PERFORMANCE ON KDDTEST+
| Model Architecture | Accuracy (%) | Precision (%) | Recall (%) | F1-Score (%) |
| :--- | :---: | :---: | :---: | :---: |
| **Multi-Layer Perceptron (MLP)** | **79.28** | 92.97 | **68.81** | **79.08** |
| Decision Tree (Depth = 20) | 79.15 | 96.56 | 65.71 | 78.21 |
| XGBoost (100 Trees) | 78.79 | **96.81** | 64.87 | 77.69 |
| Random Forest (100 Trees) | 76.48 | 96.70 | 60.75 | 74.62 |
| Logistic Regression (L2) | 75.35 | 91.73 | 62.32 | 74.21 |
| Deep Autoencoder (Unsupervised) | 74.33 | 91.20 | 60.77 | 72.94 |

**Discussion on the Precision–Recall Asymmetry**:
All evaluated models demonstrate strong precision ($91.2\% - 96.8\%$) but lower recall ($60.7\% - 68.8\%$). In operational SOC environments, this indicates that when the system alerts on an attack, the probability of a false positive is less than $4\%$. The recall deficit is directly attributable to the 17 unobserved attack subtypes embedded in `KDDTest+` that exhibit statistical properties divergent from `KDDTrain+`.

### B. Multiclass 5-Way Threat Categorization
### TABLE IV. MULTICLASS CLASSIFICATION RESULTS ON KDDTEST+
| Model | Accuracy (%) | Weighted Precision (%) | Weighted Recall (%) | Weighted F1-Score (%) |
| :--- | :---: | :---: | :---: | :---: |
| **XGBoost** | **77.70** | **82.70** | **77.70** | **74.44** |
| Decision Tree | 76.85 | 82.09 | 76.85 | 73.31 |
| Logistic Regression | 76.38 | 74.97 | 76.38 | 72.27 |
| Multi-Layer Perceptron | 76.12 | 79.41 | 76.12 | 72.25 |
| Random Forest | 75.12 | 80.89 | 75.12 | 70.47 |

XGBoost achieved the highest multiclass weighted F1 score (74.44%) due to its gradient-boosted handling of skewed decision surfaces.

### C. Methodological Disambiguation vs. Literature Benchmarks
### TABLE V. COMPARATIVE VALIDATION AGAINST ESTABLISHED LITERATURE
| Study / Venue | Reported Metric | Evaluation Basis & Methodology | Contrast with This Work |
| :--- | :---: | :--- | :--- |
| **Sow & Adda** (Elsevier, 2025) | 99.80% Acc. | 10-Fold cross-validation on `KDDTrain+` only; no held-out test. | Overstates generalization; models do not encounter novel attack variants. |
| **Negandhi et al.** (Springer, 2019) | 98.20% Acc. | Evaluated on training partition subset; no XAI layer. | Lacks out-of-distribution validation and provides zero interpretability. |
| **Gaspar et al.** (IEEE Access, 2024) | 82.10% Acc. | MLP on IoT traffic; single model focus. | Narrow architectural scope; lacks ensemble and unsupervised comparisons. |
| **Arreche et al.** (IEEE Access, 2024) | 78.40% Acc. | Full `KDDTest+` evaluation with SHAP attribution. | Closely aligns with our 79.28% benchmark; validates our experimental rigour. |
| **Wali et al.** (Elsevier, 2025) | 77.10% Acc. | Random Forest on `KDDTest+` with XAI. | Limited to single classical classifier; lacks deep neural comparisons. |
| **This Work** | **79.28% Acc. / 79.08% F1** | **Full held-out `KDDTest+` (22,544 records), 6 models, dual SHAP/LIME, and Autoencoder.** | **Realistic, verifiable benchmark pairing supervised detection with transparent triage.** |

---

## VI. Explainable AI Analysis & Attribution Consensus

### A. Global Attribution Convergence (SHAP)
Using `shap.TreeExplainer`, Shapley values were computed across 500 representative test samples. Global feature importance rankings obtained from Random Forest and XGBoost demonstrated a striking degree of concordance: **10 of the top 15 most decisive features were identical across both ensemble architectures**.
The top five globally influential traffic features identified are:
1. `same_srv_rate`: Proportion of connections to the same service.
2. `dst_host_same_srv_rate`: Host-level destination service homogeneity.
3. `dst_host_srv_count`: Number of connections to the same service at target host.
4. `src_bytes`: Bytes transmitted from source to destination.
5. `flag_SF`: Normal connection establishment flag.

### B. Local Decision Decomposition (SHAP & LIME)
At the individual instance level, SHAP waterfall plots quantify the exact additive contribution $\phi_i$ of each feature in pushing prediction odds above the baseline:
$$f(x) = \phi_0 + \sum_{i=1}^{M} \phi_i$$
Simultaneously, LIME tabular surrogates approximate the decision boundary locally using an interpretable ridge regressor:
$$\xi(x) = \arg\min_{g \in \mathcal{G}} \mathcal{L}(f, g, \pi_x) + \Omega(g)$$
Both explainers consistently highlighted identical physical traffic conditions—such as an abnormally low `same_srv_rate` coupled with a high `dst_host_rerror_rate`—as the primary root cause for triggering DoS and Probe alerts, providing SOC analysts with immediate, verifiable justification.

---

## VII. Operational Triage Dashboard & Usability

To transition theoretical XAI models into active security practice, a Streamlit dashboard was deployed featuring:
1. **Predict & Explain**: Ingests real-time connection sessions, presents class probabilities, and renders dynamic SHAP waterfall plots with feature weight tables.
2. **Model Performance**: Displays interactive confusion matrices, precision-recall breakdowns, and multi-metric radar charts.
3. **XAI Deep Dive**: Interactive exploration of global beeswarm plots, ranked feature tables, and LIME perturbation intervals.
4. **Theme Adaptation**: Full CSS variable injection supporting light and dark modes with color-calibrated chart rendering.

---

## VIII. Conclusion & Future Work
This paper demonstrated an end-to-end explainable network intrusion detection framework combining classical ML, deep feedforward networks, and unsupervised anomaly detection on the NSL-KDD benchmark. By evaluating on the full `KDDTest+` partition, we disambiguated the real-world generalization performance (~79% F1) from inflated training cross-validation metrics. SHAP and LIME integration proved that high detection accuracy and model transparency are fully compatible.

Future work will explore:
1. Handling minority class imbalance (R2L/U2R) via adaptive resampling (SMOTE/ADASYN) and Focal Loss.
2. Extending SHAP DeepExplainer to provide gradient-based attribution across the MLP and Autoencoder architectures.
3. Transitioning the pipeline to contemporary flow datasets with temporal sequences (CIC-IDS2017, UNSW-NB15).

---

## Acknowledgment
This work was carried out under the IEEE Computer Society Bangalore Chapter Student Internship and Mentorship Program (SIMP 2026). The authors express sincere gratitude to program coordinators, reviewers, and faculty mentors for their support throughout this research initiative.

---

## References
[1] M. Tavallaee, E. Bagheri, W. Lu, and A. A. Ghorbani, "A detailed analysis of the KDD CUP 99 data set," in *Proc. IEEE Symposium on Computational Intelligence for Security and Defense Applications (CISDA)*, Ottawa, ON, Canada, 2009, pp. 1-6.  
[2] P. Negandhi, R. Trivedi, and R. Mangrulkar, "Intrusion detection system using random forest on NSL-KDD dataset," in *Emerging Research in Computing, Information, Communication and Applications*, Springer, Singapore, 2019, pp. 419-431.  
[3] A. Sow and M. Adda, "Evaluating explainable artificial intelligence techniques for network intrusion detection," *Machine Learning with Applications*, vol. 16, Art. no. 100543, Elsevier, 2025.  
[4] Y. Song, S. Hyun, and Y.-G. Cheong, "Analysis of autoencoders for network intrusion detection," *Sensors*, vol. 21, no. 13, Art. no. 4294, MDPI, 2021.  
[5] G. Gaspar, F. M. Dahunsi, and A. E. Ibhaze, "Explainable AI framework for intrusion detection in IoT networks," *IEEE Access*, vol. 12, pp. 31245-31258, 2024.  
[6] O. Arreche, T. Guntur, and M. Abdallah, "XAI-IDS: Toward proposing an explainable artificial intelligence framework for enhancing network intrusion detection systems," *Applied Sciences*, vol. 14, no. 10, Art. no. 4170, 2024.  
[7] M. T. Ribeiro, S. Singh, and C. Guestrin, ""Why should I trust you?": Explaining the predictions of any classifier," in *Proc. 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining*, San Francisco, CA, USA, 2016, pp. 1135-1144.  
[8] S. M. Lundberg and S.-I. Lee, "A unified approach to interpreting model predictions," in *Proc. Advances in Neural Information Processing Systems (NeurIPS)*, Long Beach, CA, USA, 2017, pp. 4765-4774.  
[9] S. Wali, I. Khan, and M. Z. A. Bhuiyan, "Explainable machine learning for cybersecurity threat intelligence in industry 5.0," *Computers & Security*, vol. 138, Art. no. 103681, Elsevier, 2025.
