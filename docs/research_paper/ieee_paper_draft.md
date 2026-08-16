# Network Intrusion Detection Using Machine Learning and Explainable AI on the NSL-KDD Dataset

Piyush M. Borkar, Varun Gada

Piyush M. Borkar: Department of Artificial Intelligence & Data Science, Marathwada Mitramandal's Institute of Technology, Savitribai Phule Pune University, Pune, India. Email: piyushborkar.official@gmail.com.

Varun Gada: Department of Computer Engineering, Marathwada Mitramandal's Institute of Technology, Pune, India. Email: varungada2004@gmail.com.

IEEE Computer Society Bangalore Chapter, Student Internship and Mentorship Program 2026

> Draft status: content-first IEEE conference paper draft. Formatting, exact affiliations, author emails, mentor attribution, final citations, and figure/table placement should be completed after technical review.

## Abstract

Network intrusion detection systems are essential for identifying malicious traffic patterns in modern computing environments. Although traditional signature-based systems remain useful, they are limited when attack behavior changes or when alerts must be interpreted by human analysts. This paper presents an explainable artificial intelligence driven intrusion detection framework using the NSL-KDD dataset. The proposed workflow includes data cleaning, attack-category mapping, one-hot encoding, feature scaling, exploratory analysis, binary classification, multiclass attack classification, anomaly detection, and post-hoc explanation. Random Forest, XGBoost, Decision Tree, Logistic Regression, multilayer perceptron, and autoencoder models were evaluated to compare classical machine learning, deep learning, and reconstruction-based anomaly detection approaches. SHAP and LIME were integrated to explain both global model behavior and local instance-level predictions. Experimental results show that the MLP achieved 79.28% binary accuracy and 79.08% binary F1 score, while XGBoost achieved the best multiclass performance with 77.70% accuracy and 74.44% weighted F1 score. The system is also implemented as an interactive Streamlit dashboard for prediction, model comparison, confusion-matrix inspection, SHAP analysis, and LIME-based explanation. The work demonstrates a practical NIDS-XAI pipeline in which detection accuracy is paired with interpretability and analyst-facing usability.

## Keywords

Network intrusion detection, NSL-KDD, machine learning, explainable AI, SHAP, LIME, XGBoost, Random Forest, cybersecurity.

## I. Introduction

The rapid growth of digital communication, cloud services, and interconnected systems has increased the importance of reliable cybersecurity monitoring. Network intrusion detection systems (NIDS) examine network traffic to identify suspicious or malicious activity, including denial-of-service attacks, probing, unauthorized access attempts, and privilege escalation. Conventional intrusion detection methods often depend on manually written rules or known attack signatures. Although signature-based systems can be effective against previously observed threats, they are less adaptive when attack behavior changes or when new attack variants emerge.

Machine learning provides an adaptive alternative by learning decision patterns directly from network traffic data. Instead of relying only on static signatures, supervised models can classify traffic using statistical and behavioral features extracted from connection records. However, machine learning based security systems introduce an additional challenge: many high-performing models operate as black boxes. In a cybersecurity setting, a prediction alone is often insufficient. Analysts must understand why a connection was flagged, which features influenced the model, and whether the result appears trustworthy enough for operational use.

This work addresses both detection and interpretability by building an explainable AI driven NIDS framework. The system uses the NSL-KDD dataset, a benchmark dataset derived from KDD Cup 1999 and designed to reduce redundancy and improve evaluation quality [1]. The study includes data preprocessing, exploratory analysis, binary and multiclass classification, deep learning experiments, SHAP and LIME explanations, and an interactive dashboard for demonstration. The central contribution is not limited to classification accuracy; rather, it is the integration of detection, explanation, and usable model inspection in a single workflow, as shown in Fig. 1.

**Fig. 1. Proposed NIDS-XAI workflow from NSL-KDD preprocessing to prediction, explanation, and dashboard visualization.**

Suggested figure file: `docs/research_paper/nids_xai_architecture_ieee.svg`

The main contributions of this work are as follows:

1. A complete preprocessing pipeline for NSL-KDD, including attack-category mapping, binary label creation, and consistent one-hot encoding across train and test sets.
2. A comparative evaluation of classical machine learning models and deep learning models for binary and multiclass intrusion detection.
3. Integration of SHAP and LIME to explain both global model behavior and individual predictions, making the system more suitable for analyst-facing cybersecurity use.
4. A Streamlit-based dashboard that presents predictions, confidence, model metrics, confusion matrices, SHAP visualizations, and LIME explanations in an interactive interface.

## II. Related Work

Machine learning based intrusion detection has been widely studied because network traffic often contains measurable patterns that distinguish normal behavior from attacks. Earlier studies on benchmark intrusion datasets used algorithms such as Decision Trees, Naive Bayes, Support Vector Machines, k-Nearest Neighbors, and ensemble methods [2]. These approaches demonstrated that supervised learning can improve detection performance compared with purely rule-based systems, especially when the training data contains representative attack behavior.

The NSL-KDD dataset has been used extensively in intrusion detection research because it addresses several limitations of the original KDD Cup 1999 dataset, especially the presence of duplicate records that can bias model evaluation [1]. However, NSL-KDD remains challenging because of class imbalance, sparse features, skewed numerical distributions, and the very small number of U2R and R2L attack examples. These characteristics make model evaluation more complex than simple accuracy comparison.

Recent work has also explored ensemble learning and deep learning for intrusion detection. Random Forest and XGBoost are commonly used because they handle nonlinear feature interactions and mixed feature distributions effectively [3], [4]. Deep learning models such as multilayer perceptrons, autoencoders, convolutional networks, and recurrent networks have also been proposed [2], [5]. In this work, MLP and autoencoder architectures were selected over LSTM and GRU models because NSL-KDD is tabular and does not provide inherent temporal sequencing between rows. Sequence models may be more suitable for future datasets that contain time-ordered packet or flow records.

Explainable AI has become increasingly important for cybersecurity applications because AI-based security tools must justify detection, prediction, and decision-making outcomes to human operators [6]. SHAP provides feature attribution values based on cooperative game theory, making it useful for understanding global and local model decisions [7]. LIME explains individual predictions by fitting a local interpretable model around the selected instance [8]. Together, these methods help convert model outputs into analyst-friendly explanations.

The gap addressed in this work is the need for an implementation-oriented IDS pipeline that combines comparative model evaluation with interpretable, instance-level reasoning and a usable demonstration interface. Many NSL-KDD studies focus primarily on model accuracy, while practical cybersecurity workflows also require interpretability, transparency, and clear presentation of model behavior. Recent XAI-IDS work has similarly argued for end-to-end explainable frameworks that include preprocessing, model training, global explanations, and local explanations [9].

## III. Dataset and Preprocessing

### A. Dataset Description

The NSL-KDD dataset was used as the primary dataset for this project. The raw train and test files contain 41 network traffic features and a label column indicating whether the connection is normal or belongs to a specific attack subtype. The dataset includes three categorical features: `protocol_type`, `service`, and `flag`. It also includes continuous and discrete numerical features describing connection duration, byte counts, error rates, service counts, host-level statistics, and login-related behavior.

The processed training set contains 125,973 records, and the processed test set contains 22,544 records. After preprocessing and one-hot encoding, each processed record contains 125 columns, including feature columns, original labels, attack categories, and binary labels.

TABLE I. DATASET SUMMARY

| Property | Train Set | Test Set |
|---|---:|---:|
| Samples | 125,973 | 22,544 |
| Raw features | 41 | 41 |
| Processed columns | 125 | 125 |
| Categorical features | 3 | 3 |
| Null values | 0 | 0 |
| Binary label | Normal / Attack | Normal / Attack |
| Multiclass categories | Normal, DoS, Probe, R2L, U2R | Normal, DoS, Probe, R2L, U2R |

### B. Attack Category Mapping

The raw NSL-KDD labels include multiple attack subtypes. These subtypes were mapped into five broader categories: Normal, Denial of Service (DoS), Probe, Remote to Local (R2L), and User to Root (U2R). A binary label was also created where normal traffic is represented as 0 and attack traffic is represented as 1. Two test records contained attack subtypes that were mapped as Unknown and were excluded from multiclass evaluation.

TABLE II. ATTACK CATEGORY DISTRIBUTION

| Category | Train Count | Test Count |
|---|---:|---:|
| Normal | 67,343 | 9,711 |
| DoS | 45,927 | 7,458 |
| Probe | 11,656 | 2,421 |
| R2L | 995 | 2,885 |
| U2R | 52 | 67 |
| Unknown | 0 | 2 |

The class distribution in Fig. 2 highlights the imbalance between dominant categories such as Normal and DoS and rare categories such as R2L and U2R. This imbalance affects both model training and evaluation, especially in the multiclass setting.

**Fig. 2. Attack category distribution in the processed NSL-KDD training data.**

Suggested figure file: `results/graphs/attack_category_distribution.png`

### C. Preprocessing Pipeline

The preprocessing stage was implemented in `01_data_cleaning.ipynb` and supported by reusable Python utilities. The major preprocessing steps were:

1. Load `KDDTrain+.txt` and `KDDTest+.txt` with manually assigned column names.
2. Drop the NSL-KDD `difficulty_level` column because it is metadata rather than a model feature.
3. Remove duplicate rows from the training set.
4. Verify that there are no null values in train or test data.
5. Map attack subtypes into broader attack categories.
6. Create the binary classification label.
7. Apply one-hot encoding to `protocol_type`, `service`, and `flag`.
8. Align train and test feature columns using combined encoding.
9. Save cleaned outputs to `data/processed/train_cleaned.csv` and `data/processed/test_cleaned.csv`.
10. Save the scaler, feature-column list, and label encoder for later model and dashboard use.

## IV. Exploratory Data Analysis

Exploratory data analysis was performed in `02_eda.ipynb` to understand class distributions, feature distributions, correlations, categorical behavior, and outlier patterns. The analysis showed that the dataset is imbalanced, with Normal and DoS traffic dominating the training data. R2L and U2R attacks are significantly underrepresented, which can reduce model sensitivity to rare attack classes.

Several numerical features, including `src_bytes` and `dst_bytes`, were highly right-skewed. Many features were also sparse, with a large proportion of zero values. Correlation analysis showed that some feature pairs, such as `serror_rate` and `srv_serror_rate`, were highly correlated. These observations are important because skewness, sparsity, and correlation can influence model behavior and interpretation.

The EDA stage generated class-balance plots, feature-distribution plots, correlation heatmaps, categorical-feature comparisons, key feature category plots, outlier counts, and feature-label correlation charts. These outputs informed later interpretation by identifying dominant classes, sparse variables, skewed byte-count features, and strongly correlated traffic statistics.

## V. Proposed Methodology

The proposed methodology follows a complete NIDS-XAI workflow:

1. Raw NSL-KDD train and test files are loaded.
2. Data cleaning and feature transformation are applied.
3. Binary and multiclass labels are prepared.
4. Models are trained using processed features.
5. Models are evaluated using accuracy, precision, recall, F1 score, and confusion matrices.
6. SHAP and LIME explanations are generated for selected models and instances.
7. Results are presented in an interactive Streamlit dashboard.

The system supports two main classification tasks. The binary task predicts whether a connection is Normal or Attack. The multiclass task predicts one of five categories: Normal, DoS, Probe, R2L, or U2R. The autoencoder is treated separately as an anomaly detection model trained on normal traffic only.

The paper positions the system as an explainable IDS framework. Therefore, model performance validates the detection layer, while SHAP, LIME, and the dashboard validate the interpretability and usability layer. This framing is important because NSL-KDD classification alone is a heavily explored topic; the stronger research value comes from combining detection results with transparent explanations and practical inspection tools.

## VI. Experimental Setup

### A. Models Trained

Six model families were trained and evaluated to compare linear, tree-based, ensemble, neural, and reconstruction-based approaches:

TABLE III. MODELS USED IN THE PROJECT

| Model | Type | Task |
|---|---|---|
| Random Forest | Machine Learning | Binary and Multiclass |
| XGBoost | Machine Learning | Binary and Multiclass |
| Decision Tree | Machine Learning | Binary and Multiclass |
| Logistic Regression | Machine Learning | Binary and Multiclass |
| Multilayer Perceptron | Deep Learning | Binary and Multiclass |
| Autoencoder | Deep Learning | Anomaly Detection |

The Random Forest model used 100 estimators with a fixed random seed. The XGBoost model used 100 estimators and log-loss evaluation. The Decision Tree model used a maximum depth of 20. Logistic Regression was trained with a maximum of 1000 iterations.

The MLP architecture consisted of dense layers with 256, 128, and 64 neurons, ReLU activations, batch normalization, dropout, and a sigmoid or softmax output depending on the task. The autoencoder used an encoder with 128, 64, and 32 neurons followed by a mirrored decoder. It was trained on normal traffic only, and the anomaly threshold was set using the 95th percentile of reconstruction error on normal training samples.

### B. Evaluation Metrics

The models were evaluated using accuracy, precision, recall, F1 score, and confusion matrices. For binary classification, the positive class represents attack traffic. For multiclass classification, weighted precision, recall, and F1 score were used to account for class imbalance.

## VII. Results and Discussion

### A. Binary Classification Results

The binary classification results show that the evaluated supervised models were able to distinguish normal and attack traffic with moderate to strong performance. MLP achieved the highest binary F1 score, while Decision Tree and XGBoost remained close classical baselines.

TABLE IV. BINARY CLASSIFICATION AND ANOMALY DETECTION RESULTS

| Model | Accuracy | Precision | Recall | F1 Score |
|---|---:|---:|---:|---:|
| Random Forest | 76.48% | 96.70% | 60.75% | 74.62% |
| XGBoost | 78.79% | 96.81% | 64.87% | 77.69% |
| Decision Tree | 79.15% | 96.56% | 65.71% | 78.21% |
| Logistic Regression | 75.35% | 91.73% | 62.32% | 74.21% |
| MLP | 79.28% | 92.97% | 68.81% | 79.08% |
| Autoencoder | 74.33% | 91.20% | 60.77% | 72.94% |

The high precision values indicate that when these models predict an attack, they are usually correct. However, the lower recall values indicate that a meaningful portion of attack traffic is still missed. The MLP achieved the strongest binary F1 score among the evaluated models, while Decision Tree and XGBoost remained close classical baselines. The autoencoder achieved lower recall than the supervised classifiers, which is expected because it was trained as an unsupervised anomaly detector using normal traffic reconstruction error rather than direct attack labels. Fig. 3 provides a visual comparison of binary model performance. In cybersecurity applications, recall is especially important because missed attacks can create operational risk. Therefore, future work should consider threshold tuning, class weighting, resampling, or cost-sensitive learning to reduce false negatives.

**Fig. 3. Binary classification performance comparison across machine learning and deep learning models.**

Suggested figure file: `results/graphs/binary_all_models_comparison.png`

### B. Multiclass Classification Results

For multiclass classification, XGBoost achieved the best overall performance, with 77.70% accuracy and 74.44% weighted F1 score. Decision Tree and Random Forest also performed competitively, while Logistic Regression and MLP showed lower weighted F1 scores.

TABLE V. MULTICLASS CLASSIFICATION RESULTS

| Model | Accuracy | Weighted Precision | Weighted Recall | Weighted F1 Score |
|---|---:|---:|---:|---:|
| Random Forest | 75.12% | 80.89% | 75.12% | 70.47% |
| XGBoost | 77.70% | 82.70% | 77.70% | 74.44% |
| Decision Tree | 76.85% | 82.09% | 76.85% | 73.31% |
| Logistic Regression | 76.38% | 74.97% | 76.38% | 72.27% |
| MLP | 76.12% | 79.41% | 76.12% | 72.25% |

The multiclass task is more difficult than binary classification because rare attack categories such as R2L and U2R contain very few training examples. This imbalance can cause models to favor dominant classes such as Normal and DoS. The MLP multiclass confusion matrix shows this issue clearly: common classes such as Normal and DoS are detected more reliably, while R2L and U2R remain difficult due to limited and imbalanced samples. Fig. 4 summarizes multiclass model performance across the evaluated models. Weighted metrics partially address this by considering class support, but additional per-class analysis is needed before drawing strong conclusions about rare attack detection.

**Fig. 4. Multiclass classification performance comparison across evaluated models.**

Suggested figure file: `results/graphs/multiclass_all_models_comparison.png`

### C. Discussion

The results suggest that tree-based models are well suited for the NSL-KDD feature space. XGBoost and Decision Tree performed strongly across both binary and multiclass settings. The MLP achieved the best binary F1 score, showing that a dense neural network can model the processed feature space effectively. Logistic Regression provided a useful baseline but is less capable of modeling nonlinear relationships among the one-hot encoded and numerical features.

The binary results show a tradeoff between precision and recall. High attack precision is useful because it reduces false alarms, but lower recall means some attacks are not detected. For practical deployment, model selection should consider the operational cost of false positives and false negatives. In intrusion detection, false negatives are often more serious because they represent missed attacks. Therefore, a future version of the system should include recall-focused tuning.

## VIII. Explainable AI Analysis

Explainable AI was integrated using SHAP and LIME. SHAP was used to understand both global model behavior and individual predictions for tree-based models. Global SHAP summary plots and mean absolute SHAP bar charts were generated for Random Forest and XGBoost. Fig. 5 illustrates global feature impact using SHAP. Local SHAP waterfall plots were used to show how individual feature values pushed a prediction toward Normal or Attack.

**Fig. 5. Global SHAP summary showing feature-level impact on model predictions.**

Suggested figure file: `results/xai/shap_xgb_summary_beeswarm.png`

LIME was used to generate local explanations for selected normal and attack instances. LIME explanations provide a ranked list of feature conditions and their local contribution toward the predicted class. This is useful for dashboard-based demonstration because it presents explanations in a compact and instance-specific form. A representative local explanation is shown in Fig. 6.

**Fig. 6. Local explanation for an attack prediction using SHAP or LIME.**

Suggested figure file: `results/xai/lime_xgb_attack_instance.png`

Several implementation issues were resolved during the XAI phase. Newer SHAP versions can return multidimensional arrays for classification models, so SHAP values were normalized before plotting. LIME initially produced label-selection errors for normal instances, which were fixed by explicitly passing available labels and selecting the predicted label for visualization. SHAP base values were also flattened where necessary to avoid scalar conversion errors.

The explainability layer improves the practical value of the model because it allows users to inspect the reasoning behind predictions. Instead of presenting only a class label, the system can show which traffic features contributed most strongly to a decision. This is particularly useful for security analysis, where a model alert should be traceable to concrete traffic characteristics.

## IX. Dashboard Implementation

The final project includes an interactive Streamlit dashboard implemented as the front-end demonstration layer. The dashboard loads processed data, saved models, scaler objects, label encoders, and feature columns from the project directory. It is designed for live demonstration and includes both dark and light modes.

The dashboard contains four main pages:

1. Predict and Explain: Allows random or manual test-instance selection, model selection, prediction display, confidence visualization, SHAP waterfall explanation, and feature contribution table.
2. Model Performance: Displays model metrics, bar charts, radar charts, and interactive confusion matrices.
3. XAI Deep Dive: Provides global SHAP visualizations, ranked feature tables, local SHAP waterfalls, and LIME explanations.
4. Overview: Summarizes dataset statistics, attack distributions, project pipeline, model families, and team information.

The dashboard supports the project goal of making intrusion detection results explainable and demo-ready. It also provides a bridge between technical model evaluation and user-facing interpretation.

## X. Conclusion and Future Work

This paper presented a complete explainable AI driven network intrusion detection framework using the NSL-KDD dataset. The project includes preprocessing, exploratory analysis, machine learning and deep learning models, evaluation, SHAP and LIME explanations, and an interactive Streamlit dashboard. The results show that MLP achieved the highest binary F1 score, while XGBoost achieved the highest multiclass weighted F1 score. Tree-based models remained strong and interpretable classical baselines for the processed NSL-KDD feature space.

The explainability layer provides important transparency by showing feature-level contributions for model predictions. This is especially valuable in cybersecurity, where analysts must understand and validate alerts rather than simply accept automated decisions.

Future work should focus on improving recall for attack detection, handling class imbalance more directly, adding per-class rare-attack analysis, comparing additional datasets, and exploring temporal models on datasets that provide true sequence information. Further improvements may also include dashboard deployment, automated report generation, model threshold tuning, and integration with real-time traffic monitoring. This direction aligns with the project goal of producing research-oriented, practical, and implementation-backed cybersecurity work rather than a purely conceptual system.

## Acknowledgment

This project was developed under the IEEE Computer Society Bangalore Chapter Student Internship and Mentorship Program 2026. The authors thank the project mentor, program coordinators, and reviewers for their guidance throughout the development process.

## References

[1] M. Tavallaee, E. Bagheri, W. Lu, and A. A. Ghorbani, "A detailed analysis of the KDD CUP 99 data set," in Proc. IEEE Symposium on Computational Intelligence for Security and Defense Applications, Ottawa, ON, Canada, 2009, pp. 1-6.

[2] H. Liu and B. Lang, "Machine learning and deep learning methods for intrusion detection systems: A survey," Applied Sciences, vol. 9, no. 20, Art. no. 4396, 2019.

[3] L. Breiman, "Random forests," Machine Learning, vol. 45, no. 1, pp. 5-32, 2001.

[4] T. Chen and C. Guestrin, "XGBoost: A scalable tree boosting system," in Proc. 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining, San Francisco, CA, USA, 2016, pp. 785-794.

[5] Y. Song, S. Hyun, and Y.-G. Cheong, "Analysis of autoencoders for network intrusion detection," Sensors, vol. 21, no. 13, Art. no. 4294, 2021.

[6] F. Charmet et al., "Explainable artificial intelligence for cybersecurity: A literature survey," Annals of Telecommunications, vol. 77, pp. 789-812, 2022.

[7] S. M. Lundberg and S.-I. Lee, "A unified approach to interpreting model predictions," in Proc. Advances in Neural Information Processing Systems, Long Beach, CA, USA, 2017.

[8] M. T. Ribeiro, S. Singh, and C. Guestrin, "Why should I trust you? Explaining the predictions of any classifier," in Proc. 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining, San Francisco, CA, USA, 2016, pp. 1135-1144.

[9] O. Arreche, T. Guntur, and M. Abdallah, "XAI-IDS: Toward proposing an explainable artificial intelligence framework for enhancing network intrusion detection systems," Applied Sciences, vol. 14, no. 10, Art. no. 4170, 2024.

## Context Needed Before Final IEEE Formatting

1. Mentor name, title, affiliation, and whether mentor should appear as co-author or acknowledgment only.
2. Required page limit from IEEE CS Bangalore SIMP, if any.
3. Whether the submission expects IEEE conference Word format, LaTeX format, or a custom program template.
4. Mentor instructions about including dashboard screenshots in the paper.
5. Any required plagiarism/similarity constraints or citation minimums.
