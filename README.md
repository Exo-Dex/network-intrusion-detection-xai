# High-Throughput Network Intrusion Detection with Decoupled Explainable AI (NIDS-XAI)

An asymmetric cascaded network intrusion detection framework that decouples high-speed inline packet filtering from deep temporal sequence triage and post-hoc interpretability. Evaluated across **CIC-IDS2017** (2.83M flows), **UNSW-NB15** (257K flows), and **NSL-KDD**, achieving line-rate microsecond triage while saving over 96% of XAI compute time.

Developed under the **IEEE Computer Society Bangalore Chapter — Student Internship and Mentorship Program (SIMP 2026)**.

---

## 📌 Architectural Highlights & Key Innovations

1. **Asymmetric Cascaded Triage**:
   - **Tier 1 (High-Speed Inline Filter)**: Shallow quantized tree ensemble evaluating 100% of flows at sub-microsecond latency ($0.04\text{--}0.28\,\mu\text{s}$), resolving $93.27\%\text{--}99.99\%$ of routine traffic inline without invoking heavy neural models.
   - **Tier 2 (Deep Spatial-Temporal Engine)**: Unsupervised spatial Autoencoder (manifold reconstruction error $\mathcal{L}_{\text{MSE}}$) coupled with a Bidirectional LSTM processing host-aggregated temporal flow sequences ($W=10$). Executes *strictly* on ambiguous candidate flows.
2. **Principled Uncertainty Gating**:
   - **Normalized Shannon Entropy**: Dynamic gating $H(x) \in [0, 1]$ escalating flows where $H(x) \ge \tau_H = 0.80$, eliminating ad-hoc heuristic probability bounds.
   - **Split-Conformal Prediction Sets**: Finite-sample distribution-free risk guarantees ($1 - \alpha = 0.95$), escalating flows when prediction sets contain ambiguity ($|\Gamma_\alpha(x)| = 2$) or out-of-distribution shifts ($|\Gamma_\alpha(x)| = 0$).
3. **Hardware Testbed Single-Flow Streaming Latency ($\text{batch}=1$)**:
   - Strict per-flow benchmarking with 1,000-flow cache warmup and monotonic microsecond timestamps ($P_{50} = 6.61\,\mu\text{s}$, $P_{90} = 7.25\,\mu\text{s}$, $P_{99} = 29.86\,\mu\text{s}$).
4. **Decoupled Selective Post-Hoc Interpretability**:
   - Asynchronous background queue executing TreeSHAP and LIME surrogate rules strictly on confirmed alert candidates, slashing CPU attribution time by over $96\%$ to $99.9\%$.

---

## 📁 Repository Structure

```
network-intrusion-detection-xai/
├── README.md                                  <- Comprehensive project overview & documentation
├── requirements.txt                            <- Python environment dependencies
├── run_pipeline.py                             <- Unified CLI runner for cascaded evaluation
├── generate_principled_gating_benchmarks.py    <- Standalone benchmark generator for charts
├── data/
│   ├── raw/                                   <- Raw PCAP flow captures (CIC-IDS2017, UNSW-NB15, NSL-KDD)
│   └── processed/                             <- Cleaned, standardized flow partitions
├── src/
│   ├── cascade_controller.py                  <- CascadedNIDSController (Entropy, Conformal, Streaming)
│   ├── preprocessing.py                       <- Protocol scaling & feature sanitization
│   ├── sequence_builder.py                    <- Host-grouped temporal sliding sequence window builder
│   ├── train_model.py                         <- Model architecture initializers & training loops
│   ├── evaluation.py                          <- Classification & operational metrics
│   └── selective_xai.py                       <- Decoupled TreeSHAP and LIME triage queue
├── notebooks/
│   ├── 01_data_cleaning.ipynb                 <- Raw capture hygiene & missing/infinity handling
│   ├── 02_eda.ipynb                           <- Statistical distribution & correlation analysis
│   ├── 03_model_training.ipynb                <- Baseline ML/DL training & hyperparameter tuning
│   ├── 04_xai_analysis.ipynb                  <- Global & local SHAP/LIME explainability exploration
│   ├── 05_tiered_cascaded_architecture.ipynb  <- Cascaded controller architecture & prototype
│   ├── 05_cicids2017_full_cascaded_evaluation.ipynb <- Full CIC-IDS2017 multi-attack benchmarking
│   ├── 06_unsw_nb15_full_cascaded_evaluation.ipynb  <- Full UNSW-NB15 9-exploit family benchmarking
│   └── 07_principled_gating_and_streaming_benchmarks.ipynb <- Entropy, Conformal & Streaming Benchmarks
├── tests/
│   └── test_principled_gating.py              <- Automated unit tests for entropy & conformal sets
├── results/
│   ├── graphs/                                <- High-resolution publication plots & benchmark charts
│   ├── confusion_matrix/                      <- Multi-class & binary confusion matrices
│   └── xai/                                   <- Generated SHAP beeswarm & LIME rule summaries
├── docs/
│   └── research_paper/
│       ├── ieee_paper.tex                     <- Official IEEE Conference LaTeX manuscript (IEEEtran)
│       ├── ieee_paper.pdf                     <- Compiled 6-page camera-ready publication PDF
│       └── IEEEtran.cls                       <- Official IEEE Transactions & Conference class file
└── dashboard/
    └── app.py                                 <- Interactive Streamlit SOC Analyst Dashboard
```

---

## 🚀 Quickstart & Pipeline Execution

### 1. Execute End-to-End Pipeline
```bash
# Execute with Normalized Shannon Entropy gating on CIC-IDS2017 PortScan
python run_pipeline.py --dataset cic-ids2017 --file portscan --gating entropy --entropy-thresh 0.80

# Execute with Split-Conformal Prediction Sets (95% statistical coverage)
python run_pipeline.py --dataset unsw-nb15 --gating conformal --conformal-alpha 0.05

# Run standalone benchmark script generating publication graphs
python generate_principled_gating_benchmarks.py
```

### 2. Run Automated Verification Tests
```bash
python -m unittest tests/test_principled_gating.py
```

### 3. Launch Interactive Streamlit SOC Dashboard
```bash
streamlit run dashboard/app.py
```

---

## 📊 Summary of Operational Results

### Multi-Dataset Cascaded Performance
| Benchmark Dataset | Total Evaluated Flows | Tier 1 Inline Resolved | Tier 2 Escalated | Mean Latency | Sustained Throughput | XAI Compute Time Saved |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **CIC-IDS2017 Web Attacks** | 170,366 | 99.89% | 0.11% | 0.16 $\mu$s | 6,375,211 flows/s | 99.89% |
| **CIC-IDS2017 PortScan**    | 286,467 | 99.99% | 0.01% | 0.08 $\mu$s | 12,727,444 flows/s | 99.99% |
| **CIC-IDS2017 DDoS Flood**  | 225,745 | 99.95% | 0.05% | 0.07 $\mu$s | 14,808,416 flows/s | 99.95% |
| **CIC-IDS2017 Grand Stream**| 682,578 | 99.95% | 0.05% | 0.07 $\mu$s | 14,483,535 flows/s | 99.95% |
| **UNSW-NB15 Cyber-Range**   | 257,673 | 96.19% | 3.81% | 0.04 $\mu$s | 22,644,927 flows/s | 96.19% |
| **NSL-KDD (Full KDDTest+)** | 22,544  | 93.27% | 6.73% | 0.28 $\mu$s | 3,571,428 flows/s  | 93.27% |

### Single-Flow ($\text{batch}=1$) Streaming Latency Profile (with Cache Warmup)
- **Mean Latency**: $7.93\,\mu\text{s}$
- **Median ($P_{50}$)**: $6.61\,\mu\text{s}$
- **$P_{90}$ Latency**: $7.25\,\mu\text{s}$
- **$P_{99}$ Tail Latency**: $29.86\,\mu\text{s}$ (Capped well below buffer overflow danger zone)

---

## 👥 Authors & Academic Affiliations
- **Piyush M. Borkar** — Department of Artificial Intelligence & Data Science, Marathwada Mitra Mandal's Institute of Technology (MMIT), Pune. Email: `piyushborkar.official@gmail.com`
- **Varun Gada** — Department of Computer Engineering, Marathwada Mitra Mandal's Institute of Technology (MMIT), Pune. Email: `varungada2004@gmail.com`
- **Dr. Boppuru Rudra Prathap** (Mentor) — Department of Computer Science & Engineering, Faculty of Engineering & Technology, M. S. Ramaiah University of Applied Sciences, Bengaluru. Email: `brprathap@gmail.com`

---

## 📚 References & Scientific Grounding

1. **[Sharafaldin et al., 2018]** I. Sharafaldin, A. H. Lashkari, and A. A. Ghorbani, *"Toward Generating a New Intrusion Detection Dataset and Intrusion Traffic Characterization,"* in Proc. 4th International Conference on Information Systems Security and Privacy (ICISSP), 2018, pp. 108–116. [DOI: 10.5220/0006639801080116](https://doi.org/10.5220/0006639801080116)
2. **[Moustafa & Slay, 2015]** N. Moustafa and J. Slay, *"UNSW-NB15: A comprehensive data set for network intrusion detection systems (UNSW-NB15 network data set),"* in Proc. IEEE Military Communications and Information Systems Conference (MilCIS), 2015, pp. 1–6. [DOI: 10.1109/MilCIS.2015.7348942](https://doi.org/10.1109/MilCIS.2015.7348942)
3. **[Tavallaee et al., 2009]** M. Tavallaee, E. Bagheri, W. Lu, and A. A. Ghorbani, *"A detailed analysis of the KDD CUP 99 data set,"* in Proc. IEEE Symposium on Computational Intelligence for Security and Defense Applications (CISDA), 2009, pp. 1–6. [DOI: 10.1109/CISDA.2009.5356528](https://doi.org/10.1109/CISDA.2009.5356528)
4. **[Lundberg & Lee, 2017]** S. M. Lundberg and S.-I. Lee, *"A unified approach to interpreting model predictions,"* in Advances in Neural Information Processing Systems (NeurIPS), vol. 30, 2017, pp. 4765–4774. [NeurIPS Proceedings](https://proceedings.neurips.cc/paper/2017/hash/8a20a8621978632d76c43dfd28b67767-Abstract.html)
5. **[Ribeiro et al., 2016]** M. T. Ribeiro, S. Singh, and C. Guestrin, *"'Why Should I Trust You?': Explaining the Predictions of Any Classifier,"* in Proc. 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining (KDD), 2016, pp. 1135–1144. [DOI: 10.1145/2939672.2939778](https://doi.org/10.1145/2939672.2939778)
6. **[Angelopoulos & Bates, 2021]** A. N. Angelopoulos and S. Bates, *"A Gentle Introduction to Conformal Prediction and Distribution-Free Uncertainty Quantification,"* arXiv preprint arXiv:2107.07511, 2021. [arXiv:2107.07511](https://arxiv.org/abs/2107.07511)
7. **[Vovk et al., 2005]** V. Vovk, A. Gammerman, and G. Shafer, *Algorithmic Learning in a Random World*, Springer Science & Business Media, 2005. [DOI: 10.1007/b106715](https://doi.org/10.1007/b106715)
8. **[Shannon, 1948]** C. E. Shannon, *"A mathematical theory of communication,"* Bell System Technical Journal, vol. 27, no. 3, pp. 379–423, 1948. [DOI: 10.1002/j.1538-7305.1948.tb01338.x](https://doi.org/10.1002/j.1538-7305.1948.tb01338.x)
9. **[Mirsky et al., 2018]** Y. Mirsky, T. Doitshman, Y. Elovici, and A. Shabtai, *"Kitsune: An ensemble of autoencoders for online network intrusion detection,"* in Proc. Network and Distributed System Security Symposium (NDSS), 2018. [DOI: 10.14722/ndss.2018.23204](https://doi.org/10.14722/ndss.2018.23204)
10. **[Hochreiter & Schmidhuber, 1997]** S. Hochreiter and J. Schmidhuber, *"Long short-term memory,"* Neural Computation, vol. 9, no. 8, pp. 1735–1780, 1997. [DOI: 10.1162/neco.1997.9.8.1735](https://doi.org/10.1162/neco.1997.9.8.1735)
