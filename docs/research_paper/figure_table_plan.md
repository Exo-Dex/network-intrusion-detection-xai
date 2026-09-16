# Figure and Table Plan for IEEE Paper

This plan keeps the paper focused on the XAI-driven IDS contribution. Use only the strongest visuals in the final IEEE version; too many figures will crowd the two-column layout.

## Recommended Tables

1. **Dataset Summary**
   - Source: Table I in `ieee_paper_draft.md`
   - Purpose: Establishes train/test size, feature count, labels, and processed columns.

2. **Model Families**
   - Source: Table III in `ieee_paper_draft.md`
   - Purpose: Quickly shows the ML, DL, and anomaly-detection coverage.

3. **Binary Results**
   - Source: Table IV in `ieee_paper_draft.md`
   - Purpose: Compare RF, XGBoost, DT, LR, MLP, and Autoencoder using accuracy/precision/recall/F1.

4. **Multiclass Results**
   - Source: Table V in `ieee_paper_draft.md`
   - Purpose: Show XGBoost as best multiclass performer and expose the gap caused by class imbalance.

## Recommended Figures

1. **Proposed NIDS-XAI Workflow**
   - File: `docs/research_paper/nids_xai_architecture_ieee.svg`
   - Suggested caption: `Fig. 1. Proposed NIDS-XAI workflow from NSL-KDD preprocessing to prediction, explanation, and dashboard visualization.`
   - Purpose: Academic architecture diagram.

2. **Class Distribution**
   - File: `results/graphs/attack_category_distribution.png` or `results/graphs/class_balance.png`
   - Suggested caption: `Fig. 2. Attack category distribution in the processed NSL-KDD training data.`
   - Purpose: Supports class imbalance discussion.

3. **Binary Model Comparison**
   - File: `results/graphs/binary_all_models_comparison.png`
   - Suggested caption: `Fig. 3. Binary classification performance comparison across machine learning and deep learning models.`
   - Purpose: Summarizes the binary results table visually.

4. **Multiclass Model Comparison**
   - File: `results/graphs/multiclass_all_models_comparison.png`
   - Suggested caption: `Fig. 4. Multiclass classification performance comparison across evaluated models.`
   - Purpose: Supports XGBoost multiclass result.

5. **SHAP Global Explanation**
   - File: `results/xai/shap_xgb_summary_beeswarm.png` or `results/xai/shap_rf_summary_beeswarm.png`
   - Suggested caption: `Fig. 5. Global SHAP summary showing feature-level impact on model predictions.`
   - Purpose: Strongest global XAI visual.

6. **Local Explanation**
   - File: `results/xai/shap_rf_waterfall_attack.png` or `results/xai/lime_xgb_attack_instance.png`
   - Suggested caption: `Fig. 6. Local explanation for an attack prediction using SHAP or LIME.`
   - Purpose: Demonstrates instance-level interpretability.

## Final Recommendation
For a 5-6 page IEEE paper, use:
1. Workflow diagram
2. Class distribution
3. Binary model comparison
4. Multiclass model comparison
5. SHAP global summary
6. One local explanation figure
