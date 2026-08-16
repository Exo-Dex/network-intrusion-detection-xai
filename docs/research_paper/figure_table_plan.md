# Figure and Table Plan for IEEE Paper

This plan keeps the paper focused on the XAI-driven IDS contribution. Use only the strongest visuals in the final IEEE version; too many figures will crowd the two-column layout.

## Recommended Tables

1. **Dataset Summary**
   - Source: existing Table I in `ieee_paper_draft.md`
   - Purpose: establishes train/test size, feature count, labels, and processed columns.

2. **Model Families**
   - Source: existing Table III in `ieee_paper_draft.md`
   - Purpose: quickly shows the ML, DL, and anomaly-detection coverage.

3. **Binary Results**
   - Source: existing Table IV in `ieee_paper_draft.md`
   - Purpose: compare RF, XGBoost, DT, LR, MLP, and Autoencoder using accuracy/precision/recall/F1.

4. **Multiclass Results**
   - Source: existing Table V in `ieee_paper_draft.md`
   - Purpose: show XGBoost as best multiclass performer and expose the gap caused by class imbalance.

## Recommended Figures

1. **Proposed NIDS-XAI Workflow**
   - File: `docs/research_paper/nids_xai_architecture_ieee.svg`
   - Suggested caption: `Fig. 1. Proposed NIDS-XAI workflow from NSL-KDD preprocessing to prediction, explanation, and dashboard visualization.`
   - Use instead of a dashboard screenshot as the first figure because it explains the contribution more academically.

2. **Class Distribution**
   - File: `results/graphs/attack_category_distribution.png` or `results/graphs/class_balance.png`
   - Suggested caption: `Fig. 2. Attack category distribution in the processed NSL-KDD training data.`
   - Purpose: supports class imbalance discussion.

3. **Binary Model Comparison**
   - File: `results/graphs/binary_all_models_comparison.png`
   - Suggested caption: `Fig. 3. Binary classification performance comparison across machine learning and deep learning models.`
   - Purpose: summarizes the binary results table visually.

4. **Multiclass Model Comparison**
   - File: `results/graphs/multiclass_all_models_comparison.png`
   - Suggested caption: `Fig. 4. Multiclass classification performance comparison across evaluated models.`
   - Purpose: supports XGBoost multiclass result.

5. **SHAP Global Explanation**
   - File: `results/xai/shap_xgb_summary_beeswarm.png` or `results/xai/shap_rf_summary_beeswarm.png`
   - Suggested caption: `Fig. 5. Global SHAP summary showing feature-level impact on model predictions.`
   - Purpose: strongest global XAI visual.

6. **Local Explanation**
   - File: `results/xai/shap_rf_waterfall_attack.png` or `results/xai/lime_xgb_attack_instance.png`
   - Suggested caption: `Fig. 6. Local explanation for an attack prediction using SHAP or LIME.`
   - Purpose: demonstrates instance-level interpretability.

## Optional Figures

1. **Dashboard Screenshot**
   - Include only if the final page limit allows it or mentor explicitly wants a system-demo visual.
   - Better for presentation slides than for the core paper.

2. **MLP / Autoencoder Confusion Matrices**
   - Files:
     - `results/confusion_matrix/cm_mlp_binary.png`
     - `results/confusion_matrix/cm_mlp_multiclass.png`
     - `results/confusion_matrix/cm_autoencoder.png`
   - Use only if discussing DL results in depth. Otherwise, metrics table is enough.

3. **Correlation Heatmap**
   - File: `results/graphs/correlation_heatmap.png`
   - Useful for EDA, but likely too dense for two-column IEEE format.

## Final Recommendation

For a 5-6 page IEEE paper, use:

1. Workflow diagram
2. Class distribution
3. Binary model comparison
4. Multiclass model comparison
5. SHAP global summary
6. One local explanation figure

Keep dashboard screenshots for slides or appendix-style reporting unless the mentor specifically asks to include them in the research paper.
