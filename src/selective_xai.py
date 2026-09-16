"""
Selective Explainable AI (XAI) dispatch engine.
Applies SHAP and LIME strictly to candidate threats and ambiguous boundary flows
flagged by the Tier-2 triage controller, completely decoupling heavy attribution
computations from the Tier-1 high-throughput line-rate stream.
"""

import time
import numpy as np
import pandas as pd

try:
    import shap
except ImportError:
    shap = None

try:
    import lime
    import lime.lime_tabular
except ImportError:
    lime = None


class SelectiveXAIEngine:
    """
    Selective Explainability Dispatcher for Security Operations Centers (SOC).
    """

    def __init__(self, model, feature_names, background_data=None, class_names=None):
        self.model = model
        self.feature_names = feature_names
        self.background_data = background_data
        self.class_names = class_names or ['Normal', 'Attack']
        self._shap_explainer = None
        self._lime_explainer = None

    def initialize_explainers(self):
        """Lazy initialization of SHAP and LIME surrogate explainers."""
        if shap is not None and self.background_data is not None:
            try:
                # TreeExplainer for tree models, Kernel/Deep for others
                self._shap_explainer = shap.TreeExplainer(self.model)
            except Exception:
                # Sample background data for kernel explainer
                bg_sample = shap.sample(self.background_data, min(100, len(self.background_data)))
                self._shap_explainer = shap.KernelExplainer(self.model.predict_proba, bg_sample)

        if lime is not None and self.background_data is not None:
            self._lime_explainer = lime.lime_tabular.LimeTabularExplainer(
                training_data=np.array(self.background_data),
                feature_names=self.feature_names,
                class_names=self.class_names,
                mode='classification',
                verbose=False
            )

    def explain_candidate_flows(self, X_candidates, top_n_features=5):
        """
        Generate feature attributions only for flows requiring human analyst triage.
        
        Returns:
        --------
        explanations : list of dicts
            Each record contains predicted class, top positive and negative contributing
            features, attribution weights, and rule intervals.
        metrics : dict
            Compute latency and operational throughput metrics.
        """
        start_time = time.perf_counter()
        n_candidates = len(X_candidates)
        results = []

        if self._shap_explainer is None and shap is not None and self.background_data is not None:
            self.initialize_explainers()

        for idx in range(n_candidates):
            row = X_candidates[idx:idx+1]
            explanation_record = {
                'instance_index': idx,
                'top_influential_features': [],
                'shap_values': None,
                'lime_rules': None
            }

            # 1. SHAP attribution
            if self._shap_explainer is not None:
                try:
                    shap_vals = self._shap_explainer.shap_values(row)
                    if isinstance(shap_vals, list):
                        vals = shap_vals[1][0] if len(shap_vals) > 1 else shap_vals[0][0]
                    else:
                        vals = shap_vals[0]
                    
                    # Top features by absolute magnitude
                    top_indices = np.argsort(np.abs(vals))[::-1][:top_n_features]
                    feature_attributions = [
                        {
                            'feature': self.feature_names[i],
                            'shap_weight': float(vals[i]),
                            'direction': 'Increases Threat Odds' if vals[i] > 0 else 'Decreases Threat Odds'
                        }
                        for i in top_indices
                    ]
                    explanation_record['top_influential_features'] = feature_attributions
                    explanation_record['shap_values'] = vals
                except Exception as e:
                    explanation_record['shap_error'] = str(e)

            # 2. LIME rule surrogate
            if self._lime_explainer is not None:
                try:
                    exp = self._lime_explainer.explain_instance(
                        row[0],
                        self.model.predict_proba,
                        num_features=top_n_features
                    )
                    explanation_record['lime_rules'] = exp.as_list()
                except Exception as e:
                    explanation_record['lime_error'] = str(e)

            results.append(explanation_record)

        elapsed = time.perf_counter() - start_time
        metrics = {
            'num_explained_flows': n_candidates,
            'total_xai_time_seconds': elapsed,
            'mean_explanation_latency_ms': (elapsed / n_candidates) * 1000 if n_candidates > 0 else 0.0
        }
        return results, metrics


def calculate_xai_operational_savings(total_network_flows, escalated_flows_count, time_per_explanation_sec=0.085):
    """
    Quantifies the wall-clock compute and latency savings of selective triage.
    """
    monolithic_time_sec = total_network_flows * time_per_explanation_sec
    selective_time_sec = escalated_flows_count * time_per_explanation_sec
    saved_time_sec = monolithic_time_sec - selective_time_sec
    reduction_pct = ((total_network_flows - escalated_flows_count) / total_network_flows) * 100 if total_network_flows > 0 else 0.0

    return {
        'total_flows': total_network_flows,
        'explained_flows': escalated_flows_count,
        'monolithic_xai_hours': monolithic_time_sec / 3600,
        'selective_xai_hours': selective_time_sec / 3600,
        'compute_time_saved_hours': saved_time_sec / 3600,
        'compute_reduction_pct': reduction_pct
    }
