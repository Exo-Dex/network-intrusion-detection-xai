"""
Operational Tiered Cascaded Controller for Network Intrusion Detection.
Routes high-throughput traffic through a lightweight Tier-1 screening filter
and conditionally escalates ambiguous boundary states and structural anomalies
to a deep Tier-2 hybrid triage engine.
"""

import time
import numpy as np
import pandas as pd


class CascadedNIDSController:
    """
    Tiered Cascaded Deployment Controller.
    
    Tier 1: High-throughput inline classifier (e.g. Decision Tree or Quantized XGBoost).
    Tier 2: Deep hybrid anomaly and sequence engine (Autoencoder-LSTM or Deep MLP).
    """

    def __init__(
        self,
        tier1_model,
        tier2_model=None,
        tier2_autoencoder=None,
        p_high=0.85,
        p_low=0.35,
        reconstruction_threshold=None
    ):
        """
        Parameters:
        -----------
        tier1_model : trained classifier
            Lightweight model providing rapid predict_proba or predict.
        tier2_model : trained deep model, optional
            Heavyweight model for deep inspection.
        tier2_autoencoder : trained Autoencoder, optional
            Evaluates reconstruction error MSE for zero-day deviation.
        p_high : float
            Upper confidence threshold (flows with prob >= p_high resolved inline).
        p_low : float
            Lower confidence threshold (flows with prob <= p_low resolved inline as benign).
        reconstruction_threshold : float, optional
            Reconstruction error ceiling above which any flow is escalated to Tier 2.
        """
        self.tier1_model = tier1_model
        self.tier2_model = tier2_model
        self.tier2_autoencoder = tier2_autoencoder
        self.p_high = p_high
        self.p_low = p_low
        self.reconstruction_threshold = reconstruction_threshold

    def evaluate_traffic_stream(self, X_tabular, X_sequential=None, y_true=None):
        """
        Process a stream of network connection records through the cascaded pipeline.
        
        Parameters:
        -----------
        X_tabular : np.ndarray or pd.DataFrame
            Standard 2D flow features for Tier 1 screening.
        X_sequential : np.ndarray, optional
            3D sequential flow windows for Tier 2 recurrent engine (shape: N, T, D).
        y_true : np.ndarray, optional
            True binary/multiclass labels for benchmarking.
            
        Returns:
        --------
        results : dict
            Detailed execution trace, routing metrics, latency analysis, and decisions.
        """
        if isinstance(X_tabular, pd.DataFrame):
            X_mat = X_tabular.values
        else:
            X_mat = np.array(X_tabular)

        n_samples = len(X_mat)
        final_preds = np.zeros(n_samples, dtype=np.int32)
        routed_to_tier2 = np.zeros(n_samples, dtype=bool)
        tier1_confidences = np.zeros(n_samples, dtype=np.float32)

        start_total = time.perf_counter()

        # Step 1: Rapid Tier 1 Inline Inference
        start_t1 = time.perf_counter()
        if hasattr(self.tier1_model, 'predict_proba'):
            t1_probs = self.tier1_model.predict_proba(X_mat)
            if t1_probs.shape[1] == 2:
                # Binary: probability of attack
                attack_probs = t1_probs[:, 1]
            else:
                attack_probs = np.max(t1_probs, axis=1)
        else:
            # Fallback if predict_proba unavailable
            attack_probs = self.tier1_model.predict(X_mat).astype(np.float32)
            
        tier1_confidences[:] = attack_probs
        t1_elapsed = time.perf_counter() - start_t1

        # Step 2: Routing Rule Evaluation
        # Escalation criteria:
        # 1. Ambiguity in prediction: p_low < attack_probs < p_high
        ambiguous_mask = (attack_probs > self.p_low) & (attack_probs < self.p_high)
        
        # 2. Structural reconstruction anomaly (if Autoencoder threshold configured)
        if self.tier2_autoencoder is not None and self.reconstruction_threshold is not None:
            recon_preds = self.tier2_autoencoder.predict(X_mat, verbose=0)
            mse_errors = np.mean(np.square(X_mat - recon_preds), axis=-1)
            anomaly_mask = mse_errors > self.reconstruction_threshold
            escalation_mask = ambiguous_mask | anomaly_mask
        else:
            escalation_mask = ambiguous_mask

        routed_to_tier2[:] = escalation_mask
        
        # Inline resolved decisions
        final_preds[~escalation_mask] = (attack_probs[~escalation_mask] >= 0.5).astype(np.int32)

        # Step 3: Tier 2 Deep Triage for Escalated Candidate Flows
        num_escalated = int(np.sum(escalation_mask))
        t2_elapsed = 0.0

        if num_escalated > 0 and self.tier2_model is not None:
            start_t2 = time.perf_counter()
            if X_sequential is not None:
                X_t2_input = X_sequential[escalation_mask]
            else:
                X_t2_input = X_mat[escalation_mask]

            if hasattr(self.tier2_model, 'predict_proba'):
                t2_probs = self.tier2_model.predict_proba(X_t2_input)
                t2_preds = np.argmax(t2_probs, axis=-1) if t2_probs.ndim > 1 and t2_probs.shape[1] > 2 else (t2_probs[:, 1] >= 0.5).astype(np.int32)
            elif hasattr(self.tier2_model, 'predict'):
                t2_out = self.tier2_model.predict(X_t2_input)
                if isinstance(t2_out, list):
                    # Multi-output hybrid model (classification, reconstruction)
                    t2_out = t2_out[0]
                t2_preds = (t2_out >= 0.5).astype(np.int32).flatten()
            else:
                t2_preds = np.ones(num_escalated, dtype=np.int32)

            final_preds[escalation_mask] = t2_preds
            t2_elapsed = time.perf_counter() - start_t2
        elif num_escalated > 0:
            # Fallback if no tier2 model specified
            final_preds[escalation_mask] = (attack_probs[escalation_mask] >= 0.5).astype(np.int32)

        total_elapsed = time.perf_counter() - start_total
        tier1_resolved = n_samples - num_escalated

        # Latency metrics
        avg_latency_us = (total_elapsed / n_samples) * 1e6 if n_samples > 0 else 0.0
        tier1_latency_us = (t1_elapsed / n_samples) * 1e6 if n_samples > 0 else 0.0
        tier2_latency_us = (t2_elapsed / num_escalated) * 1e6 if num_escalated > 0 else 0.0

        summary = {
            'total_flows': n_samples,
            'tier1_resolved_count': tier1_resolved,
            'tier1_resolved_pct': (tier1_resolved / n_samples) * 100 if n_samples > 0 else 0.0,
            'tier2_escalated_count': num_escalated,
            'tier2_escalated_pct': (num_escalated / n_samples) * 100 if n_samples > 0 else 0.0,
            'mean_latency_microseconds_per_flow': avg_latency_us,
            'tier1_alone_latency_microseconds': tier1_latency_us,
            'tier2_deep_latency_microseconds': tier2_latency_us,
            'throughput_flows_per_second': (n_samples / total_elapsed) if total_elapsed > 0 else 0.0,
            'final_predictions': final_preds,
            'tier2_routed_mask': routed_to_tier2,
            'tier1_confidences': tier1_confidences
        }

        if y_true is not None:
            y_arr = np.array(y_true)
            acc = np.mean(final_preds == y_arr)
            summary['cascaded_accuracy'] = acc

        return summary
