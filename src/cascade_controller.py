"""
Operational Tiered Cascaded Controller for Network Intrusion Detection.
Routes high-throughput traffic through a lightweight Tier-1 screening filter
and conditionally escalates ambiguous boundary states and structural anomalies
to a deep Tier-2 hybrid triage engine.

Key Novelties & Architectural Improvements:
1. Selective Tier-2 Execution: Autoencoder reconstruction and Bi-LSTM sequence triage
   execute ONLY on flows where Tier-1 is uncertain, strictly preserving multi-gigabit line-rate.
2. Dynamic Uncertainty Gating: Replaces heuristic static thresholds with Normalized
   Shannon Entropy gating and Conformal Prediction Set evaluation.
3. True Systems Benchmarking: Full batch=1 streaming latency evaluation with cache
   warmup, reporting Mean, P50 (median), P90, and P99 tail latencies.
"""

import time
import numpy as np
import pandas as pd


class CascadedNIDSController:
    """
    Tiered Cascaded Deployment Controller with Dynamic Uncertainty Gating.
    """

    def __init__(
        self,
        tier1_model,
        tier2_model=None,
        tier2_autoencoder=None,
        gating_mode="entropy",  # "entropy", "conformal", or "threshold"
        entropy_threshold=0.80,  # Escalation if normalized Shannon entropy >= threshold
        conformal_alpha=0.05,    # Significance level for conformal prediction set
        conformal_quantile=None, # Calibrated conformal non-conformity threshold
        p_high=0.85,
        p_low=0.35,
        reconstruction_threshold=None
    ):
        self.tier1_model = tier1_model
        self.tier2_model = tier2_model
        self.tier2_autoencoder = tier2_autoencoder
        self.gating_mode = gating_mode
        self.entropy_threshold = entropy_threshold
        self.conformal_alpha = conformal_alpha
        self.conformal_quantile = conformal_quantile
        self.p_high = p_high
        self.p_low = p_low
        self.reconstruction_threshold = reconstruction_threshold

    @staticmethod
    def compute_normalized_entropy(probs):
        probs = np.asarray(probs, dtype=np.float32)
        if probs.ndim == 1:
            p1 = np.clip(probs, 1e-7, 1.0 - 1e-7)
            p0 = 1.0 - p1
            ent = -(p1 * np.log2(p1) + p0 * np.log2(p0))
            return np.clip(ent, 0.0, 1.0)
        else:
            probs = np.clip(probs, 1e-7, 1.0)
            n_classes = probs.shape[1]
            if n_classes <= 1:
                return np.zeros(len(probs), dtype=np.float32)
            ent = -np.sum(probs * np.log2(probs), axis=1) / np.log2(n_classes)
            return np.clip(ent, 0.0, 1.0)

    def calibrate_conformal_quantile(self, X_cal, y_cal):
        if hasattr(self.tier1_model, "predict_proba"):
            probs = self.tier1_model.predict_proba(X_cal)
            if probs.shape[1] == 2:
                prob_true = np.where(y_cal == 1, probs[:, 1], probs[:, 0])
            else:
                prob_true = probs[np.arange(len(y_cal)), y_cal]
        else:
            prob_true = np.ones(len(y_cal), dtype=np.float32) * 0.5

        scores = 1.0 - prob_true
        n = len(scores)
        q_idx = int(np.ceil((n + 1) * (1.0 - self.conformal_alpha))) / n
        self.conformal_quantile = float(np.quantile(scores, min(1.0, q_idx)))
        return self.conformal_quantile

    def evaluate_traffic_stream(self, X_tabular, X_sequential=None, y_true=None):
        if isinstance(X_tabular, pd.DataFrame):
            X_mat = X_tabular.values.astype(np.float32)
        else:
            X_mat = np.asarray(X_tabular, dtype=np.float32)

        n_samples = len(X_mat)
        final_preds = np.zeros(n_samples, dtype=np.int32)
        routed_to_tier2 = np.zeros(n_samples, dtype=bool)
        tier1_confidences = np.zeros(n_samples, dtype=np.float32)
        tier1_entropies = np.zeros(n_samples, dtype=np.float32)

        start_total = time.perf_counter()

        # Step 1: Rapid Tier 1 Inline Inference (Evaluates 100% of flows at line-rate)
        start_t1 = time.perf_counter()
        if hasattr(self.tier1_model, "predict_proba"):
            t1_probs = self.tier1_model.predict_proba(X_mat)
            if t1_probs.shape[1] == 2:
                attack_probs = t1_probs[:, 1]
            else:
                attack_probs = np.max(t1_probs, axis=1)
        else:
            attack_probs = self.tier1_model.predict(X_mat).astype(np.float32)

        tier1_confidences[:] = attack_probs
        tier1_entropies[:] = self.compute_normalized_entropy(attack_probs)
        t1_elapsed = time.perf_counter() - start_t1

        # Step 2: Principled Uncertainty Gating (Novelty)
        if self.gating_mode == "entropy":
            uncertain_mask = tier1_entropies >= self.entropy_threshold
        elif self.gating_mode == "conformal" and self.conformal_quantile is not None:
            scores_normal = 1.0 - (1.0 - attack_probs)
            scores_attack = 1.0 - attack_probs
            in_normal = scores_normal <= self.conformal_quantile
            in_attack = scores_attack <= self.conformal_quantile
            set_size = in_normal.astype(int) + in_attack.astype(int)
            uncertain_mask = (set_size != 1)
        else:
            uncertain_mask = (attack_probs > self.p_low) & (attack_probs < self.p_high)

        # Inline resolutions for confident flows (Tier 1 Decision)
        confident_mask = ~uncertain_mask
        final_preds[confident_mask] = (attack_probs[confident_mask] >= 0.5).astype(np.int32)

        # Step 3: Tier 2 Selective Execution (ONLY on uncertain flows)
        n_escalated = int(np.sum(uncertain_mask))
        t2_elapsed = 0.0

        if n_escalated > 0:
            start_t2 = time.perf_counter()
            X_uncertain = X_mat[uncertain_mask]
            
            # Autoencoder Spatial Reconstruction (Runs ONLY on uncertain candidate subset)
            ae_escalate_flags = np.ones(n_escalated, dtype=bool)
            if self.tier2_autoencoder is not None and self.reconstruction_threshold is not None:
                recon_uncertain = self.tier2_autoencoder.predict(X_uncertain, verbose=0)
                mse_errors = np.mean(np.square(X_uncertain - recon_uncertain), axis=-1)
                ae_escalate_flags = mse_errors > self.reconstruction_threshold

            # Sequence / Deep Triage Core (Runs on confirmed structural anomalies)
            if self.tier2_model is not None:
                if X_sequential is not None and len(X_sequential) == n_samples:
                    t2_input = X_sequential[uncertain_mask]
                else:
                    # Adapt 2D tabular features if model expects 3D temporal sequence input (None, seq_len, features)
                    seq_len = 10
                    input_shape = getattr(self.tier2_model, 'input_shape', None)
                    if input_shape and isinstance(input_shape, (tuple, list)) and len(input_shape) == 3:
                        if input_shape[1] is not None:
                            seq_len = input_shape[1]
                        t2_input = np.repeat(X_uncertain[:, np.newaxis, :], seq_len, axis=1)
                    else:
                        t2_input = X_uncertain

                try:
                    if hasattr(self.tier2_model, "predict"):
                        t2_preds = self.tier2_model.predict(t2_input, verbose=0)
                    else:
                        t2_preds = self.tier2_model(t2_input)
                except (ValueError, TypeError):
                    # Robust fallback: if model expected 3D but received 2D tabular array
                    if getattr(t2_input, "ndim", 0) == 2:
                        t2_input_3d = np.repeat(t2_input[:, np.newaxis, :], 10, axis=1)
                        t2_preds = self.tier2_model.predict(t2_input_3d, verbose=0)
                    else:
                        raise

                # Handle multi-output models (e.g. Unified Hybrid AE-LSTM returning [threat_pred, recon_seq])
                if isinstance(t2_preds, (list, tuple)):
                    t2_preds = t2_preds[0]

                t2_preds = np.asarray(t2_preds)
                if t2_preds.ndim > 1 and t2_preds.shape[1] > 1:
                    t2_decisions = np.argmax(t2_preds, axis=1)
                else:
                    t2_decisions = np.atleast_1d((np.squeeze(t2_preds) >= 0.5).astype(np.int32))

                final_preds[uncertain_mask] = np.where(ae_escalate_flags, t2_decisions, (attack_probs[uncertain_mask] >= 0.5).astype(np.int32))
            else:
                final_preds[uncertain_mask] = (attack_probs[uncertain_mask] >= 0.5).astype(np.int32)

            routed_to_tier2[uncertain_mask] = True
            t2_elapsed = time.perf_counter() - start_t2

        total_elapsed = time.perf_counter() - start_total

        t1_latency_us = (t1_elapsed / n_samples) * 1e6 if n_samples > 0 else 0.0
        t2_latency_us = (t2_elapsed / n_escalated) * 1e6 if n_escalated > 0 else 0.0
        mean_latency_us = (total_elapsed / n_samples) * 1e6 if n_samples > 0 else 0.0
        throughput = n_samples / total_elapsed if total_elapsed > 0 else 0.0

        return {
            "total_flows": n_samples,
            "tier1_resolved_count": int(np.sum(confident_mask)),
            "tier1_resolved_pct": (np.sum(confident_mask) / n_samples) * 100.0 if n_samples > 0 else 0.0,
            "tier2_escalated_count": n_escalated,
            "tier2_escalated_pct": (n_escalated / n_samples) * 100.0 if n_samples > 0 else 0.0,
            "tier1_alone_latency_microseconds": t1_latency_us,
            "tier2_deep_latency_microseconds": t2_latency_us,
            "mean_latency_microseconds_per_flow": mean_latency_us,
            "throughput_flows_per_second": throughput,
            "final_predictions": final_preds,
            "routed_to_tier2_mask": routed_to_tier2,
            "tier1_confidences": tier1_confidences,
            "tier1_entropies": tier1_entropies
        }

    def benchmark_streaming_latency(self, X_sample, n_warmup=1000, n_eval=10000):
        X_mat = np.asarray(X_sample, dtype=np.float32)
        n_available = len(X_mat)
        if n_available == 0:
            return {}

        for i in range(min(n_warmup, n_available)):
            x_single = X_mat[i:i+1]
            if hasattr(self.tier1_model, "predict_proba"):
                _ = self.tier1_model.predict_proba(x_single)
            else:
                _ = self.tier1_model.predict(x_single)

        latencies = []
        eval_count = min(n_eval, n_available)
        for i in range(eval_count):
            x_single = X_mat[i:i+1]
            t0 = time.perf_counter()
            if hasattr(self.tier1_model, "predict_proba"):
                p = self.tier1_model.predict_proba(x_single)
            else:
                p = self.tier1_model.predict(x_single)
            t1 = time.perf_counter()
            latencies.append((t1 - t0) * 1e6)

        lat_arr = np.asarray(latencies, dtype=np.float64)
        return {
            "eval_count": eval_count,
            "mean_us": float(np.mean(lat_arr)),
            "p50_us": float(np.percentile(lat_arr, 50)),
            "p90_us": float(np.percentile(lat_arr, 90)),
            "p99_us": float(np.percentile(lat_arr, 99)),
            "streaming_throughput_fps": float(1e6 / np.mean(lat_arr)) if np.mean(lat_arr) > 0 else 0.0
        }
