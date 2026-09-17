"""
Unit and Integration Tests for Principled Gating & Streaming Latency.
Verifies:
1. Normalized Shannon Entropy Gating bounds H(x) in [0, 1].
2. Split-Conformal prediction set calibration and risk coverage.
3. Single-flow streaming latency benchmark (batch=1) with cache warmup.
"""

import unittest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.cascade_controller import CascadedNIDSController


class DummyModel:
    def predict_proba(self, X):
        probs = np.zeros((len(X), 2), dtype=np.float32)
        half = len(X) // 2
        probs[:half, 0] = 0.95
        probs[:half, 1] = 0.05
        probs[half:, 0] = 0.50
        probs[half:, 1] = 0.50
        return probs

    def predict(self, X):
        p = self.predict_proba(X)
        return np.argmax(p, axis=1)


class TestPrincipledGating(unittest.TestCase):
    def setUp(self):
        self.model = DummyModel()
        self.controller = CascadedNIDSController(
            tier1_model=self.model,
            gating_mode="entropy",
            entropy_threshold=0.80,
            conformal_alpha=0.05
        )

    def test_normalized_shannon_entropy(self):
        p_ambiguous = np.array([[0.5, 0.5]], dtype=np.float32)
        h_max = CascadedNIDSController.compute_normalized_entropy(p_ambiguous)
        self.assertAlmostEqual(float(h_max[0]), 1.0, places=3)

        p_certain = np.array([[0.9999, 0.0001]], dtype=np.float32)
        h_min = CascadedNIDSController.compute_normalized_entropy(p_certain)
        self.assertLess(float(h_min[0]), 0.02)

    def test_split_conformal_calibration(self):
        X_cal = np.random.randn(200, 10).astype(np.float32)
        y_cal = np.random.randint(0, 2, size=200).astype(np.int32)
        
        controller = CascadedNIDSController(
            tier1_model=self.model,
            gating_mode="conformal",
            conformal_alpha=0.05
        )
        q = controller.calibrate_conformal_quantile(X_cal, y_cal)
        self.assertGreater(q, 0.0)
        self.assertLessEqual(q, 1.0)
        self.assertIsNotNone(controller.conformal_quantile)

    def test_single_flow_streaming_latency(self):
        X_test = np.random.randn(500, 10).astype(np.float32)
        stats = self.controller.benchmark_streaming_latency(X_test, n_warmup=50, n_eval=100)
        self.assertIn("mean_us", stats)
        self.assertIn("p50_us", stats)
        self.assertIn("p90_us", stats)
        self.assertIn("p99_us", stats)
        self.assertGreater(stats["p99_us"], 0.0)
        self.assertGreater(stats["streaming_throughput_fps"], 0.0)


    def test_host_temporal_sequence_builder(self):
        import pandas as pd
        from src.sequence_builder import build_host_temporal_sequences
        df = pd.DataFrame(np.random.randn(150, 6), columns=[f"f{j}" for j in range(6)])
        df["binary_label"] = np.random.randint(0, 2, 150)
        df["dst_host"] = np.random.choice(["hostA", "hostB", "hostC"], size=150)
        X_seq, y_seq = build_host_temporal_sequences(df, [f"f{j}" for j in range(6)], window_size=10)
        self.assertEqual(len(X_seq), 150)
        self.assertEqual(X_seq.shape, (150, 10, 6))
        X_seq_h, y_seq_h = build_host_temporal_sequences(df, [f"f{j}" for j in range(6)], host_col="dst_host", window_size=10)
        self.assertEqual(len(X_seq_h), 150)
        self.assertEqual(X_seq_h.shape, (150, 10, 6))

    def test_cascade_sequence_model_compatibility(self):
        class DummySeqModel:
            input_shape = (None, 10, 10)
            def predict(self, X, verbose=0):
                return np.ones((len(X), 1), dtype=np.float32)
        seq_controller = CascadedNIDSController(
            tier1_model=self.model,
            tier2_model=DummySeqModel(),
            gating_mode="threshold",
            p_high=0.6,
            p_low=0.4
        )
        X_test = np.random.randn(40, 10).astype(np.float32)
        X_seq = np.random.randn(40, 10, 10).astype(np.float32)
        res_with_seq = seq_controller.evaluate_traffic_stream(X_test, X_sequential=X_seq)
        self.assertEqual(res_with_seq["total_flows"], 40)
        res_without_seq = seq_controller.evaluate_traffic_stream(X_test, X_sequential=None)
        self.assertEqual(res_without_seq["total_flows"], 40)

if __name__ == "__main__":
    unittest.main()
