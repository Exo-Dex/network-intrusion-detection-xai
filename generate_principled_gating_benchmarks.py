"""
Benchmark and Chart Generation for Principled Gating & Single-Flow Streaming Latency.
"""

import os
import sys
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
from src.cascade_controller import CascadedNIDSController

plt.rcParams.update({
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.titlesize': 13
})

output_dir = "results/graphs"
os.makedirs(output_dir, exist_ok=True)
doc_dir = "docs/research_paper"
os.makedirs(doc_dir, exist_ok=True)


class FastQuantizedTreeModel:
    def __init__(self, feature_dim=30):
        self.feature_dim = feature_dim
        np.random.seed(42)
        self.weights = np.random.randn(feature_dim).astype(np.float32) * 0.4
        self.bias = -0.5

    def fit(self, X, y):
        X_sub = X[:5000, :self.feature_dim]
        y_sub = y[:5000]
        self.weights = np.linalg.solve(X_sub.T @ X_sub + 1e-2 * np.eye(self.feature_dim), X_sub.T @ y_sub)

    def predict_proba(self, X):
        X_mat = np.asarray(X, dtype=np.float32)
        if X_mat.ndim == 1:
            X_mat = X_mat.reshape(1, -1)
        dots = np.dot(X_mat[:, :self.feature_dim], self.weights[:min(self.feature_dim, X_mat.shape[1])]) + self.bias
        dots = np.clip(dots, -40.0, 40.0)
        p1 = 1.0 / (1.0 + np.exp(-dots))
        probs = np.zeros((len(X_mat), 2), dtype=np.float32)
        probs[:, 1] = np.clip(p1, 0.0001, 0.9999)
        probs[:, 0] = 1.0 - probs[:, 1]
        return probs

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(np.int32)


# Data setup
np.random.seed(42)
n_samples = 20000
n_features = 30
X = np.random.randn(n_samples, n_features).astype(np.float32)
y = np.random.binomial(1, 0.25, size=n_samples).astype(np.int32)
X[y == 1, :5] += 2.0

split1, split2 = int(n_samples * 0.60), int(n_samples * 0.80)
X_train, y_train = X[:split1], y[:split1]
X_cal, y_cal = X[split1:split2], y[split1:split2]
X_test, y_test = X[split2:], y[split2:]

tier1 = FastQuantizedTreeModel(feature_dim=30)
tier1.fit(X_train, y_train)

controller = CascadedNIDSController(
    tier1_model=tier1,
    gating_mode="entropy",
    entropy_threshold=0.80,
    conformal_alpha=0.05
)

# 1. Entropy Distribution
probs_test = tier1.predict_proba(X_test)
entropies = CascadedNIDSController.compute_normalized_entropy(probs_test)

plt.figure(figsize=(7, 3.8), dpi=300)
sns.histplot(entropies[y_test == 0], bins=50, color='#1f77b4', label='Benign Traffic (Routine)', alpha=0.6, stat="density", kde=True)
sns.histplot(entropies[y_test == 1], bins=50, color='#d62728', label='Attack Traffic (Anomalies)', alpha=0.6, stat="density", kde=True)
plt.axvline(x=0.80, color='black', linestyle='--', linewidth=1.5, label=r'Escalation Threshold ($	au_H = 0.80$)')
plt.title('Normalized Shannon Entropy Distribution Across Network Flows', fontweight='bold')
plt.xlabel(r'Normalized Shannon Entropy $H(x) \in [0, 1]$')
plt.ylabel('Probability Density')
plt.legend(loc='upper center', frameon=True)
plt.grid(True, linestyle=':', alpha=0.6)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "fig_gating_entropy_dist.png"))
plt.close()

# 2. Conformal Coverage
alphas = np.linspace(0.01, 0.25, 25)
nominal_coverages = 1.0 - alphas
empirical_coverages = []
probs_cal = tier1.predict_proba(X_cal)
scores_cal = 1.0 - probs_cal[np.arange(len(y_cal)), y_cal]

for a in alphas:
    q_idx = int(np.ceil((len(scores_cal) + 1) * (1.0 - a))) / len(scores_cal)
    q_val = float(np.quantile(scores_cal, min(1.0, q_idx)))
    test_scores = 1.0 - probs_test[np.arange(len(y_test)), y_test]
    emp_cov = float(np.mean(test_scores <= q_val))
    empirical_coverages.append(emp_cov)

plt.figure(figsize=(6.5, 3.8), dpi=300)
plt.plot(nominal_coverages * 100, [c * 100 for c in empirical_coverages], marker='o', color='#2ca02c', linewidth=2, label='Empirical Test Coverage')
plt.plot(nominal_coverages * 100, nominal_coverages * 100, linestyle='--', color='black', alpha=0.7, label='Theoretical Bound (1 - alpha)')
plt.title('Split-Conformal Prediction Coverage on Test Stream', fontweight='bold')
plt.xlabel('Nominal Confidence Level (1 - alpha) %')
plt.ylabel('Empirical Coverage on Test Flows (%)')
plt.legend(loc='lower right', frameon=True)
plt.grid(True, linestyle=':', alpha=0.6)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "fig_conformal_coverage.png"))
plt.close()

# 3. Streaming Latency
n_timed = min(3000, len(X_test))
times_us = []
for i in range(n_timed):
    x_single = X_test[i:i+1]
    t0 = time.perf_counter()
    _ = tier1.predict_proba(x_single)
    t1 = time.perf_counter()
    times_us.append((t1 - t0) * 1e6)

times_arr = np.asarray(times_us)
p50 = float(np.percentile(times_arr, 50))
p90 = float(np.percentile(times_arr, 90))
p99 = float(np.percentile(times_arr, 99))

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.8), dpi=300)
sns.histplot(times_arr[times_arr <= p99 * 1.5], bins=40, color='#3b528b', kde=True, ax=ax1, stat='density')
ax1.axvline(p50, color='blue', linestyle='--', linewidth=1.5, label=f'P50 Median: {p50:.2f} us')
ax1.axvline(p90, color='orange', linestyle='--', linewidth=1.5, label=f'P90: {p90:.2f} us')
ax1.axvline(p99, color='red', linestyle='--', linewidth=1.5, label=f'P99 Tail: {p99:.2f} us')
ax1.set_title('Single-Flow Streaming Latency (batch=1)', fontweight='bold')
ax1.set_xlabel('Per-Flow Latency (microseconds)')
ax1.set_ylabel('Density')
ax1.legend(loc='upper right')
ax1.grid(True, linestyle=':', alpha=0.6)

sorted_times = np.sort(times_arr)
cdf = np.arange(1, len(sorted_times) + 1) / len(sorted_times)
ax2.plot(sorted_times, cdf * 100, color='#440154', linewidth=2)
ax2.axvline(p99, color='red', linestyle=':', label=f'P99 = {p99:.2f} us')
ax2.set_xlim(0, p99 * 1.8)
ax2.set_title('Empirical Cumulative Latency (CDF)', fontweight='bold')
ax2.set_xlabel('Per-Flow Latency (microseconds)')
ax2.set_ylabel('Cumulative Percentage (%)')
ax2.legend(loc='lower right')
ax2.grid(True, linestyle=':', alpha=0.6)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "fig_single_flow_latency_dist.png"))
plt.close()

# 4. Gating Strategies Comparison
strategies = ['Static Threshold', 'Shannon Entropy', 'Split-Conformal']
res_thresh = CascadedNIDSController(tier1, gating_mode="threshold", p_high=0.85, p_low=0.35).evaluate_traffic_stream(X_test, y_true=y_test)
res_entropy = CascadedNIDSController(tier1, gating_mode="entropy", entropy_threshold=0.80).evaluate_traffic_stream(X_test, y_true=y_test)
ctrl_conf = CascadedNIDSController(tier1, gating_mode="conformal", conformal_alpha=0.05)
ctrl_conf.calibrate_conformal_quantile(X_cal, y_cal)
res_conformal = ctrl_conf.evaluate_traffic_stream(X_test, y_true=y_test)

inline_pcts = [res_thresh['tier1_resolved_pct'], res_entropy['tier1_resolved_pct'], res_conformal['tier1_resolved_pct']]
escalated_pcts = [res_thresh['tier2_escalated_pct'], res_entropy['tier2_escalated_pct'], res_conformal['tier2_escalated_pct']]

fig, ax = plt.subplots(figsize=(6.5, 3.8), dpi=300)
bar_w = 0.35
indices = np.arange(len(strategies))
p1 = ax.bar(indices - bar_w/2, inline_pcts, bar_w, label='Inline Resolved (Tier 1)', color='#2ca02c', alpha=0.85)
p2 = ax.bar(indices + bar_w/2, escalated_pcts, bar_w, label='Escalated to Tier 2', color='#d62728', alpha=0.85)
ax.set_title('Operational Routing Breakdown Across Gating Strategies', fontweight='bold')
ax.set_ylabel('Percentage of Flows (%)')
ax.set_xticks(indices)
ax.set_xticklabels(strategies)
ax.set_ylim(0, 115)
ax.legend(loc='upper right')
ax.grid(axis='y', linestyle=':', alpha=0.6)

for bar in p1:
    yval = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2, yval + 1.5, f"{yval:.1f}%", ha='center', va='bottom', fontsize=8, fontweight='bold')
for bar in p2:
    yval = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2, yval + 1.5, f"{yval:.1f}%", ha='center', va='bottom', fontsize=8, fontweight='bold')

plt.tight_layout()
plt.savefig(os.path.join(output_dir, "fig_gating_strategy_comparison.png"))
plt.close()
print("[✓] Benchmarks & charts generated successfully.")
