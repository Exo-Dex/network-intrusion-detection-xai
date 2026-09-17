"""
Evaluation utilities: performance metric calculation, confusion matrix plotting,
and operational latency/throughput benchmarks for the Tiered Cascaded NIDS architecture.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

try:
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score,
        f1_score, confusion_matrix, classification_report
    )
except ImportError:
    def accuracy_score(y_true, y_pred):
        return float(np.mean(np.array(y_true) == np.array(y_pred)))

    def precision_score(y_true, y_pred, average='binary', zero_division=0):
        y_t, y_p = np.array(y_true), np.array(y_pred)
        tp = np.sum((y_t == 1) & (y_p == 1))
        fp = np.sum((y_t == 0) & (y_p == 1))
        denom = tp + fp
        return float(tp / denom) if denom > 0 else float(zero_division)

    def recall_score(y_true, y_pred, average='binary', zero_division=0):
        y_t, y_p = np.array(y_true), np.array(y_pred)
        tp = np.sum((y_t == 1) & (y_p == 1))
        fn = np.sum((y_t == 1) & (y_p == 0))
        denom = tp + fn
        return float(tp / denom) if denom > 0 else float(zero_division)

    def f1_score(y_true, y_pred, average='binary', zero_division=0):
        p = precision_score(y_true, y_pred, zero_division=zero_division)
        r = recall_score(y_true, y_pred, zero_division=zero_division)
        denom = p + r
        return float(2 * p * r / denom) if denom > 0 else float(zero_division)

    def confusion_matrix(y_true, y_pred):
        y_t, y_p = np.array(y_true), np.array(y_pred)
        classes = np.unique(np.concatenate([y_t, y_p]))
        matrix = np.zeros((len(classes), len(classes)), dtype=int)
        for i, c1 in enumerate(classes):
            for j, c2 in enumerate(classes):
                matrix[i, j] = np.sum((y_t == c1) & (y_p == c2))
        return matrix

    def classification_report(y_true, y_pred, zero_division=0):
        acc = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred, zero_division=zero_division)
        rec = recall_score(y_true, y_pred, zero_division=zero_division)
        f1 = f1_score(y_true, y_pred, zero_division=zero_division)
        return f"Accuracy: {acc:.4f} | Precision: {prec:.4f} | Recall: {rec:.4f} | F1-Score: {f1:.4f}"


def evaluate_model(y_true, y_pred, is_multiclass=False):
    """Compute standard classification metrics."""
    avg = 'weighted' if is_multiclass else 'binary'
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average=avg, zero_division=0)
    rec = recall_score(y_true, y_pred, average=avg, zero_division=0)
    f1 = f1_score(y_true, y_pred, average=avg, zero_division=0)
    return {
        'accuracy': acc,
        'precision': prec,
        'recall': rec,
        'f1': f1,
        'report': classification_report(y_true, y_pred, zero_division=0)
    }


def plot_confusion_matrix(y_true, y_pred, labels, title='Confusion Matrix', save_path=None, cmap='Blues'):
    """Generate and save a styled confusion matrix heatmap."""
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5), dpi=300)
    sns.heatmap(cm, annot=True, fmt='d', cmap=cmap, cbar=True,
                xticklabels=labels, yticklabels=labels, ax=ax)
    ax.set_title(title, fontsize=12, fontweight='bold', pad=12)
    ax.set_ylabel('True Class', fontsize=10)
    ax.set_xlabel('Predicted Class', fontsize=10)
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    return fig


def evaluate_cascaded_system(results_dict, y_true):
    """
    Compute comprehensive operational and defensive performance metrics
    for the Tiered Cascaded NIDS architecture.
    """
    y_pred = results_dict['final_predictions']
    base_metrics = evaluate_model(y_true, y_pred, is_multiclass=False)

    t1_alone_us = results_dict.get('tier1_alone_latency_microseconds', 1.0)
    t2_alone_us = results_dict.get('tier2_deep_latency_microseconds', 50.0)
    mean_cascaded_us = results_dict.get('mean_latency_microseconds_per_flow', t1_alone_us)

    monolithic_latency = t2_alone_us if t2_alone_us > 0 else 50.0
    speedup = monolithic_latency / mean_cascaded_us if mean_cascaded_us > 0 else 1.0

    cascaded_metrics = {
        'accuracy': base_metrics['accuracy'],
        'precision': base_metrics['precision'],
        'recall': base_metrics['recall'],
        'f1': base_metrics['f1'],
        'total_flows': results_dict['total_flows'],
        'tier1_resolved_pct': results_dict['tier1_resolved_pct'],
        'tier2_escalated_pct': results_dict['tier2_escalated_pct'],
        'mean_latency_us': mean_cascaded_us,
        'throughput_flows_per_sec': results_dict.get('throughput_flows_per_second', 0.0),
        'speedup_vs_monolithic_t2': speedup
    }
    return cascaded_metrics
