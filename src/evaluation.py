"""
Evaluation utilities: performance metric calculation and confusion matrix plotting.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report


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
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    return fig
