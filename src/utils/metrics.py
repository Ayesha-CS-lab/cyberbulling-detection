"""
Evaluation metrics for multi-label classification.
Supports per-label and macro-averaged metrics.
"""
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
)


LABEL_NAMES = ['Aggression', 'Repetition', 'Intent']


def compute_metrics(y_true, y_pred_probs, threshold=0.5):
    """
    Compute classification metrics for multi-label predictions.

    Args:
        y_true: Ground truth labels, shape (N, 3)
        y_pred_probs: Predicted probabilities, shape (N, 3)
        threshold: Classification threshold

    Returns:
        Dictionary with all metrics
    """
    y_pred = (y_pred_probs >= threshold).astype(int)
    y_true = y_true.astype(int)

    metrics = {}

    # Overall accuracy (exact match ratio)
    metrics['accuracy'] = accuracy_score(y_true, y_pred)

    # Per-label metrics
    for i, name in enumerate(LABEL_NAMES):
        metrics[f'{name.lower()}_precision'] = precision_score(
            y_true[:, i], y_pred[:, i], zero_division=0
        )
        metrics[f'{name.lower()}_recall'] = recall_score(
            y_true[:, i], y_pred[:, i], zero_division=0
        )
        metrics[f'{name.lower()}_f1'] = f1_score(
            y_true[:, i], y_pred[:, i], zero_division=0
        )

    # Macro averaged
    metrics['precision_macro'] = precision_score(
        y_true, y_pred, average='macro', zero_division=0
    )
    metrics['recall_macro'] = recall_score(
        y_true, y_pred, average='macro', zero_division=0
    )
    metrics['f1_macro'] = f1_score(
        y_true, y_pred, average='macro', zero_division=0
    )

    # Micro averaged
    metrics['precision_micro'] = precision_score(
        y_true, y_pred, average='micro', zero_division=0
    )
    metrics['recall_micro'] = recall_score(
        y_true, y_pred, average='micro', zero_division=0
    )
    metrics['f1_micro'] = f1_score(
        y_true, y_pred, average='micro', zero_division=0
    )

    return metrics


def get_classification_report(y_true, y_pred_probs, threshold=0.5):
    """Generate a formatted classification report string."""
    y_pred = (y_pred_probs >= threshold).astype(int)
    y_true = y_true.astype(int)

    report = ""
    report += "=" * 60 + "\n"
    report += "CLASSIFICATION REPORT\n"
    report += "=" * 60 + "\n\n"

    for i, name in enumerate(LABEL_NAMES):
        report += f"--- {name} ---\n"
        report += classification_report(
            y_true[:, i], y_pred[:, i],
            target_names=['Non-Bullying', 'Bullying'],
            zero_division=0
        )
        report += "\n"

    # Overall
    metrics = compute_metrics(y_true, y_pred_probs, threshold)
    report += "--- Overall (Macro) ---\n"
    report += f"  Accuracy:  {metrics['accuracy']:.4f}\n"
    report += f"  Precision: {metrics['precision_macro']:.4f}\n"
    report += f"  Recall:    {metrics['recall_macro']:.4f}\n"
    report += f"  F1 Score:  {metrics['f1_macro']:.4f}\n"
    report += "=" * 60 + "\n"

    return report


def get_confusion_matrices(y_true, y_pred_probs, threshold=0.5):
    """Return confusion matrices for each label."""
    y_pred = (y_pred_probs >= threshold).astype(int)
    y_true = y_true.astype(int)

    matrices = {}
    for i, name in enumerate(LABEL_NAMES):
        matrices[name] = confusion_matrix(y_true[:, i], y_pred[:, i])

    return matrices
