"""
evaluate.py - metrics and plots for multiclass classification.

Accuracy alone is misleading when classes are imbalanced, so we report:
  * per-class Precision, Recall, F1  (how well each class is handled)
  * Macro F1  = plain average of the 3 per-class F1 scores (every class counts
                equally, so the small "Enrolled" class cannot be ignored)
  * Confusion matrix (which classes get mixed up)
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             classification_report, confusion_matrix, f1_score,
                             precision_score, recall_score)

from src import config

LABELS = config.CLASS_ORDER


def compute_metrics(y_true, y_pred) -> dict:
    """Overall metrics (macro = unweighted average over the 3 classes)."""
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "macro_precision": precision_score(y_true, y_pred, labels=LABELS, average="macro", zero_division=0),
        "macro_recall": recall_score(y_true, y_pred, labels=LABELS, average="macro", zero_division=0),
        "macro_f1": f1_score(y_true, y_pred, labels=LABELS, average="macro", zero_division=0),
        "weighted_f1": f1_score(y_true, y_pred, labels=LABELS, average="weighted", zero_division=0),
    }


def per_class_table(y_true, y_pred) -> pd.DataFrame:
    """Precision / Recall / F1 / support for each class."""
    report = classification_report(y_true, y_pred, labels=LABELS,
                                   output_dict=True, zero_division=0)
    table = pd.DataFrame(report).T.loc[LABELS, ["precision", "recall", "f1-score", "support"]]
    table["support"] = table["support"].astype(int)
    return table


def get_confusion_matrix(y_true, y_pred) -> pd.DataFrame:
    """Rows = TRUE class, columns = PREDICTED class."""
    cm = confusion_matrix(y_true, y_pred, labels=LABELS)
    return pd.DataFrame(cm, index=[f"true {c}" for c in LABELS],
                        columns=[f"pred {c}" for c in LABELS])


def plot_confusion_matrix(y_true, y_pred, title, save_path=None):
    """Two panels: raw counts, and row-normalised (= recall for each true class)."""
    cm = confusion_matrix(y_true, y_pred, labels=LABELS)
    cm_pct = cm / cm.sum(axis=1, keepdims=True)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for ax, data, fmt, sub in [(axes[0], cm, "d", "Counts"),
                               (axes[1], cm_pct, ".0%", "Row-normalised (recall)")]:
        im = ax.imshow(data, cmap="Blues", vmin=0)
        ax.set_xticks(range(3), LABELS)
        ax.set_yticks(range(3), LABELS)
        ax.set_xlabel("Predicted class")
        ax.set_ylabel("True class")
        ax.set_title(sub)
        threshold = data.max() / 2
        for i in range(3):
            for j in range(3):
                label = f"{data[i, j]:{fmt}}" if fmt != ".0%" else f"{data[i, j]:.0%}"
                ax.text(j, i, label, ha="center", va="center",
                        color="white" if data[i, j] > threshold else "black")
        fig.colorbar(im, ax=ax, fraction=0.046)
    fig.suptitle(title)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=130, bbox_inches="tight")
    return fig


def plot_per_class_f1(per_class_by_model: dict, save_path=None):
    """Grouped bar chart: F1 of each class for every model (test set)."""
    df = pd.DataFrame({name: t["f1-score"] for name, t in per_class_by_model.items()})
    fig, ax = plt.subplots(figsize=(10, 4.5))
    df.T.plot(kind="bar", ax=ax, width=0.8)
    ax.set_ylabel("F1-score (test set)")
    ax.set_title("Per-class F1 by model")
    ax.set_ylim(0, 1)
    ax.legend(title="Class")
    plt.setp(ax.get_xticklabels(), rotation=20, ha="right")
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=130, bbox_inches="tight")
    return fig
