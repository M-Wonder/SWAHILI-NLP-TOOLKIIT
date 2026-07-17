"""
Evaluation utilities for reporting classifier performance.
"""

from __future__ import annotations

import logging
from pathlib import Path

from sklearn.metrics import classification_report, confusion_matrix

logger = logging.getLogger(__name__)


def generate_report(y_true: list[str], y_pred: list[str]) -> str:
    """Return a text classification report (precision/recall/F1 per class)."""
    return classification_report(y_true, y_pred, digits=3, zero_division=0)


def plot_confusion_matrix(
    y_true: list[str],
    y_pred: list[str],
    labels: list[str],
    output_path: str | Path,
) -> Path:
    """Render and save a confusion matrix heatmap to ``output_path``.

    Matplotlib is imported lazily inside this function so that importing
    :mod:`swahili_nlp.evaluate` does not require a display backend or pull
    in plotting dependencies for callers that only need text metrics.
    """
    import matplotlib

    matplotlib.use("Agg")  # non-interactive backend, safe for headless runs
    import matplotlib.pyplot as plt

    cm = confusion_matrix(y_true, y_pred, labels=labels)
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted category")
    ax.set_ylabel("True category")
    ax.set_title("Swahili News Classifier — Confusion Matrix")

    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(
                j, i, str(cm[i, j]),
                ha="center", va="center",
                color="white" if cm[i, j] > cm.max() / 2 else "black",
            )

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    logger.info("Confusion matrix saved to %s", output_path)
    return output_path
