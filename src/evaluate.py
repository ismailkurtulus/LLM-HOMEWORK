"""Evaluation utilities for intent classification."""

import os
from pathlib import Path
from typing import Dict, Iterable

# Use a writable matplotlib cache directory inside project workspace.
os.environ.setdefault("MPLCONFIGDIR", str(Path(".tmp") / "matplotlib"))

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)

from src.utils import save_json


def compute_metrics(y_true: Iterable[int], y_pred: Iterable[int]) -> Dict[str, float]:
    """Compute standard classification metrics."""
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_weighted": float(precision_score(y_true, y_pred, average="weighted", zero_division=0)),
        "recall_weighted": float(recall_score(y_true, y_pred, average="weighted", zero_division=0)),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
    }


def save_confusion_matrix(
    y_true: Iterable[int],
    y_pred: Iterable[int],
    labels: Iterable[str],
    output_path: Path,
    title: str,
) -> None:
    """Save a confusion matrix image for the given predictions."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 6))
    ConfusionMatrixDisplay.from_predictions(
        np.array(list(y_true)),
        np.array(list(y_pred)),
        display_labels=list(labels),
        cmap="Blues",
        xticks_rotation=35,
        ax=ax,
        colorbar=False,
    )
    ax.set_title(title)
    plt.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def save_metrics_report(metrics: Dict[str, float], output_path: Path) -> None:
    """Persist metrics as JSON."""
    save_json(metrics, output_path)
