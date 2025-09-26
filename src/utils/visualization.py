"""Visualization utilities for the analytics engine."""
from __future__ import annotations

from typing import Any, Optional
import matplotlib.pyplot as plt
import seaborn as sns


class Visualizer:
    """Create plots for data and predictions."""

    def __init__(self, config: Optional[Any] = None) -> None:
        self.config = config
        sns.set(style="whitegrid")

    def plot_predictions(self, X: Any, y_pred: Any, y_true: Optional[Any] = None, title: str = "Predictions") -> None:
        plt.figure(figsize=(8, 5))
        if y_true is not None:
            plt.scatter(range(len(y_true)), y_true, label="True", alpha=0.6)
        plt.scatter(range(len(y_pred)), y_pred, label="Pred", alpha=0.6)
        plt.title(title)
        plt.legend()
        plt.tight_layout()
        plt.show()
