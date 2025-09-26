"""Model metrics utilities."""
from __future__ import annotations

from typing import Dict, Any
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, mean_squared_error, r2_score
import numpy as np


class ModelMetrics:
    """Compute common classification and regression metrics."""

    def calculate_metrics(self, y_true: Any, y_pred: Any, task: str | None = None) -> Dict[str, float]:
        if task is None:
            # infer task by dtype/shape
            task = "regression" if _is_regression(y_true) else "classification"
        if task == "classification":
            return {
                "accuracy": float(accuracy_score(y_true, y_pred)),
                "precision": float(precision_score(y_true, y_pred, average="weighted", zero_division=0)),
                "recall": float(recall_score(y_true, y_pred, average="weighted", zero_division=0)),
                "f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
            }
        elif task == "regression":
            return {
                "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
                "r2": float(r2_score(y_true, y_pred)),
            }
        else:
            raise ValueError(f"Unknown task type: {task}")


def _is_regression(y_true: Any) -> bool:
    try:
        y = np.asarray(y_true)
        return np.issubdtype(y.dtype, np.floating) or np.issubdtype(y.dtype, np.integer) and len(np.unique(y)) > 20
    except Exception:
        return False
