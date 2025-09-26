"""Regression Model Implementations

Provides a simple wrapper around scikit-learn regressors with a unified
interface compatible with BaseModel.
"""
from __future__ import annotations

from typing import Any, Optional
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

from .base_model import BaseModel


class RegressionModel(BaseModel):
    """Default regression model using LinearRegression."""

    def __init__(self, model: Optional[Any] = None) -> None:
        self.model = model or LinearRegression()

    def fit(self, data: Any, y: Optional[Any] = None, **kwargs) -> "RegressionModel":
        if isinstance(data, dict) and "X" in data and "y" in data:
            X, y = data["X"], data["y"]
        else:
            X = data
        if y is None:
            raise ValueError("Target y must be provided for regression")
        self.model.fit(X, y)
        return self

    def predict(self, data: Any, **kwargs) -> Any:
        X = data["X"] if isinstance(data, dict) and "X" in data else data
        return self.model.predict(X)

    def score(self, X: Any, y: Any) -> float:
        preds = self.model.predict(X)
        return float(r2_score(y, preds))
