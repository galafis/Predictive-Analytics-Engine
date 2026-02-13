"""Model implementations for the Predictive Analytics Engine."""

from .base_model import BaseModel
from .classification import ClassificationModel
from .regression import RegressionModel

MODEL_REGISTRY = {
    "classification": ClassificationModel,
    "logistic_regression": ClassificationModel,
    "regression": RegressionModel,
    "linear_regression": RegressionModel,
}


def get_model(model_type: str, **kwargs) -> BaseModel:
    """Factory function to create a model by type name.

    Args:
        model_type: Key from MODEL_REGISTRY (e.g. 'classification', 'regression')
        **kwargs: Passed to the model constructor

    Returns:
        An instance of the requested model

    Raises:
        ValueError: If model_type is not in MODEL_REGISTRY
    """
    if model_type not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model type '{model_type}'. "
            f"Available: {list(MODEL_REGISTRY.keys())}"
        )
    return MODEL_REGISTRY[model_type](**kwargs)
