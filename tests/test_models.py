"""Test suite for model components used by the Predictive Analytics Engine.

Covers:
- Model factory/creation by config
- Fit/predict API compliance
- Persistence (save/load) if implemented

Run:
    pytest tests/test_models.py -v
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch

# Import model components (adjust path as needed)
try:
    from src.models import get_model, MODEL_REGISTRY
except ImportError:
    import sys
    sys.path.append('../src')
    from models import get_model, MODEL_REGISTRY


@pytest.mark.parametrize(
    "model_key, expected_cls",
    [
        ("random_forest", object),
        ("linear_regression", object),
        ("xgboost", object),
    ],
)
def test_model_factory_returns_supported_types(model_key, expected_cls):
    """Factory returns an instance for supported model keys.
    
    Expected behavior:
    - get_model does not raise for known keys
    - Returned object provides fit and predict methods
    """
    model = get_model(model_key, config={})
    assert hasattr(model, 'fit')
    assert hasattr(model, 'predict')


def test_model_factory_unknown_key_raises():
    """Unknown model keys raise a helpful error.
    
    Expected behavior:
    - Raises KeyError or ValueError with message containing available keys
    """
    with pytest.raises((KeyError, ValueError)) as exc:
        _ = get_model("nonexistent_model", config={})
    assert any(k in str(exc.value) for k in MODEL_REGISTRY.keys())


def test_model_fit_predict_roundtrip():
    """Models implement the scikit-learn fit/predict interface.
    
    Expected behavior:
    - After fit on small dataset, predict returns array of expected length
    """
    # Use a simple stub model if available; otherwise, mock a sklearn-like model
    X = np.array([[0.0], [1.0], [2.0], [3.0]])
    y = np.array([0.0, 1.0, 2.0, 3.0])

    # Prefer a lightweight model to keep tests fast
    model = get_model("linear_regression", config={})

    model.fit(X, y)
    preds = model.predict(X)

    assert isinstance(preds, (list, np.ndarray))
    assert len(preds) == len(y)


def test_model_persistence_if_available(tmp_path):
    """Model save/load cycle preserves predictions if persistence implemented.
    
    Expected behavior:
    - If model has save/load, predictions match after reload
    - Otherwise, test is skipped
    """
    model = get_model("linear_regression", config={})

    if not all(hasattr(model, attr) for attr in ("save", "load")):
        pytest.skip("Model does not implement persistence")

    X = np.array([[0], [1], [2]])
    y = np.array([0, 1, 2])

    model.fit(X, y)
    p_before = model.predict(X)

    path = tmp_path / "model.bin"
    model.save(path)

    # Create a new instance and load
    model2 = get_model("linear_regression", config={})
    model2.load(path)
    p_after = model2.predict(X)

    np.testing.assert_allclose(p_before, p_after)


def test_registry_contains_documented_models():
    """MODEL_REGISTRY contains expected keys following documentation.
    
    Expected behavior:
    - Registry keys include baseline documented types
    """
    expected = {"random_forest", "linear_regression", "xgboost"}
    assert expected.issubset(set(MODEL_REGISTRY.keys()))


def test_invalid_fit_inputs_raise_informative_error():
    """Models raise informative errors for invalid inputs.
    
    Expected behavior:
    - Passing mismatched shapes or invalid types triggers ValueError
    """
    model = get_model("linear_regression", config={})
    with pytest.raises((ValueError, TypeError)):
        model.fit(np.array([[1, 2]]), np.array([1, 2, 3]))


def test_predict_without_fit_raises():
    """Calling predict before fit raises a clear error."""
    model = get_model("linear_regression", config={})

    with pytest.raises((RuntimeError, Exception)):
        _ = model.predict(np.array([[0.0], [1.0]]))


def test_pytest_smoke_models():
    """Basic smoke test to ensure test discovery works in this module."""
    assert True
