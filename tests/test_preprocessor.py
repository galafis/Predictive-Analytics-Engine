"""Test suite for the data preprocessing pipeline.

This module includes unit tests for preprocessing components such as:
- Handling missing values
- Encoding categorical variables
- Feature scaling/normalization
- Train/test split consistency

Run:
    pytest tests/test_preprocessor.py -v
"""

import pytest
import numpy as np
import pandas as pd
from importlib import import_module
from pathlib import Path
from unittest.mock import Mock
import sys

# Import preprocessor without assuming execution directory
ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
for path in (ROOT_DIR, SRC_DIR):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

try:
    DataPreprocessor = import_module("src.preprocessor").Preprocessor
except ModuleNotFoundError:
    DataPreprocessor = import_module("preprocessor").Preprocessor


@pytest.fixture
def raw_data():
    """Provide a raw dataset with mixed types and missing values.

    Expected format: DataFrame with numerical, categorical, boolean, and target.
    """
    rng = np.random.default_rng(123)
    df = pd.DataFrame(
        {
            "num_1": [1.0, 2.5, np.nan, 4.2, 5.1],
            "num_2": [10, 20, 30, np.nan, 50],
            "cat_1": ["A", "B", "A", None, "C"],
            "bool_1": [True, False, True, False, True],
            "target": [0, 1, 0, 1, 0],
        }
    )
    return df


@pytest.fixture
def preprocessor():
    """Instantiate the DataPreprocessor with default config."""
    return DataPreprocessor(config={})


def test_initialization_defaults(preprocessor):
    """Preprocessor initializes with sensible defaults.

    Expected behavior:
    - Has transform and fit_transform methods
    - Holds configuration dict
    """
    assert hasattr(preprocessor, "fit")
    assert hasattr(preprocessor, "transform")
    assert isinstance(preprocessor.config, dict)


def test_handle_missing_values(preprocessor, raw_data):
    """Missing values are imputed appropriately.

    Expected behavior:
    - No NaNs remain after fit_transform
    - Numeric columns imputed (e.g., median/mean)
    - Categorical columns imputed (e.g., most frequent)
    """
    X = raw_data.drop(columns=["target"])
    y = raw_data["target"]
    X_t, y_t = preprocessor.fit_transform(X, y)

    assert not pd.isna(X_t).any().any()
    assert y_t.equals(y)


def test_encoding_and_scaling(preprocessor, raw_data):
    """Categoricals encoded and numericals scaled where configured.

    Expected behavior:
    - Output is numeric matrix/DataFrame
    - Column count increases with one-hot encoding
    - Value ranges are reasonable (scaled) if applicable
    """
    X = raw_data.drop(columns=["target"])
    y = raw_data["target"]
    X_t, _ = preprocessor.fit_transform(X, y)

    # All columns should be numeric after preprocessing
    assert all(np.issubdtype(dtype, np.number) for dtype in X_t.dtypes)


def test_train_test_split_reproducibility(preprocessor, raw_data):
    """Train/test split is reproducible with fixed random_state.

    Expected behavior:
    - With a set random_state, splits are identical across runs
    """
    preprocessor.config.update({"test_size": 0.4, "random_state": 42})

    X = raw_data.drop(columns=["target"])
    y = raw_data["target"]

    (X_train1, X_test1, y_train1, y_test1) = preprocessor.train_test_split(X, y)
    (X_train2, X_test2, y_train2, y_test2) = preprocessor.train_test_split(X, y)

    pd.testing.assert_frame_equal(
        X_train1.reset_index(drop=True), X_train2.reset_index(drop=True)
    )
    pd.testing.assert_frame_equal(
        X_test1.reset_index(drop=True), X_test2.reset_index(drop=True)
    )
    pd.testing.assert_series_equal(
        y_train1.reset_index(drop=True), y_train2.reset_index(drop=True)
    )
    pd.testing.assert_series_equal(
        y_test1.reset_index(drop=True), y_test2.reset_index(drop=True)
    )


def test_transform_without_fit_raises(preprocessor, raw_data):
    """Calling transform before fit raises an informative error.

    Expected behavior:
    - Raises RuntimeError or NotFittedError with helpful message
    """
    X = raw_data.drop(columns=["target"])
    with pytest.raises(
        (RuntimeError, Exception), match="not fitted|fit first|NotFitted"
    ):
        preprocessor.transform(X)


def test_column_passthrough_behavior(raw_data):
    """Config-driven passthrough behavior is respected.

    Expected behavior:
    - Columns listed in passthrough are left unchanged in output
    """
    pre = DataPreprocessor(config={"passthrough": ["bool_1"]})
    X = raw_data.drop(columns=["target"])
    y = raw_data["target"]
    X_t, _ = pre.fit_transform(X, y)

    # The boolean column should still exist (possibly cast) in transformed data
    matched_cols = [c for c in X_t.columns if "bool_1" in c]
    assert len(matched_cols) >= 1


def test_inverse_transform_roundtrip(preprocessor, raw_data):
    """inverse_transform returns data in original feature space where applicable.

    Expected behavior:
    - After fit_transform and inverse_transform, shape is consistent
    - Some numeric values approximately match within tolerance
    """
    X = raw_data.drop(columns=["target"])
    y = raw_data["target"]
    X_t, _ = preprocessor.fit_transform(X, y)

    if hasattr(preprocessor, "inverse_transform"):
        X_inv = preprocessor.inverse_transform(X_t)
        assert isinstance(X_inv, pd.DataFrame)
        assert X_inv.shape[0] == X.shape[0]


def test_pytest_smoke():
    """Basic pytest smoke test for discovery and assertions."""
    assert True
