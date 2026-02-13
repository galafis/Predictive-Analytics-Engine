"""
Testes funcionais para Predictive-Analytics-Engine.
"""

import os
import sys
import tempfile

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from main import PredictiveAnalyticsAnalyzer, main
from src.models import get_model, MODEL_REGISTRY, ClassificationModel, RegressionModel
from src.models.base_model import BaseModel
from src.preprocessor import Preprocessor, PreprocessorConfig
from src.data_loader import DataLoader
from src.utils.metrics import ModelMetrics


# ---------------------------------------------------------------------------
# main.py — PredictiveAnalyticsAnalyzer
# ---------------------------------------------------------------------------

class TestPredictiveAnalyticsAnalyzer:

    def test_init(self):
        a = PredictiveAnalyticsAnalyzer()
        assert a.data is None
        assert a.model is None
        assert a.results == {}

    def test_load_data_synthetic(self):
        a = PredictiveAnalyticsAnalyzer()
        df = a.load_data()
        assert isinstance(df, pd.DataFrame)
        assert df.shape == (1000, 4)
        assert set(df['target'].unique()).issubset({0, 1})

    def test_load_data_custom(self):
        a = PredictiveAnalyticsAnalyzer()
        custom = pd.DataFrame({'feature1': [1, 2], 'feature2': [3, 4],
                               'feature3': [5, 6], 'target': [0, 1]})
        a.load_data(data=custom)
        assert a.data.shape == (2, 4)

    def test_analyze(self):
        a = PredictiveAnalyticsAnalyzer()
        a.load_data()
        r = a.analyze()
        assert 'accuracy' in r
        assert 'classification_report' in r
        assert 0.0 <= r['accuracy'] <= 1.0
        assert r['accuracy'] > 0.6

    def test_visualize(self):
        a = PredictiveAnalyticsAnalyzer()
        a.load_data()
        a.analyze()
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, 'out.png')
            a.visualize(output_path=p)
            assert os.path.isfile(p)
            assert os.path.getsize(p) > 0

    def test_main_function(self):
        a = main()
        assert isinstance(a, PredictiveAnalyticsAnalyzer)
        assert a.model is not None
        assert 'accuracy' in a.results


# ---------------------------------------------------------------------------
# src/models — get_model + MODEL_REGISTRY
# ---------------------------------------------------------------------------

class TestModels:

    def test_registry_has_classification(self):
        assert 'classification' in MODEL_REGISTRY

    def test_registry_has_regression(self):
        assert 'regression' in MODEL_REGISTRY

    def test_get_classification_model(self):
        m = get_model('classification')
        assert isinstance(m, ClassificationModel)
        assert isinstance(m, BaseModel)

    def test_get_regression_model(self):
        m = get_model('regression')
        assert isinstance(m, RegressionModel)
        assert isinstance(m, BaseModel)

    def test_get_model_unknown_raises(self):
        with pytest.raises(ValueError, match="Unknown model type"):
            get_model('xgboost')

    def test_classification_fit_predict(self):
        np.random.seed(0)
        X = np.random.randn(100, 3)
        y = (X[:, 0] > 0).astype(int)
        m = get_model('classification')
        m.fit(X, y=y)
        preds = m.predict(X[:5])
        assert len(preds) == 5
        assert all(p in [0, 1] for p in preds)

    def test_regression_fit_predict(self):
        np.random.seed(0)
        X = np.random.randn(100, 3)
        y = X[:, 0] * 2.0 + 1.0
        m = get_model('regression')
        m.fit(X, y=y)
        preds = m.predict(X[:5])
        assert len(preds) == 5

    def test_classification_score(self):
        np.random.seed(0)
        X = np.random.randn(200, 3)
        y = (X[:, 0] > 0).astype(int)
        m = get_model('classification')
        m.fit(X, y=y)
        score = m.score(X, y)
        assert 0.0 <= score <= 1.0

    def test_model_save_load(self):
        np.random.seed(0)
        X = np.random.randn(50, 2)
        y = (X[:, 0] > 0).astype(int)
        m = get_model('classification')
        m.fit(X, y=y)
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, 'model.pkl')
            m.save(path)
            loaded = ClassificationModel.load(path)
            assert isinstance(loaded, ClassificationModel)
            preds = loaded.predict(X[:3])
            assert len(preds) == 3


# ---------------------------------------------------------------------------
# src/preprocessor
# ---------------------------------------------------------------------------

class TestPreprocessor:

    def _sample_df(self):
        return pd.DataFrame({
            'num1': [1.0, 2.0, np.nan, 4.0, 5.0],
            'num2': [10.0, 20.0, 30.0, 40.0, 50.0],
            'cat1': ['a', 'b', 'a', 'c', 'b'],
            'target': [0, 1, 0, 1, 0],
        })

    def test_transform_returns_dict(self):
        p = Preprocessor()
        df = self._sample_df()
        result = p.transform(df, target='target')
        assert 'X' in result
        assert 'y' in result
        assert 'feature_names' in result

    def test_transform_separates_target(self):
        p = Preprocessor()
        df = self._sample_df()
        result = p.transform(df, target='target')
        assert result['y'] is not None
        assert len(result['y']) == 5

    def test_transform_handles_missing(self):
        p = Preprocessor()
        df = self._sample_df()
        result = p.transform(df, target='target')
        assert not np.isnan(result['X']).any()

    def test_split(self):
        p = Preprocessor()
        X = np.random.randn(100, 3)
        y = pd.Series(np.random.choice([0, 1], 100))
        parts = p.split(X, y)
        assert len(parts) == 4
        assert parts[0].shape[0] == 80
        assert parts[1].shape[0] == 20


# ---------------------------------------------------------------------------
# src/data_loader
# ---------------------------------------------------------------------------

class TestDataLoader:

    def test_init(self):
        dl = DataLoader()
        assert dl._data_cache == {}

    def test_load_csv(self):
        dl = DataLoader()
        with tempfile.NamedTemporaryFile(suffix='.csv', mode='w', delete=False) as f:
            f.write("a,b,c\n1,2,3\n4,5,6\n")
            f.flush()
            path = f.name
        try:
            df = dl.load(path)
            assert df.shape == (2, 3)
        finally:
            os.unlink(path)

    def test_load_caching(self):
        dl = DataLoader()
        with tempfile.NamedTemporaryFile(suffix='.csv', mode='w', delete=False) as f:
            f.write("x,y\n1,2\n")
            f.flush()
            path = f.name
        try:
            df1 = dl.load(path)
            df2 = dl.load(path)
            pd.testing.assert_frame_equal(df1, df2)
        finally:
            os.unlink(path)

    def test_clear_cache(self):
        dl = DataLoader()
        dl._data_cache['key'] = 'val'
        dl.clear_cache()
        assert len(dl._data_cache) == 0

    def test_validate_data(self):
        dl = DataLoader()
        df = pd.DataFrame({'a': [1, 2]})
        assert dl.validate_data(df, required_columns=['a']) is True

    def test_validate_data_missing_col(self):
        dl = DataLoader()
        df = pd.DataFrame({'a': [1]})
        with pytest.raises(ValueError, match="Missing required columns"):
            dl.validate_data(df, required_columns=['b'])

    def test_supported_formats(self):
        dl = DataLoader()
        fmts = dl.list_supported_formats()
        assert '.csv' in fmts
        assert '.parquet' in fmts


# ---------------------------------------------------------------------------
# src/utils/metrics
# ---------------------------------------------------------------------------

class TestMetrics:

    def test_classification_metrics(self):
        mm = ModelMetrics()
        y_true = [0, 0, 1, 1]
        y_pred = [0, 1, 1, 1]
        r = mm.calculate_metrics(y_true, y_pred, task='classification')
        assert 'accuracy' in r
        assert 'precision' in r
        assert 'recall' in r
        assert 'f1' in r
        assert r['accuracy'] == 0.75

    def test_regression_metrics(self):
        mm = ModelMetrics()
        y_true = [1.0, 2.0, 3.0]
        y_pred = [1.1, 2.1, 2.9]
        r = mm.calculate_metrics(y_true, y_pred, task='regression')
        assert 'rmse' in r
        assert 'r2' in r
        assert r['rmse'] < 0.2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
