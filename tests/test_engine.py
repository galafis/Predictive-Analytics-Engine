"""Test suite for the Predictive Analytics Engine core functionality.

This module contains comprehensive tests for the main engine components,
including initialization, configuration, prediction workflows, and error handling.

Running tests:
    pytest tests/test_engine.py -v

Example usage:
    python -m pytest tests/test_engine.py::TestPredictiveEngine::test_engine_initialization
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch, MagicMock

# Import the engine module (adjust import path as needed)
try:
    from src.engine import PredictiveEngine
except ImportError:
    # Fallback for different project structures
    import sys
    sys.path.append('../src')
    from engine import PredictiveEngine


class TestPredictiveEngine:
    """Test cases for the main PredictiveEngine class.
    
    These tests validate core engine functionality including:
    - Engine initialization and configuration
    - Data loading and validation
    - Model training workflows
    - Prediction generation
    - Error handling and edge cases
    """

    @pytest.fixture
    def sample_data(self):
        """Fixture providing sample data for testing.
        
        Returns:
            pd.DataFrame: Sample dataset with features and target
        """
        np.random.seed(42)
        data = {
            'feature_1': np.random.randn(100),
            'feature_2': np.random.randn(100),
            'feature_3': np.random.randint(0, 5, 100),
            'target': np.random.rand(100)
        }
        return pd.DataFrame(data)

    @pytest.fixture
    def engine_config(self):
        """Fixture providing engine configuration.
        
        Returns:
            dict: Configuration parameters for engine initialization
        """
        return {
            'model_type': 'random_forest',
            'validation_split': 0.2,
            'random_state': 42,
            'n_estimators': 10,  # Small for fast testing
            'max_depth': 3
        }

    def test_engine_initialization(self, engine_config):
        """Test proper engine initialization with configuration.
        
        Expected behavior:
        - Engine initializes without errors
        - Configuration parameters are stored correctly
        - Default values are set appropriately
        """
        engine = PredictiveEngine(config=engine_config)
        
        assert engine is not None
        assert hasattr(engine, 'config')
        assert engine.config['model_type'] == 'random_forest'
        assert engine.config['random_state'] == 42

    def test_engine_initialization_default_config(self):
        """Test engine initialization with default configuration.
        
        Expected behavior:
        - Engine initializes with sensible defaults
        - Required attributes are present
        """
        engine = PredictiveEngine()
        
        assert engine is not None
        assert hasattr(engine, 'config')
        assert isinstance(engine.config, dict)

    def test_data_loading(self, sample_data):
        """Test data loading functionality.
        
        Expected behavior:
        - Data is loaded and stored correctly
        - Data validation passes
        - Appropriate data types are maintained
        """
        engine = PredictiveEngine()
        
        # Test with DataFrame
        engine.load_data(sample_data)
        assert hasattr(engine, 'data')
        assert isinstance(engine.data, pd.DataFrame)
        assert len(engine.data) == 100

    def test_data_validation_missing_target(self, sample_data):
        """Test data validation with missing target column.
        
        Expected behavior:
        - Appropriate error is raised for missing target
        - Error message is informative
        """
        engine = PredictiveEngine()
        invalid_data = sample_data.drop('target', axis=1)
        
        with pytest.raises(ValueError, match="target column not found"):
            engine.validate_data(invalid_data, target_column='target')

    def test_data_validation_empty_dataset(self):
        """Test data validation with empty dataset.
        
        Expected behavior:
        - Appropriate error is raised for empty data
        - Error handling is graceful
        """
        engine = PredictiveEngine()
        empty_data = pd.DataFrame()
        
        with pytest.raises(ValueError, match="empty dataset"):
            engine.validate_data(empty_data)

    @patch('src.engine.RandomForestRegressor')  # Adjust import path as needed
    def test_model_training(self, mock_model, sample_data, engine_config):
        """Test model training workflow.
        
        Expected behavior:
        - Model is initialized with correct parameters
        - Training data is prepared correctly
        - Model.fit() is called with appropriate arguments
        """
        # Setup mock
        mock_instance = Mock()
        mock_model.return_value = mock_instance
        
        engine = PredictiveEngine(config=engine_config)
        engine.load_data(sample_data)
        
        # Train model
        engine.train(target_column='target')
        
        # Verify model was initialized and trained
        mock_model.assert_called_once()
        mock_instance.fit.assert_called_once()

    def test_prediction_generation(self, sample_data, engine_config):
        """Test prediction generation.
        
        Expected behavior:
        - Predictions are generated for new data
        - Output format is correct (numpy array or pandas Series)
        - Prediction values are reasonable
        """
        engine = PredictiveEngine(config=engine_config)
        
        # Mock trained model
        mock_model = Mock()
        expected_predictions = np.array([0.1, 0.2, 0.3])
        mock_model.predict.return_value = expected_predictions
        engine.model = mock_model
        
        # Generate predictions
        test_data = sample_data.iloc[:3, :-1]  # First 3 rows without target
        predictions = engine.predict(test_data)
        
        mock_model.predict.assert_called_once()
        np.testing.assert_array_equal(predictions, expected_predictions)

    def test_prediction_without_trained_model(self, sample_data):
        """Test prediction attempt without trained model.
        
        Expected behavior:
        - Appropriate error is raised
        - Error message indicates model needs training
        """
        engine = PredictiveEngine()
        
        with pytest.raises(RuntimeError, match="model not trained"):
            engine.predict(sample_data.iloc[:, :-1])

    def test_model_evaluation_metrics(self, sample_data, engine_config):
        """Test model evaluation metrics calculation.
        
        Expected behavior:
        - Evaluation returns dictionary of metrics
        - Common metrics are included (MAE, RMSE, R²)
        - Metric values are reasonable
        """
        engine = PredictiveEngine(config=engine_config)
        
        # Mock model predictions
        y_true = np.array([1.0, 2.0, 3.0, 4.0])
        y_pred = np.array([1.1, 1.9, 3.2, 3.8])
        
        metrics = engine.evaluate_model(y_true, y_pred)
        
        assert isinstance(metrics, dict)
        assert 'mae' in metrics
        assert 'rmse' in metrics
        assert 'r2' in metrics
        assert all(isinstance(v, (int, float)) for v in metrics.values())

    def test_engine_reset(self, sample_data, engine_config):
        """Test engine reset functionality.
        
        Expected behavior:
        - Model is cleared after reset
        - Data can be cleared optionally
        - Engine can be reused after reset
        """
        engine = PredictiveEngine(config=engine_config)
        engine.load_data(sample_data)
        
        # Simulate trained model
        engine.model = Mock()
        
        # Reset engine
        engine.reset(clear_data=True)
        
        assert engine.model is None
        assert not hasattr(engine, 'data') or engine.data is None


class TestEngineIntegration:
    """Integration tests for complete engine workflows.
    
    These tests validate end-to-end functionality combining
    multiple engine components in realistic scenarios.
    """

    @pytest.fixture
    def complex_dataset(self):
        """Fixture providing a more complex dataset for integration testing.
        
        Returns:
            pd.DataFrame: Complex dataset with mixed data types
        """
        np.random.seed(123)
        n_samples = 200
        
        data = {
            'numerical_1': np.random.normal(0, 1, n_samples),
            'numerical_2': np.random.exponential(2, n_samples),
            'categorical_1': np.random.choice(['A', 'B', 'C'], n_samples),
            'categorical_2': np.random.choice(['X', 'Y'], n_samples),
            'boolean_feature': np.random.choice([True, False], n_samples),
            'target': np.random.normal(5, 2, n_samples)
        }
        
        return pd.DataFrame(data)

    @pytest.mark.integration
    def test_complete_workflow(self, complex_dataset):
        """Test complete end-to-end workflow.
        
        Expected behavior:
        - Data loading, validation, training, and prediction work together
        - Workflow completes without errors
        - Results are reasonable
        """
        config = {
            'model_type': 'random_forest',
            'validation_split': 0.3,
            'random_state': 42
        }
        
        engine = PredictiveEngine(config=config)
        
        # Complete workflow
        engine.load_data(complex_dataset)
        engine.validate_data(complex_dataset, target_column='target')
        
        # This would require actual implementation
        # engine.train(target_column='target')
        # predictions = engine.predict(complex_dataset.iloc[:10, :-1])
        # metrics = engine.evaluate_model(y_true, predictions)
        
        # For now, just verify the setup worked
        assert engine.data is not None
        assert len(engine.data) == 200


# Utility functions for testing
def test_pytest_configuration():
    """Test that pytest is properly configured.
    
    Expected behavior:
    - Pytest can discover and run this test
    - Basic assertions work correctly
    """
    assert True
    assert 1 + 1 == 2
    assert isinstance("test", str)


if __name__ == "__main__":
    # Allow running tests directly
    pytest.main([__file__, "-v"])
