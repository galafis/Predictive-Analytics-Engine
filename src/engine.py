"""
Predictive Analytics Engine
===========================

Core engine module for the Predictive Analytics framework.
Provides the main PredictiveEngine class that orchestrates
machine learning workflows including data preprocessing,
model training, evaluation, and prediction.

Author: Predictive Analytics Team
Version: 1.0.0
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, regression_report
from typing import Dict, Any, Optional, Tuple, Union
import logging
from abc import ABC, abstractmethod

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BaseModel(ABC):
    """Abstract base class for all machine learning models."""
    
    @abstractmethod
    def train(self, X: np.ndarray, y: np.ndarray) -> None:
        """Train the model with given data."""
        pass
    
    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions on given data."""
        pass
    
    @abstractmethod
    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """Evaluate model performance."""
        pass


class PredictiveEngine:
    """
    Main predictive analytics engine class.
    
    This class orchestrates the entire machine learning pipeline
    from data preprocessing to model training and evaluation.
    """
    
    def __init__(self, model: Optional[BaseModel] = None):
        """
        Initialize the Predictive Engine.
        
        Args:
            model: Optional machine learning model instance
        """
        self.model = model
        self.is_trained = False
        self.feature_names = None
        self.target_name = None
        self.preprocessing_params = {}
        
        logger.info("Predictive Engine initialized")
    
    def load_data(self, data_source: Union[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Load data from file or DataFrame.
        
        Args:
            data_source: File path or pandas DataFrame
            
        Returns:
            Loaded DataFrame
        """
        if isinstance(data_source, str):
            if data_source.endswith('.csv'):
                data = pd.read_csv(data_source)
            elif data_source.endswith(('.xlsx', '.xls')):
                data = pd.read_excel(data_source)
            else:
                raise ValueError(f"Unsupported file format: {data_source}")
        elif isinstance(data_source, pd.DataFrame):
            data = data_source.copy()
        else:
            raise ValueError("Data source must be file path or DataFrame")
        
        logger.info(f"Data loaded: {data.shape[0]} rows, {data.shape[1]} columns")
        return data
    
    def preprocess_data(self, data: pd.DataFrame, 
                       target_column: str,
                       test_size: float = 0.2,
                       random_state: int = 42) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Preprocess data and split into train/test sets.
        
        Args:
            data: Input DataFrame
            target_column: Name of target variable column
            test_size: Proportion of data for testing
            random_state: Random seed for reproducibility
            
        Returns:
            X_train, X_test, y_train, y_test arrays
        """
        # Separate features and target
        X = data.drop(columns=[target_column])
        y = data[target_column]
        
        # Store metadata
        self.feature_names = X.columns.tolist()
        self.target_name = target_column
        
        # Handle missing values
        X = X.fillna(X.mean() if X.select_dtypes(include=[np.number]).shape[1] > 0 else X.mode().iloc[0])
        
        # Convert categorical variables to dummy variables
        X = pd.get_dummies(X, drop_first=True)
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X.values, y.values, test_size=test_size, random_state=random_state
        )
        
        # Store preprocessing parameters
        self.preprocessing_params = {
            'feature_names': self.feature_names,
            'target_name': self.target_name,
            'test_size': test_size,
            'random_state': random_state
        }
        
        logger.info(f"Data preprocessed: {X_train.shape[0]} training samples, {X_test.shape[0]} test samples")
        return X_train, X_test, y_train, y_test
    
    def train_model(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        """
        Train the machine learning model.
        
        Args:
            X_train: Training features
            y_train: Training targets
        """
        if self.model is None:
            raise ValueError("No model specified. Please set a model before training.")
        
        logger.info("Starting model training...")
        self.model.train(X_train, y_train)
        self.is_trained = True
        logger.info("Model training completed")
    
    def evaluate_model(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, Any]:
        """
        Evaluate the trained model.
        
        Args:
            X_test: Test features
            y_test: Test targets
            
        Returns:
            Evaluation metrics dictionary
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before evaluation")
        
        logger.info("Evaluating model...")
        metrics = self.model.evaluate(X_test, y_test)
        logger.info(f"Model evaluation completed: {metrics}")
        return metrics
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions with the trained model.
        
        Args:
            X: Input features
            
        Returns:
            Predictions array
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        predictions = self.model.predict(X)
        logger.info(f"Generated {len(predictions)} predictions")
        return predictions
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the current model and preprocessing.
        
        Returns:
            Model information dictionary
        """
        return {
            'model_type': type(self.model).__name__ if self.model else None,
            'is_trained': self.is_trained,
            'feature_names': self.feature_names,
            'target_name': self.target_name,
            'preprocessing_params': self.preprocessing_params
        }


def create_engine(model: Optional[BaseModel] = None) -> PredictiveEngine:
    """
    Factory function to create a new PredictiveEngine instance.
    
    Args:
        model: Optional machine learning model
        
    Returns:
        New PredictiveEngine instance
    """
    return PredictiveEngine(model=model)
