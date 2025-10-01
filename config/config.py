"""
Configuration Management
=======================

Configuration module for the Predictive Analytics Engine.
Provides centralized configuration management including
model parameters, data processing settings, logging configuration,
and environment-specific settings.

Author: Predictive Analytics Team
Version: 1.0.0
"""

import os
import json
from typing import Dict, Any, Optional, Union
from dataclasses import dataclass, asdict
from pathlib import Path
import logging


@dataclass
class ModelConfig:
    """Configuration for machine learning models."""
    model_type: str = "random_forest"
    random_state: int = 42
    test_size: float = 0.2
    validation_size: float = 0.2
    cross_validation_folds: int = 5
    max_features: Optional[Union[str, int, float]] = "auto"
    n_estimators: int = 100
    max_depth: Optional[int] = None
    min_samples_split: int = 2
    min_samples_leaf: int = 1
    bootstrap: bool = True
    oob_score: bool = False
    n_jobs: Optional[int] = -1
    verbose: int = 0


@dataclass
class DataConfig:
    """Configuration for data processing."""
    data_path: str = "data/"
    output_path: str = "output/"
    models_path: str = "models/"
    logs_path: str = "logs/"
    encoding: str = "utf-8"
    separator: str = ","
    decimal: str = "."
    date_format: str = "%Y-%m-%d"
    missing_value_strategy: str = "mean"  # mean, median, mode, drop, interpolate
    outlier_detection: bool = True
    outlier_method: str = "iqr"  # iqr, zscore, isolation_forest
    feature_scaling: str = "standard"  # standard, minmax, robust, none
    categorical_encoding: str = "onehot"  # onehot, label, target, binary


@dataclass
class LoggingConfig:
    """Configuration for logging."""
    level: str = "INFO"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    date_format: str = "%Y-%m-%d %H:%M:%S"
    file_logging: bool = True
    console_logging: bool = True
    log_file: str = "predictive_analytics.log"
    max_log_size: int = 10485760  # 10MB
    backup_count: int = 5


@dataclass
class PerformanceConfig:
    """Configuration for performance optimization."""
    use_multiprocessing: bool = True
    n_jobs: int = -1
    batch_size: int = 1000
    memory_limit: str = "2GB"
    cache_size: int = 128
    enable_gpu: bool = False
    gpu_memory_fraction: float = 0.8


class Config:
    """
    Main configuration class that manages all configuration aspects.
    
    This class provides a centralized way to manage configuration
    settings for the entire Predictive Analytics Engine.
    """
    
    def __init__(self, config_file: Optional[str] = None):
        """
        Initialize configuration.
        
        Args:
            config_file: Optional path to configuration file
        """
        self.model = ModelConfig()
        self.data = DataConfig()
        self.logging = LoggingConfig()
        self.performance = PerformanceConfig()
        
        self._config_file = config_file
        self._environment = os.getenv("ENVIRONMENT", "development")
        
        if config_file and os.path.exists(config_file):
            self.load_from_file(config_file)
        
        self._setup_paths()
        self._setup_logging()
    
    def _setup_paths(self) -> None:
        """Create necessary directories if they don't exist."""
        paths = [
            self.data.data_path,
            self.data.output_path,
            self.data.models_path,
            self.data.logs_path
        ]
        
        for path in paths:
            Path(path).mkdir(parents=True, exist_ok=True)
    
    def _setup_logging(self) -> None:
        """Setup logging configuration."""
        log_level = getattr(logging, self.logging.level.upper())
        
        # Create formatter
        formatter = logging.Formatter(
            self.logging.format,
            datefmt=self.logging.date_format
        )
        
        # Setup root logger
        logger = logging.getLogger()
        logger.setLevel(log_level)
        
        # Clear existing handlers
        logger.handlers.clear()
        
        # Console handler
        if self.logging.console_logging:
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(formatter)
            logger.addHandler(console_handler)
        
        # File handler
        if self.logging.file_logging:
            log_file_path = Path(self.data.logs_path) / self.logging.log_file
            from logging.handlers import RotatingFileHandler
            
            file_handler = RotatingFileHandler(
                log_file_path,
                maxBytes=self.logging.max_log_size,
                backupCount=self.logging.backup_count
            )
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
    
    def load_from_file(self, config_file: str) -> None:
        """
        Load configuration from JSON file.
        
        Args:
            config_file: Path to configuration file
        """
        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                config_data = json.load(f)
            
            # Update configuration sections
            if 'model' in config_data:
                self.model = ModelConfig(**config_data['model'])
            
            if 'data' in config_data:
                self.data = DataConfig(**config_data['data'])
            
            if 'logging' in config_data:
                self.logging = LoggingConfig(**config_data['logging'])
            
            if 'performance' in config_data:
                self.performance = PerformanceConfig(**config_data['performance'])
            
            logging.info(f"Configuration loaded from {config_file}")
        
        except Exception as e:
            logging.warning(f"Failed to load configuration from {config_file}: {e}")
    
    def save_to_file(self, config_file: Optional[str] = None) -> None:
        """
        Save current configuration to JSON file.
        
        Args:
            config_file: Optional path to save configuration (uses default if None)
        """
        if config_file is None:
            config_file = self._config_file or "config.json"
        
        config_data = {
            'model': asdict(self.model),
            'data': asdict(self.data),
            'logging': asdict(self.logging),
            'performance': asdict(self.performance),
            'environment': self._environment
        }
        
        try:
            with open(config_file, 'w', encoding='utf-8') as f:
                json.dump(config_data, f, indent=2, ensure_ascii=False)
            
            logging.info(f"Configuration saved to {config_file}")
        
        except Exception as e:
            logging.error(f"Failed to save configuration to {config_file}: {e}")
    
    def get_model_params(self) -> Dict[str, Any]:
        """
        Get model parameters as dictionary.
        
        Returns:
            Model parameters dictionary
        """
        params = asdict(self.model)
        # Remove non-sklearn parameters
        sklearn_params = {
            k: v for k, v in params.items() 
            if k not in ['model_type', 'test_size', 'validation_size', 'cross_validation_folds']
        }
        return sklearn_params
    
    def update_config(self, section: str, **kwargs) -> None:
        """
        Update configuration section with new values.
        
        Args:
            section: Configuration section name (model, data, logging, performance)
            **kwargs: Configuration parameters to update
        """
        if section == 'model':
            for key, value in kwargs.items():
                if hasattr(self.model, key):
                    setattr(self.model, key, value)
        
        elif section == 'data':
            for key, value in kwargs.items():
                if hasattr(self.data, key):
                    setattr(self.data, key, value)
        
        elif section == 'logging':
            for key, value in kwargs.items():
                if hasattr(self.logging, key):
                    setattr(self.logging, key, value)
            self._setup_logging()  # Re-setup logging with new config
        
        elif section == 'performance':
            for key, value in kwargs.items():
                if hasattr(self.performance, key):
                    setattr(self.performance, key, value)
        
        else:
            raise ValueError(f"Unknown configuration section: {section}")
        
        logging.info(f"Configuration section '{section}' updated")
    
    def get_environment(self) -> str:
        """
        Get current environment.
        
        Returns:
            Current environment name
        """
        return self._environment
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert configuration to dictionary.
        
        Returns:
            Configuration as dictionary
        """
        return {
            'model': asdict(self.model),
            'data': asdict(self.data),
            'logging': asdict(self.logging),
            'performance': asdict(self.performance),
            'environment': self._environment
        }


# Global configuration instance
_config_instance = None


def get_config(config_file: Optional[str] = None) -> Config:
    """
    Get global configuration instance.
    
    Args:
        config_file: Optional configuration file path
        
    Returns:
        Global Config instance
    """
    global _config_instance
    
    if _config_instance is None:
        _config_instance = Config(config_file)
    
    return _config_instance


def set_config(config: Config) -> None:
    """
    Set global configuration instance.
    
    Args:
        config: Config instance to set as global
    """
    global _config_instance
    _config_instance = config


def create_default_config_file(file_path: str = "config.json") -> None:
    """
    Create a default configuration file.
    
    Args:
        file_path: Path where to create the configuration file
    """
    config = Config()
    config.save_to_file(file_path)
    logging.info(f"Default configuration file created at {file_path}")
