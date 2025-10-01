"""Project configuration settings.

Provides a structured settings object for the Predictive Analytics Engine.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Any


@dataclass
class PreprocessorSettings:
    target_column: Optional[str] = None
    numeric_impute_strategy: str = "median"
    categorical_impute_strategy: str = "most_frequent"
    scale_numeric: bool = True
    one_hot_encode: bool = True
    test_size: float = 0.2
    random_state: int = 42


@dataclass
class Settings:
    base_dir: Path = field(default_factory=lambda: Path.cwd())
    data_dir: Path = field(default_factory=lambda: Path.cwd() / "data")
    models_dir: Path = field(default_factory=lambda: Path.cwd() / "models_store")

    preprocessor: PreprocessorSettings = field(default_factory=PreprocessorSettings)

    def __init__(self, config_path: Optional[str | Path] = None):
        # simple init that allows future loading from file (e.g., YAML)
        object.__setattr__(self, "base_dir", Path.cwd())
        object.__setattr__(self, "data_dir", Path.cwd() / "data")
        object.__setattr__(self, "models_dir", Path.cwd() / "models_store")
        object.__setattr__(self, "preprocessor", PreprocessorSettings())
        if config_path:
            # Placeholder for future: load overrides from a config file
            pass
