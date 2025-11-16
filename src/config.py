"""
Configuration management for FakeScope.
Centralized settings for data paths, model parameters, and runtime config.
"""
from dataclasses import dataclass, field
from typing import List, Dict, Any
from pathlib import Path
import os


@dataclass
class DataConfig:
    """Data loading and preprocessing configuration."""
    raw_data_paths: List[str] = field(default_factory=lambda: [
        "./datasets/input/alt/News.csv",
        "./datasets/input/alt 2/New Task.csv"
    ])
    output_path: str = "./datasets/input/alt/fake_news_total.csv"
    encoding: str = "latin1"
    test_size: float = 0.25
    random_state: int = 42
    detect_duplicates: bool = True


@dataclass
class PreprocessingConfig:
    """Text preprocessing configuration."""
    min_token_length: int = 3
    remove_stopwords: bool = True
    custom_stopwords: List[str] = field(default_factory=lambda: [
        'reuters', 'ap', 'reporting', 'editing', 'featured', 'image',
        'https', 'twitter', 'com', 'getty', 'monday', 'tuesday',
        'wednesday', 'thursday', 'friday', 'saturday', 'sunday'
    ])


@dataclass
class TFIDFConfig:
    """TF-IDF vectorization configuration."""
    max_features: int = 5000
    min_df: int = 5
    max_df: float = 0.90
    ngram_range: tuple = (1, 2)


@dataclass
class ModelConfig:
    """Model training configuration."""
    # Logistic Regression
    lr_params: Dict[str, Any] = field(default_factory=lambda: {
        'C': 1.0,
        'penalty': 'l2',
        'solver': 'lbfgs'
    })
    
    # Random Forest
    rf_params: Dict[str, Any] = field(default_factory=lambda: {
        'n_estimators': 200,
        'max_depth': 20,
        'min_samples_split': 5,
        'min_samples_leaf': 2
    })
    
    # XGBoost
    xgb_params: Dict[str, Any] = field(default_factory=lambda: {
        'n_estimators': 300,
        'max_depth': 6,
        'learning_rate': 0.1,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'reg_alpha': 0.1,
        'reg_lambda': 1.0
    })
    
    # Cross-validation
    cv_folds: int = 5


@dataclass
class MLFlowConfig:
    """MLFlow tracking configuration."""
    experiment_name: str = "FakeScope-Production"
    tracking_uri: str = "./mlruns"
    model_registry_name: str = "FakeScopeModel"


@dataclass
class FakeScopeConfig:
    """Master configuration aggregating all sub-configs."""
    data: DataConfig = field(default_factory=DataConfig)
    preprocessing: PreprocessingConfig = field(default_factory=PreprocessingConfig)
    tfidf: TFIDFConfig = field(default_factory=TFIDFConfig)
    models: ModelConfig = field(default_factory=ModelConfig)
    mlflow: MLFlowConfig = field(default_factory=MLFlowConfig)
    
    # Output directories
    models_dir: Path = Path("./trained_models")
    artifacts_dir: Path = Path("./artifacts")
    logs_dir: Path = Path("./logs")
    
    def __post_init__(self):
        """Create output directories if they don't exist."""
        for directory in [self.models_dir, self.artifacts_dir, self.logs_dir]:
            directory.mkdir(exist_ok=True, parents=True)


# Singleton config instance
config = FakeScopeConfig()
