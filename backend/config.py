"""
Configuration settings for the Blood Pressure Prediction API.
"""
import os
from pathlib import Path
from dataclasses import dataclass
from typing import Optional

@dataclass
class ModelConfiguration:
    """Configuration for model parameters."""
    sequence_length: int = 1000
    lstm_units: int = 128
    batch_size: int = 64
    epochs: int = 50
    validation_split: float = 0.2
    learning_rate: float = 0.001
    dropout_rate: float = 0.2

@dataclass
class APIConfiguration:
    """Configuration for API settings."""
    debug: bool = os.getenv('DEBUG', 'False').lower() == 'true'
    host: str = os.getenv('HOST', '0.0.0.0')
    port: int = int(os.getenv('PORT', 5000))
    cors_origins: str = os.getenv('CORS_ORIGINS', '*')
    max_content_length: int = 16 * 1024 * 1024  # 16MB

@dataclass
class DataConfiguration:
    """Configuration for data processing."""
    data_directory: Path = Path('data')
    models_directory: Path = Path('models')
    logs_directory: Path = Path('logs')
    min_signal_quality_threshold: float = 0.7
    normalization_method: str = 'minmax'  # 'minmax' or 'zscore'

class ApplicationConfig:
    """Main application configuration."""
    
    def __init__(self):
        self.model = ModelConfiguration()
        self.api = APIConfiguration()
        self.data = DataConfiguration()
        
        # Create directories if they don't exist
        self._create_directories()
    
    def _create_directories(self):
        """Create necessary directories."""
        for directory in [self.data.data_directory, 
                         self.data.models_directory, 
                         self.data.logs_directory]:
            directory.mkdir(parents=True, exist_ok=True)

# Global configuration instance
config = ApplicationConfig()
