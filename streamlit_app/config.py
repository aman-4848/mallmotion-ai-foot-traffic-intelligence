"""
Configuration Management for Production
Centralized configuration with environment variable support
"""
import os
from pathlib import Path
from typing import Optional
import json

# Project root
PROJECT_ROOT = Path(__file__).parent.parent

# Data paths
DATA_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"
LOGS_DIR = PROJECT_ROOT / "logs"
CACHE_DIR = PROJECT_ROOT / ".cache"

# Create directories if they don't exist
for dir_path in [DATA_DIR, MODELS_DIR, RESULTS_DIR, LOGS_DIR, CACHE_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

class Config:
    """Application configuration"""
    
    # App settings
    APP_NAME = os.getenv("APP_NAME", "Mall Movement Tracking Dashboard")
    APP_VERSION = os.getenv("APP_VERSION", "1.0.0")
    DEBUG = os.getenv("DEBUG", "False").lower() == "true"
    
    # Data settings
    DATA_FILE = os.getenv("DATA_FILE", str(DATA_DIR / "merged data set.csv"))
    MAX_FILE_SIZE_MB = int(os.getenv("MAX_FILE_SIZE_MB", "50"))
    
    # Model settings
    MODEL_CACHE_TTL = int(os.getenv("MODEL_CACHE_TTL", "3600"))  # 1 hour
    DATA_CACHE_TTL = int(os.getenv("DATA_CACHE_TTL", "1800"))  # 30 minutes
    
    # Performance settings
    MAX_ROWS_DISPLAY = int(os.getenv("MAX_ROWS_DISPLAY", "10000"))
    ENABLE_CACHING = os.getenv("ENABLE_CACHING", "True").lower() == "true"
    
    # Security settings
    ENABLE_AUTH = os.getenv("ENABLE_AUTH", "False").lower() == "true"
    SESSION_TIMEOUT = int(os.getenv("SESSION_TIMEOUT", "3600"))  # 1 hour
    
    # Logging settings
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
    LOG_FILE = str(LOGS_DIR / "app.log")
    
    # UI settings
    THEME = os.getenv("THEME", "dark")
    PAGE_ICON = "🏪"
    PAGE_TITLE = "Mall Movement Tracking Dashboard"
    
    @classmethod
    def load_from_file(cls, config_file: Optional[Path] = None) -> dict:
        """
        Load configuration from JSON file
        
        Args:
            config_file: Path to config file
        
        Returns:
            Configuration dictionary
        """
        if config_file is None:
            config_file = PROJECT_ROOT / "config.json"
        
        if config_file.exists():
            with open(config_file, 'r') as f:
                return json.load(f)
        return {}
    
    @classmethod
    def save_to_file(cls, config: dict, config_file: Optional[Path] = None):
        """
        Save configuration to JSON file
        
        Args:
            config: Configuration dictionary
            config_file: Path to config file
        """
        if config_file is None:
            config_file = PROJECT_ROOT / "config.json"
        
        with open(config_file, 'w') as f:
            json.dump(config, f, indent=2)

# Create config instance
config = Config()

