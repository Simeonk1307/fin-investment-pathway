"""
Global Settings Manager
Loads configuration from environment variables and YAML files
"""

import os
from pathlib import Path
from typing import Dict, Any
import yaml
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class Settings:
    """Global application settings"""
    
    # Project paths
    BASE_DIR = Path(__file__).parent.parent
    CONFIG_DIR = BASE_DIR / "config"

    
    # API Keys
    NEWS_API_KEY = os.getenv("NEWS_API_KEY")

    # Performance
    MAX_WORKERS = int(os.getenv("MAX_WORKERS", "10"))
    BATCH_SIZE = int(os.getenv("BATCH_SIZE", "100"))
    
    @classmethod
    def load_api_config(cls) -> Dict[str, Any]:
        """Load API configuration from YAML"""
        config_path = cls.CONFIG_DIR / "api.yaml"
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)



settings = Settings()