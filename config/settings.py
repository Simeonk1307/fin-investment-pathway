"""
Global Settings Manager
Loads configuration from environment variables and YAML files
"""

import os
from pathlib import Path
from typing import Dict, Any
import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict
import yaml
# Load environment variables
# load_dotenv()

class LLMSettings(BaseModel):
    model_config = SettingsConfigDict(env_prefix='LLM_',env_file=".env",extra='ignore')

    OPENAI_API_KEY: str
    PPLX_API_KEY: str
    HUGGING_FACE_API_KEY: str
    GROQ_API_KEY: str
    DEFAULT_LLM_MODEL: str = "sonar"

# class Settings:
#     """Global application settings"""
    
#     # Project paths
#     BASE_DIR = Path(__file__).parent.parent
#     CONFIG_DIR = BASE_DIR / "config" / "connections"

     
#     # API Keys
#     APIKEYS: Dict[str, str] = {
#         "NEWSAPI": os.getenv("NEWSAPI_KEY", ""),
#         "GNEWS": os.getenv("GNEWS_API_KEY", ""),
#         "FINNHUB": os.getenv("FINNHUB_API_KEY", ""),
#     }
#     for api, key in APIKEYS.items():
#         if key.startswith("your_") or key == "":
#             raise ValueError(f"API key for {api} is not set properly in environment variables.Please update the .env file.(Check .env.example for reference)")

#     # Performance
#     MAX_WORKERS = int(os.getenv("MAX_WORKERS", "10"))
#     BATCH_SIZE = int(os.getenv("BATCH_SIZE", "100"))
    
#     @classmethod
#     def load_api_config(cls):
#         config_path = cls.CONFIG_DIR / "newsAPIs.yaml"
#         with open(config_path, 'r') as f:
#             return yaml.safe_load(f)

# settings = Settings()

class NewsSettings(BaseModel):
    model_config = SettingsConfigDict(env_prefix="NEWS_", env_file=".env",extra="ignore")

    NEWSAPI_KEY: str
    GNEWS_API_KEY: str
    FINNHUB_API_KEY: str

    def apikeys(self) -> Dict[str, str]:
        return {
            "NEWSAPI": self.NEWSAPI_KEY,
            "GNEWS": self.GNEWS_API_KEY,
            "FINNHUB": self.FINNHUB_API_KEY,
        }


# Only AppSettings is BaseSettings
class AppSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        extra='ignore'
    )

    BASE_DIR: Path = Path(__file__).parent.parent
    CONFIG_DIR: Path = BASE_DIR / "config" / "connections"

    MAX_WORKERS: int = 10
    BATCH_SIZE: int = 100

    # These will be loaded from .env automatically
    OPENAI_API_KEY: str
    PPLX_API_KEY: str
    HUGGING_FACE_API_KEY: str
    GROQ_API_KEY: str
    DEFAULT_LLM_MODEL: str = "sonar"
    

    @property
    def llm(self) -> LLMSettings:
        return LLMSettings(
            OPENAI_API_KEY=self.OPENAI_API_KEY,
            PPLX_API_KEY=self.PPLX_API_KEY,
            HUGGING_FACE_API_KEY=self.HUGGING_FACE_API_KEY,
            GROQ_API_KEY=self.GROQ_API_KEY,
            DEFAULT_LLM_MODEL=self.DEFAULT_LLM_MODEL
        )

    NEWSAPI_KEY: str
    GNEWS_API_KEY: str
    FINNHUB_API_KEY: str
    
    @property
    def news(self) -> NewsSettings:
        return NewsSettings(
            NEWSAPI_KEY=self.NEWSAPI_KEY,
            GNEWS_API_KEY=self.GNEWS_API_KEY,
            FINNHUB_API_KEY=self.FINNHUB_API_KEY
        )

    def load_api_config(self):
        config_path = self.CONFIG_DIR / "newsAPIs.yaml"
        with open(config_path, "r") as f:
            return yaml.safe_load(f)
        
settings = AppSettings()