"""
Global Settings Manager
Loads configuration from environment variables and YAML files
"""

import os
from pathlib import Path
from typing import Dict, Any
import yaml
from dotenv import load_dotenv, find_dotenv
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict
import yaml
# Load environment variables
# load_dotenv()

# class LLMSettings(BaseModel):
#     model_config = SettingsConfigDict(env_prefix='LLM_',env_file=".env",extra='ignore')

#     OPENAI_API_KEY: str
#     PPLX_API_KEY: str
#     HUGGING_FACE_API_KEY: str
#     GROQ_API_KEY: str
#     DEFAULT_LLM_MODEL: str = "sonar"

# # class Settings:
# #     """Global application settings"""
    
# #     # Project paths
# #     BASE_DIR = Path(__file__).parent.parent
# #     CONFIG_DIR = BASE_DIR / "config" / "connections"

     
# #     # API Keys
# #     APIKEYS: Dict[str, str] = {
# #         "NEWSAPI": os.getenv("NEWSAPI_KEY", ""),
# #         "GNEWS": os.getenv("GNEWS_API_KEY", ""),
# #         "FINNHUB": os.getenv("FINNHUB_API_KEY", ""),
# #     }
# #     for api, key in APIKEYS.items():
# #         if key.startswith("your_") or key == "":
# #             raise ValueError(f"API key for {api} is not set properly in environment variables.Please update the .env file.(Check .env.example for reference)")

# #     # Performance
# #     MAX_WORKERS = int(os.getenv("MAX_WORKERS", "10"))
# #     BATCH_SIZE = int(os.getenv("BATCH_SIZE", "100"))
    
# #     @classmethod
# #     def load_api_config(cls):
# #         config_path = cls.CONFIG_DIR / "newsAPIs.yaml"
# #         with open(config_path, 'r') as f:
# #             return yaml.safe_load(f)

# # settings = Settings()

# class NewsSettings(BaseModel):
#     model_config = SettingsConfigDict(env_prefix="NEWS_", env_file=".env",extra="ignore")

#     NEWSAPI_KEY: str
#     GNEWS_API_KEY: str
#     FINNHUB_API_KEY: str

#     def apikeys(self) -> Dict[str, str]:
#         return {
#             "NEWSAPI": self.NEWSAPI_KEY,
#             "GNEWS": self.GNEWS_API_KEY,
#             "FINNHUB": self.FINNHUB_API_KEY,
#         }


# # Only AppSettings is BaseSettings
# class AppSettings(BaseSettings):
#     model_config = SettingsConfigDict(
#         env_file=".env",
#         extra='ignore'
#     )

#     BASE_DIR: Path = Path(__file__).parent.parent
#     CONFIG_DIR: Path = BASE_DIR / "config" / "connections"

#     MAX_WORKERS: int = 10
#     BATCH_SIZE: int = 100

#     # These will be loaded from .env automatically
#     OPENAI_API_KEY: str
#     PPLX_API_KEY: str
#     HUGGING_FACE_API_KEY: str
#     GROQ_API_KEY: str
#     DEFAULT_LLM_MODEL: str = "sonar"
    

#     @property
#     def llm(self) -> LLMSettings:
#         return LLMSettings(
#             OPENAI_API_KEY=self.OPENAI_API_KEY,
#             PPLX_API_KEY=self.PPLX_API_KEY,
#             HUGGING_FACE_API_KEY=self.HUGGING_FACE_API_KEY,
#             GROQ_API_KEY=self.GROQ_API_KEY,
#             DEFAULT_LLM_MODEL=self.DEFAULT_LLM_MODEL
#         )

#     NEWSAPI_KEY: str
#     GNEWS_API_KEY: str
#     FINNHUB_API_KEY: str
    
#     @property
#     def news(self) -> NewsSettings:
#         return NewsSettings(
#             NEWSAPI_KEY=self.NEWSAPI_KEY,
#             GNEWS_API_KEY=self.GNEWS_API_KEY,
#             FINNHUB_API_KEY=self.FINNHUB_API_KEY
#         )

#     def load_api_config(self):
#         config_path = self.CONFIG_DIR / "newsAPIs.yaml"
#         with open(config_path, "r") as f:
#             return yaml.safe_load(f)
        
# settings = AppSettings()

import logging
from typing import Dict, Optional, Type
from enum import Enum
from abc import ABC, abstractmethod

from pydantic import BaseModel, field_validator, ValidationError
from pydantic_settings import BaseSettings, SettingsConfigDict

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class APIKeyNotFoundError(Exception):
    """Raised when requested API key is not configured"""
    pass


class BaseAPISettings(BaseSettings, ABC):
    """
    Base class for all API settings.
    Main method: get_key(provider) - validates and returns API key or raises error
    """
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding='utf-8',
        case_sensitive=False,
        extra='ignore'
    )

    @field_validator('*', mode='before')
    @classmethod
    def clean_api_key(cls, v):
        """Remove placeholder/invalid/empty API key values"""
        if not v or not isinstance(v, str):
            return None
        
        # Remove whitespace
        v = v.strip()
        
        # Check for empty string
        if not v:
            return None
        
        # Remove common placeholder patterns
        invalid_patterns = ["your_", "sk-xxx", "replace_", "add_your_", "enter_"]
        if any(v.lower().startswith(pattern) for pattern in invalid_patterns):
            return None
        
        return v

    @abstractmethod
    def _get_provider_key_mapping(self) -> Dict[Enum, Optional[str]]:
        """Return mapping of provider enum to API key value"""
        pass

    @abstractmethod
    def _get_provider_enum_class(self) -> Type[Enum]:
        """Return the provider enum class"""
        pass

    def _validate_key(self, key: Optional[str], provider_name: str) -> str:
        """
        Validate API key.
        Basic validation: checks if key exists and has minimum length.
        Override this method in child classes for provider-specific validation.
        
        Args:
            key: The API key to validate
            provider_name: Name of the provider (for error messages)
            
        Returns:
            Valid API key string
            
        Raises:
            APIKeyNotFoundError: If key is invalid
        """
        # Check if key exists
        if not key:
            raise APIKeyNotFoundError(
                f"API key for '{provider_name}' is not configured or empty. "
                f"Please set it in your .env file."
            )
        
        return key

    def get_key(self, provider: str | Enum) -> str:
        """
        Get and validate API key for provider.
        Raises APIKeyNotFoundError if key is not configured.
        
        Args:
            provider: Provider name (string or enum)
            
        Returns:
            Valid API key string
            
        Raises:
            APIKeyNotFoundError: If key not found or invalid
        """
        provider_map = self._get_provider_key_mapping()
        
        # Convert string to enum if needed
        if isinstance(provider, str):
            provider_enum_class = self._get_provider_enum_class()
            try:
                provider = provider_enum_class(provider.lower())
            except ValueError:
                available = [p.value for p in provider_enum_class]
                raise APIKeyNotFoundError(
                    f"Unknown provider '{provider}'. Available: {', '.join(available)}"
                )
        
        # Get the key
        api_key = provider_map.get(provider)
        
        # Validate and return
        return self._validate_key(api_key, provider.value)

    def list_available(self) -> list[str]:
        """List all configured providers"""
        provider_map = self._get_provider_key_mapping()
        available = []
        for provider, key in provider_map.items():
            try:
                self._validate_key(key, provider.value)
                available.append(provider.value)
            except APIKeyNotFoundError:
                continue
        return available

    def has_provider(self, provider: str | Enum) -> bool:
        """Check if specific provider is configured"""
        try:
            self.get_key(provider)
            return True
        except APIKeyNotFoundError:
            return False


# ============================================================================
# LLM Settings
# ============================================================================

class LLMProvider(str, Enum):
    """Available LLM providers"""
    OPENAI = "openai"
    PERPLEXITY = "perplexity"
    GROQ = "groq"
    HUGGINGFACE = "huggingface"


class LLMSettings(BaseAPISettings):
    """
    LLM API configuration.
    
    Usage:
        llm = LLMSettings()
        try:
            key = llm.get_key("openai")
            # Use the key...
        except APIKeyNotFoundError as e:
            print(f"Error: {e}")
    """
    
    # Match your .env format exactly
    pplx_api_key: Optional[str] = None
    openai_api_key: Optional[str] = None
    hugging_face_api_key: Optional[str] = None
    groq_api_key: Optional[str] = None
    default_llm_model: str = "sonar"

    def _get_provider_key_mapping(self) -> Dict[LLMProvider, Optional[str]]:
        return {
            LLMProvider.OPENAI: self.openai_api_key,
            LLMProvider.PERPLEXITY: self.pplx_api_key,
            LLMProvider.GROQ: self.groq_api_key,
            LLMProvider.HUGGINGFACE: self.hugging_face_api_key,
        }

    def _get_provider_enum_class(self) -> Type[Enum]:
        return LLMProvider

    def _validate_key(self, key: Optional[str], provider_name: str) -> str:
        """Custom validation for LLM keys"""
        # Call base validation first
        key = super()._validate_key(key, provider_name)
        
        # Provider-specific validation (optional)
        if provider_name == "openai" and not key.startswith("sk-"):
            raise APIKeyNotFoundError(
                f"OpenAI API key should start with 'sk-'. Please check your key."
            )
        
        return key


# ============================================================================
# News Settings
# ============================================================================

class NewsProvider(str, Enum):
    """Available news API providers"""
    FINNHUB = "finnhub"


class NewsSettings(BaseAPISettings):
    """
    News API configuration.
    
    Usage:
        news = NewsSettings()
        try:
            key = news.get_key("newsapi")
        except APIKeyNotFoundError as e:
            # Handle missing key
    """
    
    # Match your .env format exactly (with underscores
    finnhub_api_key: Optional[str] = None

    def _get_provider_key_mapping(self) -> Dict[NewsProvider, Optional[str]]:
        return {
            NewsProvider.FINNHUB: self.finnhub_api_key,
        }

    def _get_provider_enum_class(self) -> Type[Enum]:
        return NewsProvider
