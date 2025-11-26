"""Application configuration using Pydantic Settings"""
# pyright: reportMissingImports=false
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Optional


class Settings(BaseSettings):
    """Application settings loaded from environment variables"""

    # OpenAI LLM Configuration
    openai_api_key: Optional[str] = None

    # Gemini LLM Configuration
    gemini_api_key: Optional[str] = None

    # Groq LLM Configuration
    groq_api_key: Optional[str] = None

    # Redis Configuration
    redis_url: str = "redis://localhost:6379"

    # Logging Configuration
    log_level: str = "INFO"
    log_file: Optional[str] = "app.log"
    log_dir: str = "logs"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )


# Global settings instance
settings = Settings()