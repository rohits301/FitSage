from functools import lru_cache
from typing import Optional

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Environment-backed settings. Credentials are optional in demo mode."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    app_mode: str = "demo"
    openai_api_key: Optional[str] = None
    openai_model: Optional[str] = None
    pinecone_api_key: Optional[str] = None
    pinecone_index_name: Optional[str] = None

    @property
    def live_configured(self) -> bool:
        return bool(
            self.openai_api_key
            and self.pinecone_api_key
            and self.pinecone_index_name
        )


@lru_cache
def get_settings() -> Settings:
    return Settings()
