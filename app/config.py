from functools import lru_cache
from typing import Optional

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Environment-backed settings. Nothing here is required for the default mode."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # "extractive": the answer is the top retrieved passage, verbatim (default, no keys).
    # "openai":     an LLM writes a short answer constrained to the retrieved passages.
    generation: str = "extractive"
    openai_api_key: Optional[str] = None
    openai_model: Optional[str] = None

    @property
    def llm_configured(self) -> bool:
        return bool(self.openai_api_key and self.openai_model)


@lru_cache
def get_settings() -> Settings:
    return Settings()
