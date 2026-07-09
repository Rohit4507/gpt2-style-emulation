"""
Application settings, loaded from environment variables (or a .env file).

Using pydantic-settings means these are type-checked at startup -- a bad
MAX_RESIDENT_MODELS value fails fast on boot instead of misbehaving deep
inside model_manager.py at request time.
"""
from typing import List, Optional

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # -- Hugging Face Hub --
    model_repo: str = "your-username/gpt2-style-emulation"
    hf_token: Optional[str] = None  # only needed if model_repo is private

    # -- Model manager --
    max_resident_models: int = 3  # LRU cap; each GPT-2-small checkpoint is ~500MB

    # -- Generation defaults (mirror the paper's Table I) --
    max_new_tokens: int = 50
    top_k: int = 50
    top_p: float = 0.95
    temperature: float = 0.9

    # -- API behavior --
    rate_limit_per_minute: int = 30
    cors_origins: List[str] = ["*"]
    log_level: str = "INFO"

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")


settings = Settings()
