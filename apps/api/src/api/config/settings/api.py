"""Settings specific to this app. Anything shared lives in `core.config`."""

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class ApiSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="API_", env_file=".env", extra="ignore")

    title: str = "take-home api"
    cors_origins: list[str] = ["*"]
    max_upload_bytes: int = 50 * 1024 * 1024
    search_limit_default: int = 10
    search_limit_max: int = 100


@lru_cache
def get_api_settings() -> ApiSettings:
    return ApiSettings()
