"""Configuration management."""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="DELEGATION_")

    database_url: str = "postgresql+asyncpg://localhost/delegation_service"

    # Globus configuration
    globus_client_id: str = ""
    globus_client_secret: str = ""

    # For development: skip actual Globus calls
    globus_dry_run: bool = True

    # API settings
    api_prefix: str = "/v1"


settings = Settings()
