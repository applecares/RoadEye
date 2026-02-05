"""Pydantic settings for application configuration."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class DatabaseSettings(BaseSettings):
    """Database configuration."""

    url: str = Field(default="sqlite:///roadkill.db", description="Database URL")
    pool_size: int = Field(default=5, description="Connection pool size")
    echo: bool = Field(default=False, description="Echo SQL statements")


class APISettings(BaseSettings):
    """API server configuration."""

    host: str = Field(default="0.0.0.0", description="Host to bind to")
    port: int = Field(default=8000, description="Port to bind to")
    api_key: Optional[str] = Field(default=None, description="API key for authentication")
    cors_origins: list[str] = Field(
        default=["http://localhost:3000", "http://localhost:8000"],
        description="Allowed CORS origins",
    )


class DetectionSettings(BaseSettings):
    """Detection engine configuration."""

    confidence_threshold: float = Field(
        default=0.5, description="Minimum confidence for detection"
    )
    dedup_distance_metres: float = Field(
        default=50.0, description="Distance threshold for deduplication"
    )
    dedup_time_seconds: float = Field(
        default=300.0, description="Time threshold for deduplication"
    )
    auto_approve_threshold: float = Field(
        default=0.95, description="Confidence threshold for auto-approval"
    )
    review_threshold: float = Field(
        default=0.75, description="Confidence threshold for review queue"
    )


class EdgeSettings(BaseSettings):
    """Edge device configuration."""

    save_images: bool = Field(default=True, description="Save detection images")
    image_quality: int = Field(default=85, description="JPEG quality (0-100)")
    upload_batch_size: int = Field(default=10, description="Batch size for uploads")
    retry_delay_seconds: int = Field(default=30, description="Retry delay on failure")
    max_retries: int = Field(default=5, description="Maximum retry attempts")
    offline_queue_max_size: int = Field(default=1000, description="Max offline queue size")


class Settings(BaseSettings):
    """Main application settings."""

    model_config = SettingsConfigDict(
        env_prefix="ROADKILL_",
        env_nested_delimiter="__",
        extra="ignore",
    )

    # Application
    app_name: str = Field(default="Wildlife Roadkill Detection System")
    environment: str = Field(default="development")
    debug: bool = Field(default=False)

    # State configuration
    state_code: str = Field(default="TAS", description="State code (TAS, VIC, NSW, QLD)")
    state_config_path: Optional[Path] = Field(
        default=None, description="Path to state-specific config YAML"
    )

    # Nested settings
    database: DatabaseSettings = Field(default_factory=DatabaseSettings)
    api: APISettings = Field(default_factory=APISettings)
    detection: DetectionSettings = Field(default_factory=DetectionSettings)
    edge: EdgeSettings = Field(default_factory=EdgeSettings)

    @classmethod
    def from_yaml(cls, yaml_path: Path) -> "Settings":
        """Load settings from YAML file."""
        import yaml

        with open(yaml_path) as f:
            data = yaml.safe_load(f)

        return cls(**data)


# Global settings instance (lazy loaded)
_settings: Optional[Settings] = None


def get_settings() -> Settings:
    """Get the global settings instance."""
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings


def configure_settings(settings: Settings) -> None:
    """Configure the global settings instance."""
    global _settings
    _settings = settings
