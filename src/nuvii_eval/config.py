"""
Configuration management for Nuvii CDI Evaluation Framework.

Uses pydantic-settings for environment variable loading with validation.
"""

from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class NuviiAPIConfig(BaseSettings):
    """Nuvii CDI Agent API configuration."""

    model_config = SettingsConfigDict(
        env_prefix="NUVII_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    api_url: str = Field(
        default="http://localhost:8000",
        description="Nuvii API base URL",
    )
    api_key: SecretStr = Field(
        default=SecretStr(""),
        description="API authentication key",
    )
    timeout_seconds: int = Field(
        default=30,
        ge=1,
        le=300,
        description="Request timeout in seconds",
    )
    max_retries: int = Field(
        default=3,
        ge=0,
        le=10,
        description="Maximum retry attempts for failed requests",
    )

    @field_validator("api_url")
    @classmethod
    def validate_url(cls, v: str) -> str:
        """Ensure URL doesn't have trailing slash."""
        return v.rstrip("/")


class PhoenixConfig(BaseSettings):
    """Phoenix tracing configuration."""

    model_config = SettingsConfigDict(
        env_prefix="PHOENIX_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    enabled: bool = Field(
        default=True,
        description="Enable Phoenix tracing",
    )
    endpoint: str = Field(
        default="http://localhost:6006",
        description="Phoenix endpoint URL",
    )
    project_name: str = Field(
        default="nuvii-cdi-eval",
        description="Phoenix project name for trace organization",
    )
    collect_inputs: bool = Field(
        default=False,
        description="Collect input data in traces (disable for PHI safety)",
    )
    collect_outputs: bool = Field(
        default=True,
        description="Collect output data in traces",
    )


class EvalConfig(BaseSettings):
    """Evaluation runtime configuration."""

    model_config = SettingsConfigDict(
        env_prefix="EVAL_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    phi_safe_mode: bool = Field(
        default=True,
        description="Enable PHI redaction in logs and traces",
    )
    concurrency: int = Field(
        default=5,
        ge=1,
        le=50,
        description="Maximum concurrent API calls",
    )
    rate_limit_rpm: int = Field(
        default=60,
        ge=1,
        le=1000,
        description="Rate limit in requests per minute",
    )
    deterministic_mode: bool = Field(
        default=True,
        description="Use temperature=0 for reproducible results",
    )
    output_dir: Path = Field(
        default=Path("./runs"),
        description="Output directory for evaluation results",
    )
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = Field(
        default="INFO",
        description="Logging level",
    )

    @field_validator("output_dir")
    @classmethod
    def ensure_path(cls, v: Path | str) -> Path:
        """Ensure output_dir is a Path object."""
        return Path(v)


class LLMJudgeConfig(BaseSettings):
    """Configuration for LLM-based evaluation judges."""

    model_config = SettingsConfigDict(
        env_prefix="LLM_JUDGE_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    enabled: bool = Field(
        default=False,
        description="Enable LLM-based judging for query quality",
    )
    model: str = Field(
        default="gpt-4-turbo-preview",
        description="Model to use for LLM judging",
    )
    api_key: SecretStr = Field(
        default=SecretStr(""),
        description="API key for LLM provider",
    )
    temperature: float = Field(
        default=0.0,
        ge=0.0,
        le=2.0,
        description="Temperature for LLM judge",
    )
    max_tokens: int = Field(
        default=1000,
        ge=100,
        le=4000,
        description="Maximum tokens for judge response",
    )


class Settings(BaseSettings):
    """
    Root settings aggregating all configuration sections.

    Usage:
        settings = Settings()
        print(settings.nuvii.api_url)
        print(settings.eval.phi_safe_mode)
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    nuvii: NuviiAPIConfig = Field(default_factory=NuviiAPIConfig)
    phoenix: PhoenixConfig = Field(default_factory=PhoenixConfig)
    eval: EvalConfig = Field(default_factory=EvalConfig)
    llm_judge: LLMJudgeConfig = Field(default_factory=LLMJudgeConfig)

    def __init__(self, **kwargs):
        """Initialize settings, creating nested configs."""
        super().__init__(**kwargs)
        # Re-initialize nested configs to pick up env vars
        self.nuvii = NuviiAPIConfig()
        self.phoenix = PhoenixConfig()
        self.eval = EvalConfig()
        self.llm_judge = LLMJudgeConfig()


# Singleton instance for easy access
_settings: Settings | None = None


def get_settings() -> Settings:
    """Get or create the global settings instance."""
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings


def reload_settings() -> Settings:
    """Force reload of settings (useful for testing)."""
    global _settings
    _settings = Settings()
    return _settings
