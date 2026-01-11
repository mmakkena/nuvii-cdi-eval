"""Tests for configuration management."""

import os
from pathlib import Path

import pytest

from nuvii_eval.config import (
    EvalConfig,
    NuviiAPIConfig,
    PhoenixConfig,
    Settings,
    get_settings,
    reload_settings,
)


class TestNuviiAPIConfig:
    """Tests for NuviiAPIConfig."""

    def test_default_values(self):
        """Test default configuration values."""
        config = NuviiAPIConfig()

        assert config.api_url == "http://localhost:8000"
        assert config.timeout_seconds == 30
        assert config.max_retries == 3

    def test_url_trailing_slash_removed(self):
        """Test that trailing slashes are removed from URL."""
        # Set env var with trailing slash
        os.environ["NUVII_API_URL"] = "http://example.com/"

        config = NuviiAPIConfig()
        assert config.api_url == "http://example.com"

        del os.environ["NUVII_API_URL"]

    def test_api_key_is_secret(self):
        """Test that API key is stored securely."""
        os.environ["NUVII_API_KEY"] = "secret_key_123"

        config = NuviiAPIConfig()

        # Should not expose key in string representation
        assert "secret_key_123" not in str(config)
        assert "secret_key_123" not in repr(config)

        # But should be accessible via get_secret_value
        assert config.api_key.get_secret_value() == "secret_key_123"

        del os.environ["NUVII_API_KEY"]

    def test_timeout_bounds(self):
        """Test timeout validation bounds."""
        # Valid timeout
        os.environ["NUVII_TIMEOUT_SECONDS"] = "60"
        config = NuviiAPIConfig()
        assert config.timeout_seconds == 60
        del os.environ["NUVII_TIMEOUT_SECONDS"]

        # Invalid - too high
        os.environ["NUVII_TIMEOUT_SECONDS"] = "500"
        with pytest.raises(ValueError):
            NuviiAPIConfig()
        del os.environ["NUVII_TIMEOUT_SECONDS"]


class TestPhoenixConfig:
    """Tests for PhoenixConfig."""

    def test_default_values(self):
        """Test default Phoenix configuration."""
        config = PhoenixConfig()

        assert config.enabled is True
        assert config.endpoint == "http://localhost:6006"
        assert config.project_name == "nuvii-cdi-eval"
        assert config.collect_inputs is False  # PHI safety default
        assert config.collect_outputs is True

    def test_env_override(self):
        """Test environment variable override."""
        os.environ["PHOENIX_ENABLED"] = "false"
        os.environ["PHOENIX_PROJECT_NAME"] = "test-project"

        config = PhoenixConfig()

        assert config.enabled is False
        assert config.project_name == "test-project"

        del os.environ["PHOENIX_ENABLED"]
        del os.environ["PHOENIX_PROJECT_NAME"]


class TestEvalConfig:
    """Tests for EvalConfig."""

    def test_default_values(self):
        """Test default evaluation configuration."""
        config = EvalConfig()

        assert config.phi_safe_mode is True
        assert config.concurrency == 5
        assert config.rate_limit_rpm == 60
        assert config.deterministic_mode is True
        assert config.log_level == "INFO"

    def test_output_dir_as_path(self):
        """Test that output_dir is converted to Path."""
        os.environ["EVAL_OUTPUT_DIR"] = "/tmp/test_runs"

        config = EvalConfig()

        assert isinstance(config.output_dir, Path)
        assert config.output_dir == Path("/tmp/test_runs")

        del os.environ["EVAL_OUTPUT_DIR"]

    def test_concurrency_bounds(self):
        """Test concurrency validation."""
        os.environ["EVAL_CONCURRENCY"] = "100"

        with pytest.raises(ValueError):
            EvalConfig()

        del os.environ["EVAL_CONCURRENCY"]


class TestSettings:
    """Tests for aggregated Settings."""

    def test_nested_configs_created(self):
        """Test that nested configs are properly initialized."""
        settings = Settings()

        assert isinstance(settings.nuvii, NuviiAPIConfig)
        assert isinstance(settings.phoenix, PhoenixConfig)
        assert isinstance(settings.eval, EvalConfig)

    def test_get_settings_singleton(self):
        """Test that get_settings returns singleton."""
        settings1 = get_settings()
        settings2 = get_settings()

        assert settings1 is settings2

    def test_reload_settings(self):
        """Test that reload_settings creates new instance."""
        settings1 = get_settings()
        settings2 = reload_settings()

        assert settings1 is not settings2


class TestSettingsIntegration:
    """Integration tests for settings with environment."""

    def test_full_config_from_env(self, env_vars):
        """Test loading full configuration from environment."""
        settings = reload_settings()

        assert settings.nuvii.api_url == "http://test-api.local:8000"
        assert settings.nuvii.api_key.get_secret_value() == "test_key_12345"
        assert settings.phoenix.enabled is False
        assert settings.eval.phi_safe_mode is True
