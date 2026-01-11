"""
Promptfoo configuration generator.

Generates Promptfoo YAML configuration files from CDI evaluation settings.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import structlog
import yaml

from nuvii_eval.config import get_settings

logger = structlog.get_logger(__name__)


# =============================================================================
# Configuration Models
# =============================================================================


@dataclass
class PromptfooProvider:
    """
    Promptfoo provider configuration.

    Represents an API endpoint or model to test against.
    """

    id: str
    label: str | None = None
    config: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to Promptfoo YAML format."""
        result: dict[str, Any] = {"id": self.id}
        if self.label:
            result["label"] = self.label
        if self.config:
            result["config"] = self.config
        return result


@dataclass
class PromptfooAssertion:
    """
    Promptfoo assertion configuration.

    Defines a check to run against model output.
    """

    type: str
    value: Any = None
    threshold: float | None = None
    weight: float = 1.0
    metric: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to Promptfoo YAML format."""
        result: dict[str, Any] = {"type": self.type}
        if self.value is not None:
            result["value"] = self.value
        if self.threshold is not None:
            result["threshold"] = self.threshold
        if self.weight != 1.0:
            result["weight"] = self.weight
        if self.metric:
            result["metric"] = self.metric
        return result


@dataclass
class PromptfooTest:
    """
    Promptfoo test case configuration.

    Represents a single test case with variables and assertions.
    """

    description: str
    vars: dict[str, Any]
    assert_: list[PromptfooAssertion] = field(default_factory=list)
    options: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to Promptfoo YAML format."""
        result: dict[str, Any] = {
            "description": self.description,
            "vars": self.vars,
        }
        if self.assert_:
            result["assert"] = [a.to_dict() for a in self.assert_]
        if self.options:
            result["options"] = self.options
        if self.metadata:
            result["metadata"] = self.metadata
        return result


@dataclass
class PromptfooConfig:
    """
    Complete Promptfoo configuration.

    Represents a full promptfoo.yaml configuration file.
    """

    description: str
    providers: list[PromptfooProvider]
    prompts: list[str]
    tests: list[PromptfooTest]
    default_test: dict[str, Any] | None = None
    output_path: str | None = None
    sharing: bool = False
    env: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to Promptfoo YAML format."""
        result: dict[str, Any] = {
            "description": self.description,
            "providers": [p.to_dict() for p in self.providers],
            "prompts": self.prompts,
            "tests": [t.to_dict() for t in self.tests],
        }
        if self.default_test:
            result["defaultTest"] = self.default_test
        if self.output_path:
            result["outputPath"] = self.output_path
        if not self.sharing:
            result["sharing"] = False
        if self.env:
            result["env"] = self.env
        return result

    def to_yaml(self) -> str:
        """Convert to YAML string."""
        return yaml.dump(
            self.to_dict(),
            default_flow_style=False,
            sort_keys=False,
            allow_unicode=True,
        )

    def save(self, path: str | Path) -> None:
        """Save configuration to file."""
        path = Path(path)
        path.write_text(self.to_yaml())
        logger.info("promptfoo_config_saved", path=str(path))


# =============================================================================
# Provider Configurations
# =============================================================================


def create_nuvii_provider(
    base_url: str | None = None,
    api_key_env: str = "NUVII_API_KEY",
    endpoint: str = "/api/v2/coding/suggest",
    timeout_ms: int = 30000,
) -> PromptfooProvider:
    """
    Create a Nuvii API provider configuration.

    Args:
        base_url: API base URL (defaults to settings)
        api_key_env: Environment variable for API key
        endpoint: API endpoint path
        timeout_ms: Request timeout in milliseconds

    Returns:
        Configured PromptfooProvider
    """
    settings = get_settings()
    url = base_url or settings.nuvii_api.base_url

    return PromptfooProvider(
        id="http",
        label="Nuvii CDI API",
        config={
            "url": f"{url}{endpoint}",
            "method": "POST",
            "headers": {
                "Authorization": f"Bearer {{{{{api_key_env}}}}}",
                "Content-Type": "application/json",
            },
            "body": {
                "clinical_note": "{{clinical_note}}",
                "options": "{{options}}",
            },
            "timeout": timeout_ms,
        },
    )


def create_openai_provider(
    model: str = "gpt-4o-mini",
    temperature: float = 0.0,
) -> PromptfooProvider:
    """
    Create an OpenAI provider for comparison testing.

    Args:
        model: OpenAI model name
        temperature: Sampling temperature

    Returns:
        Configured PromptfooProvider
    """
    return PromptfooProvider(
        id=f"openai:{model}",
        label=f"OpenAI {model}",
        config={
            "temperature": temperature,
        },
    )


# =============================================================================
# Configuration Generator
# =============================================================================


@dataclass
class ConfigGeneratorOptions:
    """Options for configuration generation."""

    description: str = "Nuvii CDI Agent Evaluation"
    output_path: str = "promptfoo_output.json"
    include_baseline: bool = True
    baseline_model: str = "gpt-4o-mini"
    timeout_ms: int = 30000
    max_concurrency: int = 5
    sharing: bool = False


def generate_promptfoo_config(
    tests: list[PromptfooTest],
    task_type: str,
    options: ConfigGeneratorOptions | None = None,
) -> PromptfooConfig:
    """
    Generate a complete Promptfoo configuration.

    Args:
        tests: List of test cases
        task_type: Type of CDI task (icd, hcc, gap, query, em)
        options: Configuration options

    Returns:
        Complete PromptfooConfig
    """
    opts = options or ConfigGeneratorOptions()

    # Configure providers based on task type
    providers = []

    # Add Nuvii API provider
    endpoint_map = {
        "icd": "/api/v2/coding/suggest",
        "hcc": "/api/v2/risk/analyze",
        "gap": "/api/v2/cdi/gaps",
        "query": "/api/v2/cdi/queries",
        "em": "/api/v2/coding/em",
    }
    endpoint = endpoint_map.get(task_type, "/api/v2/coding/suggest")

    providers.append(create_nuvii_provider(
        endpoint=endpoint,
        timeout_ms=opts.timeout_ms,
    ))

    # Optionally add baseline for comparison
    if opts.include_baseline:
        providers.append(create_openai_provider(model=opts.baseline_model))

    # Configure prompts based on task type
    prompt_map = {
        "icd": "Analyze the following clinical note and suggest ICD-10 codes:\n\n{{clinical_note}}",
        "hcc": "Analyze the following clinical note for HCC opportunities:\n\n{{clinical_note}}",
        "gap": "Identify documentation gaps in the following clinical note:\n\n{{clinical_note}}",
        "query": "Generate a CDI query for the following gap:\n\nGap: {{gap}}\n\nClinical Note:\n{{clinical_note}}",
        "em": "Determine the E/M level for this encounter:\n\n{{clinical_note}}",
    }
    prompts = [prompt_map.get(task_type, "{{clinical_note}}")]

    # Create default test configuration
    default_test = {
        "options": {
            "timeout": opts.timeout_ms,
        },
    }

    return PromptfooConfig(
        description=f"{opts.description} - {task_type.upper()}",
        providers=providers,
        prompts=prompts,
        tests=tests,
        default_test=default_test,
        output_path=opts.output_path,
        sharing=opts.sharing,
        env={
            "NUVII_API_KEY": "${NUVII_API_KEY}",
        },
    )


# =============================================================================
# Task-Specific Configuration Templates
# =============================================================================


def generate_icd_config(
    tests: list[PromptfooTest],
    options: ConfigGeneratorOptions | None = None,
) -> PromptfooConfig:
    """Generate configuration for ICD code evaluation."""
    return generate_promptfoo_config(tests, "icd", options)


def generate_hcc_config(
    tests: list[PromptfooTest],
    options: ConfigGeneratorOptions | None = None,
) -> PromptfooConfig:
    """Generate configuration for HCC evaluation."""
    return generate_promptfoo_config(tests, "hcc", options)


def generate_gap_config(
    tests: list[PromptfooTest],
    options: ConfigGeneratorOptions | None = None,
) -> PromptfooConfig:
    """Generate configuration for gap detection evaluation."""
    return generate_promptfoo_config(tests, "gap", options)


def generate_query_config(
    tests: list[PromptfooTest],
    options: ConfigGeneratorOptions | None = None,
) -> PromptfooConfig:
    """Generate configuration for query quality evaluation."""
    return generate_promptfoo_config(tests, "query", options)


def generate_em_config(
    tests: list[PromptfooTest],
    options: ConfigGeneratorOptions | None = None,
) -> PromptfooConfig:
    """Generate configuration for E/M level evaluation."""
    return generate_promptfoo_config(tests, "em", options)
