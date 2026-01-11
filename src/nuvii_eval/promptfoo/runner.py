"""
Promptfoo CI runner and result parser.

Provides utilities for running Promptfoo evaluations and parsing results.
"""

import json
import subprocess
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import structlog

logger = structlog.get_logger(__name__)


# =============================================================================
# Configuration
# =============================================================================


@dataclass
class RunConfig:
    """Configuration for Promptfoo run."""

    config_path: str
    output_path: str = "promptfoo_output.json"
    max_concurrency: int = 5
    timeout_seconds: int = 300
    env_file: str | None = None
    verbose: bool = False
    no_cache: bool = False
    filter_pattern: str | None = None
    grader: str | None = None

    def to_args(self) -> list[str]:
        """Convert to command line arguments."""
        args = [
            "promptfoo", "eval",
            "--config", self.config_path,
            "--output", self.output_path,
            "--max-concurrency", str(self.max_concurrency),
        ]

        if self.env_file:
            args.extend(["--env-file", self.env_file])
        if self.verbose:
            args.append("--verbose")
        if self.no_cache:
            args.append("--no-cache")
        if self.filter_pattern:
            args.extend(["--filter-pattern", self.filter_pattern])
        if self.grader:
            args.extend(["--grader", self.grader])

        return args


# =============================================================================
# Result Models
# =============================================================================


@dataclass
class AssertionResult:
    """Result of a single assertion."""

    type: str
    pass_: bool
    score: float
    reason: str | None = None
    metric: str | None = None
    threshold: float | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AssertionResult":
        """Create from Promptfoo output dict."""
        return cls(
            type=data.get("type", "unknown"),
            pass_=data.get("pass", False),
            score=data.get("score", 0.0),
            reason=data.get("reason"),
            metric=data.get("metric"),
            threshold=data.get("threshold"),
        )


@dataclass
class TestResult:
    """Result of a single test case."""

    test_id: str
    description: str
    pass_: bool
    score: float
    assertions: list[AssertionResult]
    latency_ms: int | None = None
    output: str | None = None
    error: str | None = None
    provider: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TestResult":
        """Create from Promptfoo output dict."""
        assertions = [
            AssertionResult.from_dict(a)
            for a in data.get("assertions", [])
        ]

        # Calculate overall score
        if assertions:
            passed = sum(1 for a in assertions if a.pass_)
            score = passed / len(assertions)
        else:
            score = 1.0 if data.get("pass", False) else 0.0

        return cls(
            test_id=data.get("vars", {}).get("test_case_id", "unknown"),
            description=data.get("description", ""),
            pass_=data.get("pass", False),
            score=score,
            assertions=assertions,
            latency_ms=data.get("latencyMs"),
            output=data.get("output"),
            error=data.get("error"),
            provider=data.get("provider"),
            metadata=data.get("metadata", {}),
        )

    @property
    def failed_assertions(self) -> list[AssertionResult]:
        """Get list of failed assertions."""
        return [a for a in self.assertions if not a.pass_]


@dataclass
class PromptfooResult:
    """Complete Promptfoo evaluation result."""

    timestamp: datetime
    config_path: str
    tests: list[TestResult]
    stats: dict[str, Any] = field(default_factory=dict)
    version: str | None = None

    @classmethod
    def from_output_file(cls, path: str) -> "PromptfooResult":
        """Load result from Promptfoo output file."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Output file not found: {path}")

        with path.open() as f:
            data = json.load(f)

        return cls.from_dict(data, str(path))

    @classmethod
    def from_dict(cls, data: dict[str, Any], config_path: str = "") -> "PromptfooResult":
        """Create from Promptfoo output dict."""
        results = data.get("results", [])
        tests = [TestResult.from_dict(r) for r in results]

        return cls(
            timestamp=datetime.utcnow(),
            config_path=config_path,
            tests=tests,
            stats=data.get("stats", {}),
            version=data.get("version"),
        )

    @property
    def total_tests(self) -> int:
        """Total number of tests."""
        return len(self.tests)

    @property
    def passed_tests(self) -> int:
        """Number of passed tests."""
        return sum(1 for t in self.tests if t.pass_)

    @property
    def failed_tests(self) -> int:
        """Number of failed tests."""
        return sum(1 for t in self.tests if not t.pass_)

    @property
    def pass_rate(self) -> float:
        """Pass rate as percentage."""
        if not self.tests:
            return 0.0
        return (self.passed_tests / self.total_tests) * 100

    @property
    def average_score(self) -> float:
        """Average score across all tests."""
        if not self.tests:
            return 0.0
        return sum(t.score for t in self.tests) / len(self.tests)

    @property
    def average_latency_ms(self) -> float | None:
        """Average latency in milliseconds."""
        latencies = [t.latency_ms for t in self.tests if t.latency_ms is not None]
        if not latencies:
            return None
        return sum(latencies) / len(latencies)

    def get_failed_tests(self) -> list[TestResult]:
        """Get list of failed tests."""
        return [t for t in self.tests if not t.pass_]

    def get_test_by_id(self, test_id: str) -> TestResult | None:
        """Get a specific test by ID."""
        for t in self.tests:
            if t.test_id == test_id:
                return t
        return None

    def to_summary(self) -> dict[str, Any]:
        """Create a summary dict."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "total_tests": self.total_tests,
            "passed": self.passed_tests,
            "failed": self.failed_tests,
            "pass_rate": f"{self.pass_rate:.1f}%",
            "average_score": f"{self.average_score:.3f}",
            "average_latency_ms": self.average_latency_ms,
        }


# =============================================================================
# Runner
# =============================================================================


class PromptfooRunner:
    """
    Runs Promptfoo evaluations.

    Wraps the promptfoo CLI and parses results.
    """

    def __init__(self, working_dir: str | Path | None = None):
        """
        Initialize the runner.

        Args:
            working_dir: Working directory for promptfoo commands
        """
        self.working_dir = Path(working_dir) if working_dir else Path.cwd()

    def run(self, config: RunConfig) -> PromptfooResult:
        """
        Run a Promptfoo evaluation.

        Args:
            config: Run configuration

        Returns:
            PromptfooResult with evaluation results

        Raises:
            RuntimeError: If promptfoo command fails
        """
        args = config.to_args()

        logger.info(
            "running_promptfoo",
            config=config.config_path,
            output=config.output_path,
        )

        try:
            result = subprocess.run(
                args,
                cwd=self.working_dir,
                capture_output=True,
                text=True,
                timeout=config.timeout_seconds,
            )

            if result.returncode != 0:
                logger.error(
                    "promptfoo_failed",
                    returncode=result.returncode,
                    stderr=result.stderr,
                )
                raise RuntimeError(f"Promptfoo failed: {result.stderr}")

            logger.info("promptfoo_completed", output=config.output_path)

            # Parse output
            output_path = self.working_dir / config.output_path
            return PromptfooResult.from_output_file(str(output_path))

        except subprocess.TimeoutExpired:
            logger.error("promptfoo_timeout", timeout=config.timeout_seconds)
            raise RuntimeError(f"Promptfoo timed out after {config.timeout_seconds}s")

        except FileNotFoundError:
            logger.error("promptfoo_not_found")
            raise RuntimeError("promptfoo CLI not found. Install with: npm install -g promptfoo")

    def run_from_file(self, config_path: str, **kwargs) -> PromptfooResult:
        """
        Run evaluation from a config file.

        Args:
            config_path: Path to promptfoo.yaml
            **kwargs: Additional RunConfig options

        Returns:
            PromptfooResult
        """
        config = RunConfig(config_path=config_path, **kwargs)
        return self.run(config)

    def check_installation(self) -> bool:
        """
        Check if promptfoo is installed.

        Returns:
            True if promptfoo is available
        """
        try:
            result = subprocess.run(
                ["promptfoo", "--version"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            return result.returncode == 0
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return False


# =============================================================================
# Result Analysis
# =============================================================================


def analyze_results(result: PromptfooResult) -> dict[str, Any]:
    """
    Analyze Promptfoo results for CI reporting.

    Args:
        result: Promptfoo evaluation result

    Returns:
        Analysis dict with metrics and insights
    """
    analysis = {
        "summary": result.to_summary(),
        "metrics": {},
        "failing_tests": [],
        "assertion_breakdown": {},
    }

    # Assertion type breakdown
    assertion_types: dict[str, dict[str, int]] = {}
    for test in result.tests:
        for assertion in test.assertions:
            if assertion.type not in assertion_types:
                assertion_types[assertion.type] = {"passed": 0, "failed": 0}
            if assertion.pass_:
                assertion_types[assertion.type]["passed"] += 1
            else:
                assertion_types[assertion.type]["failed"] += 1

    analysis["assertion_breakdown"] = assertion_types

    # Failing tests details
    for test in result.get_failed_tests():
        failure_info = {
            "test_id": test.test_id,
            "description": test.description,
            "score": test.score,
            "failed_assertions": [
                {
                    "type": a.type,
                    "reason": a.reason,
                }
                for a in test.failed_assertions
            ],
        }
        if test.error:
            failure_info["error"] = test.error
        analysis["failing_tests"].append(failure_info)

    # Calculate metrics
    analysis["metrics"] = {
        "pass_rate": result.pass_rate,
        "average_score": result.average_score,
        "average_latency_ms": result.average_latency_ms,
        "assertion_pass_rate": _calc_assertion_pass_rate(assertion_types),
    }

    return analysis


def _calc_assertion_pass_rate(assertion_types: dict[str, dict[str, int]]) -> float:
    """Calculate overall assertion pass rate."""
    total_passed = sum(v["passed"] for v in assertion_types.values())
    total_failed = sum(v["failed"] for v in assertion_types.values())
    total = total_passed + total_failed
    if total == 0:
        return 0.0
    return (total_passed / total) * 100


def format_ci_report(result: PromptfooResult) -> str:
    """
    Format results as a CI-friendly report.

    Args:
        result: Promptfoo evaluation result

    Returns:
        Formatted report string
    """
    lines = [
        "=" * 60,
        "PROMPTFOO EVALUATION REPORT",
        "=" * 60,
        "",
        f"Timestamp: {result.timestamp.isoformat()}",
        f"Tests: {result.passed_tests}/{result.total_tests} passed ({result.pass_rate:.1f}%)",
        f"Average Score: {result.average_score:.3f}",
    ]

    if result.average_latency_ms:
        lines.append(f"Average Latency: {result.average_latency_ms:.0f}ms")

    lines.append("")

    # Failed tests
    failed = result.get_failed_tests()
    if failed:
        lines.append(f"FAILED TESTS ({len(failed)}):")
        lines.append("-" * 40)
        for test in failed:
            lines.append(f"  - {test.test_id}: {test.description}")
            for assertion in test.failed_assertions:
                lines.append(f"      [{assertion.type}] {assertion.reason or 'Failed'}")
        lines.append("")

    # Status
    if result.pass_rate >= 90:
        lines.append("STATUS: PASS (>= 90% pass rate)")
    elif result.pass_rate >= 70:
        lines.append("STATUS: WARNING (70-90% pass rate)")
    else:
        lines.append("STATUS: FAIL (< 70% pass rate)")

    lines.append("=" * 60)

    return "\n".join(lines)
