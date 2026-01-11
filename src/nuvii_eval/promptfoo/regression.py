"""
Regression detection utilities for Promptfoo.

Compares evaluation results across runs to detect regressions.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

import structlog

from nuvii_eval.promptfoo.runner import PromptfooResult, TestResult

logger = structlog.get_logger(__name__)


# =============================================================================
# Regression Types
# =============================================================================


class RegressionSeverity(str, Enum):
    """Severity level of a regression."""

    CRITICAL = "critical"  # Major functionality broken
    HIGH = "high"  # Significant quality degradation
    MEDIUM = "medium"  # Noticeable quality change
    LOW = "low"  # Minor quality change
    INFO = "info"  # Informational change


class RegressionType(str, Enum):
    """Type of regression detected."""

    NEW_FAILURE = "new_failure"  # Test that was passing now fails
    SCORE_DROP = "score_drop"  # Significant score decrease
    LATENCY_INCREASE = "latency_increase"  # Significant latency increase
    NEW_ASSERTION_FAILURE = "new_assertion_failure"  # New assertion type failing
    PASS_RATE_DROP = "pass_rate_drop"  # Overall pass rate decreased


# =============================================================================
# Regression Models
# =============================================================================


@dataclass
class Regression:
    """
    A single detected regression.

    Represents a change in behavior between baseline and current results.
    """

    type: RegressionType
    severity: RegressionSeverity
    test_id: str | None
    description: str
    baseline_value: Any
    current_value: Any
    delta: float | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "type": self.type.value,
            "severity": self.severity.value,
            "test_id": self.test_id,
            "description": self.description,
            "baseline_value": self.baseline_value,
            "current_value": self.current_value,
            "delta": self.delta,
        }


@dataclass
class RegressionReport:
    """
    Complete regression analysis report.

    Contains all detected regressions and summary statistics.
    """

    timestamp: datetime
    baseline_timestamp: datetime | None
    regressions: list[Regression]
    improvements: list[Regression]  # Positive changes
    stats: dict[str, Any] = field(default_factory=dict)

    @property
    def has_regressions(self) -> bool:
        """Check if any regressions were detected."""
        return len(self.regressions) > 0

    @property
    def has_critical_regressions(self) -> bool:
        """Check if any critical regressions were detected."""
        return any(r.severity == RegressionSeverity.CRITICAL for r in self.regressions)

    @property
    def has_blocking_regressions(self) -> bool:
        """Check if any blocking (critical/high) regressions were detected."""
        blocking = {RegressionSeverity.CRITICAL, RegressionSeverity.HIGH}
        return any(r.severity in blocking for r in self.regressions)

    @property
    def regression_count(self) -> int:
        """Total number of regressions."""
        return len(self.regressions)

    @property
    def improvement_count(self) -> int:
        """Total number of improvements."""
        return len(self.improvements)

    def get_regressions_by_severity(
        self,
        severity: RegressionSeverity,
    ) -> list[Regression]:
        """Get regressions of a specific severity."""
        return [r for r in self.regressions if r.severity == severity]

    def get_regressions_by_type(
        self,
        reg_type: RegressionType,
    ) -> list[Regression]:
        """Get regressions of a specific type."""
        return [r for r in self.regressions if r.type == reg_type]

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "baseline_timestamp": self.baseline_timestamp.isoformat() if self.baseline_timestamp else None,
            "has_regressions": self.has_regressions,
            "has_blocking_regressions": self.has_blocking_regressions,
            "regression_count": self.regression_count,
            "improvement_count": self.improvement_count,
            "regressions": [r.to_dict() for r in self.regressions],
            "improvements": [r.to_dict() for r in self.improvements],
            "stats": self.stats,
        }

    def format_report(self) -> str:
        """Format as a human-readable report."""
        lines = [
            "=" * 60,
            "REGRESSION ANALYSIS REPORT",
            "=" * 60,
            "",
            f"Timestamp: {self.timestamp.isoformat()}",
        ]

        if self.baseline_timestamp:
            lines.append(f"Baseline: {self.baseline_timestamp.isoformat()}")

        lines.extend([
            "",
            f"Regressions: {self.regression_count}",
            f"Improvements: {self.improvement_count}",
            "",
        ])

        # Regressions by severity
        if self.regressions:
            lines.append("REGRESSIONS:")
            lines.append("-" * 40)

            for severity in RegressionSeverity:
                regs = self.get_regressions_by_severity(severity)
                if regs:
                    lines.append(f"\n  [{severity.value.upper()}]")
                    for r in regs:
                        lines.append(f"    - {r.description}")
                        if r.test_id:
                            lines.append(f"      Test: {r.test_id}")
                        lines.append(f"      {r.baseline_value} -> {r.current_value}")

            lines.append("")

        # Improvements
        if self.improvements:
            lines.append("IMPROVEMENTS:")
            lines.append("-" * 40)
            for r in self.improvements:
                lines.append(f"  - {r.description}")
                lines.append(f"    {r.baseline_value} -> {r.current_value}")
            lines.append("")

        # Status
        if self.has_critical_regressions:
            lines.append("STATUS: BLOCKED (Critical regressions detected)")
        elif self.has_blocking_regressions:
            lines.append("STATUS: BLOCKED (High severity regressions detected)")
        elif self.has_regressions:
            lines.append("STATUS: WARNING (Non-blocking regressions detected)")
        else:
            lines.append("STATUS: PASS (No regressions detected)")

        lines.append("=" * 60)

        return "\n".join(lines)


# =============================================================================
# Regression Detector
# =============================================================================


@dataclass
class DetectorConfig:
    """Configuration for regression detection."""

    score_drop_threshold: float = 0.1  # 10% score drop
    latency_increase_threshold: float = 0.5  # 50% latency increase
    pass_rate_drop_threshold: float = 5.0  # 5% pass rate drop
    min_tests_for_comparison: int = 1


class RegressionDetector:
    """
    Detects regressions between evaluation runs.

    Compares baseline and current results to identify quality changes.
    """

    def __init__(self, config: DetectorConfig | None = None):
        """
        Initialize the detector.

        Args:
            config: Detection configuration
        """
        self.config = config or DetectorConfig()

    def compare(
        self,
        baseline: PromptfooResult,
        current: PromptfooResult,
    ) -> RegressionReport:
        """
        Compare two evaluation runs.

        Args:
            baseline: Baseline (previous) evaluation result
            current: Current evaluation result

        Returns:
            RegressionReport with detected changes
        """
        regressions: list[Regression] = []
        improvements: list[Regression] = []

        # Compare overall pass rate
        pass_rate_changes = self._compare_pass_rates(baseline, current)
        regressions.extend([r for r in pass_rate_changes if r.delta and r.delta < 0])
        improvements.extend([r for r in pass_rate_changes if r.delta and r.delta > 0])

        # Compare individual tests
        test_changes = self._compare_tests(baseline, current)
        regressions.extend([r for r in test_changes if r.severity != RegressionSeverity.INFO])
        improvements.extend([r for r in test_changes if r.severity == RegressionSeverity.INFO and r.delta and r.delta > 0])

        # Compare latency
        latency_changes = self._compare_latency(baseline, current)
        regressions.extend([r for r in latency_changes if r.delta and r.delta > 0])
        improvements.extend([r for r in latency_changes if r.delta and r.delta < 0])

        # Compile stats
        stats = {
            "baseline_pass_rate": baseline.pass_rate,
            "current_pass_rate": current.pass_rate,
            "baseline_avg_score": baseline.average_score,
            "current_avg_score": current.average_score,
            "baseline_tests": baseline.total_tests,
            "current_tests": current.total_tests,
        }

        return RegressionReport(
            timestamp=current.timestamp,
            baseline_timestamp=baseline.timestamp,
            regressions=regressions,
            improvements=improvements,
            stats=stats,
        )

    def _compare_pass_rates(
        self,
        baseline: PromptfooResult,
        current: PromptfooResult,
    ) -> list[Regression]:
        """Compare overall pass rates."""
        changes = []

        delta = current.pass_rate - baseline.pass_rate

        if abs(delta) >= self.config.pass_rate_drop_threshold:
            if delta < 0:
                severity = RegressionSeverity.HIGH if abs(delta) >= 10 else RegressionSeverity.MEDIUM
                changes.append(Regression(
                    type=RegressionType.PASS_RATE_DROP,
                    severity=severity,
                    test_id=None,
                    description=f"Pass rate dropped by {abs(delta):.1f}%",
                    baseline_value=f"{baseline.pass_rate:.1f}%",
                    current_value=f"{current.pass_rate:.1f}%",
                    delta=delta,
                ))
            else:
                changes.append(Regression(
                    type=RegressionType.PASS_RATE_DROP,
                    severity=RegressionSeverity.INFO,
                    test_id=None,
                    description=f"Pass rate improved by {delta:.1f}%",
                    baseline_value=f"{baseline.pass_rate:.1f}%",
                    current_value=f"{current.pass_rate:.1f}%",
                    delta=delta,
                ))

        return changes

    def _compare_tests(
        self,
        baseline: PromptfooResult,
        current: PromptfooResult,
    ) -> list[Regression]:
        """Compare individual test results."""
        changes = []

        # Build lookup for baseline tests
        baseline_tests = {t.test_id: t for t in baseline.tests}

        for current_test in current.tests:
            baseline_test = baseline_tests.get(current_test.test_id)

            if baseline_test is None:
                # New test, skip comparison
                continue

            # Check for new failures
            if baseline_test.pass_ and not current_test.pass_:
                changes.append(Regression(
                    type=RegressionType.NEW_FAILURE,
                    severity=RegressionSeverity.HIGH,
                    test_id=current_test.test_id,
                    description=f"Test '{current_test.description}' now failing",
                    baseline_value="PASS",
                    current_value="FAIL",
                    delta=-1.0,
                ))

            # Check for score drops
            score_delta = current_test.score - baseline_test.score
            if score_delta < -self.config.score_drop_threshold:
                severity = RegressionSeverity.MEDIUM if abs(score_delta) < 0.3 else RegressionSeverity.HIGH
                changes.append(Regression(
                    type=RegressionType.SCORE_DROP,
                    severity=severity,
                    test_id=current_test.test_id,
                    description=f"Score dropped for '{current_test.description}'",
                    baseline_value=f"{baseline_test.score:.3f}",
                    current_value=f"{current_test.score:.3f}",
                    delta=score_delta,
                ))
            elif score_delta > self.config.score_drop_threshold:
                # Improvement
                changes.append(Regression(
                    type=RegressionType.SCORE_DROP,
                    severity=RegressionSeverity.INFO,
                    test_id=current_test.test_id,
                    description=f"Score improved for '{current_test.description}'",
                    baseline_value=f"{baseline_test.score:.3f}",
                    current_value=f"{current_test.score:.3f}",
                    delta=score_delta,
                ))

            # Check for new assertion failures
            baseline_failed_types = {a.type for a in baseline_test.assertions if not a.pass_}
            current_failed_types = {a.type for a in current_test.assertions if not a.pass_}
            new_failures = current_failed_types - baseline_failed_types

            for failure_type in new_failures:
                changes.append(Regression(
                    type=RegressionType.NEW_ASSERTION_FAILURE,
                    severity=RegressionSeverity.MEDIUM,
                    test_id=current_test.test_id,
                    description=f"New '{failure_type}' assertion failure",
                    baseline_value="PASS",
                    current_value="FAIL",
                    delta=-1.0,
                ))

        return changes

    def _compare_latency(
        self,
        baseline: PromptfooResult,
        current: PromptfooResult,
    ) -> list[Regression]:
        """Compare latency metrics."""
        changes = []

        baseline_latency = baseline.average_latency_ms
        current_latency = current.average_latency_ms

        if baseline_latency is None or current_latency is None:
            return changes

        if baseline_latency > 0:
            latency_increase = (current_latency - baseline_latency) / baseline_latency

            if abs(latency_increase) >= self.config.latency_increase_threshold:
                if latency_increase > 0:
                    severity = RegressionSeverity.LOW if latency_increase < 1.0 else RegressionSeverity.MEDIUM
                    changes.append(Regression(
                        type=RegressionType.LATENCY_INCREASE,
                        severity=severity,
                        test_id=None,
                        description=f"Average latency increased by {latency_increase*100:.0f}%",
                        baseline_value=f"{baseline_latency:.0f}ms",
                        current_value=f"{current_latency:.0f}ms",
                        delta=latency_increase,
                    ))
                else:
                    changes.append(Regression(
                        type=RegressionType.LATENCY_INCREASE,
                        severity=RegressionSeverity.INFO,
                        test_id=None,
                        description=f"Average latency decreased by {abs(latency_increase)*100:.0f}%",
                        baseline_value=f"{baseline_latency:.0f}ms",
                        current_value=f"{current_latency:.0f}ms",
                        delta=latency_increase,
                    ))

        return changes


# =============================================================================
# Convenience Functions
# =============================================================================


def compare_results(
    baseline: PromptfooResult,
    current: PromptfooResult,
    config: DetectorConfig | None = None,
) -> RegressionReport:
    """
    Compare two evaluation results for regressions.

    Args:
        baseline: Baseline evaluation result
        current: Current evaluation result
        config: Detection configuration

    Returns:
        RegressionReport with analysis
    """
    detector = RegressionDetector(config)
    return detector.compare(baseline, current)


def check_for_blockers(report: RegressionReport) -> bool:
    """
    Check if regression report contains blocking issues.

    Args:
        report: Regression report to check

    Returns:
        True if there are blocking regressions
    """
    return report.has_blocking_regressions


def get_regression_summary(report: RegressionReport) -> str:
    """
    Get a short summary of regressions.

    Args:
        report: Regression report

    Returns:
        Summary string
    """
    if not report.has_regressions:
        return "No regressions detected"

    parts = []
    for severity in [RegressionSeverity.CRITICAL, RegressionSeverity.HIGH, RegressionSeverity.MEDIUM, RegressionSeverity.LOW]:
        count = len(report.get_regressions_by_severity(severity))
        if count > 0:
            parts.append(f"{count} {severity.value}")

    return f"Regressions: {', '.join(parts)}"
