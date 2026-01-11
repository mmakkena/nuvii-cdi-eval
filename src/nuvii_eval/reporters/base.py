"""
Base reporter classes.

Provides abstract base for all report generators.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import structlog

logger = structlog.get_logger(__name__)


# =============================================================================
# Report Options
# =============================================================================


@dataclass
class ReportOptions:
    """Configuration options for report generation."""

    title: str = "CDI Evaluation Report"
    include_details: bool = True
    include_charts: bool = True
    include_raw_data: bool = False
    max_failures_shown: int = 20
    theme: str = "default"
    custom_css: str | None = None
    custom_js: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


# =============================================================================
# Report Data
# =============================================================================


@dataclass
class ReportData:
    """Normalized data structure for report generation."""

    timestamp: datetime
    title: str
    summary: dict[str, Any]
    results: list[dict[str, Any]]
    metrics: dict[str, float]
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_batch_result(cls, result: Any, title: str = "Evaluation Report") -> "ReportData":
        """Create from BatchResult."""
        return cls(
            timestamp=result.timestamp,
            title=title,
            summary={
                "total": result.total_count,
                "passed": result.passed_count,
                "failed": result.failed_count,
                "pass_rate": result.pass_rate,
                "average_score": result.average_score,
                "duration_seconds": result.duration_seconds,
            },
            results=[e.to_dict() for e in result.evaluations],
            metrics={
                "pass_rate": result.pass_rate,
                "average_score": result.average_score,
            },
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any], title: str = "Evaluation Report") -> "ReportData":
        """Create from dictionary (loaded from JSON)."""
        stats = data.get("stats", {})
        return cls(
            timestamp=datetime.fromisoformat(data.get("timestamp", datetime.utcnow().isoformat())),
            title=title,
            summary={
                "total": stats.get("total", len(data.get("results", []))),
                "passed": stats.get("passed", 0),
                "failed": stats.get("failed", 0),
                "pass_rate": stats.get("pass_rate", 0),
                "average_score": stats.get("average_score", 0),
                "duration_seconds": data.get("duration_seconds"),
            },
            results=data.get("results", []),
            metrics={
                "pass_rate": stats.get("pass_rate", 0),
                "average_score": stats.get("average_score", 0),
            },
            metadata=data.get("metadata", {}),
        )


# =============================================================================
# Base Reporter
# =============================================================================


class BaseReporter(ABC):
    """
    Abstract base class for report generators.

    All reporters inherit from this and implement format-specific generation.
    """

    def __init__(self, options: ReportOptions | None = None):
        """
        Initialize the reporter.

        Args:
            options: Report generation options
        """
        self.options = options or ReportOptions()

    @abstractmethod
    def generate(self, data: ReportData) -> str:
        """
        Generate report content.

        Args:
            data: Report data

        Returns:
            Report content as string
        """
        pass

    def save(self, data: ReportData, path: str) -> None:
        """
        Generate and save report to file.

        Args:
            data: Report data
            path: Output file path
        """
        content = self.generate(data)
        Path(path).write_text(content)
        logger.info("report_saved", path=path, format=self.__class__.__name__)


# =============================================================================
# Report Generator (Factory)
# =============================================================================


class ReportGenerator:
    """
    Factory for generating reports in various formats.

    Provides a unified interface for report generation.
    """

    def __init__(self, options: ReportOptions | None = None):
        """
        Initialize the generator.

        Args:
            options: Report generation options
        """
        self.options = options or ReportOptions()

    def generate(
        self,
        results_path: str,
        output_path: str,
        format: str = "html",
    ) -> None:
        """
        Generate a report from results file.

        Args:
            results_path: Path to results JSON file
            output_path: Output file path
            format: Report format (html, json, csv, markdown)
        """
        import json

        # Load results
        with open(results_path) as f:
            data = json.load(f)

        # Create report data
        report_data = ReportData.from_dict(data, self.options.title)

        # Get appropriate reporter
        reporter = self._get_reporter(format)

        # Generate and save
        reporter.save(report_data, output_path)

    def _get_reporter(self, format: str) -> BaseReporter:
        """Get reporter for format."""
        from nuvii_eval.reporters.csv_reporter import CSVReporter
        from nuvii_eval.reporters.html_reporter import HTMLReporter
        from nuvii_eval.reporters.json_reporter import JSONReporter
        from nuvii_eval.reporters.markdown_reporter import MarkdownReporter

        reporters = {
            "html": HTMLReporter,
            "json": JSONReporter,
            "csv": CSVReporter,
            "markdown": MarkdownReporter,
            "md": MarkdownReporter,
        }

        if format not in reporters:
            raise ValueError(f"Unknown format: {format}. Supported: {list(reporters.keys())}")

        return reporters[format](self.options)
