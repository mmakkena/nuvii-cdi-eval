"""
Result reporters and exporters.

Provides report generation in various formats.
"""

from nuvii_eval.reporters.base import (
    BaseReporter,
    ReportData,
    ReportGenerator,
    ReportOptions,
)
from nuvii_eval.reporters.csv_reporter import CSVReporter, DetailedCSVReporter
from nuvii_eval.reporters.dashboard import CombinedReporter, DashboardGenerator
from nuvii_eval.reporters.html_reporter import HTMLReporter
from nuvii_eval.reporters.json_reporter import JSONReporter
from nuvii_eval.reporters.markdown_reporter import MarkdownReporter, PRCommentGenerator

__all__ = [
    # Base
    "BaseReporter",
    "ReportData",
    "ReportGenerator",
    "ReportOptions",
    # HTML
    "HTMLReporter",
    # JSON
    "JSONReporter",
    # CSV
    "CSVReporter",
    "DetailedCSVReporter",
    # Markdown
    "MarkdownReporter",
    "PRCommentGenerator",
    # Dashboard
    "DashboardGenerator",
    "CombinedReporter",
]
