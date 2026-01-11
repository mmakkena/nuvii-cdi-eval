"""
CSV report generator.

Generates CSV reports for spreadsheet analysis.
"""

import csv
import io
from typing import Any

from nuvii_eval.reporters.base import BaseReporter, ReportData


class CSVReporter(BaseReporter):
    """
    Generates CSV evaluation reports.

    Creates tabular CSV output for spreadsheet analysis.
    """

    def generate(self, data: ReportData) -> str:
        """Generate CSV report."""
        output = io.StringIO()
        writer = csv.writer(output)

        # Write header
        headers = self._get_headers(data)
        writer.writerow(headers)

        # Write data rows
        for result in data.results:
            row = self._format_row(result, headers)
            writer.writerow(row)

        return output.getvalue()

    def _get_headers(self, data: ReportData) -> list[str]:
        """Determine CSV headers from data."""
        base_headers = [
            "test_id",
            "pass",
            "score",
            "duration_ms",
        ]

        # Add metric columns if available
        metric_keys = set()
        for result in data.results:
            metrics = result.get("metrics", {})
            metric_keys.update(metrics.keys())

        # Add metadata columns
        metadata_keys = set()
        for result in data.results:
            metadata = result.get("metadata", {})
            metadata_keys.update(metadata.keys())

        headers = base_headers.copy()
        headers.extend(sorted(metric_keys))
        headers.extend([f"meta_{k}" for k in sorted(metadata_keys)])
        headers.append("errors")

        return headers

    def _format_row(self, result: dict[str, Any], headers: list[str]) -> list[str]:
        """Format a result as a CSV row."""
        row = []

        for header in headers:
            if header == "test_id":
                row.append(result.get("test_id", ""))
            elif header == "pass":
                row.append("1" if result.get("pass", False) else "0")
            elif header == "score":
                row.append(f"{result.get('score', 0):.4f}")
            elif header == "duration_ms":
                row.append(str(result.get("duration_ms", "")))
            elif header == "errors":
                errors = result.get("errors", [])
                row.append("; ".join(errors) if errors else "")
            elif header.startswith("meta_"):
                key = header[5:]  # Remove "meta_" prefix
                metadata = result.get("metadata", {})
                row.append(str(metadata.get(key, "")))
            else:
                # Metric column
                metrics = result.get("metrics", {})
                value = metrics.get(header)
                row.append(f"{value:.4f}" if value is not None else "")

        return row


class DetailedCSVReporter(CSVReporter):
    """
    Generates detailed CSV reports with additional columns.

    Includes assertion-level details and timing information.
    """

    def _get_headers(self, data: ReportData) -> list[str]:
        """Get extended headers."""
        headers = super()._get_headers(data)

        # Add assertion columns if available
        assertion_types = set()
        for result in data.results:
            for assertion in result.get("assertions", []):
                assertion_types.add(assertion.get("type", "unknown"))

        for atype in sorted(assertion_types):
            headers.append(f"assertion_{atype}")

        return headers

    def _format_row(self, result: dict[str, Any], headers: list[str]) -> list[str]:
        """Format with assertion details."""
        row = super()._format_row(result, headers)

        # Build assertion lookup
        assertion_results = {}
        for assertion in result.get("assertions", []):
            atype = assertion.get("type", "unknown")
            assertion_results[atype] = "1" if assertion.get("pass", False) else "0"

        # Add assertion columns
        for header in headers:
            if header.startswith("assertion_"):
                atype = header[10:]  # Remove "assertion_" prefix
                row.append(assertion_results.get(atype, ""))

        return row
