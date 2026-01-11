"""
JSON report generator.

Generates structured JSON reports for programmatic consumption.
"""

import json
from typing import Any

from nuvii_eval.reporters.base import BaseReporter, ReportData


class JSONReporter(BaseReporter):
    """
    Generates JSON evaluation reports.

    Creates structured JSON output suitable for further processing.
    """

    def generate(self, data: ReportData) -> str:
        """Generate JSON report."""
        report = {
            "title": data.title,
            "generated_at": data.timestamp.isoformat(),
            "summary": data.summary,
            "metrics": data.metrics,
            "metadata": data.metadata,
        }

        if self.options.include_details:
            report["results"] = data.results

        if self.options.include_raw_data:
            report["raw_data"] = self._get_raw_data(data)

        return json.dumps(report, indent=2, default=str)

    def _get_raw_data(self, data: ReportData) -> dict[str, Any]:
        """Get additional raw data for analysis."""
        # Calculate distributions
        score_distribution = {}
        task_type_distribution = {}

        for result in data.results:
            # Score bins
            score = result.get("score", 0)
            bin_key = f"{int(score * 10) * 10}-{int(score * 10) * 10 + 10}%"
            score_distribution[bin_key] = score_distribution.get(bin_key, 0) + 1

            # Task types
            task_type = result.get("metadata", {}).get("task_type", "unknown")
            if task_type not in task_type_distribution:
                task_type_distribution[task_type] = {"total": 0, "passed": 0}
            task_type_distribution[task_type]["total"] += 1
            if result.get("pass", False):
                task_type_distribution[task_type]["passed"] += 1

        return {
            "score_distribution": score_distribution,
            "task_type_distribution": task_type_distribution,
            "test_count": len(data.results),
        }
