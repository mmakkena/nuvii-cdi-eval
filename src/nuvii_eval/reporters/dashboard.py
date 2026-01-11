"""
Dashboard generator.

Generates static HTML dashboard for evaluation metrics.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import structlog

logger = structlog.get_logger(__name__)


class DashboardGenerator:
    """
    Generates a static HTML dashboard.

    Creates an interactive dashboard showing evaluation trends and metrics.
    """

    def __init__(self, output_dir: str):
        """
        Initialize the generator.

        Args:
            output_dir: Directory for dashboard files
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.history_file = self.output_dir / "history.json"

    def update(self, results_path: str) -> None:
        """
        Update dashboard with new results.

        Args:
            results_path: Path to results JSON file
        """
        # Load new results
        with open(results_path) as f:
            new_results = json.load(f)

        # Load or create history
        history = self._load_history()

        # Add new entry
        entry = self._create_history_entry(new_results)
        history["entries"].append(entry)

        # Keep last 100 entries
        history["entries"] = history["entries"][-100:]

        # Save history
        self._save_history(history)

        # Generate dashboard
        self._generate_dashboard(history)

        logger.info("dashboard_updated", entries=len(history["entries"]))

    def _load_history(self) -> dict[str, Any]:
        """Load history from file."""
        if self.history_file.exists():
            with open(self.history_file) as f:
                return json.load(f)
        return {"entries": []}

    def _save_history(self, history: dict[str, Any]) -> None:
        """Save history to file."""
        with open(self.history_file, "w") as f:
            json.dump(history, f, indent=2)

    def _create_history_entry(self, results: dict[str, Any]) -> dict[str, Any]:
        """Create a history entry from results."""
        stats = results.get("stats", {})
        return {
            "timestamp": results.get("timestamp", datetime.utcnow().isoformat()),
            "total": stats.get("total", len(results.get("results", []))),
            "passed": stats.get("passed", 0),
            "failed": stats.get("failed", 0),
            "pass_rate": stats.get("pass_rate", 0),
            "average_score": stats.get("average_score", 0),
        }

    def _generate_dashboard(self, history: dict[str, Any]) -> None:
        """Generate the dashboard HTML."""
        entries = history["entries"]

        # Prepare chart data
        timestamps = [e["timestamp"][:10] for e in entries]  # Date only
        pass_rates = [e["pass_rate"] for e in entries]
        avg_scores = [e["average_score"] for e in entries]

        # Latest stats
        latest = entries[-1] if entries else {}

        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>CDI Evaluation Dashboard</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        :root {{
            --primary: #3b82f6;
            --success: #10b981;
            --warning: #f59e0b;
            --danger: #ef4444;
            --bg: #f3f4f6;
            --card-bg: #ffffff;
            --text: #1f2937;
            --text-muted: #6b7280;
        }}

        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}

        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: var(--bg);
            color: var(--text);
            line-height: 1.6;
        }}

        .dashboard {{
            max-width: 1400px;
            margin: 0 auto;
            padding: 2rem;
        }}

        header {{
            text-align: center;
            margin-bottom: 2rem;
        }}

        header h1 {{
            font-size: 2rem;
            margin-bottom: 0.5rem;
        }}

        .last-updated {{
            color: var(--text-muted);
            font-size: 0.875rem;
        }}

        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 1.5rem;
            margin-bottom: 2rem;
        }}

        .stat-card {{
            background: var(--card-bg);
            border-radius: 0.75rem;
            padding: 1.5rem;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        }}

        .stat-value {{
            font-size: 2.5rem;
            font-weight: 700;
            line-height: 1;
        }}

        .stat-label {{
            color: var(--text-muted);
            font-size: 0.875rem;
            margin-top: 0.5rem;
        }}

        .stat-card.success .stat-value {{ color: var(--success); }}
        .stat-card.warning .stat-value {{ color: var(--warning); }}
        .stat-card.danger .stat-value {{ color: var(--danger); }}

        .charts-section {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(500px, 1fr));
            gap: 1.5rem;
        }}

        .chart-card {{
            background: var(--card-bg);
            border-radius: 0.75rem;
            padding: 1.5rem;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        }}

        .chart-card h2 {{
            font-size: 1.125rem;
            margin-bottom: 1rem;
        }}

        .chart-container {{
            position: relative;
            height: 300px;
        }}

        footer {{
            text-align: center;
            margin-top: 2rem;
            color: var(--text-muted);
            font-size: 0.875rem;
        }}

        @media (max-width: 768px) {{
            .charts-section {{
                grid-template-columns: 1fr;
            }}
        }}
    </style>
</head>
<body>
    <div class="dashboard">
        <header>
            <h1>CDI Evaluation Dashboard</h1>
            <p class="last-updated">Last updated: {latest.get('timestamp', 'N/A')}</p>
        </header>

        <div class="stats-grid">
            <div class="stat-card">
                <div class="stat-value">{latest.get('total', 0)}</div>
                <div class="stat-label">Total Tests</div>
            </div>
            <div class="stat-card success">
                <div class="stat-value">{latest.get('passed', 0)}</div>
                <div class="stat-label">Passed</div>
            </div>
            <div class="stat-card danger">
                <div class="stat-value">{latest.get('failed', 0)}</div>
                <div class="stat-label">Failed</div>
            </div>
            <div class="stat-card {'success' if latest.get('pass_rate', 0) >= 90 else 'warning' if latest.get('pass_rate', 0) >= 70 else 'danger'}">
                <div class="stat-value">{latest.get('pass_rate', 0):.1f}%</div>
                <div class="stat-label">Pass Rate</div>
            </div>
        </div>

        <div class="charts-section">
            <div class="chart-card">
                <h2>Pass Rate Trend</h2>
                <div class="chart-container">
                    <canvas id="passRateChart"></canvas>
                </div>
            </div>
            <div class="chart-card">
                <h2>Average Score Trend</h2>
                <div class="chart-container">
                    <canvas id="avgScoreChart"></canvas>
                </div>
            </div>
        </div>

        <footer>
            <p>Nuvii CDI Evaluation Framework</p>
        </footer>
    </div>

    <script>
        const timestamps = {json.dumps(timestamps)};
        const passRates = {json.dumps(pass_rates)};
        const avgScores = {json.dumps(avg_scores)};

        // Pass Rate Chart
        new Chart(document.getElementById('passRateChart'), {{
            type: 'line',
            data: {{
                labels: timestamps,
                datasets: [{{
                    label: 'Pass Rate (%)',
                    data: passRates,
                    borderColor: '#3b82f6',
                    backgroundColor: 'rgba(59, 130, 246, 0.1)',
                    fill: true,
                    tension: 0.3,
                }}]
            }},
            options: {{
                responsive: true,
                maintainAspectRatio: false,
                scales: {{
                    y: {{
                        min: 0,
                        max: 100,
                    }}
                }},
                plugins: {{
                    legend: {{
                        display: false,
                    }}
                }}
            }}
        }});

        // Average Score Chart
        new Chart(document.getElementById('avgScoreChart'), {{
            type: 'line',
            data: {{
                labels: timestamps,
                datasets: [{{
                    label: 'Average Score',
                    data: avgScores,
                    borderColor: '#10b981',
                    backgroundColor: 'rgba(16, 185, 129, 0.1)',
                    fill: true,
                    tension: 0.3,
                }}]
            }},
            options: {{
                responsive: true,
                maintainAspectRatio: false,
                scales: {{
                    y: {{
                        min: 0,
                        max: 1,
                    }}
                }},
                plugins: {{
                    legend: {{
                        display: false,
                    }}
                }}
            }}
        }});
    </script>
</body>
</html>"""

        # Write dashboard
        index_path = self.output_dir / "index.html"
        index_path.write_text(html)


class CombinedReporter:
    """
    Generates combined reports from multiple evaluation runs.

    Used for batch evaluation reports.
    """

    def __init__(self, results: list[tuple[str, Any]]):
        """
        Initialize the reporter.

        Args:
            results: List of (name, BatchResult) tuples
        """
        self.results = results

    def generate_html(self, output_path: str) -> None:
        """Generate combined HTML report."""
        from nuvii_eval.reporters.html_reporter import HTMLReporter
        from nuvii_eval.reporters.base import ReportData, ReportOptions

        # Combine all results
        all_evaluations = []
        total_passed = 0
        total_failed = 0

        for name, result in self.results:
            for eval in result.evaluations:
                eval_dict = eval.to_dict()
                eval_dict["batch_name"] = name
                all_evaluations.append(eval_dict)

            total_passed += result.passed_count
            total_failed += result.failed_count

        total = total_passed + total_failed
        pass_rate = (total_passed / total * 100) if total > 0 else 0

        # Create combined report data
        data = ReportData(
            timestamp=datetime.utcnow(),
            title="Combined Batch Evaluation Report",
            summary={
                "total": total,
                "passed": total_passed,
                "failed": total_failed,
                "pass_rate": pass_rate,
                "average_score": sum(e.get("score", 0) for e in all_evaluations) / total if total > 0 else 0,
                "batches": len(self.results),
            },
            results=all_evaluations,
            metrics={
                "pass_rate": pass_rate,
            },
        )

        options = ReportOptions(title="Combined Batch Evaluation Report")
        reporter = HTMLReporter(options)
        reporter.save(data, output_path)
