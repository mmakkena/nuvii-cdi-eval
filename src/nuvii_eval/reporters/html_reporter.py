"""
HTML report generator.

Generates interactive HTML reports with charts and styling.
"""

from nuvii_eval.reporters.base import BaseReporter, ReportData, ReportOptions


class HTMLReporter(BaseReporter):
    """
    Generates HTML evaluation reports.

    Creates styled HTML reports with optional charts and interactivity.
    """

    def generate(self, data: ReportData) -> str:
        """Generate HTML report."""
        # Build HTML sections
        header = self._build_header(data)
        summary = self._build_summary(data)
        charts = self._build_charts(data) if self.options.include_charts else ""
        details = self._build_details(data) if self.options.include_details else ""
        footer = self._build_footer(data)

        # Combine into full HTML
        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{data.title}</title>
    {self._get_styles()}
    {self._get_chart_scripts() if self.options.include_charts else ""}
</head>
<body>
    <div class="container">
        {header}
        {summary}
        {charts}
        {details}
        {footer}
    </div>
    {self._get_scripts()}
</body>
</html>"""

        return html

    def _build_header(self, data: ReportData) -> str:
        """Build header section."""
        status_class = "status-pass" if data.summary.get("pass_rate", 0) >= 70 else "status-fail"
        status_text = "PASS" if data.summary.get("pass_rate", 0) >= 70 else "FAIL"

        return f"""
        <header>
            <h1>{data.title}</h1>
            <div class="meta">
                <span class="timestamp">Generated: {data.timestamp.strftime("%Y-%m-%d %H:%M:%S")}</span>
                <span class="status {status_class}">{status_text}</span>
            </div>
        </header>"""

    def _build_summary(self, data: ReportData) -> str:
        """Build summary section."""
        s = data.summary
        pass_rate = s.get("pass_rate", 0)
        pass_class = "good" if pass_rate >= 90 else "warning" if pass_rate >= 70 else "bad"

        duration = s.get("duration_seconds")
        duration_str = f"{duration:.1f}s" if duration else "N/A"

        return f"""
        <section class="summary">
            <h2>Summary</h2>
            <div class="metrics-grid">
                <div class="metric">
                    <div class="metric-value">{s.get('total', 0)}</div>
                    <div class="metric-label">Total Tests</div>
                </div>
                <div class="metric good">
                    <div class="metric-value">{s.get('passed', 0)}</div>
                    <div class="metric-label">Passed</div>
                </div>
                <div class="metric bad">
                    <div class="metric-value">{s.get('failed', 0)}</div>
                    <div class="metric-label">Failed</div>
                </div>
                <div class="metric {pass_class}">
                    <div class="metric-value">{pass_rate:.1f}%</div>
                    <div class="metric-label">Pass Rate</div>
                </div>
                <div class="metric">
                    <div class="metric-value">{s.get('average_score', 0):.3f}</div>
                    <div class="metric-label">Avg Score</div>
                </div>
                <div class="metric">
                    <div class="metric-value">{duration_str}</div>
                    <div class="metric-label">Duration</div>
                </div>
            </div>
        </section>"""

    def _build_charts(self, data: ReportData) -> str:
        """Build charts section."""
        # Calculate data for charts
        passed = data.summary.get("passed", 0)
        failed = data.summary.get("failed", 0)

        # Group by task type if available
        task_types = {}
        for result in data.results:
            task_type = result.get("metadata", {}).get("task_type", "unknown")
            if task_type not in task_types:
                task_types[task_type] = {"passed": 0, "failed": 0}
            if result.get("pass", False):
                task_types[task_type]["passed"] += 1
            else:
                task_types[task_type]["failed"] += 1

        task_labels = list(task_types.keys())
        task_passed = [task_types[t]["passed"] for t in task_labels]
        task_failed = [task_types[t]["failed"] for t in task_labels]

        return f"""
        <section class="charts">
            <h2>Visualizations</h2>
            <div class="charts-grid">
                <div class="chart-container">
                    <canvas id="passFailChart"></canvas>
                </div>
                <div class="chart-container">
                    <canvas id="taskTypeChart"></canvas>
                </div>
            </div>
        </section>
        <script>
            // Pass/Fail Pie Chart
            new Chart(document.getElementById('passFailChart'), {{
                type: 'doughnut',
                data: {{
                    labels: ['Passed', 'Failed'],
                    datasets: [{{
                        data: [{passed}, {failed}],
                        backgroundColor: ['#10b981', '#ef4444'],
                    }}]
                }},
                options: {{
                    responsive: true,
                    plugins: {{
                        title: {{
                            display: true,
                            text: 'Pass/Fail Distribution'
                        }}
                    }}
                }}
            }});

            // Task Type Bar Chart
            new Chart(document.getElementById('taskTypeChart'), {{
                type: 'bar',
                data: {{
                    labels: {task_labels},
                    datasets: [
                        {{
                            label: 'Passed',
                            data: {task_passed},
                            backgroundColor: '#10b981',
                        }},
                        {{
                            label: 'Failed',
                            data: {task_failed},
                            backgroundColor: '#ef4444',
                        }}
                    ]
                }},
                options: {{
                    responsive: true,
                    plugins: {{
                        title: {{
                            display: true,
                            text: 'Results by Task Type'
                        }}
                    }},
                    scales: {{
                        x: {{ stacked: true }},
                        y: {{ stacked: true }}
                    }}
                }}
            }});
        </script>"""

    def _build_details(self, data: ReportData) -> str:
        """Build detailed results section."""
        # Get failed tests
        failed_tests = [r for r in data.results if not r.get("pass", True)]
        shown_failures = failed_tests[:self.options.max_failures_shown]

        if not failed_tests:
            return """
            <section class="details">
                <h2>Test Results</h2>
                <p class="success-message">All tests passed!</p>
            </section>"""

        rows = []
        for result in shown_failures:
            test_id = result.get("test_id", "unknown")
            score = result.get("score", 0)
            errors = result.get("errors", [])
            error_str = "<br>".join(errors[:3]) if errors else "No details"

            rows.append(f"""
                <tr>
                    <td>{test_id}</td>
                    <td>{score:.3f}</td>
                    <td class="error-cell">{error_str}</td>
                </tr>""")

        more_count = len(failed_tests) - len(shown_failures)
        more_msg = f"<p class='more-indicator'>... and {more_count} more failures</p>" if more_count > 0 else ""

        return f"""
        <section class="details">
            <h2>Failed Tests ({len(failed_tests)})</h2>
            <table class="results-table">
                <thead>
                    <tr>
                        <th>Test ID</th>
                        <th>Score</th>
                        <th>Errors</th>
                    </tr>
                </thead>
                <tbody>
                    {"".join(rows)}
                </tbody>
            </table>
            {more_msg}
        </section>"""

    def _build_footer(self, data: ReportData) -> str:
        """Build footer section."""
        return f"""
        <footer>
            <p>Generated by Nuvii CDI Evaluation Framework</p>
            <p class="timestamp">{data.timestamp.isoformat()}</p>
        </footer>"""

    def _get_styles(self) -> str:
        """Get CSS styles."""
        custom_css = self.options.custom_css or ""

        return f"""
        <style>
            :root {{
                --primary: #3b82f6;
                --success: #10b981;
                --warning: #f59e0b;
                --danger: #ef4444;
                --gray-50: #f9fafb;
                --gray-100: #f3f4f6;
                --gray-200: #e5e7eb;
                --gray-600: #4b5563;
                --gray-800: #1f2937;
            }}

            * {{
                margin: 0;
                padding: 0;
                box-sizing: border-box;
            }}

            body {{
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                line-height: 1.6;
                color: var(--gray-800);
                background: var(--gray-50);
            }}

            .container {{
                max-width: 1200px;
                margin: 0 auto;
                padding: 2rem;
            }}

            header {{
                text-align: center;
                margin-bottom: 2rem;
                padding-bottom: 1rem;
                border-bottom: 1px solid var(--gray-200);
            }}

            header h1 {{
                font-size: 2rem;
                margin-bottom: 0.5rem;
            }}

            .meta {{
                display: flex;
                justify-content: center;
                gap: 1rem;
                color: var(--gray-600);
            }}

            .status {{
                padding: 0.25rem 0.75rem;
                border-radius: 9999px;
                font-weight: 600;
                font-size: 0.875rem;
            }}

            .status-pass {{
                background: var(--success);
                color: white;
            }}

            .status-fail {{
                background: var(--danger);
                color: white;
            }}

            section {{
                background: white;
                border-radius: 0.5rem;
                padding: 1.5rem;
                margin-bottom: 1.5rem;
                box-shadow: 0 1px 3px rgba(0,0,0,0.1);
            }}

            section h2 {{
                font-size: 1.25rem;
                margin-bottom: 1rem;
                color: var(--gray-800);
            }}

            .metrics-grid {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(120px, 1fr));
                gap: 1rem;
            }}

            .metric {{
                text-align: center;
                padding: 1rem;
                background: var(--gray-50);
                border-radius: 0.5rem;
            }}

            .metric-value {{
                font-size: 1.5rem;
                font-weight: 700;
            }}

            .metric-label {{
                font-size: 0.875rem;
                color: var(--gray-600);
            }}

            .metric.good .metric-value {{ color: var(--success); }}
            .metric.warning .metric-value {{ color: var(--warning); }}
            .metric.bad .metric-value {{ color: var(--danger); }}

            .charts-grid {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
                gap: 1.5rem;
            }}

            .chart-container {{
                position: relative;
                height: 300px;
            }}

            .results-table {{
                width: 100%;
                border-collapse: collapse;
            }}

            .results-table th,
            .results-table td {{
                padding: 0.75rem;
                text-align: left;
                border-bottom: 1px solid var(--gray-200);
            }}

            .results-table th {{
                background: var(--gray-50);
                font-weight: 600;
            }}

            .error-cell {{
                font-size: 0.875rem;
                color: var(--danger);
            }}

            .success-message {{
                color: var(--success);
                font-weight: 500;
            }}

            .more-indicator {{
                color: var(--gray-600);
                font-style: italic;
                margin-top: 1rem;
            }}

            footer {{
                text-align: center;
                color: var(--gray-600);
                font-size: 0.875rem;
                padding-top: 1rem;
                border-top: 1px solid var(--gray-200);
            }}

            {custom_css}
        </style>"""

    def _get_chart_scripts(self) -> str:
        """Get chart library scripts."""
        return """
        <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>"""

    def _get_scripts(self) -> str:
        """Get additional scripts."""
        custom_js = self.options.custom_js or ""
        return f"""
        <script>
            {custom_js}
        </script>"""
