"""
Compare command for regression detection.

Provides commands to compare evaluation runs and detect regressions.
"""

from pathlib import Path
from typing import Optional

import typer
from rich.console import Console

app = typer.Typer(help="Compare evaluation runs for regression detection")
console = Console()


@app.command("runs")
def compare_runs(
    baseline: Path = typer.Argument(
        ...,
        help="Path to baseline results file",
        exists=True,
    ),
    current: Path = typer.Argument(
        ...,
        help="Path to current results file",
        exists=True,
    ),
    output: Optional[Path] = typer.Option(
        None,
        "--output",
        help="Output file for comparison report",
    ),
    threshold: float = typer.Option(
        0.1,
        "--threshold",
        help="Score drop threshold for regression detection",
        min=0.01,
        max=0.5,
    ),
    fail_on_regression: bool = typer.Option(
        True,
        "--fail-on-regression",
        help="Exit with error code on blocking regressions",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        help="Show detailed regression information",
    ),
) -> None:
    """
    Compare two evaluation runs for regressions.

    Analyzes the difference between baseline and current results to detect
    quality regressions, score drops, and new failures.

    Examples:

        # Basic comparison
        nuvii-eval compare runs baseline.json current.json

        # With custom threshold
        nuvii-eval compare runs baseline.json current.json -t 0.05

        # Export comparison report
        nuvii-eval compare runs baseline.json current.json -o comparison.md
    """
    import json

    from nuvii_eval.promptfoo import (
        DetectorConfig,
        PromptfooResult,
        RegressionDetector,
        check_for_blockers,
    )

    try:
        # Load results
        with open(baseline) as f:
            baseline_data = json.load(f)
        with open(current) as f:
            current_data = json.load(f)

        baseline_result = PromptfooResult.from_dict(baseline_data, str(baseline))
        current_result = PromptfooResult.from_dict(current_data, str(current))

        # Configure detector
        config = DetectorConfig(
            score_drop_threshold=threshold,
        )
        detector = RegressionDetector(config)

        # Run comparison
        report = detector.compare(baseline_result, current_result)

        # Display report
        console.print(report.format_report())

        if verbose:
            _display_detailed_regressions(report)

        # Export if requested
        if output:
            _export_comparison(report, output)
            console.print(f"\n[green]Report saved to {output}[/green]")

        # Check for blockers
        if fail_on_regression and check_for_blockers(report):
            console.print("\n[red]Blocking regressions detected![/red]")
            raise typer.Exit(1)

    except FileNotFoundError as e:
        console.print(f"[red]File not found: {e}[/red]")
        raise typer.Exit(1)
    except json.JSONDecodeError as e:
        console.print(f"[red]Invalid JSON file: {e}[/red]")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"[red]Comparison failed: {e}[/red]")
        raise typer.Exit(1)


@app.command("trend")
def compare_trend(
    results_dir: Path = typer.Argument(
        ...,
        help="Directory containing historical result files",
        exists=True,
    ),
    output: Optional[Path] = typer.Option(
        None,
        "--output",
        help="Output file for trend report",
    ),
    last_n: int = typer.Option(
        10,
        "--last",
        help="Number of recent runs to analyze",
        min=2,
        max=100,
    ),
) -> None:
    """
    Analyze trends across multiple evaluation runs.

    Generates a trend report showing score changes over time.
    """
    import json
    from datetime import datetime

    from rich.table import Table

    try:
        # Find result files
        result_files = sorted(
            results_dir.glob("*.json"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )[:last_n]

        if len(result_files) < 2:
            console.print("[yellow]Need at least 2 result files for trend analysis[/yellow]")
            raise typer.Exit(0)

        # Load and analyze results
        results = []
        for path in reversed(result_files):  # Oldest first
            with open(path) as f:
                data = json.load(f)
            results.append({
                "file": path.name,
                "timestamp": datetime.fromtimestamp(path.stat().st_mtime),
                "pass_rate": data.get("stats", {}).get("pass_rate", 0),
                "avg_score": data.get("stats", {}).get("average_score", 0),
                "total_tests": len(data.get("results", [])),
            })

        # Display trend table
        table = Table(title="Evaluation Trend Analysis")
        table.add_column("Date", style="cyan")
        table.add_column("File", style="dim")
        table.add_column("Tests", justify="right")
        table.add_column("Pass Rate", justify="right")
        table.add_column("Avg Score", justify="right")
        table.add_column("Change", justify="right")

        prev_rate = None
        for r in results:
            change = ""
            if prev_rate is not None:
                delta = r["pass_rate"] - prev_rate
                if delta > 0:
                    change = f"[green]+{delta:.1f}%[/green]"
                elif delta < 0:
                    change = f"[red]{delta:.1f}%[/red]"
                else:
                    change = "[dim]--[/dim]"
            prev_rate = r["pass_rate"]

            table.add_row(
                r["timestamp"].strftime("%Y-%m-%d %H:%M"),
                r["file"],
                str(r["total_tests"]),
                f"{r['pass_rate']:.1f}%",
                f"{r['avg_score']:.3f}",
                change,
            )

        console.print(table)

        # Export if requested
        if output:
            import csv

            with open(output, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=["timestamp", "file", "pass_rate", "avg_score", "total_tests"])
                writer.writeheader()
                for r in results:
                    writer.writerow({
                        "timestamp": r["timestamp"].isoformat(),
                        "file": r["file"],
                        "pass_rate": r["pass_rate"],
                        "avg_score": r["avg_score"],
                        "total_tests": r["total_tests"],
                    })
            console.print(f"\n[green]Trend data saved to {output}[/green]")

    except Exception as e:
        console.print(f"[red]Trend analysis failed: {e}[/red]")
        raise typer.Exit(1)


@app.command("ci")
def compare_ci(
    baseline_ref: str = typer.Argument(
        "main",
        help="Git ref for baseline (branch, tag, or commit)",
    ),
    results_path: str = typer.Option(
        "results/latest.json",
        "--results",
        help="Path to current results file",
    ),
    artifact_dir: str = typer.Option(
        ".eval-artifacts",
        "--artifacts",
        help="Directory for baseline artifacts",
    ),
) -> None:
    """
    CI-optimized comparison against baseline.

    Fetches baseline results from git and compares with current run.
    Designed for use in CI/CD pipelines.
    """
    import json
    import subprocess
    import tempfile

    from nuvii_eval.promptfoo import (
        PromptfooResult,
        RegressionDetector,
        check_for_blockers,
        get_regression_summary,
    )

    try:
        # Fetch baseline from git
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as tmp:
            result = subprocess.run(
                ["git", "show", f"{baseline_ref}:{results_path}"],
                capture_output=True,
                text=True,
            )
            if result.returncode != 0:
                console.print(f"[yellow]No baseline found at {baseline_ref}:{results_path}[/yellow]")
                console.print("[yellow]Treating this as a baseline run[/yellow]")
                raise typer.Exit(0)

            tmp.write(result.stdout)
            baseline_path = tmp.name

        # Load results
        with open(baseline_path) as f:
            baseline_data = json.load(f)
        with open(results_path) as f:
            current_data = json.load(f)

        baseline_result = PromptfooResult.from_dict(baseline_data)
        current_result = PromptfooResult.from_dict(current_data)

        # Compare
        detector = RegressionDetector()
        report = detector.compare(baseline_result, current_result)

        # Output for CI
        summary = get_regression_summary(report)
        console.print(f"\n[bold]Regression Summary:[/bold] {summary}")

        if report.has_regressions:
            console.print("\n[yellow]Regressions detected:[/yellow]")
            for r in report.regressions[:5]:  # Show top 5
                console.print(f"  - [{r.severity.value}] {r.description}")
            if len(report.regressions) > 5:
                console.print(f"  ... and {len(report.regressions) - 5} more")

        if report.improvements:
            console.print(f"\n[green]Improvements: {report.improvement_count}[/green]")

        # Exit with error on blocking regressions
        if check_for_blockers(report):
            console.print("\n[red]BLOCKED: Critical or high severity regressions detected[/red]")
            raise typer.Exit(1)

        console.print("\n[green]PASSED: No blocking regressions[/green]")

    except subprocess.SubprocessError as e:
        console.print(f"[red]Git operation failed: {e}[/red]")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"[red]CI comparison failed: {e}[/red]")
        raise typer.Exit(1)


def _display_detailed_regressions(report) -> None:
    """Display detailed regression information."""
    from rich.table import Table

    if not report.regressions:
        return

    console.print("\n[bold]Detailed Regressions:[/bold]")

    table = Table()
    table.add_column("Severity", style="bold")
    table.add_column("Type")
    table.add_column("Test ID")
    table.add_column("Description")
    table.add_column("Change")

    for r in report.regressions:
        severity_style = {
            "critical": "red bold",
            "high": "red",
            "medium": "yellow",
            "low": "dim",
        }.get(r.severity.value, "")

        table.add_row(
            f"[{severity_style}]{r.severity.value.upper()}[/{severity_style}]",
            r.type.value,
            r.test_id or "-",
            r.description,
            f"{r.baseline_value} -> {r.current_value}",
        )

    console.print(table)


def _export_comparison(report, output: Path) -> None:
    """Export comparison report."""
    suffix = output.suffix.lower()

    if suffix == ".md":
        content = _format_markdown_report(report)
    elif suffix == ".json":
        import json

        content = json.dumps(report.to_dict(), indent=2)
    else:
        content = report.format_report()

    output.write_text(content)


def _format_markdown_report(report) -> str:
    """Format report as Markdown."""
    lines = [
        "# Regression Analysis Report",
        "",
        f"**Generated:** {report.timestamp.isoformat()}",
        "",
        "## Summary",
        "",
        f"- **Regressions:** {report.regression_count}",
        f"- **Improvements:** {report.improvement_count}",
        f"- **Status:** {'BLOCKED' if report.has_blocking_regressions else 'PASS'}",
        "",
    ]

    if report.regressions:
        lines.extend([
            "## Regressions",
            "",
            "| Severity | Type | Test ID | Description |",
            "|----------|------|---------|-------------|",
        ])
        for r in report.regressions:
            test_id = r.test_id or "-"
            lines.append(f"| {r.severity.value} | {r.type.value} | {test_id} | {r.description} |")
        lines.append("")

    if report.improvements:
        lines.extend([
            "## Improvements",
            "",
            "| Description | Change |",
            "|-------------|--------|",
        ])
        for r in report.improvements:
            lines.append(f"| {r.description} | {r.baseline_value} -> {r.current_value} |")
        lines.append("")

    return "\n".join(lines)
