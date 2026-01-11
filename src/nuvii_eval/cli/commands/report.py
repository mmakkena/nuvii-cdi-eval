"""
Report command for generating evaluation reports.

Provides commands to generate various report formats.
"""

from enum import Enum
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console

app = typer.Typer(help="Generate evaluation reports")
console = Console()


class ReportFormat(str, Enum):
    """Report format options."""

    HTML = "html"
    MARKDOWN = "markdown"
    JSON = "json"
    CSV = "csv"
    PDF = "pdf"


@app.command("generate")
def generate_report(
    results: Path = typer.Argument(
        ...,
        help="Path to results file or directory",
        exists=True,
    ),
    output: Path = typer.Option(
        Path("./report.html"),
        "--output",
        help="Output file path",
    ),
    format: ReportFormat = typer.Option(
        ReportFormat.HTML,
        "--format",
        help="Report format",
    ),
    title: str = typer.Option(
        "CDI Evaluation Report",
        "--title",
        help="Report title",
    ),
    include_details: bool = typer.Option(
        True,
        "--details",
        help="Include detailed test results",
    ),
    include_charts: bool = typer.Option(
        True,
        "--charts",
        help="Include charts and visualizations (HTML only)",
    ),
) -> None:
    """
    Generate an evaluation report from results.

    Examples:

        # Generate HTML report
        nuvii-eval report generate results.json -o report.html

        # Generate Markdown summary
        nuvii-eval report generate results.json -f markdown -o summary.md

        # Generate CSV for data analysis
        nuvii-eval report generate results.json -f csv -o data.csv
    """
    from nuvii_eval.reporters import ReportGenerator, ReportOptions

    try:
        options = ReportOptions(
            title=title,
            include_details=include_details,
            include_charts=include_charts,
        )

        generator = ReportGenerator(options)

        with console.status("[bold cyan]Generating report..."):
            generator.generate(
                results_path=str(results),
                output_path=str(output),
                format=format.value,
            )

        console.print(f"[green]Report generated: {output}[/green]")

        # Show file size
        size = output.stat().st_size
        if size > 1024 * 1024:
            size_str = f"{size / (1024 * 1024):.1f} MB"
        elif size > 1024:
            size_str = f"{size / 1024:.1f} KB"
        else:
            size_str = f"{size} bytes"
        console.print(f"[dim]Size: {size_str}[/dim]")

    except Exception as e:
        console.print(f"[red]Report generation failed: {e}[/red]")
        raise typer.Exit(1)


@app.command("pr")
def generate_pr_comment(
    results: Path = typer.Argument(
        ...,
        help="Path to current results file",
        exists=True,
    ),
    baseline: Optional[Path] = typer.Option(
        None,
        "--baseline",
        help="Path to baseline results for comparison",
    ),
    output: Optional[Path] = typer.Option(
        None,
        "--output",
        help="Output file (defaults to stdout)",
    ),
    max_failures: int = typer.Option(
        10,
        "--max-failures",
        help="Maximum failed tests to include",
    ),
) -> None:
    """
    Generate a PR comment summary.

    Creates a concise Markdown summary suitable for GitHub/GitLab PR comments.
    """
    import json

    from nuvii_eval.reporters import PRCommentGenerator

    try:
        with open(results) as f:
            current_data = json.load(f)

        baseline_data = None
        if baseline and baseline.exists():
            with open(baseline) as f:
                baseline_data = json.load(f)

        generator = PRCommentGenerator(max_failures=max_failures)
        comment = generator.generate(current_data, baseline_data)

        if output:
            output.write_text(comment)
            console.print(f"[green]PR comment saved to {output}[/green]")
        else:
            console.print(comment)

    except Exception as e:
        console.print(f"[red]PR comment generation failed: {e}[/red]")
        raise typer.Exit(1)


@app.command("dashboard")
def update_dashboard(
    results: Path = typer.Argument(
        ...,
        help="Path to results file",
        exists=True,
    ),
    dashboard_dir: Path = typer.Option(
        Path("./dashboard"),
        "--dir",
        help="Dashboard output directory",
    ),
) -> None:
    """
    Update evaluation dashboard with latest results.

    Generates static HTML dashboard files that can be hosted.
    """
    from nuvii_eval.reporters import DashboardGenerator

    try:
        dashboard_dir.mkdir(parents=True, exist_ok=True)

        generator = DashboardGenerator(output_dir=str(dashboard_dir))

        with console.status("[bold cyan]Updating dashboard..."):
            generator.update(str(results))

        console.print(f"[green]Dashboard updated: {dashboard_dir}[/green]")
        console.print(f"[dim]Open {dashboard_dir}/index.html in a browser[/dim]")

    except Exception as e:
        console.print(f"[red]Dashboard update failed: {e}[/red]")
        raise typer.Exit(1)


@app.command("summary")
def show_summary(
    results: Path = typer.Argument(
        ...,
        help="Path to results file",
        exists=True,
    ),
    by_task: bool = typer.Option(
        False,
        "--by-task",
        help="Group results by task type",
    ),
    by_specialty: bool = typer.Option(
        False,
        "--by-specialty",
        help="Group results by medical specialty",
    ),
) -> None:
    """
    Display a summary of evaluation results.

    Shows key metrics in the terminal.
    """
    import json

    from rich.panel import Panel
    from rich.table import Table

    try:
        with open(results) as f:
            data = json.load(f)

        results_list = data.get("results", [])
        stats = data.get("stats", {})

        # Main summary
        summary_table = Table(show_header=False, box=None)
        summary_table.add_column("Metric", style="cyan")
        summary_table.add_column("Value", style="green")

        summary_table.add_row("Total Tests", str(len(results_list)))
        summary_table.add_row("Passed", str(sum(1 for r in results_list if r.get("pass", False))))
        summary_table.add_row("Failed", str(sum(1 for r in results_list if not r.get("pass", True))))
        summary_table.add_row("Pass Rate", f"{stats.get('pass_rate', 0):.1f}%")
        summary_table.add_row("Avg Score", f"{stats.get('average_score', 0):.3f}")

        console.print(Panel(summary_table, title="[bold]Evaluation Summary[/bold]"))

        # Group by task type
        if by_task:
            _show_grouped_summary(results_list, "task_type", "By Task Type")

        # Group by specialty
        if by_specialty:
            _show_grouped_summary(results_list, "specialty", "By Specialty")

    except Exception as e:
        console.print(f"[red]Failed to load results: {e}[/red]")
        raise typer.Exit(1)


def _show_grouped_summary(results_list: list, group_key: str, title: str) -> None:
    """Show summary grouped by a key."""
    from rich.table import Table

    groups = {}
    for r in results_list:
        key = r.get("metadata", {}).get(group_key, "unknown")
        if key not in groups:
            groups[key] = {"passed": 0, "failed": 0, "total_score": 0}
        groups[key]["total_score"] += r.get("score", 0)
        if r.get("pass", False):
            groups[key]["passed"] += 1
        else:
            groups[key]["failed"] += 1

    table = Table(title=title)
    table.add_column("Group", style="cyan")
    table.add_column("Passed", justify="right", style="green")
    table.add_column("Failed", justify="right", style="red")
    table.add_column("Pass Rate", justify="right")
    table.add_column("Avg Score", justify="right")

    for key, stats in sorted(groups.items()):
        total = stats["passed"] + stats["failed"]
        pass_rate = (stats["passed"] / total * 100) if total > 0 else 0
        avg_score = stats["total_score"] / total if total > 0 else 0

        table.add_row(
            key,
            str(stats["passed"]),
            str(stats["failed"]),
            f"{pass_rate:.1f}%",
            f"{avg_score:.3f}",
        )

    console.print(table)
