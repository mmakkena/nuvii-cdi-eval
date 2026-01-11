"""
Run command for executing evaluations.

Provides commands to run evaluations against the Nuvii CDI API.
"""

from enum import Enum
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

app = typer.Typer(help="Run evaluations against the Nuvii CDI API")
console = Console()


class TaskType(str, Enum):
    """Evaluation task types."""

    ICD = "icd"
    HCC = "hcc"
    GAP = "gap"
    QUERY = "query"
    EM = "em"
    ALL = "all"


class OutputFormat(str, Enum):
    """Output format options."""

    JSON = "json"
    CSV = "csv"
    HTML = "html"
    MARKDOWN = "markdown"


@app.command("eval")
def run_evaluation(
    dataset: Path = typer.Argument(
        ...,
        help="Path to dataset file (JSON/YAML) or directory",
        exists=True,
    ),
    task_type: TaskType = typer.Option(
        TaskType.ALL,
        "--task",
        help="Type of evaluation task to run",
    ),
    output: Optional[Path] = typer.Option(
        None,
        "--output",
        help="Output file path for results",
    ),
    output_format: OutputFormat = typer.Option(
        OutputFormat.JSON,
        "--format",
        help="Output format",
    ),
    max_concurrency: int = typer.Option(
        5,
        "--concurrency",
        help="Maximum concurrent API requests",
        min=1,
        max=20,
    ),
    timeout: int = typer.Option(
        60,
        "--timeout",
        help="Request timeout in seconds",
        min=10,
        max=300,
    ),
    fail_fast: bool = typer.Option(
        False,
        "--fail-fast",
        help="Stop on first failure",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        help="Enable verbose output",
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Validate configuration without running",
    ),
) -> None:
    """
    Run evaluation against the Nuvii CDI API.

    Examples:

        # Run all evaluations on a dataset
        nuvii-eval run eval datasets/icd_tests.json

        # Run specific task type
        nuvii-eval run eval datasets/ --task icd

        # Export results to HTML
        nuvii-eval run eval datasets/test.json -o report.html -f html
    """
    from nuvii_eval.runner import BatchRunner, RunConfig

    try:
        # Build configuration
        config = RunConfig(
            dataset_path=str(dataset),
            task_type=task_type.value if task_type != TaskType.ALL else None,
            max_concurrency=max_concurrency,
            timeout_seconds=timeout,
            fail_fast=fail_fast,
            verbose=verbose,
        )

        if dry_run:
            console.print("[yellow]Dry run mode - validating configuration...[/yellow]")
            # Validate dataset loads correctly
            from nuvii_eval.datasets import load_dataset
            test_cases = load_dataset(str(dataset))
            console.print(f"[green]Validated {len(test_cases)} test cases[/green]")
            return

        # Run evaluation
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            progress.add_task("Running evaluation...", total=None)

            runner = BatchRunner(config)
            result = runner.run()

        # Display summary
        _display_summary(result)

        # Export results
        if output:
            _export_results(result, output, output_format)
            console.print(f"[green]Results saved to {output}[/green]")

        # Exit with error if failures
        if result.failed_count > 0 and fail_fast:
            raise typer.Exit(1)

    except FileNotFoundError as e:
        console.print(f"[red]Dataset not found: {e}[/red]")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"[red]Evaluation failed: {e}[/red]")
        if verbose:
            console.print_exception()
        raise typer.Exit(1)


@app.command("batch")
def run_batch(
    config_file: Path = typer.Argument(
        ...,
        help="Path to batch configuration file (YAML)",
        exists=True,
    ),
    output_dir: Path = typer.Option(
        Path("./results"),
        "--output-dir",
        help="Output directory for results",
    ),
    parallel: bool = typer.Option(
        False,
        "--parallel",
        help="Run task types in parallel",
    ),
) -> None:
    """
    Run batch evaluation from configuration file.

    The configuration file should specify multiple datasets and task types
    to evaluate in a single run.
    """
    import yaml

    from nuvii_eval.runner import BatchRunner

    try:
        # Load batch configuration
        with open(config_file) as f:
            batch_config = yaml.safe_load(f)

        console.print(f"[cyan]Running batch evaluation from {config_file}[/cyan]")

        # Create output directory
        output_dir.mkdir(parents=True, exist_ok=True)

        # Run each evaluation
        results = []
        for eval_config in batch_config.get("evaluations", []):
            name = eval_config.get("name", "unnamed")
            console.print(f"\n[bold]Running: {name}[/bold]")

            runner = BatchRunner.from_dict(eval_config)
            result = runner.run()
            results.append((name, result))

            _display_summary(result)

        # Generate combined report
        from nuvii_eval.reporters import CombinedReporter

        reporter = CombinedReporter(results)
        report_path = output_dir / "batch_report.html"
        reporter.generate_html(str(report_path))

        console.print(f"\n[green]Batch complete. Report: {report_path}[/green]")

    except Exception as e:
        console.print(f"[red]Batch evaluation failed: {e}[/red]")
        raise typer.Exit(1)


@app.command("promptfoo")
def run_promptfoo(
    config: Path = typer.Argument(
        ...,
        help="Path to Promptfoo configuration file",
        exists=True,
    ),
    output: Optional[Path] = typer.Option(
        None,
        "--output",
        help="Output file for results",
    ),
    no_cache: bool = typer.Option(
        False,
        "--no-cache",
        help="Disable result caching",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        help="Enable verbose output",
    ),
) -> None:
    """
    Run evaluation using Promptfoo.

    This wraps the Promptfoo CLI for CDI-specific evaluations.
    """
    from nuvii_eval.promptfoo import PromptfooRunner, RunConfig as PfRunConfig

    try:
        runner = PromptfooRunner()

        # Check installation
        if not runner.check_installation():
            console.print(
                "[red]Promptfoo not installed. Run: npm install -g promptfoo[/red]"
            )
            raise typer.Exit(1)

        pf_config = PfRunConfig(
            config_path=str(config),
            output_path=str(output) if output else "promptfoo_output.json",
            no_cache=no_cache,
            verbose=verbose,
        )

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            progress.add_task("Running Promptfoo evaluation...", total=None)
            result = runner.run(pf_config)

        # Display results
        from nuvii_eval.promptfoo import format_ci_report

        console.print(format_ci_report(result))

    except Exception as e:
        console.print(f"[red]Promptfoo evaluation failed: {e}[/red]")
        raise typer.Exit(1)


def _display_summary(result) -> None:
    """Display evaluation result summary."""
    from rich.panel import Panel
    from rich.table import Table

    table = Table(show_header=False, box=None)
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")

    table.add_row("Total Tests", str(result.total_count))
    table.add_row("Passed", f"[green]{result.passed_count}[/green]")
    table.add_row("Failed", f"[red]{result.failed_count}[/red]")
    table.add_row("Pass Rate", f"{result.pass_rate:.1f}%")
    table.add_row("Avg Score", f"{result.average_score:.3f}")
    if result.duration_seconds:
        table.add_row("Duration", f"{result.duration_seconds:.1f}s")

    status = "[green]PASS[/green]" if result.passed else "[red]FAIL[/red]"
    console.print(Panel(table, title=f"[bold]Evaluation Result: {status}[/bold]"))


def _export_results(result, output: Path, format: OutputFormat) -> None:
    """Export results to file."""
    from nuvii_eval.reporters import (
        CSVReporter,
        HTMLReporter,
        JSONReporter,
        MarkdownReporter,
    )

    reporters = {
        OutputFormat.JSON: JSONReporter,
        OutputFormat.CSV: CSVReporter,
        OutputFormat.HTML: HTMLReporter,
        OutputFormat.MARKDOWN: MarkdownReporter,
    }

    reporter_class = reporters[format]
    reporter = reporter_class(result)
    reporter.save(str(output))
