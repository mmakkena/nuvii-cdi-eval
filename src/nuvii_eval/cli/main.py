"""
Main CLI application for Nuvii CDI Evaluation Framework.

Provides the root command and subcommand registration.
"""

from typing import Optional

import typer
from rich.console import Console

from nuvii_eval.cli import commands

app = typer.Typer(
    name="nuvii-eval",
    help="Nuvii CDI Agent Evaluation Framework",
    add_completion=False,
    rich_markup_mode="rich",
)

console = Console()

# Register subcommands
app.add_typer(commands.run_app, name="run", help="Run evaluations")
app.add_typer(commands.compare_app, name="compare", help="Compare evaluation runs")
app.add_typer(commands.report_app, name="report", help="Generate reports")
app.add_typer(commands.dataset_app, name="dataset", help="Manage test datasets")


def _version_callback(value: bool) -> None:
    """Print version and exit."""
    if value:
        from nuvii_eval import __version__
        console.print(f"nuvii-eval version {__version__}")
        raise typer.Exit()


@app.callback(invoke_without_command=True)
def main(
    ctx: typer.Context,
    version: Optional[bool] = typer.Option(
        None,
        "--version",
        "-v",
        help="Show version and exit",
        is_eager=True,
        callback=_version_callback,
    ),
) -> None:
    """
    Nuvii CDI Agent Evaluation Framework.

    Run evaluations, compare results, and generate reports for
    CDI agent quality assessment.
    """
    if ctx.invoked_subcommand is None and not version:
        console.print(ctx.get_help())


def get_settings():
    """Lazy import of settings to avoid circular imports."""
    from nuvii_eval.config import get_settings as _get_settings
    return _get_settings()


@app.command()
def info() -> None:
    """Show framework information and configuration."""
    from rich.panel import Panel
    from rich.table import Table

    try:
        settings = get_settings()

        table = Table(show_header=False, box=None)
        table.add_column("Setting", style="cyan")
        table.add_column("Value", style="green")

        table.add_row("Environment", settings.environment)
        table.add_row("Debug Mode", str(settings.debug))
        table.add_row("Log Level", settings.log_level)
        table.add_row("API Base URL", settings.nuvii_api.base_url)
        table.add_row("API Timeout", f"{settings.nuvii_api.timeout}s")
        table.add_row("Phoenix Enabled", str(settings.phoenix.enabled))
        if settings.phoenix.enabled:
            table.add_row("Phoenix Endpoint", settings.phoenix.collector_endpoint)

        console.print(Panel(table, title="[bold]Nuvii Eval Configuration[/bold]"))

    except Exception as e:
        console.print(f"[red]Error loading configuration: {e}[/red]")
        raise typer.Exit(1)


if __name__ == "__main__":
    app()
