"""
Dataset command for managing test datasets.

Provides commands to validate, inspect, and manage test datasets.
"""

from enum import Enum
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console

app = typer.Typer(help="Manage test datasets")
console = Console()


class DatasetFormat(str, Enum):
    """Dataset format options."""

    JSON = "json"
    YAML = "yaml"


@app.command("validate")
def validate_dataset(
    path: Path = typer.Argument(
        ...,
        help="Path to dataset file or directory",
        exists=True,
    ),
    strict: bool = typer.Option(
        False,
        "--strict",
        help="Enable strict validation",
    ),
    fix: bool = typer.Option(
        False,
        "--fix",
        help="Attempt to fix common issues",
    ),
) -> None:
    """
    Validate a test dataset.

    Checks that all test cases conform to the schema and have valid data.

    Examples:

        # Validate a single file
        nuvii-eval dataset validate datasets/icd_tests.json

        # Validate all files in a directory
        nuvii-eval dataset validate datasets/

        # Strict validation with detailed errors
        nuvii-eval dataset validate datasets/ --strict
    """
    from nuvii_eval.datasets import DatasetValidator

    try:
        validator = DatasetValidator(strict=strict)

        if path.is_file():
            files = [path]
        else:
            files = list(path.glob("**/*.json")) + list(path.glob("**/*.yaml"))

        if not files:
            console.print("[yellow]No dataset files found[/yellow]")
            raise typer.Exit(0)

        total_errors = 0
        total_warnings = 0

        for file_path in files:
            console.print(f"\n[bold]Validating: {file_path}[/bold]")

            result = validator.validate(str(file_path))

            if result.is_valid:
                console.print(f"  [green]Valid[/green] - {result.test_count} test cases")
            else:
                console.print(f"  [red]Invalid[/red]")
                for error in result.errors[:10]:  # Show first 10 errors
                    console.print(f"    [red]Error:[/red] {error}")
                    total_errors += 1
                if len(result.errors) > 10:
                    console.print(f"    ... and {len(result.errors) - 10} more errors")

            for warning in result.warnings[:5]:
                console.print(f"    [yellow]Warning:[/yellow] {warning}")
                total_warnings += 1

            if fix and result.fixable_issues:
                console.print(f"  [cyan]Fixing {len(result.fixable_issues)} issues...[/cyan]")
                validator.fix(str(file_path), result.fixable_issues)
                console.print(f"  [green]Fixed[/green]")

        # Summary
        console.print("\n[bold]Summary:[/bold]")
        console.print(f"  Files: {len(files)}")
        console.print(f"  Errors: {total_errors}")
        console.print(f"  Warnings: {total_warnings}")

        if total_errors > 0:
            raise typer.Exit(1)

    except Exception as e:
        console.print(f"[red]Validation failed: {e}[/red]")
        raise typer.Exit(1)


@app.command("inspect")
def inspect_dataset(
    path: Path = typer.Argument(
        ...,
        help="Path to dataset file",
        exists=True,
    ),
    show_samples: int = typer.Option(
        3,
        "--samples",
        help="Number of sample test cases to show",
        min=0,
        max=10,
    ),
) -> None:
    """
    Inspect a dataset and show statistics.

    Displays information about the dataset structure and content.
    """
    from rich.panel import Panel
    from rich.table import Table
    from rich.syntax import Syntax

    from nuvii_eval.datasets import load_dataset

    try:
        test_cases = load_dataset(str(path))

        # Overview
        overview = Table(show_header=False, box=None)
        overview.add_column("Property", style="cyan")
        overview.add_column("Value", style="green")

        overview.add_row("File", str(path))
        overview.add_row("Total Test Cases", str(len(test_cases)))

        # Count by task type
        task_types = {}
        specialties = {}
        complexities = {}

        for tc in test_cases:
            # Get task type from the class name
            task_type = type(tc).__name__.replace("TestCase", "").lower()
            task_types[task_type] = task_types.get(task_type, 0) + 1

            if hasattr(tc, "specialty"):
                spec = tc.specialty.value
                specialties[spec] = specialties.get(spec, 0) + 1

            if hasattr(tc, "complexity"):
                comp = tc.complexity.value
                complexities[comp] = complexities.get(comp, 0) + 1

        console.print(Panel(overview, title="[bold]Dataset Overview[/bold]"))

        # Task types breakdown
        if task_types:
            task_table = Table(title="By Task Type")
            task_table.add_column("Type", style="cyan")
            task_table.add_column("Count", justify="right")
            task_table.add_column("Percentage", justify="right")

            for task, count in sorted(task_types.items()):
                pct = count / len(test_cases) * 100
                task_table.add_row(task, str(count), f"{pct:.1f}%")

            console.print(task_table)

        # Specialties breakdown
        if specialties:
            spec_table = Table(title="By Specialty")
            spec_table.add_column("Specialty", style="cyan")
            spec_table.add_column("Count", justify="right")

            for spec, count in sorted(specialties.items(), key=lambda x: -x[1]):
                spec_table.add_row(spec, str(count))

            console.print(spec_table)

        # Sample test cases
        if show_samples > 0 and test_cases:
            console.print(f"\n[bold]Sample Test Cases ({min(show_samples, len(test_cases))}):[/bold]")

            for i, tc in enumerate(test_cases[:show_samples]):
                console.print(f"\n[cyan]#{i+1} {tc.id}[/cyan]")
                console.print(f"  Type: {type(tc).__name__}")
                if hasattr(tc, "specialty"):
                    console.print(f"  Specialty: {tc.specialty.value}")
                if hasattr(tc, "complexity"):
                    console.print(f"  Complexity: {tc.complexity.value}")

                # Show truncated clinical note
                note = tc.clinical_note[:200] + "..." if len(tc.clinical_note) > 200 else tc.clinical_note
                console.print(f"  Note: [dim]{note}[/dim]")

    except Exception as e:
        console.print(f"[red]Inspection failed: {e}[/red]")
        raise typer.Exit(1)


@app.command("convert")
def convert_dataset(
    input_path: Path = typer.Argument(
        ...,
        help="Input dataset file",
        exists=True,
    ),
    output_path: Path = typer.Argument(
        ...,
        help="Output file path",
    ),
    to_format: DatasetFormat = typer.Option(
        DatasetFormat.JSON,
        "--to",
        help="Output format",
    ),
) -> None:
    """
    Convert dataset between formats.

    Converts between JSON and YAML formats.
    """
    import json

    import yaml

    try:
        # Load input
        with open(input_path) as f:
            if input_path.suffix in [".yaml", ".yml"]:
                data = yaml.safe_load(f)
            else:
                data = json.load(f)

        # Write output
        with open(output_path, "w") as f:
            if to_format == DatasetFormat.YAML:
                yaml.dump(data, f, default_flow_style=False, allow_unicode=True)
            else:
                json.dump(data, f, indent=2)

        console.print(f"[green]Converted {input_path} -> {output_path}[/green]")

    except Exception as e:
        console.print(f"[red]Conversion failed: {e}[/red]")
        raise typer.Exit(1)


@app.command("split")
def split_dataset(
    input_path: Path = typer.Argument(
        ...,
        help="Input dataset file",
        exists=True,
    ),
    output_dir: Path = typer.Option(
        Path("./split"),
        "--output",
        help="Output directory",
    ),
    train_ratio: float = typer.Option(
        0.8,
        "--train",
        help="Training set ratio",
        min=0.1,
        max=0.95,
    ),
    seed: int = typer.Option(
        42,
        "--seed",
        help="Random seed for reproducibility",
    ),
) -> None:
    """
    Split dataset into train/test sets.

    Creates stratified splits maintaining distribution of task types.
    """
    import json
    import random

    try:
        with open(input_path) as f:
            data = json.load(f)

        test_cases = data.get("test_cases", data) if isinstance(data, dict) else data

        # Shuffle with seed
        random.seed(seed)
        shuffled = list(test_cases)
        random.shuffle(shuffled)

        # Split
        split_idx = int(len(shuffled) * train_ratio)
        train_cases = shuffled[:split_idx]
        test_cases = shuffled[split_idx:]

        # Write splits
        output_dir.mkdir(parents=True, exist_ok=True)

        with open(output_dir / "train.json", "w") as f:
            json.dump({"test_cases": train_cases}, f, indent=2)

        with open(output_dir / "test.json", "w") as f:
            json.dump({"test_cases": test_cases}, f, indent=2)

        console.print(f"[green]Split complete:[/green]")
        console.print(f"  Train: {len(train_cases)} cases -> {output_dir}/train.json")
        console.print(f"  Test: {len(test_cases)} cases -> {output_dir}/test.json")

    except Exception as e:
        console.print(f"[red]Split failed: {e}[/red]")
        raise typer.Exit(1)


@app.command("generate")
def generate_promptfoo_config(
    dataset: Path = typer.Argument(
        ...,
        help="Input dataset file",
        exists=True,
    ),
    output: Path = typer.Option(
        Path("./promptfoo.yaml"),
        "--output",
        help="Output Promptfoo config file",
    ),
    task_type: str = typer.Option(
        ...,
        "--task",
        help="Task type (icd, hcc, gap, query, em)",
    ),
) -> None:
    """
    Generate Promptfoo configuration from dataset.

    Creates a Promptfoo YAML configuration file from CDI test cases.
    """
    from nuvii_eval.datasets import load_dataset
    from nuvii_eval.promptfoo import convert_test_suite, generate_promptfoo_config

    try:
        # Load test cases
        test_cases = load_dataset(str(dataset))

        # Filter by task type if needed
        if task_type != "all":
            expected_type = f"{task_type.upper()}TestCase"
            test_cases = [tc for tc in test_cases if type(tc).__name__ == expected_type]

        if not test_cases:
            console.print(f"[yellow]No test cases found for task type: {task_type}[/yellow]")
            raise typer.Exit(0)

        # Convert to Promptfoo format
        pf_tests = convert_test_suite(test_cases, task_type)

        # Generate config
        from nuvii_eval.promptfoo import ConfigGeneratorOptions

        options = ConfigGeneratorOptions(
            description=f"CDI Evaluation - {task_type.upper()}",
        )
        config = generate_promptfoo_config(pf_tests, task_type, options)

        # Save
        config.save(output)

        console.print(f"[green]Generated Promptfoo config: {output}[/green]")
        console.print(f"  Test cases: {len(pf_tests)}")
        console.print(f"  Task type: {task_type}")

    except Exception as e:
        console.print(f"[red]Config generation failed: {e}[/red]")
        raise typer.Exit(1)
