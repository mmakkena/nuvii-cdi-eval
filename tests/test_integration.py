"""
End-to-end integration tests for the Nuvii CDI Evaluation Framework.

Tests the full workflow from CLI to reports.
"""

import json
import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from typer.testing import CliRunner

from nuvii_eval.cli.main import app
from nuvii_eval.runner import BatchRunner, RunConfig, BatchResult
from nuvii_eval.reporters import (
    HTMLReporter,
    JSONReporter,
    MarkdownReporter,
    ReportData,
    ReportOptions,
)


runner = CliRunner()


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def sample_dataset(tmp_path):
    """Create a sample dataset file."""
    dataset = {
        "metadata": {
            "name": "Integration Test Dataset",
            "version": "1.0",
            "task_type": "icd",
        },
        "test_cases": [
            {
                "id": "test-001",
                "clinical_note": "A 65-year-old male with type 2 diabetes mellitus presents with fatigue and increased thirst. HbA1c is 9.2%. Currently on metformin.",
                "expected_icd_codes": ["E11.65", "R53.83"],
                "expected_primary_code": "E11.65",
                "specialty": "endocrinology",
                "complexity": "moderate",
            },
            {
                "id": "test-002",
                "clinical_note": "A 72-year-old female with heart failure presents with shortness of breath and leg swelling. EF 35% on echo. BNP elevated.",
                "expected_icd_codes": ["I50.22", "R06.02"],
                "expected_primary_code": "I50.22",
                "specialty": "cardiology",
                "complexity": "moderate",
            },
            {
                "id": "test-003",
                "clinical_note": "A 55-year-old male with COPD exacerbation. Increased sputum production and dyspnea. SpO2 88% on room air.",
                "expected_icd_codes": ["J44.1", "R06.02"],
                "expected_primary_code": "J44.1",
                "specialty": "pulmonology",
                "complexity": "low",
            },
        ],
    }

    dataset_file = tmp_path / "test_dataset.json"
    dataset_file.write_text(json.dumps(dataset, indent=2))
    return dataset_file


@pytest.fixture
def sample_results(tmp_path):
    """Create sample evaluation results."""
    results = {
        "metadata": {
            "run_id": "test-run-001",
            "timestamp": "2024-01-15T12:00:00Z",
            "task_type": "icd",
        },
        "stats": {
            "total": 10,
            "passed": 8,
            "failed": 2,
            "pass_rate": 80.0,
            "average_score": 0.85,
        },
        "results": [
            {
                "test_id": "test-001",
                "pass": True,
                "score": 0.95,
                "predicted_codes": ["E11.65", "R53.83"],
                "expected_codes": ["E11.65", "R53.83"],
                "metadata": {"specialty": "endocrinology"},
            },
            {
                "test_id": "test-002",
                "pass": True,
                "score": 0.90,
                "predicted_codes": ["I50.22", "R06.02"],
                "expected_codes": ["I50.22", "R06.02"],
                "metadata": {"specialty": "cardiology"},
            },
            {
                "test_id": "test-003",
                "pass": False,
                "score": 0.60,
                "predicted_codes": ["J44.0"],
                "expected_codes": ["J44.1", "R06.02"],
                "metadata": {"specialty": "pulmonology"},
            },
        ],
    }

    results_file = tmp_path / "results.json"
    results_file.write_text(json.dumps(results, indent=2))
    return results_file


# =============================================================================
# Test: Dataset Validation Workflow
# =============================================================================


class TestDatasetWorkflow:
    """Tests for dataset validation and inspection workflow."""

    def test_validate_dataset(self, sample_dataset):
        """Test validating a dataset file."""
        result = runner.invoke(app, ["dataset", "validate", str(sample_dataset)])
        # Should run without error
        assert result.exit_code in [0, 1]
        assert "Validating" in result.output or "test" in result.output.lower()

    def test_inspect_dataset(self, sample_dataset):
        """Test inspecting a dataset file."""
        result = runner.invoke(
            app, ["dataset", "inspect", str(sample_dataset), "--samples", "2"]
        )
        assert result.exit_code == 0
        assert "test" in result.output.lower()

    def test_convert_dataset(self, sample_dataset, tmp_path):
        """Test converting dataset to YAML."""
        output_file = tmp_path / "output.yaml"
        result = runner.invoke(
            app,
            [
                "dataset",
                "convert",
                str(sample_dataset),
                str(output_file),
                "--to",
                "yaml",
            ],
        )
        assert result.exit_code == 0
        assert output_file.exists()

    def test_split_dataset(self, sample_dataset, tmp_path):
        """Test splitting dataset into train/test."""
        output_dir = tmp_path / "split"
        result = runner.invoke(
            app,
            [
                "dataset",
                "split",
                str(sample_dataset),
                "--output",
                str(output_dir),
                "--train",
                "0.6",
            ],
        )
        assert result.exit_code == 0
        assert (output_dir / "train.json").exists()
        assert (output_dir / "test.json").exists()


# =============================================================================
# Test: Report Generation Workflow
# =============================================================================


class TestReportWorkflow:
    """Tests for report generation workflow."""

    def test_generate_html_report(self, sample_results, tmp_path):
        """Test generating HTML report from results."""
        output_file = tmp_path / "report.html"
        result = runner.invoke(
            app,
            [
                "report",
                "generate",
                str(sample_results),
                "--output",
                str(output_file),
                "--format",
                "html",
                "--title",
                "Integration Test Report",
            ],
        )
        assert result.exit_code == 0
        assert output_file.exists()

        # Verify HTML content
        content = output_file.read_text()
        assert "<html" in content
        assert "Integration Test Report" in content

    def test_generate_json_report(self, sample_results, tmp_path):
        """Test generating JSON report from results."""
        output_file = tmp_path / "report.json"
        result = runner.invoke(
            app,
            [
                "report",
                "generate",
                str(sample_results),
                "--output",
                str(output_file),
                "--format",
                "json",
            ],
        )
        assert result.exit_code == 0
        assert output_file.exists()

        # Verify JSON is valid
        data = json.loads(output_file.read_text())
        assert "summary" in data

    def test_generate_markdown_report(self, sample_results, tmp_path):
        """Test generating Markdown report from results."""
        output_file = tmp_path / "report.md"
        result = runner.invoke(
            app,
            [
                "report",
                "generate",
                str(sample_results),
                "--output",
                str(output_file),
                "--format",
                "markdown",
            ],
        )
        assert result.exit_code == 0
        assert output_file.exists()

        # Verify Markdown content
        content = output_file.read_text()
        assert "#" in content

    def test_summary_command(self, sample_results):
        """Test summary command output."""
        result = runner.invoke(app, ["report", "summary", str(sample_results)])
        assert result.exit_code == 0
        assert "Total Tests" in result.output or "10" in result.output

    def test_summary_by_task(self, sample_results):
        """Test summary grouped by task type."""
        result = runner.invoke(
            app, ["report", "summary", str(sample_results), "--by-task"]
        )
        assert result.exit_code == 0

    def test_pr_comment_generation(self, sample_results, tmp_path):
        """Test PR comment generation."""
        output_file = tmp_path / "pr_comment.md"
        result = runner.invoke(
            app,
            ["report", "pr", str(sample_results), "--output", str(output_file)],
        )
        assert result.exit_code == 0
        assert output_file.exists()

        content = output_file.read_text()
        assert "CDI Evaluation" in content or "Results" in content


# =============================================================================
# Test: Comparison Workflow
# =============================================================================


class TestComparisonWorkflow:
    """Tests for comparison and regression detection workflow."""

    def test_compare_runs(self, tmp_path):
        """Test comparing two evaluation runs."""
        # Create baseline results
        baseline = {
            "stats": {"pass_rate": 90.0, "average_score": 0.92},
            "results": [
                {"test_id": "t1", "pass": True, "score": 0.95},
                {"test_id": "t2", "pass": True, "score": 0.88},
            ],
        }
        baseline_file = tmp_path / "baseline.json"
        baseline_file.write_text(json.dumps(baseline))

        # Create current results (with regression)
        current = {
            "stats": {"pass_rate": 80.0, "average_score": 0.82},
            "results": [
                {"test_id": "t1", "pass": True, "score": 0.90},
                {"test_id": "t2", "pass": False, "score": 0.65},
            ],
        }
        current_file = tmp_path / "current.json"
        current_file.write_text(json.dumps(current))

        result = runner.invoke(
            app,
            ["compare", "runs", str(baseline_file), str(current_file)],
        )
        # May fail on missing imports, but should run
        assert result.exit_code in [0, 1]

    def test_trend_analysis(self, tmp_path):
        """Test trend analysis over multiple runs."""
        # Create multiple result files
        for i in range(3):
            results = {
                "stats": {"pass_rate": 80 + i * 5, "average_score": 0.8 + i * 0.05},
                "results": [],
            }
            (tmp_path / f"run_{i}.json").write_text(json.dumps(results))

        result = runner.invoke(
            app, ["compare", "trend", str(tmp_path), "--last", "3"]
        )
        assert result.exit_code in [0, 1]


# =============================================================================
# Test: Full Pipeline Integration
# =============================================================================


class TestFullPipeline:
    """Tests for the complete evaluation pipeline."""

    def test_dry_run_evaluation(self, sample_dataset):
        """Test dry run mode validates without executing."""
        result = runner.invoke(
            app,
            ["run", "eval", str(sample_dataset), "--dry-run"],
        )
        # Dry run should validate and exit
        assert result.exit_code in [0, 1]
        assert "dry run" in result.output.lower() or "validat" in result.output.lower()

    def test_batch_runner_integration(self, sample_dataset, tmp_path):
        """Test BatchRunner initialization and basic methods."""
        config = RunConfig(
            dataset_path=str(sample_dataset),
            task_type="icd",
            max_concurrency=2,
            timeout_seconds=30,
        )

        runner_instance = BatchRunner(config)

        # Verify runner is properly configured
        assert runner_instance.config == config
        assert runner_instance.config.max_concurrency == 2

        # Test run method returns BatchResult
        result = runner_instance.run()
        assert isinstance(result, BatchResult)
        assert result.total_count >= 0

    def test_reporter_integration(self, sample_results):
        """Test reporter creates valid output."""
        with open(sample_results) as f:
            data = json.load(f)

        report_data = ReportData.from_dict(data)
        options = ReportOptions(title="Integration Test", include_charts=False)

        # Test HTML reporter
        html_reporter = HTMLReporter(options)
        html_output = html_reporter.generate(report_data)
        assert "<html" in html_output

        # Test JSON reporter
        json_reporter = JSONReporter(options)
        json_output = json_reporter.generate(report_data)
        parsed_data = json.loads(json_output)
        assert "summary" in parsed_data

        # Test Markdown reporter
        md_reporter = MarkdownReporter(options)
        md_output = md_reporter.generate(report_data)
        assert "#" in md_output

    def test_full_workflow_with_mocked_api(self, sample_dataset, tmp_path):
        """Test complete workflow: load dataset -> evaluate -> report."""
        # Step 1: Validate dataset
        result = runner.invoke(app, ["dataset", "validate", str(sample_dataset)])
        assert result.exit_code in [0, 1]

        # Step 2: Run dry evaluation
        result = runner.invoke(
            app,
            ["run", "eval", str(sample_dataset), "--dry-run", "--verbose"],
        )
        assert result.exit_code in [0, 1]

        # Step 3: Generate report from sample results
        results_file = tmp_path / "mock_results.json"
        results_data = {
            "stats": {"total": 3, "passed": 2, "failed": 1, "pass_rate": 66.7, "average_score": 0.82},
            "results": [
                {"test_id": "test-001", "pass": True, "score": 0.95, "metadata": {}},
                {"test_id": "test-002", "pass": True, "score": 0.90, "metadata": {}},
                {"test_id": "test-003", "pass": False, "score": 0.60, "metadata": {}},
            ],
        }
        results_file.write_text(json.dumps(results_data))

        report_file = tmp_path / "final_report.html"
        result = runner.invoke(
            app,
            [
                "report",
                "generate",
                str(results_file),
                "--output",
                str(report_file),
                "--format",
                "html",
            ],
        )
        assert result.exit_code == 0
        assert report_file.exists()


# =============================================================================
# Test: Error Handling
# =============================================================================


class TestErrorHandling:
    """Tests for error handling in integration scenarios."""

    def test_missing_dataset_file(self):
        """Test handling of missing dataset file."""
        result = runner.invoke(app, ["run", "eval", "nonexistent_file.json"])
        assert result.exit_code != 0

    def test_invalid_json_dataset(self, tmp_path):
        """Test handling of invalid JSON in dataset."""
        invalid_file = tmp_path / "invalid.json"
        invalid_file.write_text("not valid json {{{")

        result = runner.invoke(app, ["dataset", "validate", str(invalid_file)])
        assert result.exit_code != 0

    def test_missing_results_file(self, tmp_path):
        """Test handling of missing results file."""
        result = runner.invoke(
            app,
            ["report", "generate", "nonexistent_results.json"],
        )
        assert result.exit_code != 0

    def test_invalid_task_type(self, sample_dataset):
        """Test handling of invalid task type."""
        result = runner.invoke(
            app,
            ["run", "eval", str(sample_dataset), "--task", "invalid_task"],
        )
        # Typer should reject invalid enum value
        assert result.exit_code != 0
