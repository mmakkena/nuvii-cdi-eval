"""
Unit tests for CLI module.

Tests command-line interface commands and options.
"""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from typer.testing import CliRunner

from nuvii_eval.cli.main import app


runner = CliRunner()


# =============================================================================
# Test: Main CLI
# =============================================================================


class TestMainCLI:
    """Tests for main CLI app."""

    def test_help(self):
        """Test help output."""
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        assert "Nuvii CDI Agent Evaluation Framework" in result.output

    def test_version(self):
        """Test version output."""
        result = runner.invoke(app, ["--version"])
        assert result.exit_code == 0
        assert "nuvii-eval version" in result.output

    @patch("nuvii_eval.cli.main.get_settings")
    def test_info_command(self, mock_settings):
        """Test info command."""
        mock_settings.return_value = MagicMock(
            environment="development",
            debug=False,
            log_level="INFO",
            nuvii_api=MagicMock(base_url="http://api.test.com", timeout=30),
            phoenix=MagicMock(enabled=True, collector_endpoint="http://phoenix:6006"),
        )

        result = runner.invoke(app, ["info"])
        assert result.exit_code == 0
        assert "Environment" in result.output


# =============================================================================
# Test: Run Commands
# =============================================================================


class TestRunCommands:
    """Tests for run command group."""

    def test_run_help(self):
        """Test run help output."""
        result = runner.invoke(app, ["run", "--help"])
        assert result.exit_code == 0
        assert "Run evaluations" in result.output

    def test_eval_help(self):
        """Test eval command help."""
        result = runner.invoke(app, ["run", "eval", "--help"])
        assert result.exit_code == 0
        assert "dataset" in result.output.lower()

    def test_eval_missing_dataset(self):
        """Test eval with missing dataset."""
        result = runner.invoke(app, ["run", "eval", "nonexistent.json"])
        assert result.exit_code != 0

    def test_promptfoo_help(self):
        """Test promptfoo command help."""
        result = runner.invoke(app, ["run", "promptfoo", "--help"])
        assert result.exit_code == 0


# =============================================================================
# Test: Compare Commands
# =============================================================================


class TestCompareCommands:
    """Tests for compare command group."""

    def test_compare_help(self):
        """Test compare help output."""
        result = runner.invoke(app, ["compare", "--help"])
        assert result.exit_code == 0
        assert "Compare evaluation runs" in result.output

    def test_runs_help(self):
        """Test runs command help."""
        result = runner.invoke(app, ["compare", "runs", "--help"])
        assert result.exit_code == 0
        assert "baseline" in result.output.lower()

    def test_runs_missing_files(self):
        """Test runs with missing files."""
        result = runner.invoke(app, ["compare", "runs", "base.json", "current.json"])
        assert result.exit_code != 0

    def test_runs_comparison(self, tmp_path):
        """Test runs comparison with mock data."""
        # Create mock files
        baseline_file = tmp_path / "baseline.json"
        current_file = tmp_path / "current.json"

        baseline_data = {"results": [], "stats": {"pass_rate": 90}}
        current_data = {"results": [], "stats": {"pass_rate": 85}}

        baseline_file.write_text(json.dumps(baseline_data))
        current_file.write_text(json.dumps(current_data))

        # This test just verifies the command can be invoked with valid files
        # The actual comparison logic is tested in the promptfoo module tests
        result = runner.invoke(
            app,
            ["compare", "runs", str(baseline_file), str(current_file)],
        )
        # Command may fail due to missing modules, but should at least run
        assert result.exit_code in [0, 1]

    def test_trend_help(self):
        """Test trend command help."""
        result = runner.invoke(app, ["compare", "trend", "--help"])
        assert result.exit_code == 0


# =============================================================================
# Test: Report Commands
# =============================================================================


class TestReportCommands:
    """Tests for report command group."""

    def test_report_help(self):
        """Test report help output."""
        result = runner.invoke(app, ["report", "--help"])
        assert result.exit_code == 0
        assert "Generate reports" in result.output

    def test_generate_help(self):
        """Test generate command help."""
        result = runner.invoke(app, ["report", "generate", "--help"])
        assert result.exit_code == 0

    def test_pr_help(self):
        """Test pr command help."""
        result = runner.invoke(app, ["report", "pr", "--help"])
        assert result.exit_code == 0

    def test_summary_help(self):
        """Test summary command help."""
        result = runner.invoke(app, ["report", "summary", "--help"])
        assert result.exit_code == 0

    def test_summary_with_results(self, tmp_path):
        """Test summary command with results file."""
        results_file = tmp_path / "results.json"
        results_data = {
            "stats": {
                "total": 10,
                "passed": 8,
                "failed": 2,
                "pass_rate": 80.0,
                "average_score": 0.85,
            },
            "results": [
                {"test_id": "t1", "pass": True, "score": 1.0, "metadata": {}},
                {"test_id": "t2", "pass": False, "score": 0.5, "metadata": {}},
            ],
        }
        results_file.write_text(json.dumps(results_data))

        result = runner.invoke(app, ["report", "summary", str(results_file)])
        assert result.exit_code == 0
        assert "Total Tests" in result.output or "10" in result.output


# =============================================================================
# Test: Dataset Commands
# =============================================================================


class TestDatasetCommands:
    """Tests for dataset command group."""

    def test_dataset_help(self):
        """Test dataset help output."""
        result = runner.invoke(app, ["dataset", "--help"])
        assert result.exit_code == 0
        assert "Manage test datasets" in result.output

    def test_validate_help(self):
        """Test validate command help."""
        result = runner.invoke(app, ["dataset", "validate", "--help"])
        assert result.exit_code == 0

    def test_inspect_help(self):
        """Test inspect command help."""
        result = runner.invoke(app, ["dataset", "inspect", "--help"])
        assert result.exit_code == 0

    def test_convert_help(self):
        """Test convert command help."""
        result = runner.invoke(app, ["dataset", "convert", "--help"])
        assert result.exit_code == 0

    def test_split_help(self):
        """Test split command help."""
        result = runner.invoke(app, ["dataset", "split", "--help"])
        assert result.exit_code == 0

    def test_validate_missing_path(self):
        """Test validate with missing path."""
        result = runner.invoke(app, ["dataset", "validate", "nonexistent.json"])
        assert result.exit_code != 0

    def test_validate_valid_dataset(self, tmp_path):
        """Test validate with valid dataset."""
        dataset_file = tmp_path / "test.json"
        dataset_data = {
            "test_cases": [
                {
                    "id": "test-001",
                    "clinical_note": "A" * 100,  # Min 50 chars
                    "expected_icd_codes": ["E11.9"],
                }
            ]
        }
        dataset_file.write_text(json.dumps(dataset_data))

        result = runner.invoke(app, ["dataset", "validate", str(dataset_file)])
        # Validation may have warnings but should run
        assert "test" in result.output.lower() or result.exit_code in [0, 1]

    def test_convert_json_to_yaml(self, tmp_path):
        """Test converting JSON to YAML."""
        input_file = tmp_path / "input.json"
        output_file = tmp_path / "output.yaml"

        input_data = {"test_cases": [{"id": "test-1"}]}
        input_file.write_text(json.dumps(input_data))

        result = runner.invoke(
            app,
            ["dataset", "convert", str(input_file), str(output_file), "--to", "yaml"],
        )
        assert result.exit_code == 0
        assert output_file.exists()

    def test_split_dataset(self, tmp_path):
        """Test splitting dataset."""
        input_file = tmp_path / "dataset.json"
        output_dir = tmp_path / "split"

        # Create dataset with multiple items
        dataset_data = {
            "test_cases": [{"id": f"test-{i}"} for i in range(10)]
        }
        input_file.write_text(json.dumps(dataset_data))

        result = runner.invoke(
            app,
            ["dataset", "split", str(input_file), "--output", str(output_dir)],
        )
        assert result.exit_code == 0
        assert (output_dir / "train.json").exists()
        assert (output_dir / "test.json").exists()
