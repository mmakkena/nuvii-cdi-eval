"""
Unit tests for reporters module.

Tests report generation in various formats.
"""

import json
from datetime import datetime
from pathlib import Path

import pytest

from nuvii_eval.reporters import (
    BaseReporter,
    CSVReporter,
    DashboardGenerator,
    HTMLReporter,
    JSONReporter,
    MarkdownReporter,
    PRCommentGenerator,
    ReportData,
    ReportGenerator,
    ReportOptions,
)


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def sample_report_data():
    """Create sample report data."""
    return ReportData(
        timestamp=datetime(2024, 1, 15, 10, 30, 0),
        title="Test Evaluation Report",
        summary={
            "total": 100,
            "passed": 85,
            "failed": 15,
            "pass_rate": 85.0,
            "average_score": 0.87,
            "duration_seconds": 45.5,
        },
        results=[
            {
                "test_id": "t1",
                "pass": True,
                "score": 1.0,
                "metrics": {"accuracy": 0.95},
                "metadata": {"task_type": "icd"},
            },
            {
                "test_id": "t2",
                "pass": False,
                "score": 0.3,
                "errors": ["Assertion failed"],
                "metadata": {"task_type": "icd"},
            },
            {
                "test_id": "t3",
                "pass": True,
                "score": 0.9,
                "metadata": {"task_type": "hcc"},
            },
        ],
        metrics={
            "pass_rate": 85.0,
            "average_score": 0.87,
        },
    )


@pytest.fixture
def sample_results_dict():
    """Create sample results dictionary."""
    return {
        "timestamp": "2024-01-15T10:30:00",
        "stats": {
            "total": 100,
            "passed": 85,
            "failed": 15,
            "pass_rate": 85.0,
            "average_score": 0.87,
        },
        "results": [
            {"test_id": "t1", "pass": True, "score": 1.0},
            {"test_id": "t2", "pass": False, "score": 0.3},
        ],
    }


# =============================================================================
# Test: ReportOptions
# =============================================================================


class TestReportOptions:
    """Tests for ReportOptions."""

    def test_default_options(self):
        """Test default options."""
        options = ReportOptions()
        assert options.title == "CDI Evaluation Report"
        assert options.include_details is True
        assert options.include_charts is True

    def test_custom_options(self):
        """Test custom options."""
        options = ReportOptions(
            title="Custom Report",
            include_details=False,
            max_failures_shown=5,
        )
        assert options.title == "Custom Report"
        assert options.include_details is False
        assert options.max_failures_shown == 5


# =============================================================================
# Test: ReportData
# =============================================================================


class TestReportData:
    """Tests for ReportData."""

    def test_from_dict(self, sample_results_dict):
        """Test creating from dictionary."""
        data = ReportData.from_dict(sample_results_dict, "Test Report")

        assert data.title == "Test Report"
        assert data.summary["total"] == 100
        assert data.summary["pass_rate"] == 85.0
        assert len(data.results) == 2

    def test_from_batch_result(self):
        """Test creating from BatchResult."""
        from nuvii_eval.runner import BatchResult, RunConfig, TestEvaluation

        evaluations = [
            TestEvaluation(test_id="t1", passed=True, score=1.0),
            TestEvaluation(test_id="t2", passed=False, score=0.5),
        ]
        batch_result = BatchResult(
            timestamp=datetime.utcnow(),
            config=RunConfig(dataset_path="data.json"),
            evaluations=evaluations,
            duration_seconds=10.0,
        )

        data = ReportData.from_batch_result(batch_result, "Batch Report")

        assert data.title == "Batch Report"
        assert data.summary["total"] == 2
        assert data.summary["passed"] == 1


# =============================================================================
# Test: HTML Reporter
# =============================================================================


class TestHTMLReporter:
    """Tests for HTMLReporter."""

    def test_generate(self, sample_report_data):
        """Test HTML generation."""
        reporter = HTMLReporter()
        html = reporter.generate(sample_report_data)

        assert "<!DOCTYPE html>" in html
        assert "Test Evaluation Report" in html
        assert "85.0%" in html or "85%" in html
        assert "chart" in html.lower()

    def test_generate_without_charts(self, sample_report_data):
        """Test HTML without charts."""
        options = ReportOptions(include_charts=False)
        reporter = HTMLReporter(options)
        html = reporter.generate(sample_report_data)

        assert "<!DOCTYPE html>" in html
        assert "chart.js" not in html

    def test_generate_without_details(self, sample_report_data):
        """Test HTML without details."""
        options = ReportOptions(include_details=False)
        reporter = HTMLReporter(options)
        html = reporter.generate(sample_report_data)

        assert "<!DOCTYPE html>" in html
        # Should not have failed tests section
        assert "Failed Tests" not in html or "All tests passed" in html

    def test_save(self, sample_report_data, tmp_path):
        """Test saving HTML report."""
        reporter = HTMLReporter()
        output_path = tmp_path / "report.html"

        reporter.save(sample_report_data, str(output_path))

        assert output_path.exists()
        content = output_path.read_text()
        assert "<!DOCTYPE html>" in content


# =============================================================================
# Test: JSON Reporter
# =============================================================================


class TestJSONReporter:
    """Tests for JSONReporter."""

    def test_generate(self, sample_report_data):
        """Test JSON generation."""
        reporter = JSONReporter()
        json_str = reporter.generate(sample_report_data)

        data = json.loads(json_str)
        assert data["title"] == "Test Evaluation Report"
        assert data["summary"]["total"] == 100
        assert "results" in data

    def test_generate_without_details(self, sample_report_data):
        """Test JSON without details."""
        options = ReportOptions(include_details=False)
        reporter = JSONReporter(options)
        json_str = reporter.generate(sample_report_data)

        data = json.loads(json_str)
        assert "results" not in data

    def test_save(self, sample_report_data, tmp_path):
        """Test saving JSON report."""
        reporter = JSONReporter()
        output_path = tmp_path / "report.json"

        reporter.save(sample_report_data, str(output_path))

        assert output_path.exists()
        with open(output_path) as f:
            data = json.load(f)
        assert data["title"] == "Test Evaluation Report"


# =============================================================================
# Test: CSV Reporter
# =============================================================================


class TestCSVReporter:
    """Tests for CSVReporter."""

    def test_generate(self, sample_report_data):
        """Test CSV generation."""
        reporter = CSVReporter()
        csv_str = reporter.generate(sample_report_data)

        lines = csv_str.strip().split("\n")
        assert len(lines) == 4  # Header + 3 results

        # Check header
        header = lines[0]
        assert "test_id" in header
        assert "pass" in header
        assert "score" in header

    def test_save(self, sample_report_data, tmp_path):
        """Test saving CSV report."""
        reporter = CSVReporter()
        output_path = tmp_path / "report.csv"

        reporter.save(sample_report_data, str(output_path))

        assert output_path.exists()


# =============================================================================
# Test: Markdown Reporter
# =============================================================================


class TestMarkdownReporter:
    """Tests for MarkdownReporter."""

    def test_generate(self, sample_report_data):
        """Test Markdown generation."""
        reporter = MarkdownReporter()
        md = reporter.generate(sample_report_data)

        assert "# Test Evaluation Report" in md
        assert "## Summary" in md
        assert "85" in md  # Pass rate

    def test_generate_with_failures(self, sample_report_data):
        """Test Markdown with failures."""
        reporter = MarkdownReporter()
        md = reporter.generate(sample_report_data)

        assert "Failed Tests" in md
        assert "t2" in md  # Failed test ID

    def test_save(self, sample_report_data, tmp_path):
        """Test saving Markdown report."""
        reporter = MarkdownReporter()
        output_path = tmp_path / "report.md"

        reporter.save(sample_report_data, str(output_path))

        assert output_path.exists()
        content = output_path.read_text()
        assert "# Test Evaluation Report" in content


# =============================================================================
# Test: PR Comment Generator
# =============================================================================


class TestPRCommentGenerator:
    """Tests for PRCommentGenerator."""

    def test_generate_basic(self, sample_results_dict):
        """Test basic PR comment generation."""
        generator = PRCommentGenerator()
        comment = generator.generate(sample_results_dict)

        assert "## CDI Evaluation Results" in comment
        assert "Status" in comment
        assert "Pass Rate" in comment or "85" in comment

    def test_generate_with_baseline(self, sample_results_dict):
        """Test PR comment with baseline comparison."""
        baseline = {
            "stats": {"pass_rate": 80.0},
        }

        generator = PRCommentGenerator()
        comment = generator.generate(sample_results_dict, baseline)

        assert "Baseline" in comment
        assert "+" in comment  # Positive delta

    def test_max_failures_limit(self):
        """Test max failures limit."""
        data = {
            "stats": {"pass_rate": 50.0},
            "results": [
                {"test_id": f"t{i}", "pass": False, "score": 0.0}
                for i in range(20)
            ],
        }

        generator = PRCommentGenerator(max_failures=5)
        comment = generator.generate(data)

        # Should show max 5 failures plus "more" indicator
        assert "more" in comment


# =============================================================================
# Test: Dashboard Generator
# =============================================================================


class TestDashboardGenerator:
    """Tests for DashboardGenerator."""

    def test_init(self, tmp_path):
        """Test dashboard initialization."""
        output_dir = tmp_path / "dashboard"
        generator = DashboardGenerator(str(output_dir))

        assert generator.output_dir == output_dir
        assert output_dir.exists()

    def test_update(self, tmp_path):
        """Test dashboard update."""
        # Create results file
        results_file = tmp_path / "results.json"
        results_data = {
            "timestamp": "2024-01-15T10:30:00",
            "stats": {
                "total": 100,
                "passed": 85,
                "failed": 15,
                "pass_rate": 85.0,
                "average_score": 0.87,
            },
            "results": [],
        }
        results_file.write_text(json.dumps(results_data))

        # Create dashboard
        output_dir = tmp_path / "dashboard"
        generator = DashboardGenerator(str(output_dir))
        generator.update(str(results_file))

        # Check outputs
        assert (output_dir / "index.html").exists()
        assert (output_dir / "history.json").exists()

        # Check history
        with open(output_dir / "history.json") as f:
            history = json.load(f)
        assert len(history["entries"]) == 1

    def test_multiple_updates(self, tmp_path):
        """Test multiple dashboard updates."""
        output_dir = tmp_path / "dashboard"
        generator = DashboardGenerator(str(output_dir))

        # Create and update multiple times
        for i in range(3):
            results_file = tmp_path / f"results_{i}.json"
            results_data = {
                "timestamp": f"2024-01-1{i}T10:30:00",
                "stats": {
                    "pass_rate": 80 + i * 5,
                    "average_score": 0.8 + i * 0.05,
                },
                "results": [],
            }
            results_file.write_text(json.dumps(results_data))
            generator.update(str(results_file))

        # Check history has all entries
        with open(output_dir / "history.json") as f:
            history = json.load(f)
        assert len(history["entries"]) == 3


# =============================================================================
# Test: Report Generator Factory
# =============================================================================


class TestReportGenerator:
    """Tests for ReportGenerator factory."""

    def test_generate_html(self, sample_results_dict, tmp_path):
        """Test generating HTML report."""
        results_file = tmp_path / "results.json"
        results_file.write_text(json.dumps(sample_results_dict))

        output_file = tmp_path / "report.html"

        generator = ReportGenerator()
        generator.generate(
            results_path=str(results_file),
            output_path=str(output_file),
            format="html",
        )

        assert output_file.exists()
        content = output_file.read_text()
        assert "<!DOCTYPE html>" in content

    def test_generate_json(self, sample_results_dict, tmp_path):
        """Test generating JSON report."""
        results_file = tmp_path / "results.json"
        results_file.write_text(json.dumps(sample_results_dict))

        output_file = tmp_path / "report_out.json"

        generator = ReportGenerator()
        generator.generate(
            results_path=str(results_file),
            output_path=str(output_file),
            format="json",
        )

        assert output_file.exists()

    def test_generate_csv(self, sample_results_dict, tmp_path):
        """Test generating CSV report."""
        results_file = tmp_path / "results.json"
        results_file.write_text(json.dumps(sample_results_dict))

        output_file = tmp_path / "report.csv"

        generator = ReportGenerator()
        generator.generate(
            results_path=str(results_file),
            output_path=str(output_file),
            format="csv",
        )

        assert output_file.exists()

    def test_generate_markdown(self, sample_results_dict, tmp_path):
        """Test generating Markdown report."""
        results_file = tmp_path / "results.json"
        results_file.write_text(json.dumps(sample_results_dict))

        output_file = tmp_path / "report.md"

        generator = ReportGenerator()
        generator.generate(
            results_path=str(results_file),
            output_path=str(output_file),
            format="markdown",
        )

        assert output_file.exists()

    def test_unknown_format(self, sample_results_dict, tmp_path):
        """Test unknown format raises error."""
        results_file = tmp_path / "results.json"
        results_file.write_text(json.dumps(sample_results_dict))

        generator = ReportGenerator()

        with pytest.raises(ValueError, match="Unknown format"):
            generator.generate(
                results_path=str(results_file),
                output_path=str(tmp_path / "report.xyz"),
                format="xyz",
            )
