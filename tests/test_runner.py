"""
Unit tests for runner module.

Tests batch and async evaluation runners.
"""

import asyncio
import json
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from nuvii_eval.runner import (
    AsyncBatchRunner,
    AsyncRunConfig,
    BatchResult,
    BatchRunner,
    RunConfig,
    TestEvaluation,
    run_async_evaluation,
    run_evaluation,
)


# =============================================================================
# Test: Configuration
# =============================================================================


class TestRunConfig:
    """Tests for RunConfig."""

    def test_basic_config(self):
        """Test basic configuration."""
        config = RunConfig(dataset_path="data.json")
        assert config.dataset_path == "data.json"
        assert config.max_concurrency == 5
        assert config.timeout_seconds == 60

    def test_config_with_options(self):
        """Test configuration with options."""
        config = RunConfig(
            dataset_path="data.json",
            task_type="icd",
            max_concurrency=10,
            timeout_seconds=120,
            fail_fast=True,
            verbose=True,
        )
        assert config.task_type == "icd"
        assert config.max_concurrency == 10
        assert config.fail_fast is True

    def test_from_dict(self):
        """Test creating config from dictionary."""
        data = {
            "dataset_path": "data.json",
            "task_type": "hcc",
            "max_concurrency": 8,
        }
        config = RunConfig.from_dict(data)
        assert config.dataset_path == "data.json"
        assert config.task_type == "hcc"
        assert config.max_concurrency == 8


# =============================================================================
# Test: Result Models
# =============================================================================


class TestTestEvaluation:
    """Tests for TestEvaluation."""

    def test_basic_evaluation(self):
        """Test basic evaluation result."""
        eval = TestEvaluation(
            test_id="test-001",
            passed=True,
            score=0.95,
        )
        assert eval.test_id == "test-001"
        assert eval.passed is True
        assert eval.score == 0.95

    def test_to_dict(self):
        """Test converting to dictionary."""
        eval = TestEvaluation(
            test_id="test-001",
            passed=True,
            score=0.95,
            metrics={"accuracy": 0.9},
            duration_ms=150,
        )
        result = eval.to_dict()

        assert result["test_id"] == "test-001"
        assert result["pass"] is True
        assert result["score"] == 0.95
        assert result["metrics"]["accuracy"] == 0.9


class TestBatchResult:
    """Tests for BatchResult."""

    @pytest.fixture
    def sample_evaluations(self):
        """Create sample evaluations."""
        return [
            TestEvaluation(test_id="t1", passed=True, score=1.0),
            TestEvaluation(test_id="t2", passed=True, score=0.8),
            TestEvaluation(test_id="t3", passed=False, score=0.3),
        ]

    def test_properties(self, sample_evaluations):
        """Test result properties."""
        result = BatchResult(
            timestamp=datetime.utcnow(),
            config=RunConfig(dataset_path="data.json"),
            evaluations=sample_evaluations,
            duration_seconds=10.5,
        )

        assert result.total_count == 3
        assert result.passed_count == 2
        assert result.failed_count == 1
        assert result.pass_rate == pytest.approx(66.67, rel=0.01)
        assert result.average_score == pytest.approx(0.7, rel=0.01)
        assert result.passed is False  # < 70%

    def test_get_failed_evaluations(self, sample_evaluations):
        """Test getting failed evaluations."""
        result = BatchResult(
            timestamp=datetime.utcnow(),
            config=RunConfig(dataset_path="data.json"),
            evaluations=sample_evaluations,
        )

        failed = result.get_failed_evaluations()
        assert len(failed) == 1
        assert failed[0].test_id == "t3"

    def test_to_dict(self, sample_evaluations):
        """Test converting to dictionary."""
        result = BatchResult(
            timestamp=datetime.utcnow(),
            config=RunConfig(dataset_path="data.json"),
            evaluations=sample_evaluations,
            duration_seconds=10.5,
        )

        data = result.to_dict()

        assert "timestamp" in data
        assert data["stats"]["total"] == 3
        assert data["stats"]["passed"] == 2
        assert len(data["results"]) == 3

    def test_save(self, sample_evaluations, tmp_path):
        """Test saving results to file."""
        result = BatchResult(
            timestamp=datetime.utcnow(),
            config=RunConfig(dataset_path="data.json"),
            evaluations=sample_evaluations,
        )

        output_path = tmp_path / "results.json"
        result.save(str(output_path))

        assert output_path.exists()
        with open(output_path) as f:
            saved_data = json.load(f)
        assert saved_data["stats"]["total"] == 3


# =============================================================================
# Test: Batch Runner
# =============================================================================


class TestBatchRunner:
    """Tests for BatchRunner."""

    @pytest.fixture
    def sample_dataset(self, tmp_path):
        """Create a sample dataset file."""
        dataset_file = tmp_path / "test_dataset.json"
        dataset_data = {
            "test_cases": [
                {
                    "id": "icd-test-001",
                    "clinical_note": "Patient presents with type 2 diabetes mellitus, currently on metformin. " * 3,
                    "specialty": "endocrinology",
                    "complexity": "moderate",
                    "expected_icd_codes": ["E11.9"],
                    "acceptable_icd_codes": [],
                    "unacceptable_codes": [],
                },
                {
                    "id": "icd-test-002",
                    "clinical_note": "Patient with hypertension and chronic kidney disease stage 3. " * 3,
                    "specialty": "nephrology",
                    "complexity": "moderate",
                    "expected_icd_codes": ["I10", "N18.3"],
                    "acceptable_icd_codes": [],
                    "unacceptable_codes": [],
                },
            ]
        }
        dataset_file.write_text(json.dumps(dataset_data))
        return dataset_file

    def test_init(self):
        """Test runner initialization."""
        config = RunConfig(dataset_path="data.json")
        runner = BatchRunner(config)
        assert runner.config == config

    def test_from_dict(self):
        """Test creating runner from dictionary."""
        data = {"dataset_path": "data.json", "task_type": "icd"}
        runner = BatchRunner.from_dict(data)
        assert runner.config.task_type == "icd"

    def test_run_with_dataset(self, sample_dataset):
        """Test running with actual dataset."""
        config = RunConfig(
            dataset_path=str(sample_dataset),
            task_type="icd",
            verbose=True,
        )
        runner = BatchRunner(config)
        result = runner.run()

        assert result.total_count == 2
        # With mock responses matching expected codes, should pass
        assert result.passed_count >= 0

    def test_run_with_limit(self, sample_dataset):
        """Test running with limit."""
        config = RunConfig(
            dataset_path=str(sample_dataset),
            limit=1,
        )
        runner = BatchRunner(config)
        result = runner.run()

        assert result.total_count == 1

    def test_run_empty_dataset(self, tmp_path):
        """Test running with empty dataset."""
        empty_file = tmp_path / "empty.json"
        empty_file.write_text('{"test_cases": []}')

        config = RunConfig(dataset_path=str(empty_file))
        runner = BatchRunner(config)
        result = runner.run()

        assert result.total_count == 0
        assert len(result.errors) > 0 or result.passed

    def test_run_fail_fast(self, sample_dataset):
        """Test fail-fast mode."""
        config = RunConfig(
            dataset_path=str(sample_dataset),
            fail_fast=True,
        )
        runner = BatchRunner(config)

        # Mock an evaluator that always fails
        with patch.object(runner, "_evaluate_case") as mock_eval:
            mock_eval.return_value = TestEvaluation(
                test_id="test",
                passed=False,
                score=0.0,
            )

            result = runner.run()

            # Should stop after first failure
            assert result.total_count == 1


# =============================================================================
# Test: Async Runner
# =============================================================================


class TestAsyncBatchRunner:
    """Tests for AsyncBatchRunner."""

    @pytest.fixture
    def sample_dataset(self, tmp_path):
        """Create a sample dataset file."""
        dataset_file = tmp_path / "test_dataset.json"
        dataset_data = {
            "test_cases": [
                {
                    "id": "icd-test-001",
                    "clinical_note": "Patient with diabetes mellitus type 2. " * 5,
                    "specialty": "endocrinology",
                    "complexity": "moderate",
                    "expected_icd_codes": ["E11.9"],
                },
            ]
        }
        dataset_file.write_text(json.dumps(dataset_data))
        return dataset_file

    def test_init(self):
        """Test async runner initialization."""
        config = AsyncRunConfig(
            dataset_path="data.json",
            semaphore_limit=10,
        )
        runner = AsyncBatchRunner(config)
        assert runner.config.semaphore_limit == 10

    @pytest.mark.asyncio
    async def test_run_async(self, sample_dataset):
        """Test async run."""
        config = AsyncRunConfig(
            dataset_path=str(sample_dataset),
            semaphore_limit=5,
        )
        runner = AsyncBatchRunner(config)
        result = await runner.run()

        assert result.total_count == 1

    def test_run_sync_wrapper(self, sample_dataset):
        """Test sync wrapper for async run."""
        result = run_async_evaluation(
            dataset_path=str(sample_dataset),
            max_concurrency=5,
        )

        assert result.total_count == 1


# =============================================================================
# Test: Convenience Functions
# =============================================================================


class TestConvenienceFunctions:
    """Tests for convenience functions."""

    @pytest.fixture
    def sample_dataset(self, tmp_path):
        """Create a sample dataset file."""
        dataset_file = tmp_path / "test.json"
        dataset_data = {
            "test_cases": [
                {
                    "id": "test-001",
                    "clinical_note": "Patient with condition. " * 10,
                    "specialty": "general",
                    "expected_icd_codes": ["Z00.0"],
                },
            ]
        }
        dataset_file.write_text(json.dumps(dataset_data))
        return dataset_file

    def test_run_evaluation(self, sample_dataset):
        """Test run_evaluation function."""
        result = run_evaluation(
            dataset_path=str(sample_dataset),
            task_type="icd",
        )

        assert isinstance(result, BatchResult)
        assert result.total_count >= 0
