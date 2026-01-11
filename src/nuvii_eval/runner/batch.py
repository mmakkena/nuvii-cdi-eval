"""
Batch evaluation runner.

Provides utilities for running evaluations across datasets.
"""

import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import structlog

from nuvii_eval.datasets import load_dataset
from nuvii_eval.datasets.schemas import BaseTestCase
from nuvii_eval.evaluators import get_evaluator

logger = structlog.get_logger(__name__)


# =============================================================================
# Configuration
# =============================================================================


@dataclass
class RunConfig:
    """Configuration for batch evaluation run."""

    dataset_path: str
    task_type: str | None = None
    max_concurrency: int = 5
    timeout_seconds: int = 60
    fail_fast: bool = False
    verbose: bool = False
    tags_filter: list[str] | None = None
    specialty_filter: str | None = None
    limit: int | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RunConfig":
        """Create from dictionary."""
        return cls(
            dataset_path=data.get("dataset_path", data.get("dataset", "")),
            task_type=data.get("task_type"),
            max_concurrency=data.get("max_concurrency", 5),
            timeout_seconds=data.get("timeout_seconds", 60),
            fail_fast=data.get("fail_fast", False),
            verbose=data.get("verbose", False),
            tags_filter=data.get("tags_filter"),
            specialty_filter=data.get("specialty_filter"),
            limit=data.get("limit"),
        )


# =============================================================================
# Result Models
# =============================================================================


@dataclass
class TestEvaluation:
    """Result of evaluating a single test case."""

    test_id: str
    passed: bool
    score: float
    metrics: dict[str, float] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)
    duration_ms: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "test_id": self.test_id,
            "pass": self.passed,
            "score": self.score,
            "metrics": self.metrics,
            "errors": self.errors,
            "duration_ms": self.duration_ms,
            "metadata": self.metadata,
        }


@dataclass
class BatchResult:
    """Result of a batch evaluation run."""

    timestamp: datetime
    config: RunConfig
    evaluations: list[TestEvaluation]
    duration_seconds: float | None = None
    errors: list[str] = field(default_factory=list)

    @property
    def total_count(self) -> int:
        """Total number of evaluations."""
        return len(self.evaluations)

    @property
    def passed_count(self) -> int:
        """Number of passed evaluations."""
        return sum(1 for e in self.evaluations if e.passed)

    @property
    def failed_count(self) -> int:
        """Number of failed evaluations."""
        return sum(1 for e in self.evaluations if not e.passed)

    @property
    def pass_rate(self) -> float:
        """Pass rate as percentage."""
        if not self.evaluations:
            return 0.0
        return (self.passed_count / self.total_count) * 100

    @property
    def average_score(self) -> float:
        """Average score across all evaluations."""
        if not self.evaluations:
            return 0.0
        return sum(e.score for e in self.evaluations) / len(self.evaluations)

    @property
    def passed(self) -> bool:
        """Whether the batch passed (>= 70% pass rate)."""
        return self.pass_rate >= 70.0

    def get_failed_evaluations(self) -> list[TestEvaluation]:
        """Get list of failed evaluations."""
        return [e for e in self.evaluations if not e.passed]

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "stats": {
                "total": self.total_count,
                "passed": self.passed_count,
                "failed": self.failed_count,
                "pass_rate": self.pass_rate,
                "average_score": self.average_score,
            },
            "duration_seconds": self.duration_seconds,
            "results": [e.to_dict() for e in self.evaluations],
            "errors": self.errors,
        }

    def save(self, path: str) -> None:
        """Save results to JSON file."""
        import json

        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)


# =============================================================================
# Batch Runner
# =============================================================================


class BatchRunner:
    """
    Runs batch evaluations across test cases.

    Supports filtering, concurrency control, and progress tracking.
    """

    def __init__(self, config: RunConfig):
        """
        Initialize the batch runner.

        Args:
            config: Run configuration
        """
        self.config = config
        self._evaluators: dict[str, Any] = {}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "BatchRunner":
        """Create runner from dictionary configuration."""
        config = RunConfig.from_dict(data)
        return cls(config)

    def run(self) -> BatchResult:
        """
        Run the batch evaluation.

        Returns:
            BatchResult with all evaluations
        """
        start_time = time.time()
        evaluations: list[TestEvaluation] = []
        errors: list[str] = []

        logger.info(
            "starting_batch_evaluation",
            dataset=self.config.dataset_path,
            task_type=self.config.task_type,
        )

        try:
            # Load test cases
            test_cases = self._load_and_filter_cases()

            if not test_cases:
                logger.warning("no_test_cases_found")
                return BatchResult(
                    timestamp=datetime.utcnow(),
                    config=self.config,
                    evaluations=[],
                    errors=["No test cases found matching criteria"],
                )

            logger.info("loaded_test_cases", count=len(test_cases))

            # Run evaluations
            for i, test_case in enumerate(test_cases):
                try:
                    evaluation = self._evaluate_case(test_case)
                    evaluations.append(evaluation)

                    if self.config.verbose:
                        status = "PASS" if evaluation.passed else "FAIL"
                        logger.info(
                            "test_completed",
                            test_id=test_case.id,
                            status=status,
                            score=evaluation.score,
                            progress=f"{i+1}/{len(test_cases)}",
                        )

                    if self.config.fail_fast and not evaluation.passed:
                        logger.info("fail_fast_triggered", test_id=test_case.id)
                        break

                except Exception as e:
                    error_msg = f"Error evaluating {test_case.id}: {str(e)}"
                    errors.append(error_msg)
                    logger.error("evaluation_error", test_id=test_case.id, error=str(e))

                    evaluations.append(TestEvaluation(
                        test_id=test_case.id,
                        passed=False,
                        score=0.0,
                        errors=[str(e)],
                    ))

                    if self.config.fail_fast:
                        break

        except Exception as e:
            errors.append(f"Batch run failed: {str(e)}")
            logger.exception("batch_run_failed")

        duration = time.time() - start_time

        result = BatchResult(
            timestamp=datetime.utcnow(),
            config=self.config,
            evaluations=evaluations,
            duration_seconds=duration,
            errors=errors,
        )

        logger.info(
            "batch_evaluation_complete",
            total=result.total_count,
            passed=result.passed_count,
            failed=result.failed_count,
            pass_rate=f"{result.pass_rate:.1f}%",
            duration=f"{duration:.1f}s",
        )

        return result

    def _load_and_filter_cases(self) -> list[BaseTestCase]:
        """Load and filter test cases based on configuration."""
        # Load dataset
        path = Path(self.config.dataset_path)
        if path.is_dir():
            # Load all files in directory
            test_cases = []
            for file_path in path.glob("**/*.json"):
                test_cases.extend(load_dataset(str(file_path)))
            for file_path in path.glob("**/*.yaml"):
                test_cases.extend(load_dataset(str(file_path)))
        else:
            test_cases = load_dataset(str(path))

        # Filter by task type
        if self.config.task_type:
            expected_type = f"{self.config.task_type.upper()}TestCase"
            test_cases = [
                tc for tc in test_cases
                if type(tc).__name__.upper() == expected_type.upper()
            ]

        # Filter by specialty
        if self.config.specialty_filter:
            test_cases = [
                tc for tc in test_cases
                if hasattr(tc, "specialty") and tc.specialty.value == self.config.specialty_filter
            ]

        # Filter by tags
        if self.config.tags_filter:
            test_cases = [
                tc for tc in test_cases
                if any(tag in tc.tags for tag in self.config.tags_filter)
            ]

        # Apply limit
        if self.config.limit:
            test_cases = test_cases[:self.config.limit]

        return test_cases

    def _evaluate_case(self, test_case: BaseTestCase) -> TestEvaluation:
        """Evaluate a single test case."""
        start_time = time.time()

        # Determine task type from test case class
        task_type = type(test_case).__name__.replace("TestCase", "").lower()

        # Get or create evaluator
        if task_type not in self._evaluators:
            self._evaluators[task_type] = get_evaluator(task_type)

        evaluator = self._evaluators[task_type]

        # For now, create a mock response since we don't have the actual API
        # In production, this would call the Nuvii API
        mock_response = self._create_mock_response(test_case, task_type)

        # Run evaluation
        eval_result = evaluator.evaluate(test_case, mock_response)

        duration_ms = int((time.time() - start_time) * 1000)

        # Extract metrics from scores
        metrics = {}
        for score in eval_result.scores:
            metrics[score.name] = score.value

        return TestEvaluation(
            test_id=test_case.id,
            passed=eval_result.passed,
            score=eval_result.composite_score,
            metrics=metrics,
            errors=[],
            duration_ms=duration_ms,
            metadata={
                "task_type": task_type,
                "specialty": test_case.specialty.value if hasattr(test_case, "specialty") else None,
                "complexity": test_case.complexity.value if hasattr(test_case, "complexity") else None,
            },
        )

    def _create_mock_response(self, test_case: BaseTestCase, task_type: str) -> Any:
        """Create a mock response for testing.

        In production, this would be replaced with actual API calls.
        """
        from nuvii_eval.schemas.api_responses import (
            CodeSuggestion,
            ConfidenceLevel,
            EMResponse,
            GapResponse,
            HCCResponse,
            ICDResponse,
            IdentifiedGap,
            MDMComponent,
            QueryResponse,
        )

        if task_type == "icd":
            from nuvii_eval.datasets.schemas import ICDTestCase
            if isinstance(test_case, ICDTestCase):
                return ICDResponse(
                    request_id="mock-001",
                    codes=[
                        CodeSuggestion(
                            code=code,
                            description=f"Mock description for {code}",
                            confidence=ConfidenceLevel.HIGH,
                            evidence_spans=["Mock evidence"],
                        )
                        for code in test_case.expected_icd_codes
                    ],
                    primary_code=test_case.primary_code,
                )

        elif task_type == "hcc":
            from nuvii_eval.datasets.schemas import HCCTestCase
            if isinstance(test_case, HCCTestCase):
                return HCCResponse(
                    request_id="mock-001",
                    hcc_codes=test_case.expected_hccs,
                    raf_score=(test_case.expected_raf_range[0] + test_case.expected_raf_range[1]) / 2,
                    opportunities=test_case.expected_opportunities,
                )

        elif task_type == "gap":
            from nuvii_eval.datasets.schemas import GapTestCase
            if isinstance(test_case, GapTestCase):
                return GapResponse(
                    request_id="mock-001",
                    facts_cache_key="mock-cache",
                    gaps=[
                        IdentifiedGap(
                            gap_id=f"gap-{i}",
                            gap_type=gap.gap_type,
                            condition=gap.condition,
                            evidence_text="Mock evidence",
                            priority=gap.min_priority,
                            confidence=ConfidenceLevel.HIGH,
                        )
                        for i, gap in enumerate(test_case.expected_gaps)
                    ],
                )

        elif task_type == "query":
            return QueryResponse(
                request_id="mock-001",
                query_id="query-001",
                query_text="Mock query text for clarification",
                gap_addressed="Mock gap",
                evidence_cited=["Mock evidence 1"],
                response_options=["Option A", "Option B"],
            )

        elif task_type == "em":
            from nuvii_eval.datasets.schemas import EMTestCase
            if isinstance(test_case, EMTestCase):
                return EMResponse(
                    request_id="mock-001",
                    recommended_code=test_case.expected_code,
                    recommended_level=test_case.expected_level,
                    mdm_analysis=MDMComponent(
                        problems=test_case.expected_mdm.problems,
                        data=test_case.expected_mdm.data,
                        risk=test_case.expected_mdm.risk,
                    ),
                    supporting_documentation=["Mock documentation"],
                )

        # Return a generic mock if type not matched
        return {"mock": True, "test_id": test_case.id}


# =============================================================================
# Convenience Functions
# =============================================================================


def run_evaluation(
    dataset_path: str,
    task_type: str | None = None,
    **kwargs: Any,
) -> BatchResult:
    """
    Run evaluation on a dataset.

    Args:
        dataset_path: Path to dataset file or directory
        task_type: Optional task type filter
        **kwargs: Additional RunConfig options

    Returns:
        BatchResult with evaluation results
    """
    config = RunConfig(
        dataset_path=dataset_path,
        task_type=task_type,
        **kwargs,
    )
    runner = BatchRunner(config)
    return runner.run()
