"""
Async evaluation runner.

Provides utilities for running evaluations with async/parallel execution.
"""

import asyncio
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import structlog

from nuvii_eval.datasets.schemas import BaseTestCase
from nuvii_eval.runner.batch import BatchResult, RunConfig, TestEvaluation

logger = structlog.get_logger(__name__)


# =============================================================================
# Async Runner Configuration
# =============================================================================


@dataclass
class AsyncRunConfig(RunConfig):
    """Configuration for async evaluation run."""

    semaphore_limit: int = 10  # Max concurrent tasks
    retry_count: int = 2
    retry_delay_seconds: float = 1.0
    progress_callback: Any = None  # Callable for progress updates


# =============================================================================
# Async Batch Runner
# =============================================================================


class AsyncBatchRunner:
    """
    Runs batch evaluations asynchronously.

    Supports concurrent API calls with rate limiting.
    """

    def __init__(self, config: AsyncRunConfig):
        """
        Initialize the async runner.

        Args:
            config: Async run configuration
        """
        self.config = config
        self._semaphore: asyncio.Semaphore | None = None
        self._evaluators: dict[str, Any] = {}
        self._progress: dict[str, Any] = {
            "total": 0,
            "completed": 0,
            "passed": 0,
            "failed": 0,
        }

    async def run(self) -> BatchResult:
        """
        Run the batch evaluation asynchronously.

        Returns:
            BatchResult with all evaluations
        """
        start_time = time.time()
        errors: list[str] = []

        logger.info(
            "starting_async_batch_evaluation",
            dataset=self.config.dataset_path,
            concurrency=self.config.semaphore_limit,
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

            self._progress["total"] = len(test_cases)
            logger.info("loaded_test_cases", count=len(test_cases))

            # Create semaphore for concurrency control
            self._semaphore = asyncio.Semaphore(self.config.semaphore_limit)

            # Run evaluations concurrently
            tasks = [
                self._evaluate_with_semaphore(test_case)
                for test_case in test_cases
            ]

            evaluations = await asyncio.gather(*tasks, return_exceptions=True)

            # Process results
            processed_evaluations: list[TestEvaluation] = []
            for i, result in enumerate(evaluations):
                if isinstance(result, Exception):
                    error_msg = f"Error evaluating test case: {str(result)}"
                    errors.append(error_msg)
                    processed_evaluations.append(TestEvaluation(
                        test_id=test_cases[i].id,
                        passed=False,
                        score=0.0,
                        errors=[str(result)],
                    ))
                else:
                    processed_evaluations.append(result)

        except Exception as e:
            errors.append(f"Async batch run failed: {str(e)}")
            logger.exception("async_batch_run_failed")
            processed_evaluations = []

        duration = time.time() - start_time

        result = BatchResult(
            timestamp=datetime.utcnow(),
            config=self.config,
            evaluations=processed_evaluations,
            duration_seconds=duration,
            errors=errors,
        )

        logger.info(
            "async_batch_evaluation_complete",
            total=result.total_count,
            passed=result.passed_count,
            failed=result.failed_count,
            pass_rate=f"{result.pass_rate:.1f}%",
            duration=f"{duration:.1f}s",
        )

        return result

    async def _evaluate_with_semaphore(
        self,
        test_case: BaseTestCase,
    ) -> TestEvaluation:
        """Evaluate a test case with semaphore control."""
        async with self._semaphore:
            return await self._evaluate_case_async(test_case)

    async def _evaluate_case_async(
        self,
        test_case: BaseTestCase,
    ) -> TestEvaluation:
        """Evaluate a single test case asynchronously."""
        from nuvii_eval.evaluators import get_evaluator

        start_time = time.time()

        # Determine task type
        task_type = type(test_case).__name__.replace("TestCase", "").lower()

        # Get or create evaluator
        if task_type not in self._evaluators:
            self._evaluators[task_type] = get_evaluator(task_type)

        evaluator = self._evaluators[task_type]

        # Simulate async API call (in production, this would use httpx async client)
        response = await self._call_api_async(test_case, task_type)

        # Run evaluation (evaluators are sync, but fast)
        eval_result = evaluator.evaluate(test_case, response)

        duration_ms = int((time.time() - start_time) * 1000)

        # Update progress
        self._progress["completed"] += 1
        if eval_result.passed:
            self._progress["passed"] += 1
        else:
            self._progress["failed"] += 1

        # Call progress callback if provided
        if self.config.progress_callback:
            self.config.progress_callback(self._progress.copy())

        # Extract metrics
        metrics = {score.name: score.value for score in eval_result.scores}

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
            },
        )

    async def _call_api_async(
        self,
        test_case: BaseTestCase,
        task_type: str,
    ) -> Any:
        """
        Make async API call.

        In production, this would use httpx async client.
        For now, returns mock response.
        """
        # Simulate network latency
        await asyncio.sleep(0.01)

        # Return mock response (same as sync runner)
        from nuvii_eval.runner.batch import BatchRunner

        sync_runner = BatchRunner(self.config)
        return sync_runner._create_mock_response(test_case, task_type)

    def _load_and_filter_cases(self) -> list[BaseTestCase]:
        """Load and filter test cases (reuse from sync runner)."""
        from nuvii_eval.runner.batch import BatchRunner

        sync_runner = BatchRunner(self.config)
        return sync_runner._load_and_filter_cases()


# =============================================================================
# Parallel Evaluation Utilities
# =============================================================================


async def run_parallel_evaluations(
    test_cases: list[BaseTestCase],
    max_concurrency: int = 10,
    timeout_seconds: float = 60.0,
) -> list[TestEvaluation]:
    """
    Run evaluations in parallel with concurrency limit.

    Args:
        test_cases: List of test cases to evaluate
        max_concurrency: Maximum concurrent evaluations
        timeout_seconds: Timeout for each evaluation

    Returns:
        List of evaluation results
    """
    semaphore = asyncio.Semaphore(max_concurrency)
    evaluators: dict[str, Any] = {}

    async def evaluate_one(test_case: BaseTestCase) -> TestEvaluation:
        async with semaphore:
            from nuvii_eval.evaluators import get_evaluator

            task_type = type(test_case).__name__.replace("TestCase", "").lower()

            if task_type not in evaluators:
                evaluators[task_type] = get_evaluator(task_type)

            evaluator = evaluators[task_type]

            # Mock response for now
            from nuvii_eval.runner.batch import BatchRunner

            mock_config = RunConfig(dataset_path="")
            mock_runner = BatchRunner(mock_config)
            response = mock_runner._create_mock_response(test_case, task_type)

            result = evaluator.evaluate(test_case, response)

            return TestEvaluation(
                test_id=test_case.id,
                passed=result.passed,
                score=result.composite_score,
                metrics={s.name: s.value for s in result.scores},
            )

    tasks = [
        asyncio.wait_for(evaluate_one(tc), timeout=timeout_seconds)
        for tc in test_cases
    ]

    results = await asyncio.gather(*tasks, return_exceptions=True)

    # Convert exceptions to failed evaluations
    evaluations = []
    for i, result in enumerate(results):
        if isinstance(result, Exception):
            evaluations.append(TestEvaluation(
                test_id=test_cases[i].id,
                passed=False,
                score=0.0,
                errors=[str(result)],
            ))
        else:
            evaluations.append(result)

    return evaluations


# =============================================================================
# Convenience Functions
# =============================================================================


def run_async_evaluation(
    dataset_path: str,
    task_type: str | None = None,
    max_concurrency: int = 10,
    **kwargs: Any,
) -> BatchResult:
    """
    Run async evaluation on a dataset.

    This is a sync wrapper around the async runner for convenience.

    Args:
        dataset_path: Path to dataset file or directory
        task_type: Optional task type filter
        max_concurrency: Maximum concurrent evaluations
        **kwargs: Additional AsyncRunConfig options

    Returns:
        BatchResult with evaluation results
    """
    config = AsyncRunConfig(
        dataset_path=dataset_path,
        task_type=task_type,
        semaphore_limit=max_concurrency,
        **kwargs,
    )
    runner = AsyncBatchRunner(config)
    return asyncio.run(runner.run())
