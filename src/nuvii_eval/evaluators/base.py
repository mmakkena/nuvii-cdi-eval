"""
Base evaluator framework for CDI agent evaluation.

Provides abstract base classes and common utilities for implementing
domain-specific evaluators (ICD, HCC, Gap, Query, E/M).
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Generic, TypeVar

import structlog

from nuvii_eval.datasets.schemas import BaseTestCase

logger = structlog.get_logger(__name__)

# Type variables for generic evaluator
T = TypeVar("T", bound=BaseTestCase)  # Test case type
R = TypeVar("R")  # API response type


@dataclass
class EvalScore:
    """
    Individual evaluation score for a specific metric.

    Attributes:
        name: Metric name (e.g., "top_1_accuracy", "precision")
        value: Score value (typically 0.0 to 1.0)
        weight: Weight for composite score calculation
        details: Additional details about the score
    """

    name: str
    value: float
    weight: float = 1.0
    details: dict[str, Any] = field(default_factory=dict)

    @property
    def weighted_value(self) -> float:
        """Calculate weighted score value."""
        return self.value * self.weight

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "name": self.name,
            "value": self.value,
            "weight": self.weight,
            "weighted_value": self.weighted_value,
            "details": self.details,
        }


@dataclass
class EvalResult:
    """
    Complete evaluation result for a single test case.

    Attributes:
        test_case_id: ID of the evaluated test case
        evaluator_type: Type of evaluator used
        timestamp: When the evaluation was performed
        scores: List of individual metric scores
        passed: Whether the evaluation passed threshold
        details: Additional evaluation details
        errors: Any errors encountered during evaluation
        latency_ms: API call latency in milliseconds
    """

    test_case_id: str
    evaluator_type: str
    timestamp: datetime
    scores: list[EvalScore]
    passed: bool
    details: dict[str, Any] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)
    latency_ms: int | None = None

    @property
    def composite_score(self) -> float:
        """
        Calculate weighted composite score.

        Returns:
            Weighted average of all scores, or 0.0 if no scores
        """
        if not self.scores:
            return 0.0

        total_weight = sum(s.weight for s in self.scores)
        if total_weight == 0:
            return 0.0

        return sum(s.weighted_value for s in self.scores) / total_weight

    @property
    def score_dict(self) -> dict[str, float]:
        """Get scores as a simple name -> value dictionary."""
        return {s.name: s.value for s in self.scores}

    def get_score(self, name: str) -> float | None:
        """Get a specific score by name."""
        for score in self.scores:
            if score.name == name:
                return score.value
        return None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "test_case_id": self.test_case_id,
            "evaluator_type": self.evaluator_type,
            "timestamp": self.timestamp.isoformat(),
            "composite_score": self.composite_score,
            "passed": self.passed,
            "scores": {s.name: {"value": s.value, "weight": s.weight} for s in self.scores},
            "details": self.details,
            "errors": self.errors,
            "latency_ms": self.latency_ms,
        }

    def to_flat_dict(self) -> dict[str, Any]:
        """Convert to flat dictionary (useful for CSV export)."""
        flat = {
            "test_case_id": self.test_case_id,
            "evaluator_type": self.evaluator_type,
            "timestamp": self.timestamp.isoformat(),
            "composite_score": self.composite_score,
            "passed": self.passed,
            "latency_ms": self.latency_ms,
            "error_count": len(self.errors),
        }

        # Add individual scores
        for score in self.scores:
            flat[f"score_{score.name}"] = score.value

        return flat


class BaseEvaluator(ABC, Generic[T, R]):
    """
    Abstract base class for all evaluators.

    Subclasses must implement the `evaluate` method to perform
    domain-specific evaluation logic.

    Type Parameters:
        T: Test case schema type (e.g., ICDTestCase)
        R: API response type (e.g., CodingSuggestResponse)

    Usage:
        class ICDEvaluator(BaseEvaluator[ICDTestCase, CodingSuggestResponse]):
            evaluator_type = "icd"

            def evaluate(self, test_case, response):
                # Implementation
                pass
    """

    # Override in subclasses
    evaluator_type: str = "base"
    pass_threshold: float = 0.7

    def __init__(self, config: dict[str, Any] | None = None):
        """
        Initialize the evaluator.

        Args:
            config: Optional configuration dictionary
        """
        self.config = config or {}
        self._setup()

    def _setup(self) -> None:
        """
        Hook for subclass initialization.

        Override this method to perform custom setup without
        overriding __init__.
        """
        pass

    @abstractmethod
    def evaluate(self, test_case: T, response: R) -> EvalResult:
        """
        Evaluate an API response against expected values.

        Args:
            test_case: The test case with expected values
            response: The API response to evaluate

        Returns:
            EvalResult with scores and details
        """
        pass

    def _create_result(
        self,
        test_case: T,
        scores: list[EvalScore],
        details: dict[str, Any] | None = None,
        errors: list[str] | None = None,
        latency_ms: int | None = None,
        custom_pass_check: bool | None = None,
    ) -> EvalResult:
        """
        Helper to create an EvalResult.

        Args:
            test_case: The test case being evaluated
            scores: List of evaluation scores
            details: Additional details
            errors: Any errors encountered
            latency_ms: API latency
            custom_pass_check: Override automatic pass/fail determination

        Returns:
            Configured EvalResult
        """
        # Determine pass/fail
        if custom_pass_check is not None:
            passed = custom_pass_check
        elif errors:
            passed = False
        else:
            # Pass if composite score meets threshold
            composite = self._calculate_composite(scores)
            passed = composite >= self.pass_threshold

        result = EvalResult(
            test_case_id=test_case.id,
            evaluator_type=self.evaluator_type,
            timestamp=datetime.utcnow(),
            scores=scores,
            passed=passed,
            details=details or {},
            errors=errors or [],
            latency_ms=latency_ms,
        )

        logger.debug(
            "evaluation_complete",
            test_case_id=test_case.id,
            evaluator=self.evaluator_type,
            composite_score=result.composite_score,
            passed=result.passed,
        )

        return result

    def _calculate_composite(self, scores: list[EvalScore]) -> float:
        """Calculate composite score from individual scores."""
        if not scores:
            return 0.0

        total_weight = sum(s.weight for s in scores)
        if total_weight == 0:
            return 0.0

        return sum(s.weighted_value for s in scores) / total_weight

    def _create_error_result(
        self,
        test_case: T,
        error: str | Exception,
    ) -> EvalResult:
        """Create an error result when evaluation fails."""
        error_msg = str(error)

        logger.error(
            "evaluation_error",
            test_case_id=test_case.id,
            evaluator=self.evaluator_type,
            error=error_msg,
        )

        return EvalResult(
            test_case_id=test_case.id,
            evaluator_type=self.evaluator_type,
            timestamp=datetime.utcnow(),
            scores=[],
            passed=False,
            errors=[error_msg],
        )


# =============================================================================
# Common Evaluation Utilities
# =============================================================================


def precision(predicted: set, expected: set) -> float:
    """
    Calculate precision: correct predictions / total predictions.

    Args:
        predicted: Set of predicted values
        expected: Set of expected values

    Returns:
        Precision score (0.0 to 1.0)
    """
    if not predicted:
        return 0.0
    return len(predicted & expected) / len(predicted)


def recall(predicted: set, expected: set) -> float:
    """
    Calculate recall: correct predictions / total expected.

    Args:
        predicted: Set of predicted values
        expected: Set of expected values

    Returns:
        Recall score (0.0 to 1.0)
    """
    if not expected:
        return 1.0  # Nothing to find = perfect recall
    return len(predicted & expected) / len(expected)


def f1_score(prec: float, rec: float) -> float:
    """
    Calculate F1 score: harmonic mean of precision and recall.

    Args:
        prec: Precision score
        rec: Recall score

    Returns:
        F1 score (0.0 to 1.0)
    """
    if prec + rec == 0:
        return 0.0
    return 2 * (prec * rec) / (prec + rec)


def top_n_hit(predicted: list, expected: set, n: int) -> bool:
    """
    Check if any expected value is in top N predictions.

    Args:
        predicted: Ordered list of predictions (best first)
        expected: Set of expected values
        n: Number of top predictions to check

    Returns:
        True if any expected value is in top N
    """
    return bool(set(predicted[:n]) & expected)


def jaccard_similarity(set1: set, set2: set) -> float:
    """
    Calculate Jaccard similarity between two sets.

    Args:
        set1: First set
        set2: Second set

    Returns:
        Jaccard similarity (0.0 to 1.0)
    """
    if not set1 and not set2:
        return 1.0
    if not set1 or not set2:
        return 0.0

    intersection = len(set1 & set2)
    union = len(set1 | set2)

    return intersection / union


def normalize_code(code: str) -> str:
    """
    Normalize a medical code for comparison.

    - Uppercase
    - Remove extra whitespace
    - Standardize decimal format

    Args:
        code: Code to normalize

    Returns:
        Normalized code string
    """
    return code.strip().upper()


def codes_match(code1: str, code2: str, allow_truncated: bool = False) -> bool:
    """
    Check if two codes match, with optional truncation allowance.

    Args:
        code1: First code
        code2: Second code
        allow_truncated: If True, E11 matches E11.9

    Returns:
        True if codes match
    """
    c1 = normalize_code(code1)
    c2 = normalize_code(code2)

    if c1 == c2:
        return True

    if allow_truncated:
        # Check if one is a prefix of the other
        return c1.startswith(c2) or c2.startswith(c1)

    return False
