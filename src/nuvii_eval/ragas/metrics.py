"""
RAGAS metrics implementation for CDI evaluation.

Provides wrappers around RAGAS metrics with CDI-specific adaptations
and graceful fallbacks when RAGAS is not available.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import structlog

from nuvii_eval.config import get_settings

logger = structlog.get_logger(__name__)

# Try to import RAGAS - graceful fallback if not available
try:
    from ragas import evaluate as ragas_evaluate
    from ragas.metrics import (
        faithfulness,
        answer_relevancy,
        context_precision,
        context_recall,
    )
    from datasets import Dataset

    RAGAS_AVAILABLE = True
except ImportError:
    RAGAS_AVAILABLE = False
    logger.warning("ragas_not_available", message="RAGAS not installed, using fallback metrics")


# =============================================================================
# Configuration
# =============================================================================


@dataclass
class RAGASConfig:
    """
    Configuration for RAGAS evaluation.

    Attributes:
        llm_model: Model to use for LLM-based metrics
        embeddings_model: Model to use for embedding-based metrics
        batch_size: Batch size for evaluation
        timeout_seconds: Timeout for LLM calls
        use_async: Whether to use async evaluation
        phi_safe_mode: Enable PHI-safe evaluation (redact before sending to LLM)
    """

    llm_model: str = "gpt-4o-mini"
    embeddings_model: str = "text-embedding-3-small"
    batch_size: int = 10
    timeout_seconds: int = 120
    use_async: bool = True
    phi_safe_mode: bool = True

    @classmethod
    def from_settings(cls) -> "RAGASConfig":
        """Create config from application settings."""
        settings = get_settings()
        return cls(
            llm_model=settings.llm_judge.model,
            timeout_seconds=settings.llm_judge.timeout_seconds,
            phi_safe_mode=settings.eval.phi_safe_mode,
        )


# =============================================================================
# Result Types
# =============================================================================


@dataclass
class RAGASMetricResult:
    """
    Result from a single RAGAS metric evaluation.

    Attributes:
        metric_name: Name of the metric
        score: Score value (0.0 to 1.0)
        timestamp: When the evaluation was performed
        details: Additional metric-specific details
        error: Error message if evaluation failed
    """

    metric_name: str
    score: float
    timestamp: datetime = field(default_factory=datetime.utcnow)
    details: dict[str, Any] = field(default_factory=dict)
    error: str | None = None

    @property
    def passed(self) -> bool:
        """Check if metric passed (score >= 0.5)."""
        return self.score >= 0.5 and self.error is None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "metric_name": self.metric_name,
            "score": self.score,
            "timestamp": self.timestamp.isoformat(),
            "passed": self.passed,
            "details": self.details,
            "error": self.error,
        }


@dataclass
class RAGASEvaluationResult:
    """
    Complete RAGAS evaluation result.

    Attributes:
        test_case_id: ID of the evaluated test case
        metrics: Individual metric results
        overall_score: Weighted average of all metrics
        timestamp: When the evaluation was performed
    """

    test_case_id: str
    metrics: list[RAGASMetricResult]
    timestamp: datetime = field(default_factory=datetime.utcnow)

    @property
    def overall_score(self) -> float:
        """Calculate overall score as average of all metrics."""
        valid_scores = [m.score for m in self.metrics if m.error is None]
        return sum(valid_scores) / len(valid_scores) if valid_scores else 0.0

    @property
    def passed(self) -> bool:
        """Check if all metrics passed."""
        return all(m.passed for m in self.metrics)

    def get_metric(self, name: str) -> RAGASMetricResult | None:
        """Get a specific metric result by name."""
        for m in self.metrics:
            if m.metric_name == name:
                return m
        return None


# =============================================================================
# Base RAGAS Evaluator
# =============================================================================


class RAGASEvaluator(ABC):
    """
    Abstract base class for RAGAS metric evaluators.

    Provides common functionality for all RAGAS-based evaluations.
    """

    metric_name: str = "base"

    def __init__(self, config: RAGASConfig | None = None):
        """
        Initialize the evaluator.

        Args:
            config: RAGAS configuration (uses defaults if not provided)
        """
        self.config = config or RAGASConfig()
        self._setup()

    def _setup(self) -> None:
        """Initialize evaluator-specific configuration. Override in subclasses."""
        pass

    @abstractmethod
    def evaluate(
        self,
        question: str,
        answer: str,
        contexts: list[str],
        ground_truth: str | None = None,
    ) -> RAGASMetricResult:
        """
        Evaluate a single sample.

        Args:
            question: The input question/query
            answer: The generated answer/response
            contexts: Retrieved context passages
            ground_truth: Expected answer (for reference-based metrics)

        Returns:
            RAGASMetricResult with the evaluation score
        """
        pass

    def evaluate_batch(
        self,
        samples: list[dict[str, Any]],
    ) -> list[RAGASMetricResult]:
        """
        Evaluate a batch of samples.

        Args:
            samples: List of dicts with keys: question, answer, contexts, ground_truth

        Returns:
            List of RAGASMetricResults
        """
        results = []
        for sample in samples:
            result = self.evaluate(
                question=sample["question"],
                answer=sample["answer"],
                contexts=sample.get("contexts", []),
                ground_truth=sample.get("ground_truth"),
            )
            results.append(result)
        return results


# =============================================================================
# Faithfulness Evaluator
# =============================================================================


class FaithfulnessEvaluator(RAGASEvaluator):
    """
    Evaluates faithfulness of generated answers to source context.

    Faithfulness measures how factually consistent the generated answer
    is with the provided context. High faithfulness means the answer
    doesn't contain hallucinated information.

    For CDI:
    - Ensures code suggestions are supported by clinical note
    - Validates that gap detections cite actual evidence
    - Checks query recommendations match documented findings
    """

    metric_name = "faithfulness"

    def evaluate(
        self,
        question: str,
        answer: str,
        contexts: list[str],
        ground_truth: str | None = None,
    ) -> RAGASMetricResult:
        """
        Evaluate faithfulness of answer to contexts.

        Args:
            question: The clinical query
            answer: Generated response (e.g., code suggestion, query text)
            contexts: Clinical note passages used as context
            ground_truth: Not used for faithfulness

        Returns:
            RAGASMetricResult with faithfulness score
        """
        if not RAGAS_AVAILABLE:
            return self._fallback_evaluate(question, answer, contexts)

        try:
            # Create dataset for RAGAS
            data = {
                "question": [question],
                "answer": [answer],
                "contexts": [contexts],
            }
            dataset = Dataset.from_dict(data)

            # Run RAGAS evaluation
            result = ragas_evaluate(
                dataset,
                metrics=[faithfulness],
            )

            score = result["faithfulness"]

            return RAGASMetricResult(
                metric_name=self.metric_name,
                score=float(score),
                details={
                    "context_count": len(contexts),
                    "answer_length": len(answer),
                },
            )

        except Exception as e:
            logger.error("faithfulness_evaluation_failed", error=str(e))
            return RAGASMetricResult(
                metric_name=self.metric_name,
                score=0.0,
                error=str(e),
            )

    def _fallback_evaluate(
        self,
        question: str,
        answer: str,
        contexts: list[str],
    ) -> RAGASMetricResult:
        """
        Fallback evaluation when RAGAS is not available.

        Uses simple heuristics:
        - Token overlap between answer and contexts
        - Presence of context-specific terms in answer
        """
        if not contexts:
            return RAGASMetricResult(
                metric_name=self.metric_name,
                score=0.0,
                details={"fallback": True, "reason": "no_contexts"},
            )

        # Tokenize
        answer_tokens = set(answer.lower().split())
        context_tokens = set()
        for ctx in contexts:
            context_tokens.update(ctx.lower().split())

        # Calculate overlap
        if not answer_tokens:
            score = 0.0
        else:
            overlap = len(answer_tokens & context_tokens)
            score = min(1.0, overlap / len(answer_tokens))

        return RAGASMetricResult(
            metric_name=self.metric_name,
            score=score,
            details={
                "fallback": True,
                "token_overlap": overlap if answer_tokens else 0,
                "answer_tokens": len(answer_tokens),
            },
        )


# =============================================================================
# Answer Relevancy Evaluator
# =============================================================================


class AnswerRelevancyEvaluator(RAGASEvaluator):
    """
    Evaluates relevancy of generated answers to the question.

    Answer relevancy measures how well the generated answer addresses
    the original question. High relevancy means the answer is on-topic
    and provides useful information.

    For CDI:
    - Ensures code suggestions address the documented conditions
    - Validates queries are relevant to identified gaps
    - Checks E/M recommendations relate to visit complexity
    """

    metric_name = "answer_relevancy"

    def evaluate(
        self,
        question: str,
        answer: str,
        contexts: list[str],
        ground_truth: str | None = None,
    ) -> RAGASMetricResult:
        """
        Evaluate relevancy of answer to question.

        Args:
            question: The clinical query
            answer: Generated response
            contexts: Clinical note passages (used for context)
            ground_truth: Not used for relevancy

        Returns:
            RAGASMetricResult with relevancy score
        """
        if not RAGAS_AVAILABLE:
            return self._fallback_evaluate(question, answer, contexts)

        try:
            data = {
                "question": [question],
                "answer": [answer],
                "contexts": [contexts],
            }
            dataset = Dataset.from_dict(data)

            result = ragas_evaluate(
                dataset,
                metrics=[answer_relevancy],
            )

            score = result["answer_relevancy"]

            return RAGASMetricResult(
                metric_name=self.metric_name,
                score=float(score),
                details={
                    "question_length": len(question),
                    "answer_length": len(answer),
                },
            )

        except Exception as e:
            logger.error("answer_relevancy_evaluation_failed", error=str(e))
            return RAGASMetricResult(
                metric_name=self.metric_name,
                score=0.0,
                error=str(e),
            )

    def _fallback_evaluate(
        self,
        question: str,
        answer: str,
        contexts: list[str],
    ) -> RAGASMetricResult:
        """
        Fallback evaluation using question-answer token overlap.
        """
        question_tokens = set(question.lower().split())
        answer_tokens = set(answer.lower().split())

        # Remove common stop words
        stop_words = {"the", "a", "an", "is", "are", "was", "were", "what", "how", "why", "when", "where", "which", "who", "for", "to", "of", "in", "on", "at", "by", "with"}
        question_tokens -= stop_words
        answer_tokens -= stop_words

        if not question_tokens:
            score = 0.5  # Neutral if question has no meaningful tokens
        else:
            overlap = len(question_tokens & answer_tokens)
            # Relevancy based on how many question terms appear in answer
            score = min(1.0, overlap / len(question_tokens) * 2)

        return RAGASMetricResult(
            metric_name=self.metric_name,
            score=score,
            details={
                "fallback": True,
                "question_tokens": len(question_tokens),
                "overlap": overlap if question_tokens else 0,
            },
        )


# =============================================================================
# Context Precision Evaluator
# =============================================================================


class ContextPrecisionEvaluator(RAGASEvaluator):
    """
    Evaluates precision of retrieved contexts.

    Context precision measures whether the retrieved context passages
    are relevant to answering the question. High precision means
    less noise in the retrieved contexts.

    For CDI:
    - Validates that retrieved note sections are relevant to the task
    - Ensures gap detection uses pertinent clinical evidence
    - Checks that code suggestions cite relevant documentation
    """

    metric_name = "context_precision"

    def evaluate(
        self,
        question: str,
        answer: str,
        contexts: list[str],
        ground_truth: str | None = None,
    ) -> RAGASMetricResult:
        """
        Evaluate precision of retrieved contexts.

        Args:
            question: The clinical query
            answer: Generated response
            contexts: Retrieved clinical note passages
            ground_truth: Expected answer (used for precision calculation)

        Returns:
            RAGASMetricResult with context precision score
        """
        if not RAGAS_AVAILABLE:
            return self._fallback_evaluate(question, answer, contexts, ground_truth)

        try:
            data = {
                "question": [question],
                "answer": [answer],
                "contexts": [contexts],
                "ground_truth": [ground_truth or answer],
            }
            dataset = Dataset.from_dict(data)

            result = ragas_evaluate(
                dataset,
                metrics=[context_precision],
            )

            score = result["context_precision"]

            return RAGASMetricResult(
                metric_name=self.metric_name,
                score=float(score),
                details={
                    "context_count": len(contexts),
                    "total_context_length": sum(len(c) for c in contexts),
                },
            )

        except Exception as e:
            logger.error("context_precision_evaluation_failed", error=str(e))
            return RAGASMetricResult(
                metric_name=self.metric_name,
                score=0.0,
                error=str(e),
            )

    def _fallback_evaluate(
        self,
        question: str,
        answer: str,
        contexts: list[str],
        ground_truth: str | None = None,
    ) -> RAGASMetricResult:
        """
        Fallback evaluation using context-answer overlap.
        """
        if not contexts:
            return RAGASMetricResult(
                metric_name=self.metric_name,
                score=0.0,
                details={"fallback": True, "reason": "no_contexts"},
            )

        # Use answer or ground truth for comparison
        reference = ground_truth or answer
        ref_tokens = set(reference.lower().split())

        # Calculate precision for each context
        precisions = []
        for ctx in contexts:
            ctx_tokens = set(ctx.lower().split())
            if ctx_tokens:
                overlap = len(ctx_tokens & ref_tokens)
                precision = overlap / len(ctx_tokens)
                precisions.append(precision)

        # Average precision across contexts
        score = sum(precisions) / len(precisions) if precisions else 0.0

        return RAGASMetricResult(
            metric_name=self.metric_name,
            score=min(1.0, score * 2),  # Scale up slightly
            details={
                "fallback": True,
                "context_count": len(contexts),
                "avg_precision": score,
            },
        )


# =============================================================================
# Context Recall Evaluator
# =============================================================================


class ContextRecallEvaluator(RAGASEvaluator):
    """
    Evaluates recall of retrieved contexts.

    Context recall measures whether the retrieved contexts contain
    all the information needed to answer the question. High recall
    means the retrieval captured necessary information.

    For CDI:
    - Validates retrieval captured all relevant clinical findings
    - Ensures gap detection has access to complete patient history
    - Checks that code suggestions have full diagnostic context
    """

    metric_name = "context_recall"

    def evaluate(
        self,
        question: str,
        answer: str,
        contexts: list[str],
        ground_truth: str | None = None,
    ) -> RAGASMetricResult:
        """
        Evaluate recall of retrieved contexts.

        Args:
            question: The clinical query
            answer: Generated response
            contexts: Retrieved clinical note passages
            ground_truth: Expected answer (required for recall)

        Returns:
            RAGASMetricResult with context recall score
        """
        if not ground_truth:
            return RAGASMetricResult(
                metric_name=self.metric_name,
                score=0.5,  # Neutral score when no ground truth
                details={"reason": "no_ground_truth"},
            )

        if not RAGAS_AVAILABLE:
            return self._fallback_evaluate(question, answer, contexts, ground_truth)

        try:
            data = {
                "question": [question],
                "answer": [answer],
                "contexts": [contexts],
                "ground_truth": [ground_truth],
            }
            dataset = Dataset.from_dict(data)

            result = ragas_evaluate(
                dataset,
                metrics=[context_recall],
            )

            score = result["context_recall"]

            return RAGASMetricResult(
                metric_name=self.metric_name,
                score=float(score),
                details={
                    "context_count": len(contexts),
                    "ground_truth_length": len(ground_truth),
                },
            )

        except Exception as e:
            logger.error("context_recall_evaluation_failed", error=str(e))
            return RAGASMetricResult(
                metric_name=self.metric_name,
                score=0.0,
                error=str(e),
            )

    def _fallback_evaluate(
        self,
        question: str,
        answer: str,
        contexts: list[str],
        ground_truth: str,
    ) -> RAGASMetricResult:
        """
        Fallback evaluation using ground truth coverage.
        """
        if not contexts:
            return RAGASMetricResult(
                metric_name=self.metric_name,
                score=0.0,
                details={"fallback": True, "reason": "no_contexts"},
            )

        # Check how much of ground truth is covered by contexts
        gt_tokens = set(ground_truth.lower().split())
        context_tokens = set()
        for ctx in contexts:
            context_tokens.update(ctx.lower().split())

        if not gt_tokens:
            score = 0.5
        else:
            covered = len(gt_tokens & context_tokens)
            score = covered / len(gt_tokens)

        return RAGASMetricResult(
            metric_name=self.metric_name,
            score=score,
            details={
                "fallback": True,
                "gt_tokens": len(gt_tokens),
                "covered_tokens": covered if gt_tokens else 0,
            },
        )


# =============================================================================
# Combined RAGAS Evaluator
# =============================================================================


class CombinedRAGASEvaluator:
    """
    Combines multiple RAGAS metrics into a single evaluation.

    Runs all configured metrics and returns a comprehensive result.
    """

    def __init__(
        self,
        config: RAGASConfig | None = None,
        metrics: list[str] | None = None,
    ):
        """
        Initialize the combined evaluator.

        Args:
            config: RAGAS configuration
            metrics: List of metric names to use (default: all)
        """
        self.config = config or RAGASConfig()
        self.metrics = metrics or ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]

        # Initialize evaluators
        self.evaluators: dict[str, RAGASEvaluator] = {}
        metric_classes = {
            "faithfulness": FaithfulnessEvaluator,
            "answer_relevancy": AnswerRelevancyEvaluator,
            "context_precision": ContextPrecisionEvaluator,
            "context_recall": ContextRecallEvaluator,
        }

        for metric in self.metrics:
            if metric in metric_classes:
                self.evaluators[metric] = metric_classes[metric](self.config)

    def evaluate(
        self,
        test_case_id: str,
        question: str,
        answer: str,
        contexts: list[str],
        ground_truth: str | None = None,
    ) -> RAGASEvaluationResult:
        """
        Run all metrics on a single sample.

        Args:
            test_case_id: Identifier for the test case
            question: The input question
            answer: Generated answer
            contexts: Retrieved contexts
            ground_truth: Expected answer

        Returns:
            RAGASEvaluationResult with all metric scores
        """
        metric_results = []

        for name, evaluator in self.evaluators.items():
            result = evaluator.evaluate(
                question=question,
                answer=answer,
                contexts=contexts,
                ground_truth=ground_truth,
            )
            metric_results.append(result)

        return RAGASEvaluationResult(
            test_case_id=test_case_id,
            metrics=metric_results,
        )

    def evaluate_batch(
        self,
        samples: list[dict[str, Any]],
    ) -> list[RAGASEvaluationResult]:
        """
        Evaluate a batch of samples.

        Args:
            samples: List of dicts with keys: test_case_id, question, answer, contexts, ground_truth

        Returns:
            List of RAGASEvaluationResults
        """
        results = []
        for sample in samples:
            result = self.evaluate(
                test_case_id=sample.get("test_case_id", "unknown"),
                question=sample["question"],
                answer=sample["answer"],
                contexts=sample.get("contexts", []),
                ground_truth=sample.get("ground_truth"),
            )
            results.append(result)
        return results
