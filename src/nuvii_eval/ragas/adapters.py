"""
Adapters for integrating RAGAS with CDI evaluation.

Provides utilities to convert CDI test cases and responses
into RAGAS-compatible formats.
"""

from dataclasses import dataclass, field
from typing import Any

import structlog

from nuvii_eval.datasets.schemas import (
    BaseTestCase,
    GapTestCase,
    ICDTestCase,
    QueryTestCase,
)
from nuvii_eval.ragas.metrics import (
    CombinedRAGASEvaluator,
    RAGASConfig,
    RAGASEvaluationResult,
)

logger = structlog.get_logger(__name__)


# =============================================================================
# Data Conversion Utilities
# =============================================================================


def convert_to_ragas_format(
    test_case: BaseTestCase,
    response: Any,
    task_type: str,
) -> dict[str, Any]:
    """
    Convert a CDI test case and response to RAGAS format.

    Args:
        test_case: The CDI test case
        response: The API response
        task_type: Type of task (icd, gap, query, etc.)

    Returns:
        Dict with question, answer, contexts, ground_truth
    """
    converters = {
        "icd": _convert_icd_to_ragas,
        "gap": _convert_gap_to_ragas,
        "query": _convert_query_to_ragas,
    }

    converter = converters.get(task_type)
    if converter:
        return converter(test_case, response)

    # Default conversion
    return _default_conversion(test_case, response)


def _convert_icd_to_ragas(
    test_case: ICDTestCase,
    response: Any,
) -> dict[str, Any]:
    """Convert ICD test case to RAGAS format."""
    # Question: What codes should be assigned?
    question = f"What ICD-10 codes should be assigned based on this clinical documentation?"

    # Answer: The suggested codes with descriptions
    if hasattr(response, "suggested_codes") and response.suggested_codes:
        answer_parts = []
        for code in response.suggested_codes:
            desc = getattr(code, "description", "")
            answer_parts.append(f"{code.icd10_code}: {desc}")
        answer = "\n".join(answer_parts)
    else:
        answer = "No codes suggested"

    # Contexts: The clinical note (split into chunks if long)
    contexts = _chunk_text(test_case.clinical_note, max_length=500)

    # Ground truth: Expected codes
    ground_truth = ", ".join(test_case.expected_icd_codes)

    return {
        "test_case_id": test_case.id,
        "question": question,
        "answer": answer,
        "contexts": contexts,
        "ground_truth": ground_truth,
    }


def _convert_gap_to_ragas(
    test_case: GapTestCase,
    response: Any,
) -> dict[str, Any]:
    """Convert gap detection test case to RAGAS format."""
    # Question: What documentation gaps exist?
    question = "What documentation gaps should be identified in this clinical note?"

    # Answer: Detected gaps
    if hasattr(response, "gaps") and response.gaps:
        answer_parts = []
        for gap in response.gaps:
            answer_parts.append(f"Gap: {gap.condition} ({gap.gap_type})")
        answer = "\n".join(answer_parts)
    else:
        answer = "No documentation gaps identified"

    # Contexts: Clinical note chunks
    contexts = _chunk_text(test_case.clinical_note, max_length=500)

    # Ground truth: Expected gaps
    gt_parts = [f"{g.condition} ({g.gap_type})" for g in test_case.expected_gaps]
    ground_truth = "\n".join(gt_parts)

    return {
        "test_case_id": test_case.id,
        "question": question,
        "answer": answer,
        "contexts": contexts,
        "ground_truth": ground_truth,
    }


def _convert_query_to_ragas(
    test_case: QueryTestCase,
    response: Any,
) -> dict[str, Any]:
    """Convert query test case to RAGAS format."""
    # Question: What query should be sent for this gap?
    question = f"Generate a CDI query for the following documentation gap: {test_case.gap.condition} ({test_case.gap.gap_type})"

    # Answer: The generated query
    answer = getattr(response, "query_text", str(response))

    # Contexts: Clinical note chunks
    contexts = _chunk_text(test_case.clinical_note, max_length=500)

    # Ground truth: Reference query if available
    ground_truth = test_case.reference_query or f"Query for {test_case.gap.condition}"

    return {
        "test_case_id": test_case.id,
        "question": question,
        "answer": answer,
        "contexts": contexts,
        "ground_truth": ground_truth,
    }


def _default_conversion(
    test_case: BaseTestCase,
    response: Any,
) -> dict[str, Any]:
    """Default conversion for unknown task types."""
    return {
        "test_case_id": test_case.id,
        "question": "Analyze this clinical documentation",
        "answer": str(response),
        "contexts": _chunk_text(test_case.clinical_note, max_length=500),
        "ground_truth": "",
    }


def _chunk_text(text: str, max_length: int = 500) -> list[str]:
    """
    Split text into chunks for context passages.

    Args:
        text: Text to chunk
        max_length: Maximum chunk length

    Returns:
        List of text chunks
    """
    if len(text) <= max_length:
        return [text]

    chunks = []
    words = text.split()
    current_chunk = []
    current_length = 0

    for word in words:
        if current_length + len(word) + 1 > max_length:
            if current_chunk:
                chunks.append(" ".join(current_chunk))
            current_chunk = [word]
            current_length = len(word)
        else:
            current_chunk.append(word)
            current_length += len(word) + 1

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks


def create_ragas_dataset(
    test_cases: list[BaseTestCase],
    responses: list[Any],
    task_type: str,
) -> list[dict[str, Any]]:
    """
    Create a RAGAS-compatible dataset from CDI test cases.

    Args:
        test_cases: List of CDI test cases
        responses: Corresponding API responses
        task_type: Type of CDI task

    Returns:
        List of RAGAS-formatted samples
    """
    if len(test_cases) != len(responses):
        raise ValueError("test_cases and responses must have same length")

    samples = []
    for tc, resp in zip(test_cases, responses):
        sample = convert_to_ragas_format(tc, resp, task_type)
        samples.append(sample)

    return samples


# =============================================================================
# CDI-RAGAS Adapter
# =============================================================================


@dataclass
class CDIRAGASResult:
    """
    Combined result from CDI and RAGAS evaluation.

    Attributes:
        test_case_id: Test case identifier
        task_type: Type of CDI task
        ragas_result: RAGAS evaluation result
        cdi_scores: CDI-specific scores (from domain evaluator)
        combined_score: Weighted combination of RAGAS and CDI scores
    """

    test_case_id: str
    task_type: str
    ragas_result: RAGASEvaluationResult
    cdi_scores: dict[str, float] = field(default_factory=dict)
    ragas_weight: float = 0.3  # Weight for RAGAS in combined score

    @property
    def combined_score(self) -> float:
        """Calculate combined CDI + RAGAS score."""
        ragas_score = self.ragas_result.overall_score

        if not self.cdi_scores:
            return ragas_score

        cdi_avg = sum(self.cdi_scores.values()) / len(self.cdi_scores)
        return (cdi_avg * (1 - self.ragas_weight)) + (ragas_score * self.ragas_weight)

    @property
    def passed(self) -> bool:
        """Check if evaluation passed."""
        return self.combined_score >= 0.7

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "test_case_id": self.test_case_id,
            "task_type": self.task_type,
            "ragas_scores": {
                m.metric_name: m.score for m in self.ragas_result.metrics
            },
            "cdi_scores": self.cdi_scores,
            "combined_score": self.combined_score,
            "passed": self.passed,
        }


class CDIRAGASAdapter:
    """
    Adapter that combines CDI evaluators with RAGAS metrics.

    Provides a unified interface for running both CDI-specific
    evaluations and RAGAS RAG quality metrics.
    """

    def __init__(
        self,
        ragas_config: RAGASConfig | None = None,
        ragas_metrics: list[str] | None = None,
        ragas_weight: float = 0.3,
    ):
        """
        Initialize the adapter.

        Args:
            ragas_config: Configuration for RAGAS evaluators
            ragas_metrics: Which RAGAS metrics to use
            ragas_weight: Weight for RAGAS scores in combined result
        """
        self.ragas_config = ragas_config or RAGASConfig()
        self.ragas_evaluator = CombinedRAGASEvaluator(
            config=self.ragas_config,
            metrics=ragas_metrics,
        )
        self.ragas_weight = ragas_weight

    def evaluate_with_ragas(
        self,
        test_case: BaseTestCase,
        response: Any,
        task_type: str,
        cdi_scores: dict[str, float] | None = None,
    ) -> CDIRAGASResult:
        """
        Evaluate a single test case with RAGAS metrics.

        Args:
            test_case: The CDI test case
            response: API response
            task_type: Type of CDI task
            cdi_scores: Pre-computed CDI evaluation scores

        Returns:
            CDIRAGASResult with combined scores
        """
        # Convert to RAGAS format
        ragas_data = convert_to_ragas_format(test_case, response, task_type)

        # Run RAGAS evaluation
        ragas_result = self.ragas_evaluator.evaluate(
            test_case_id=ragas_data["test_case_id"],
            question=ragas_data["question"],
            answer=ragas_data["answer"],
            contexts=ragas_data["contexts"],
            ground_truth=ragas_data.get("ground_truth"),
        )

        return CDIRAGASResult(
            test_case_id=test_case.id,
            task_type=task_type,
            ragas_result=ragas_result,
            cdi_scores=cdi_scores or {},
            ragas_weight=self.ragas_weight,
        )

    def evaluate_batch_with_ragas(
        self,
        test_cases: list[BaseTestCase],
        responses: list[Any],
        task_type: str,
        cdi_scores_list: list[dict[str, float]] | None = None,
    ) -> list[CDIRAGASResult]:
        """
        Evaluate a batch with RAGAS metrics.

        Args:
            test_cases: List of CDI test cases
            responses: Corresponding API responses
            task_type: Type of CDI task
            cdi_scores_list: Pre-computed CDI scores for each case

        Returns:
            List of CDIRAGASResults
        """
        if cdi_scores_list is None:
            cdi_scores_list = [{}] * len(test_cases)

        results = []
        for tc, resp, scores in zip(test_cases, responses, cdi_scores_list):
            result = self.evaluate_with_ragas(tc, resp, task_type, scores)
            results.append(result)

        return results


# =============================================================================
# CDI-Specific RAGAS Metrics
# =============================================================================


class CDIFaithfulnessEvaluator:
    """
    CDI-specific faithfulness evaluation.

    Extends standard faithfulness with CDI-specific checks:
    - Code suggestions must be supported by clinical findings
    - Gap detections must cite actual documented evidence
    - Query recommendations must match documented symptoms
    """

    def __init__(self, config: RAGASConfig | None = None):
        """Initialize with optional config."""
        self.config = config or RAGASConfig()
        from nuvii_eval.ragas.metrics import FaithfulnessEvaluator
        self.base_evaluator = FaithfulnessEvaluator(config)

    def evaluate_icd_faithfulness(
        self,
        clinical_note: str,
        suggested_codes: list[Any],
    ) -> dict[str, Any]:
        """
        Evaluate faithfulness of ICD code suggestions.

        Checks that each suggested code has supporting evidence
        in the clinical note.

        Args:
            clinical_note: The source clinical documentation
            suggested_codes: List of ICD suggestions with evidence

        Returns:
            Dict with overall score and per-code details
        """
        if not suggested_codes:
            return {"overall_score": 0.0, "codes": [], "reason": "no_codes"}

        note_lower = clinical_note.lower()
        code_results = []

        for code in suggested_codes:
            code_str = getattr(code, "icd10_code", str(code))
            evidence = getattr(code, "evidence_spans", [])

            # Check if evidence spans exist in note
            evidence_found = 0
            for ev in evidence:
                if ev.lower() in note_lower:
                    evidence_found += 1

            if evidence:
                code_score = evidence_found / len(evidence)
            else:
                # No evidence cited - check if code description terms appear
                desc = getattr(code, "description", "").lower()
                desc_terms = set(desc.split()) - {"the", "a", "an", "of", "with", "and", "or"}
                terms_found = sum(1 for t in desc_terms if t in note_lower)
                code_score = min(1.0, terms_found / max(len(desc_terms), 1))

            code_results.append({
                "code": code_str,
                "score": code_score,
                "evidence_count": len(evidence),
                "evidence_found": evidence_found,
            })

        overall_score = sum(c["score"] for c in code_results) / len(code_results)

        return {
            "overall_score": overall_score,
            "codes": code_results,
        }

    def evaluate_gap_faithfulness(
        self,
        clinical_note: str,
        detected_gaps: list[Any],
    ) -> dict[str, Any]:
        """
        Evaluate faithfulness of gap detections.

        Checks that each detected gap cites actual evidence from the note.

        Args:
            clinical_note: The source clinical documentation
            detected_gaps: List of detected gaps with evidence

        Returns:
            Dict with overall score and per-gap details
        """
        if not detected_gaps:
            return {"overall_score": 1.0, "gaps": [], "reason": "no_gaps"}

        note_lower = clinical_note.lower()
        gap_results = []

        for gap in detected_gaps:
            condition = getattr(gap, "condition", str(gap))
            evidence = getattr(gap, "current_evidence", [])

            # Check evidence presence
            evidence_found = 0
            for ev in evidence:
                if ev.lower() in note_lower:
                    evidence_found += 1

            if evidence:
                gap_score = evidence_found / len(evidence)
            else:
                # Check if condition terms appear in note
                condition_terms = set(condition.lower().split())
                terms_found = sum(1 for t in condition_terms if t in note_lower)
                gap_score = min(1.0, terms_found / max(len(condition_terms), 1))

            gap_results.append({
                "condition": condition,
                "score": gap_score,
                "evidence_count": len(evidence),
                "evidence_found": evidence_found,
            })

        overall_score = sum(g["score"] for g in gap_results) / len(gap_results)

        return {
            "overall_score": overall_score,
            "gaps": gap_results,
        }
