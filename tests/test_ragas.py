"""
Unit tests for RAGAS integration.

Tests cover:
- RAGAS metric evaluators (with fallback mode)
- CDI-RAGAS adapters
- Data conversion utilities
"""

import pytest

from nuvii_eval.datasets.schemas import (
    ExpectedGap,
    GapTestCase,
    ICDTestCase,
    QueryQualityCriteria,
    QueryTestCase,
)
from nuvii_eval.ragas.adapters import (
    CDIFaithfulnessEvaluator,
    CDIRAGASAdapter,
    CDIRAGASResult,
    convert_to_ragas_format,
    create_ragas_dataset,
    _chunk_text,
)
from nuvii_eval.ragas.metrics import (
    AnswerRelevancyEvaluator,
    CombinedRAGASEvaluator,
    ContextPrecisionEvaluator,
    ContextRecallEvaluator,
    FaithfulnessEvaluator,
    RAGASConfig,
    RAGASEvaluationResult,
    RAGASMetricResult,
)


# =============================================================================
# RAGASConfig Tests
# =============================================================================


class TestRAGASConfig:
    """Tests for RAGAS configuration."""

    def test_default_config(self):
        """Test default configuration values."""
        config = RAGASConfig()
        assert config.llm_model == "gpt-4o-mini"
        assert config.batch_size == 10
        assert config.timeout_seconds == 120
        assert config.phi_safe_mode is True

    def test_custom_config(self):
        """Test custom configuration."""
        config = RAGASConfig(
            llm_model="gpt-4",
            batch_size=20,
            phi_safe_mode=False,
        )
        assert config.llm_model == "gpt-4"
        assert config.batch_size == 20
        assert config.phi_safe_mode is False


# =============================================================================
# RAGASMetricResult Tests
# =============================================================================


class TestRAGASMetricResult:
    """Tests for RAGAS metric results."""

    def test_metric_result_creation(self):
        """Test creating a metric result."""
        result = RAGASMetricResult(
            metric_name="faithfulness",
            score=0.85,
            details={"token_overlap": 42},
        )
        assert result.metric_name == "faithfulness"
        assert result.score == 0.85
        assert result.passed is True
        assert result.error is None

    def test_metric_result_failed(self):
        """Test metric result with error."""
        result = RAGASMetricResult(
            metric_name="faithfulness",
            score=0.0,
            error="API timeout",
        )
        assert result.passed is False
        assert result.error == "API timeout"

    def test_metric_result_below_threshold(self):
        """Test metric result below pass threshold."""
        result = RAGASMetricResult(
            metric_name="faithfulness",
            score=0.3,
        )
        assert result.passed is False

    def test_to_dict(self):
        """Test converting result to dictionary."""
        result = RAGASMetricResult(
            metric_name="test",
            score=0.75,
        )
        d = result.to_dict()
        assert d["metric_name"] == "test"
        assert d["score"] == 0.75
        assert d["passed"] is True
        assert "timestamp" in d


# =============================================================================
# RAGASEvaluationResult Tests
# =============================================================================


class TestRAGASEvaluationResult:
    """Tests for combined RAGAS evaluation results."""

    def test_overall_score_calculation(self):
        """Test overall score as average of metrics."""
        metrics = [
            RAGASMetricResult(metric_name="faithfulness", score=0.8),
            RAGASMetricResult(metric_name="relevancy", score=0.6),
            RAGASMetricResult(metric_name="precision", score=1.0),
        ]
        result = RAGASEvaluationResult(
            test_case_id="test_001",
            metrics=metrics,
        )
        assert result.overall_score == pytest.approx(0.8)

    def test_overall_score_with_errors(self):
        """Test overall score excludes errored metrics."""
        metrics = [
            RAGASMetricResult(metric_name="faithfulness", score=0.8),
            RAGASMetricResult(metric_name="relevancy", score=0.0, error="Failed"),
        ]
        result = RAGASEvaluationResult(
            test_case_id="test_001",
            metrics=metrics,
        )
        # Only faithfulness counts
        assert result.overall_score == 0.8

    def test_passed_all_metrics(self):
        """Test passed when all metrics pass."""
        metrics = [
            RAGASMetricResult(metric_name="faithfulness", score=0.8),
            RAGASMetricResult(metric_name="relevancy", score=0.7),
        ]
        result = RAGASEvaluationResult(
            test_case_id="test_001",
            metrics=metrics,
        )
        assert result.passed is True

    def test_passed_some_fail(self):
        """Test not passed when some metrics fail."""
        metrics = [
            RAGASMetricResult(metric_name="faithfulness", score=0.8),
            RAGASMetricResult(metric_name="relevancy", score=0.3),  # Below threshold
        ]
        result = RAGASEvaluationResult(
            test_case_id="test_001",
            metrics=metrics,
        )
        assert result.passed is False

    def test_get_metric(self):
        """Test retrieving specific metric."""
        metrics = [
            RAGASMetricResult(metric_name="faithfulness", score=0.8),
            RAGASMetricResult(metric_name="relevancy", score=0.6),
        ]
        result = RAGASEvaluationResult(
            test_case_id="test_001",
            metrics=metrics,
        )
        faith = result.get_metric("faithfulness")
        assert faith is not None
        assert faith.score == 0.8

        missing = result.get_metric("nonexistent")
        assert missing is None


# =============================================================================
# Faithfulness Evaluator Tests (Fallback Mode)
# =============================================================================


class TestFaithfulnessEvaluator:
    """Tests for faithfulness evaluator (fallback mode)."""

    def test_fallback_high_overlap(self):
        """Test high faithfulness with good token overlap."""
        evaluator = FaithfulnessEvaluator()

        result = evaluator.evaluate(
            question="What diagnosis codes apply?",
            answer="The patient has diabetes mellitus with hyperglycemia",
            contexts=["Patient presents with diabetes mellitus and elevated blood glucose indicating hyperglycemia"],
        )

        # Should have high score due to overlap
        assert result.score > 0.5
        assert result.details.get("fallback") is True

    def test_fallback_no_overlap(self):
        """Test low faithfulness with no token overlap."""
        evaluator = FaithfulnessEvaluator()

        result = evaluator.evaluate(
            question="What diagnosis codes apply?",
            answer="The patient has cancer",
            contexts=["Patient presents with diabetes mellitus"],
        )

        # Should have lower score due to poor overlap
        assert result.score < 0.5

    def test_empty_contexts(self):
        """Test handling of empty contexts."""
        evaluator = FaithfulnessEvaluator()

        result = evaluator.evaluate(
            question="What codes?",
            answer="Some answer",
            contexts=[],
        )

        assert result.score == 0.0
        assert result.details.get("reason") == "no_contexts"


# =============================================================================
# Answer Relevancy Evaluator Tests (Fallback Mode)
# =============================================================================


class TestAnswerRelevancyEvaluator:
    """Tests for answer relevancy evaluator (fallback mode)."""

    def test_fallback_relevant_answer(self):
        """Test relevancy with matching terms."""
        evaluator = AnswerRelevancyEvaluator()

        result = evaluator.evaluate(
            question="What ICD-10 codes should be assigned for diabetes?",
            answer="The patient should be assigned code E11.65 for type 2 diabetes with hyperglycemia",
            contexts=["Clinical note"],
        )

        # Should have good score - "diabetes" and "codes" appear in answer
        assert result.score > 0.3
        assert result.details.get("fallback") is True

    def test_fallback_irrelevant_answer(self):
        """Test low relevancy with unrelated answer."""
        evaluator = AnswerRelevancyEvaluator()

        result = evaluator.evaluate(
            question="What ICD-10 codes should be assigned for diabetes?",
            answer="The weather today is sunny",
            contexts=["Clinical note"],
        )

        # Should have lower score - no question terms in answer
        assert result.score <= 0.5


# =============================================================================
# Context Precision Evaluator Tests (Fallback Mode)
# =============================================================================


class TestContextPrecisionEvaluator:
    """Tests for context precision evaluator (fallback mode)."""

    def test_fallback_precise_context(self):
        """Test high precision when context is relevant."""
        evaluator = ContextPrecisionEvaluator()

        result = evaluator.evaluate(
            question="Diagnose the patient",
            answer="Patient has diabetes mellitus type 2",
            contexts=["Patient presents with diabetes mellitus type 2 with A1c of 9.0%"],
            ground_truth="diabetes mellitus type 2",
        )

        assert result.score > 0.3
        assert result.details.get("fallback") is True

    def test_fallback_imprecise_context(self):
        """Test lower precision with noisy context."""
        evaluator = ContextPrecisionEvaluator()

        result = evaluator.evaluate(
            question="Diagnose the patient",
            answer="diabetes",
            contexts=[
                "Lorem ipsum dolor sit amet consectetur adipiscing elit sed do eiusmod tempor incididunt ut labore et dolore magna aliqua"
            ],
            ground_truth="diabetes",
        )

        # Low overlap between context and answer
        assert result.score < 0.5


# =============================================================================
# Context Recall Evaluator Tests (Fallback Mode)
# =============================================================================


class TestContextRecallEvaluator:
    """Tests for context recall evaluator (fallback mode)."""

    def test_no_ground_truth(self):
        """Test handling when no ground truth provided."""
        evaluator = ContextRecallEvaluator()

        result = evaluator.evaluate(
            question="What codes?",
            answer="Some answer",
            contexts=["Some context"],
            ground_truth=None,
        )

        # Should return neutral score
        assert result.score == 0.5
        assert result.details.get("reason") == "no_ground_truth"

    def test_fallback_high_recall(self):
        """Test high recall when context covers ground truth."""
        evaluator = ContextRecallEvaluator()

        result = evaluator.evaluate(
            question="Diagnose patient",
            answer="Diabetes",
            contexts=["Patient has diabetes mellitus with complications"],
            ground_truth="diabetes mellitus complications",
        )

        # Context should cover most of ground truth
        assert result.score > 0.5
        assert result.details.get("fallback") is True

    def test_fallback_low_recall(self):
        """Test low recall when context missing information."""
        evaluator = ContextRecallEvaluator()

        result = evaluator.evaluate(
            question="Diagnose patient",
            answer="Diabetes",
            contexts=["Patient feels fine"],
            ground_truth="diabetes mellitus hyperglycemia nephropathy",
        )

        # Context doesn't cover ground truth terms
        assert result.score < 0.5


# =============================================================================
# Combined RAGAS Evaluator Tests
# =============================================================================


class TestCombinedRAGASEvaluator:
    """Tests for combined RAGAS evaluator."""

    def test_evaluate_all_metrics(self):
        """Test running all metrics on a sample."""
        evaluator = CombinedRAGASEvaluator()

        result = evaluator.evaluate(
            test_case_id="test_001",
            question="What ICD-10 codes apply to this diabetic patient?",
            answer="E11.65 - Type 2 diabetes mellitus with hyperglycemia",
            contexts=["Patient presents with type 2 diabetes and elevated blood glucose"],
            ground_truth="E11.65",
        )

        assert result.test_case_id == "test_001"
        assert len(result.metrics) == 4  # All 4 metrics

        # Check each metric is present
        metric_names = {m.metric_name for m in result.metrics}
        assert "faithfulness" in metric_names
        assert "answer_relevancy" in metric_names
        assert "context_precision" in metric_names
        assert "context_recall" in metric_names

    def test_evaluate_specific_metrics(self):
        """Test running only specific metrics."""
        evaluator = CombinedRAGASEvaluator(
            metrics=["faithfulness", "answer_relevancy"]
        )

        result = evaluator.evaluate(
            test_case_id="test_002",
            question="Question",
            answer="Answer",
            contexts=["Context"],
        )

        assert len(result.metrics) == 2
        metric_names = {m.metric_name for m in result.metrics}
        assert "faithfulness" in metric_names
        assert "answer_relevancy" in metric_names

    def test_evaluate_batch(self):
        """Test batch evaluation."""
        evaluator = CombinedRAGASEvaluator(metrics=["faithfulness"])

        samples = [
            {
                "test_case_id": "test_001",
                "question": "Q1",
                "answer": "A1 has diabetes",
                "contexts": ["diabetes context"],
            },
            {
                "test_case_id": "test_002",
                "question": "Q2",
                "answer": "A2",
                "contexts": ["context"],
            },
        ]

        results = evaluator.evaluate_batch(samples)

        assert len(results) == 2
        assert results[0].test_case_id == "test_001"
        assert results[1].test_case_id == "test_002"


# =============================================================================
# Data Conversion Tests
# =============================================================================


class TestDataConversion:
    """Tests for CDI to RAGAS data conversion."""

    def test_chunk_text_short(self):
        """Test chunking short text."""
        text = "Short text"
        chunks = _chunk_text(text, max_length=500)
        assert len(chunks) == 1
        assert chunks[0] == text

    def test_chunk_text_long(self):
        """Test chunking long text."""
        text = "word " * 200  # 1000 chars
        chunks = _chunk_text(text, max_length=100)
        assert len(chunks) > 1
        # All chunks should be under limit
        for chunk in chunks:
            assert len(chunk) <= 100 + 10  # Some tolerance

    def test_convert_icd_to_ragas(self):
        """Test converting ICD test case to RAGAS format."""
        test_case = ICDTestCase(
            id="icd_001",
            clinical_note="Patient presents with type 2 diabetes mellitus with hyperglycemia. A1c is 9.0%.",
            expected_icd_codes=["E11.65"],
        )

        # Mock response
        class MockICDSuggestion:
            icd10_code = "E11.65"
            description = "Type 2 diabetes with hyperglycemia"

        class MockResponse:
            suggested_codes = [MockICDSuggestion()]

        result = convert_to_ragas_format(test_case, MockResponse(), "icd")

        assert result["test_case_id"] == "icd_001"
        assert "ICD-10" in result["question"]
        assert "E11.65" in result["answer"]
        assert len(result["contexts"]) >= 1
        assert "E11.65" in result["ground_truth"]

    def test_convert_gap_to_ragas(self):
        """Test converting gap test case to RAGAS format."""
        test_case = GapTestCase(
            id="gap_001",
            clinical_note="Patient with elevated creatinine and reduced GFR. History of diabetes documented.",
            expected_gaps=[
                ExpectedGap(
                    condition="chronic kidney disease",
                    gap_type="specificity",
                    min_priority=1,
                ),
            ],
        )

        class MockGap:
            condition = "CKD"
            gap_type = "specificity"

        class MockResponse:
            gaps = [MockGap()]

        result = convert_to_ragas_format(test_case, MockResponse(), "gap")

        assert result["test_case_id"] == "gap_001"
        assert "gap" in result["question"].lower()
        assert "CKD" in result["answer"]

    def test_convert_query_to_ragas(self):
        """Test converting query test case to RAGAS format."""
        test_case = QueryTestCase(
            id="query_001",
            clinical_note="Patient with fever, tachycardia, elevated WBC. Possible sepsis based on clinical presentation.",
            gap=ExpectedGap(
                condition="sepsis",
                gap_type="clarification",
                min_priority=1,
            ),
            quality_criteria=QueryQualityCriteria(
                must_mention=["sepsis"],
                min_evidence_citations=1,
            ),
            reference_query="Please clarify if sepsis is present",
        )

        class MockResponse:
            query_text = "Based on clinical indicators, please clarify sepsis status"

        result = convert_to_ragas_format(test_case, MockResponse(), "query")

        assert result["test_case_id"] == "query_001"
        assert "sepsis" in result["question"].lower()
        assert "clarify sepsis" in result["answer"].lower()
        assert result["ground_truth"] == "Please clarify if sepsis is present"

    def test_create_ragas_dataset(self):
        """Test creating dataset from multiple test cases."""
        test_cases = [
            ICDTestCase(
                id="icd_001",
                clinical_note="Patient with diabetes mellitus type 2 presenting for routine follow-up and management.",
                expected_icd_codes=["E11.9"],
            ),
            ICDTestCase(
                id="icd_002",
                clinical_note="Patient with essential hypertension, well controlled on current medications and treatment plan.",
                expected_icd_codes=["I10"],
            ),
        ]

        class MockResponse:
            suggested_codes = []

        responses = [MockResponse(), MockResponse()]

        dataset = create_ragas_dataset(test_cases, responses, "icd")

        assert len(dataset) == 2
        assert dataset[0]["test_case_id"] == "icd_001"
        assert dataset[1]["test_case_id"] == "icd_002"


# =============================================================================
# CDI-RAGAS Adapter Tests
# =============================================================================


class TestCDIRAGASAdapter:
    """Tests for CDI-RAGAS adapter."""

    def test_evaluate_with_ragas(self):
        """Test evaluating single test case with RAGAS."""
        adapter = CDIRAGASAdapter(ragas_metrics=["faithfulness"])

        test_case = ICDTestCase(
            id="icd_001",
            clinical_note="Patient presents with type 2 diabetes mellitus with hyperglycemia and elevated blood glucose.",
            expected_icd_codes=["E11.65"],
        )

        class MockResponse:
            suggested_codes = []

        result = adapter.evaluate_with_ragas(
            test_case=test_case,
            response=MockResponse(),
            task_type="icd",
            cdi_scores={"top_1_accuracy": 0.9, "precision": 0.8},
        )

        assert isinstance(result, CDIRAGASResult)
        assert result.test_case_id == "icd_001"
        assert result.task_type == "icd"
        assert "top_1_accuracy" in result.cdi_scores

    def test_combined_score_calculation(self):
        """Test combined CDI + RAGAS score."""
        metrics = [
            RAGASMetricResult(metric_name="faithfulness", score=0.8),
        ]
        ragas_result = RAGASEvaluationResult(
            test_case_id="test_001",
            metrics=metrics,
        )

        result = CDIRAGASResult(
            test_case_id="test_001",
            task_type="icd",
            ragas_result=ragas_result,
            cdi_scores={"accuracy": 0.9},
            ragas_weight=0.3,
        )

        # Combined = (0.9 * 0.7) + (0.8 * 0.3) = 0.63 + 0.24 = 0.87
        assert result.combined_score == pytest.approx(0.87)

    def test_result_to_dict(self):
        """Test converting result to dictionary."""
        metrics = [
            RAGASMetricResult(metric_name="faithfulness", score=0.8),
        ]
        ragas_result = RAGASEvaluationResult(
            test_case_id="test_001",
            metrics=metrics,
        )

        result = CDIRAGASResult(
            test_case_id="test_001",
            task_type="icd",
            ragas_result=ragas_result,
            cdi_scores={"accuracy": 0.9},
        )

        d = result.to_dict()

        assert d["test_case_id"] == "test_001"
        assert d["task_type"] == "icd"
        assert "ragas_scores" in d
        assert "cdi_scores" in d
        assert "combined_score" in d


# =============================================================================
# CDI Faithfulness Evaluator Tests
# =============================================================================


class TestCDIFaithfulnessEvaluator:
    """Tests for CDI-specific faithfulness evaluator."""

    def test_icd_faithfulness_with_evidence(self):
        """Test ICD faithfulness when evidence is present."""
        evaluator = CDIFaithfulnessEvaluator()

        class MockCode:
            icd10_code = "E11.65"
            description = "Type 2 diabetes with hyperglycemia"
            evidence_spans = ["diabetes mellitus", "hyperglycemia"]

        result = evaluator.evaluate_icd_faithfulness(
            clinical_note="Patient has diabetes mellitus with hyperglycemia and elevated A1c",
            suggested_codes=[MockCode()],
        )

        assert result["overall_score"] == 1.0  # Both evidence spans found
        assert len(result["codes"]) == 1
        assert result["codes"][0]["evidence_found"] == 2

    def test_icd_faithfulness_partial_evidence(self):
        """Test ICD faithfulness with partial evidence."""
        evaluator = CDIFaithfulnessEvaluator()

        class MockCode:
            icd10_code = "E11.65"
            description = "Type 2 diabetes with hyperglycemia"
            evidence_spans = ["diabetes", "cancer"]  # cancer not in note

        result = evaluator.evaluate_icd_faithfulness(
            clinical_note="Patient has diabetes mellitus",
            suggested_codes=[MockCode()],
        )

        assert result["overall_score"] == 0.5  # 1 of 2 evidence found

    def test_gap_faithfulness(self):
        """Test gap detection faithfulness."""
        evaluator = CDIFaithfulnessEvaluator()

        class MockGap:
            condition = "sepsis"
            gap_type = "clarification"
            current_evidence = ["fever", "elevated WBC"]

        result = evaluator.evaluate_gap_faithfulness(
            clinical_note="Patient with fever 101.5F, elevated WBC 18000, tachycardia",
            detected_gaps=[MockGap()],
        )

        assert result["overall_score"] == 1.0  # Both evidence found
        assert result["gaps"][0]["evidence_found"] == 2

    def test_empty_codes(self):
        """Test handling of empty codes list."""
        evaluator = CDIFaithfulnessEvaluator()

        result = evaluator.evaluate_icd_faithfulness(
            clinical_note="Some note",
            suggested_codes=[],
        )

        assert result["overall_score"] == 0.0
        assert result["reason"] == "no_codes"
