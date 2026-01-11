"""
Unit tests for CDI evaluators.

Tests cover:
- Base evaluator utilities
- ICD-10 code evaluation
- HCC detection and RAF scoring
- Gap detection evaluation
- Query quality evaluation
- E/M level evaluation
"""

import pytest

from nuvii_eval.datasets.schemas import (
    EMTestCase,
    ExpectedGap,
    ExpectedMDM,
    GapTestCase,
    HCCTestCase,
    ICDTestCase,
    QueryQualityCriteria,
    QueryTestCase,
)
from nuvii_eval.evaluators import (
    # Base utilities
    EvalResult,
    EvalScore,
    f1_score,
    get_evaluator,
    jaccard_similarity,
    normalize_code,
    precision,
    recall,
    top_n_hit,
    # ICD
    ICDEvaluator,
    ICDEvaluatorLenient,
    ICDEvaluatorStrict,
    # HCC
    HCCEvaluator,
    HCC_GROUPS,
    HCC_SUPERSESSIONS,
    get_hcc_group,
    get_superseded_hccs,
    # Gap
    GapEvaluator,
    GapEvaluatorStrict,
    normalize_condition,
    tokenize_condition,
    # Query
    QueryEvaluator,
    QueryEvaluatorStrict,
    # E/M
    EMEvaluator,
    EMEvaluatorStrict,
    EM_CODE_LEVELS,
    get_code_level,
    get_code_family,
    codes_in_same_family,
)
from nuvii_eval.schemas.api_responses import (
    ConfidenceLevel,
    CodingSuggestResponse,
    EMAnalysisResult,
    GapCandidate,
    GapDetectionResponse,
    HCCOpportunity,
    ICDSuggestion,
    MDMComponent,
    ProviderQuery,
    RiskAnalysisResult,
)


# =============================================================================
# Base Evaluator Utility Tests
# =============================================================================


class TestBaseUtilities:
    """Tests for base evaluator utility functions."""

    def test_normalize_code(self):
        """Test code normalization."""
        assert normalize_code("e11.65") == "E11.65"
        assert normalize_code("  E11.65  ") == "E11.65"
        assert normalize_code("e11") == "E11"

    def test_precision(self):
        """Test precision calculation."""
        predicted = {"A", "B", "C"}
        expected = {"A", "B", "D"}
        assert precision(predicted, expected) == pytest.approx(2 / 3)

        # Edge cases
        assert precision(set(), {"A"}) == 0.0
        assert precision({"A"}, set()) == 0.0

    def test_recall(self):
        """Test recall calculation."""
        predicted = {"A", "B", "C"}
        expected = {"A", "B", "D"}
        assert recall(predicted, expected) == pytest.approx(2 / 3)

        # Edge cases
        assert recall(set(), {"A"}) == 0.0
        assert recall({"A"}, set()) == 1.0  # All expected found (none)

    def test_f1_score(self):
        """Test F1 score calculation."""
        assert f1_score(1.0, 1.0) == 1.0
        assert f1_score(0.0, 0.0) == 0.0
        assert f1_score(0.5, 0.5) == 0.5
        assert f1_score(1.0, 0.5) == pytest.approx(2 / 3)

    def test_top_n_hit(self):
        """Test top-N hit detection."""
        predicted = ["A", "B", "C", "D", "E"]
        expected = {"C", "F"}

        assert top_n_hit(predicted, expected, 1) is False
        assert top_n_hit(predicted, expected, 2) is False
        assert top_n_hit(predicted, expected, 3) is True
        assert top_n_hit(predicted, expected, 5) is True

    def test_jaccard_similarity(self):
        """Test Jaccard similarity calculation."""
        set1 = {"A", "B", "C"}
        set2 = {"B", "C", "D"}
        # Intersection: {B, C} = 2, Union: {A, B, C, D} = 4
        assert jaccard_similarity(set1, set2) == pytest.approx(0.5)

        # Perfect overlap
        assert jaccard_similarity({"A", "B"}, {"A", "B"}) == 1.0

        # No overlap
        assert jaccard_similarity({"A"}, {"B"}) == 0.0


class TestEvalScore:
    """Tests for EvalScore dataclass."""

    def test_eval_score_creation(self):
        """Test EvalScore creation."""
        score = EvalScore(
            name="test_metric",
            value=0.85,
            weight=1.5,
            details={"key": "value"},
        )
        assert score.name == "test_metric"
        assert score.value == 0.85
        assert score.weight == 1.5
        assert score.details == {"key": "value"}

    def test_eval_score_defaults(self):
        """Test EvalScore default values."""
        score = EvalScore(name="test", value=0.5)
        assert score.weight == 1.0
        assert score.details == {}


class TestEvaluatorRegistry:
    """Tests for evaluator registry and factory."""

    def test_get_evaluator_valid(self):
        """Test getting valid evaluator types."""
        icd_eval = get_evaluator("icd")
        assert isinstance(icd_eval, ICDEvaluator)

        hcc_eval = get_evaluator("hcc")
        assert isinstance(hcc_eval, HCCEvaluator)

    def test_get_evaluator_with_config(self):
        """Test getting evaluator with configuration."""
        eval_instance = get_evaluator("icd", hierarchy_credit=0.3)
        assert eval_instance.hierarchy_credit == 0.3

    def test_get_evaluator_invalid(self):
        """Test error on invalid evaluator type."""
        with pytest.raises(ValueError) as exc_info:
            get_evaluator("invalid_type")
        assert "Unknown evaluator type" in str(exc_info.value)


# =============================================================================
# ICD Evaluator Tests
# =============================================================================


class TestICDEvaluator:
    """Tests for ICD-10 code evaluation."""

    @pytest.fixture
    def icd_test_case(self) -> ICDTestCase:
        """Create a sample ICD test case."""
        return ICDTestCase(
            id="icd_test_001",
            clinical_note="Patient presents with type 2 diabetes mellitus with hyperglycemia and chronic kidney disease stage 3. Labs show A1c 9.2% and eGFR 45.",
            expected_icd_codes=["E11.65", "N18.3"],
            acceptable_icd_codes=["E11.9"],
            unacceptable_codes=["E10.65"],  # Type 1 diabetes
            primary_code="E11.65",
            code_sequence_matters=True,
        )

    @pytest.fixture
    def icd_response_perfect(self) -> CodingSuggestResponse:
        """Create a perfect ICD response."""
        return CodingSuggestResponse(
            request_id="req_001",
            suggested_codes=[
                ICDSuggestion(
                    icd10_code="E11.65",
                    description="Type 2 diabetes mellitus with hyperglycemia",
                    confidence=ConfidenceLevel.HIGH,
                    evidence_spans=["Patient presents with type 2 diabetes"],
                ),
                ICDSuggestion(
                    icd10_code="N18.3",
                    description="Chronic kidney disease, stage 3",
                    confidence=ConfidenceLevel.HIGH,
                    evidence_spans=["chronic kidney disease stage 3"],
                ),
            ],
            model_version="1.0.0",
            processing_time_ms=150,
        )

    @pytest.fixture
    def icd_response_partial(self) -> CodingSuggestResponse:
        """Create a partial match ICD response."""
        return CodingSuggestResponse(
            request_id="req_002",
            suggested_codes=[
                ICDSuggestion(
                    icd10_code="E11.9",  # Acceptable but not primary
                    description="Type 2 diabetes mellitus without complications",
                    confidence=ConfidenceLevel.HIGH,
                    evidence_spans=["Patient presents with type 2 diabetes"],
                ),
            ],
            model_version="1.0.0",
            processing_time_ms=120,
        )

    def test_perfect_match(self, icd_test_case, icd_response_perfect):
        """Test evaluation with perfect code match."""
        evaluator = ICDEvaluator()
        result = evaluator.evaluate(icd_test_case, icd_response_perfect)

        assert result.passed is True
        assert result.composite_score >= 0.9

        # Check top-1 accuracy
        top1_score = next(s for s in result.scores if s.name == "top_1_accuracy")
        assert top1_score.value == 1.0

    def test_partial_match(self, icd_test_case, icd_response_partial):
        """Test evaluation with partial match."""
        evaluator = ICDEvaluator()
        result = evaluator.evaluate(icd_test_case, icd_response_partial)

        # Top-1 should be 0 (E11.9 is acceptable but not in expected)
        top1_score = next(s for s in result.scores if s.name == "top_1_accuracy")
        assert top1_score.value == 0.0

        # But recall should have some value (E11.9 is acceptable)
        recall_score = next(s for s in result.scores if s.name == "recall")
        assert recall_score.value > 0

    def test_false_positive_penalty(self, icd_test_case):
        """Test penalty for unacceptable codes."""
        response = CodingSuggestResponse(
            request_id="req_003",
            suggested_codes=[
                ICDSuggestion(
                    icd10_code="E10.65",  # Type 1 - explicitly unacceptable
                    description="Type 1 diabetes",
                    confidence=ConfidenceLevel.HIGH,
                    evidence_spans=[],
                ),
            ],
            model_version="1.0.0",
            processing_time_ms=100,
        )

        evaluator = ICDEvaluator()
        result = evaluator.evaluate(icd_test_case, response)

        fp_score = next(s for s in result.scores if s.name == "false_positive_penalty")
        assert fp_score.value < 1.0

    def test_strict_evaluator(self, icd_test_case, icd_response_perfect):
        """Test strict evaluator has higher threshold."""
        evaluator = ICDEvaluatorStrict()
        assert evaluator.pass_threshold == 0.85
        assert evaluator.hierarchy_credit == 0.0

    def test_lenient_evaluator(self, icd_test_case, icd_response_partial):
        """Test lenient evaluator is more forgiving."""
        evaluator = ICDEvaluatorLenient()
        assert evaluator.pass_threshold == 0.6
        assert evaluator.hierarchy_credit == 0.8


# =============================================================================
# HCC Evaluator Tests
# =============================================================================


class TestHCCEvaluator:
    """Tests for HCC detection and RAF evaluation."""

    @pytest.fixture
    def hcc_test_case(self) -> HCCTestCase:
        """Create a sample HCC test case."""
        return HCCTestCase(
            id="hcc_test_001",
            clinical_note="Patient with type 2 diabetes with chronic complications including nephropathy and retinopathy. History of CHF per cardiology notes.",
            expected_hccs=["HCC18"],
            expected_opportunities=["HCC85"],
            expected_raf_range=(1.2, 1.5),
            patient_age=72,
            patient_gender="M",
        )

    @pytest.fixture
    def hcc_response(self) -> RiskAnalysisResult:
        """Create a sample HCC response."""
        return RiskAnalysisResult(
            request_id="req_hcc_001",
            current_hccs=["HCC18"],
            current_raf=1.35,
            projected_raf=1.85,
            raf_gap=0.50,
            opportunities=[
                HCCOpportunity(
                    hcc_code="HCC85",
                    hcc_description="Heart failure",
                    raf_weight=0.35,
                    confidence=ConfidenceLevel.HIGH,
                    supporting_evidence=["History of CHF"],
                )
            ],
            processing_time_ms=200,
        )

    def test_hcc_utilities(self):
        """Test HCC utility functions."""
        # Test get_hcc_group
        assert get_hcc_group("HCC18") == "diabetes"
        assert get_hcc_group("HCC85") == "heart_failure"
        assert get_hcc_group("HCC999") is None

        # Test get_superseded_hccs
        superseded = get_superseded_hccs("HCC17")
        assert "HCC18" in superseded
        assert "HCC19" in superseded

    def test_perfect_hcc_match(self, hcc_test_case, hcc_response):
        """Test evaluation with perfect HCC match."""
        evaluator = HCCEvaluator()
        result = evaluator.evaluate(hcc_test_case, hcc_response)

        assert result.passed is True

        # Check HCC recall
        recall_score = next(s for s in result.scores if s.name == "hcc_recall")
        assert recall_score.value == 1.0

        # Check RAF in range
        raf_score = next(s for s in result.scores if s.name == "raf_accuracy")
        assert raf_score.value == 1.0

    def test_raf_out_of_range(self, hcc_test_case):
        """Test RAF accuracy when out of range."""
        response = RiskAnalysisResult(
            request_id="req_hcc_002",
            current_hccs=["HCC18"],
            current_raf=2.0,  # Above expected range
            projected_raf=2.5,
            raf_gap=0.5,
            opportunities=[],
            processing_time_ms=100,
        )

        evaluator = HCCEvaluator()
        result = evaluator.evaluate(hcc_test_case, response)

        raf_score = next(s for s in result.scores if s.name == "raf_accuracy")
        assert raf_score.value < 1.0

    def test_supersession_violation(self):
        """Test detection of HCC supersession violations."""
        evaluator = HCCEvaluator()

        # HCC17 supersedes HCC18, both present = violation
        violations_present = evaluator._supersession_accuracy({"HCC17", "HCC18"})
        assert violations_present < 1.0

        # Only superior HCC = no violation
        no_violations = evaluator._supersession_accuracy({"HCC17"})
        assert no_violations == 1.0


# =============================================================================
# Gap Evaluator Tests
# =============================================================================


class TestGapEvaluator:
    """Tests for documentation gap detection evaluation."""

    @pytest.fixture
    def gap_test_case(self) -> GapTestCase:
        """Create a sample gap test case."""
        return GapTestCase(
            id="gap_test_001",
            clinical_note="Patient presents with fever 101.5F, elevated WBC 18,000, tachycardia HR 112, elevated lactate 3.2. Creatinine elevated to 2.1 from baseline 1.0.",
            expected_gaps=[
                ExpectedGap(
                    condition="sepsis",
                    gap_type="clarification",
                    min_priority=1,
                ),
                ExpectedGap(
                    condition="acute kidney injury",
                    gap_type="specificity",
                    min_priority=2,
                ),
            ],
            false_positive_conditions=["pneumonia"],
        )

    @pytest.fixture
    def gap_response(self) -> GapDetectionResponse:
        """Create a sample gap response."""
        return GapDetectionResponse(
            request_id="req_gap_001",
            facts_cache_key="cache_001",
            gaps=[
                GapCandidate(
                    gap_id="gap_001",
                    condition="Sepsis",
                    gap_type="clarification",
                    priority=1,
                    confidence=ConfidenceLevel.HIGH,
                    current_evidence=["fever", "elevated WBC", "tachycardia"],
                ),
                GapCandidate(
                    gap_id="gap_002",
                    condition="Acute Kidney Injury",
                    gap_type="specificity",
                    priority=2,
                    confidence=ConfidenceLevel.HIGH,
                    current_evidence=["elevated creatinine"],
                ),
            ],
            processing_time_ms=180,
        )

    def test_gap_utilities(self):
        """Test gap utility functions."""
        # Test normalize_condition
        assert normalize_condition("  SEPSIS  ") == "sepsis"

        # Test tokenize_condition
        tokens = tokenize_condition("acute kidney injury")
        assert "acute" in tokens
        assert "kidney" in tokens
        assert "injury" in tokens
        assert "the" not in tokens  # Stop word removed

    def test_perfect_gap_detection(self, gap_test_case, gap_response):
        """Test evaluation with perfect gap detection."""
        evaluator = GapEvaluator()
        result = evaluator.evaluate(gap_test_case, gap_response)

        assert result.passed is True

        # Check precision and recall
        precision_score = next(s for s in result.scores if s.name == "precision")
        recall_score = next(s for s in result.scores if s.name == "recall")
        assert precision_score.value == 1.0
        assert recall_score.value == 1.0

    def test_false_positive_detection(self, gap_test_case):
        """Test penalty for explicit false positives."""
        response = GapDetectionResponse(
            request_id="req_gap_002",
            facts_cache_key="cache_002",
            gaps=[
                GapCandidate(
                    gap_id="gap_fp_001",
                    condition="Pneumonia",  # Explicitly a false positive
                    gap_type="clarification",
                    priority=1,
                    confidence=ConfidenceLevel.MEDIUM,
                    current_evidence=["cough"],
                ),
            ],
            processing_time_ms=100,
        )

        evaluator = GapEvaluator()
        result = evaluator.evaluate(gap_test_case, response)

        fp_score = next(s for s in result.scores if s.name == "false_positive_penalty")
        assert fp_score.value < 1.0

    def test_no_gaps_expected(self):
        """Test case where no gaps are expected."""
        # Create a test case that expects gaps (for the no-gap response test)
        # The GapTestCase schema requires at least 1 expected gap when no_gaps_expected=False
        # So we test a different scenario: response returns no gaps when we expected some
        test_case = GapTestCase(
            id="no_gaps_001",
            clinical_note="Complete, well-documented case with all diagnoses specified and documented. Patient has CHF documented with type and stage.",
            expected_gaps=[
                ExpectedGap(
                    condition="placeholder",
                    gap_type="clarification",
                    min_priority=1,
                ),
            ],
        )

        # Response with no gaps = should have low recall
        response_empty = GapDetectionResponse(
            request_id="req_gap_003",
            facts_cache_key="cache_003",
            gaps=[],
            processing_time_ms=50,
        )
        evaluator = GapEvaluator()
        result = evaluator.evaluate(test_case, response_empty)

        # Check that recall is 0 since we expected a gap but found none
        recall_score = next(s for s in result.scores if s.name == "recall")
        assert recall_score.value == 0.0

        # Response with matching gap = should pass
        response_with_gaps = GapDetectionResponse(
            request_id="req_gap_004",
            facts_cache_key="cache_004",
            gaps=[
                GapCandidate(
                    gap_id="gap_match_001",
                    condition="placeholder",
                    gap_type="clarification",
                    priority=1,
                    confidence=ConfidenceLevel.HIGH,
                    current_evidence=[],
                ),
            ],
            processing_time_ms=50,
        )
        result = evaluator.evaluate(test_case, response_with_gaps)
        recall_score = next(s for s in result.scores if s.name == "recall")
        assert recall_score.value == 1.0


# =============================================================================
# Query Evaluator Tests
# =============================================================================


class TestQueryEvaluator:
    """Tests for CDI query quality evaluation."""

    @pytest.fixture
    def query_test_case(self) -> QueryTestCase:
        """Create a sample query test case."""
        return QueryTestCase(
            id="query_test_001",
            clinical_note="Patient presents with fever 101.5F, tachycardia HR 112, elevated WBC 18000. Labs show elevated lactate 3.2 and creatinine 2.1.",
            gap=ExpectedGap(
                condition="sepsis",
                gap_type="clarification",
                min_priority=1,
            ),
            quality_criteria=QueryQualityCriteria(
                must_mention=["sepsis", "fever"],
                must_not_mention=["definitely", "obviously"],
                min_evidence_citations=2,
                expected_query_type="clarification",
            ),
        )

    @pytest.fixture
    def query_response_good(self) -> ProviderQuery:
        """Create a high-quality query response."""
        return ProviderQuery(
            query_id="query_001",
            gap_id="gap_001",
            query_text=(
                "Based on the documented fever and elevated WBC, "
                "please clarify if the clinical picture is consistent with sepsis. "
                "If clinically appropriate, please specify the underlying source."
            ),
            query_type="clarification",
            evidence_cited=["fever documented on admission", "WBC 15,000"],
            suggested_responses=[
                "Sepsis due to UTI",
                "Sepsis due to pneumonia",
                "Clinical picture not consistent with sepsis",
            ],
            icd_impact=["A41.9 - Sepsis, unspecified organism"],
        )

    @pytest.fixture
    def query_response_leading(self) -> ProviderQuery:
        """Create a query with leading language."""
        return ProviderQuery(
            query_id="query_002",
            gap_id="gap_001",
            query_text=(
                "The patient obviously has sepsis. "
                "Please confirm that the diagnosis is sepsis."
            ),
            query_type="clarification",
            evidence_cited=[],
            suggested_responses=["Yes"],
            icd_impact=[],
        )

    def test_good_query_evaluation(self, query_test_case, query_response_good):
        """Test evaluation of high-quality query."""
        evaluator = QueryEvaluator()
        result = evaluator.evaluate(query_test_case, query_response_good)

        assert result.passed is True

        # Check non-leading score
        non_leading = next(s for s in result.scores if s.name == "non_leading")
        assert non_leading.value == 1.0

        # Check compliance
        compliance = next(s for s in result.scores if s.name == "compliance")
        assert compliance.value > 0.5

    def test_leading_query_penalty(self, query_test_case, query_response_leading):
        """Test penalty for leading language."""
        evaluator = QueryEvaluator()
        result = evaluator.evaluate(query_test_case, query_response_leading)

        non_leading = next(s for s in result.scores if s.name == "non_leading")
        assert non_leading.value < 1.0
        assert "obviously" in non_leading.details.get("violations", []) or \
               "please_confirm" in non_leading.details.get("violations", [])

    def test_forbidden_terms_penalty(self, query_test_case):
        """Test penalty for forbidden terms."""
        response = ProviderQuery(
            query_id="query_003",
            gap_id="gap_001",
            query_text="This patient definitely has sepsis based on the clinical presentation.",
            query_type="clarification",
            evidence_cited=["fever"],
            suggested_responses=["Yes", "No"],
            icd_impact=[],
        )

        evaluator = QueryEvaluator()
        result = evaluator.evaluate(query_test_case, response)

        forbidden = next(s for s in result.scores if s.name == "no_forbidden_terms")
        assert forbidden.value < 1.0

    def test_evidence_grounding(self, query_test_case, query_response_good):
        """Test evidence citation scoring."""
        evaluator = QueryEvaluator()
        result = evaluator.evaluate(query_test_case, query_response_good)

        evidence = next(s for s in result.scores if s.name == "evidence_grounding")
        assert evidence.value == 1.0  # 2 citations meets minimum of 2


# =============================================================================
# E/M Evaluator Tests
# =============================================================================


class TestEMEvaluator:
    """Tests for E/M level evaluation."""

    @pytest.fixture
    def em_test_case(self) -> EMTestCase:
        """Create a sample E/M test case."""
        return EMTestCase(
            id="em_test_001",
            clinical_note="Established patient with multiple chronic conditions presenting for follow-up of diabetes, hypertension, and chronic kidney disease. Reviewed labs and adjusted medications.",
            encounter_type="outpatient",
            patient_type="established",
            expected_code="99214",
            expected_level=4,
            expected_mdm=ExpectedMDM(problems=3, data=2, risk=3),
            acceptable_codes=["99215"],
        )

    @pytest.fixture
    def em_response_correct(self) -> EMAnalysisResult:
        """Create a correct E/M response."""
        return EMAnalysisResult(
            request_id="req_em_001",
            recommended_code="99214",
            recommended_level=4,
            mdm_score=MDMComponent(problems=3, data=2, risk=3),
            justification="Moderate complexity based on chronic conditions",
            upcoding_risk=False,
            downcoding_risk=False,
            processing_time_ms=180,
        )

    @pytest.fixture
    def em_response_upcode(self) -> EMAnalysisResult:
        """Create an upcoded E/M response."""
        return EMAnalysisResult(
            request_id="req_em_002",
            recommended_code="99215",
            recommended_level=5,
            mdm_score=MDMComponent(problems=4, data=3, risk=4),
            justification="High complexity",
            upcoding_risk=True,
            downcoding_risk=False,
            processing_time_ms=150,
        )

    def test_em_utilities(self):
        """Test E/M utility functions."""
        # Test get_code_level
        assert get_code_level("99214") == 4
        assert get_code_level("99215") == 5
        assert get_code_level("99999") is None

        # Test get_code_family
        assert get_code_family("99214") == "office_established"
        assert get_code_family("99203") == "office_new"
        assert get_code_family("99285") == "emergency"

        # Test codes_in_same_family
        assert codes_in_same_family("99213", "99214") is True
        assert codes_in_same_family("99213", "99203") is False

    def test_em_code_levels_complete(self):
        """Test that E/M code levels are properly defined."""
        # Office established
        assert EM_CODE_LEVELS["99211"] == 1
        assert EM_CODE_LEVELS["99215"] == 5

        # Emergency
        assert EM_CODE_LEVELS["99281"] == 1
        assert EM_CODE_LEVELS["99285"] == 5

        # Inpatient
        assert EM_CODE_LEVELS["99221"] == 1
        assert EM_CODE_LEVELS["99223"] == 3

    def test_correct_em_evaluation(self, em_test_case, em_response_correct):
        """Test evaluation with correct E/M level."""
        evaluator = EMEvaluator()
        result = evaluator.evaluate(em_test_case, em_response_correct)

        assert result.passed is True

        # Check exact match
        exact_match = next(s for s in result.scores if s.name == "exact_match")
        assert exact_match.value == 1.0

        # Check level accuracy
        level_acc = next(s for s in result.scores if s.name == "level_accuracy")
        assert level_acc.value == 1.0

        # Check no upcoding penalty
        upcode = next(s for s in result.scores if s.name == "upcoding_penalty")
        assert upcode.value == 1.0

    def test_upcoding_detection(self, em_test_case, em_response_upcode):
        """Test detection and penalty for upcoding."""
        evaluator = EMEvaluator()
        result = evaluator.evaluate(em_test_case, em_response_upcode)

        # Check upcoding penalty applied
        upcode = next(s for s in result.scores if s.name == "upcoding_penalty")
        assert upcode.value < 1.0

        # Within one level should still pass
        within_one = next(s for s in result.scores if s.name == "within_one_level")
        assert within_one.value == 1.0

    def test_downcoding_detection(self, em_test_case):
        """Test detection of downcoding."""
        response = EMAnalysisResult(
            request_id="req_em_003",
            recommended_code="99213",
            recommended_level=3,
            mdm_score=MDMComponent(problems=2, data=2, risk=2),
            justification="Low complexity",
            upcoding_risk=False,
            downcoding_risk=True,
            processing_time_ms=150,
        )

        evaluator = EMEvaluator()
        result = evaluator.evaluate(em_test_case, response)

        downcode = next(s for s in result.scores if s.name == "downcoding_penalty")
        assert downcode.value < 1.0

    def test_strict_em_evaluator(self, em_test_case, em_response_upcode):
        """Test strict E/M evaluator has higher penalties."""
        evaluator = EMEvaluatorStrict()
        assert evaluator.pass_threshold == 0.85
        assert evaluator.upcoding_weight == 2.0

        result = evaluator.evaluate(em_test_case, em_response_upcode)
        upcode = next(s for s in result.scores if s.name == "upcoding_penalty")
        assert upcode.value == 0.5  # Stricter penalty for 1-level upcode

    def test_mdm_accuracy(self, em_test_case, em_response_correct):
        """Test MDM component accuracy scoring."""
        evaluator = EMEvaluator()
        result = evaluator.evaluate(em_test_case, em_response_correct)

        mdm_score = next(s for s in result.scores if s.name == "mdm_accuracy")
        assert mdm_score.value == 1.0  # All components match


# =============================================================================
# Integration Tests
# =============================================================================


class TestEvaluatorIntegration:
    """Integration tests for evaluator workflows."""

    def test_evaluator_result_structure(self):
        """Test that all evaluators return properly structured results."""
        # Create minimal test cases
        icd_case = ICDTestCase(
            id="test",
            clinical_note="Test note for patient with type 2 diabetes mellitus presenting for routine follow up evaluation and management.",
            expected_icd_codes=["E11.65"],
        )
        icd_response = CodingSuggestResponse(
            request_id="req_int_001",
            suggested_codes=[
                ICDSuggestion(
                    icd10_code="E11.65",
                    description="Diabetes",
                    confidence=ConfidenceLevel.HIGH,
                    evidence_spans=["evidence"],
                )
            ],
            model_version="1.0",
            processing_time_ms=100,
        )

        evaluator = ICDEvaluator()
        result = evaluator.evaluate(icd_case, icd_response)

        # Check result structure
        assert isinstance(result, EvalResult)
        assert result.test_case_id == "test"
        assert result.evaluator_type == "icd"
        assert len(result.scores) > 0
        assert 0 <= result.composite_score <= 1
        assert isinstance(result.passed, bool)

    def test_all_evaluator_types_registered(self):
        """Test that all evaluator variants can be instantiated."""
        evaluator_types = [
            "icd", "icd_strict", "icd_lenient",
            "hcc", "hcc_v28",
            "gap", "gap_strict",
            "query", "query_strict", "query_lenient",
            "em", "em_strict", "em_lenient",
        ]

        for eval_type in evaluator_types:
            evaluator = get_evaluator(eval_type)
            assert evaluator is not None
            assert evaluator.evaluator_type == eval_type
