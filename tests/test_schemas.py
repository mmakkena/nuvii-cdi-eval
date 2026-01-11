"""Tests for API response and test case schemas."""

import pytest
from pydantic import ValidationError

from nuvii_eval.schemas.api_responses import (
    CodingSuggestResponse,
    ConfidenceLevel,
    EMAnalysisResult,
    GapCandidate,
    GapDetectionResponse,
    ICDSuggestion,
    MDMComponent,
    RiskAnalysisResult,
)
from nuvii_eval.datasets.schemas import (
    BaseTestCase,
    Complexity,
    EMTestCase,
    ExpectedGap,
    ExpectedMDM,
    GapTestCase,
    HCCTestCase,
    ICDTestCase,
    QueryQualityCriteria,
    QueryTestCase,
    Specialty,
)


class TestAPIResponseSchemas:
    """Tests for API response schemas."""

    def test_icd_suggestion_valid(self):
        """Test valid ICD suggestion."""
        suggestion = ICDSuggestion(
            icd10_code="E11.9",
            description="Type 2 diabetes mellitus without complications",
            confidence=ConfidenceLevel.HIGH,
            evidence_spans=["diabetes mellitus", "HbA1c 7.2%"],
        )

        assert suggestion.icd10_code == "E11.9"
        assert suggestion.confidence == ConfidenceLevel.HIGH
        assert len(suggestion.evidence_spans) == 2

    def test_coding_suggest_response(self, sample_api_response_coding):
        """Test coding suggest response parsing."""
        response = CodingSuggestResponse(**sample_api_response_coding)

        assert response.request_id == "req_123"
        assert len(response.suggested_codes) == 2
        assert response.suggested_codes[0].icd10_code == "E11.9"
        assert response.processing_time_ms == 250

    def test_gap_candidate(self):
        """Test gap candidate schema."""
        gap = GapCandidate(
            gap_id="gap_001",
            gap_type="missing_specificity",
            condition="diabetes",
            priority=2,
            confidence=ConfidenceLevel.MEDIUM,
        )

        assert gap.gap_type == "missing_specificity"
        assert gap.priority == 2

    def test_gap_candidate_priority_bounds(self):
        """Test gap priority validation."""
        with pytest.raises(ValidationError):
            GapCandidate(
                gap_id="gap_001",
                gap_type="test",
                condition="test",
                priority=6,  # Invalid: max is 5
                confidence=ConfidenceLevel.LOW,
            )

    def test_gap_detection_response(self, sample_api_response_gaps):
        """Test gap detection response parsing."""
        response = GapDetectionResponse(**sample_api_response_gaps)

        assert len(response.gaps) == 1
        assert response.facts_cache_key == "cache_abc123"

    def test_mdm_component_level(self):
        """Test MDM level calculation."""
        mdm = MDMComponent(problems=4, data=3, risk=2)

        assert mdm.mdm_level == 3  # Second highest of [4, 3, 2]

    def test_em_analysis_result(self):
        """Test E/M analysis result."""
        result = EMAnalysisResult(
            request_id="req_789",
            recommended_code="99214",
            recommended_level=4,
            mdm_score=MDMComponent(problems=3, data=3, risk=3),
            justification="Moderate complexity visit",
            processing_time_ms=200,
        )

        assert result.recommended_level == 4
        assert result.mdm_score.mdm_level == 3

    def test_risk_analysis_result(self):
        """Test risk analysis result."""
        result = RiskAnalysisResult(
            request_id="req_999",
            current_hccs=["HCC18", "HCC85"],
            current_raf=1.25,
            opportunities=[],
            projected_raf=1.25,
            processing_time_ms=150,
        )

        assert len(result.current_hccs) == 2
        assert result.raf_gap == 0.0


class TestTestCaseSchemas:
    """Tests for test case schemas."""

    def test_base_test_case_id_validation(self):
        """Test ID format validation."""
        # Valid IDs
        valid_ids = ["test_001", "test-001", "Test123", "a1b2c3"]
        for test_id in valid_ids:
            tc = BaseTestCase(
                id=test_id,
                clinical_note="A" * 50,  # Min 50 chars
            )
            assert tc.id == test_id

        # Invalid IDs
        with pytest.raises(ValidationError):
            BaseTestCase(id="test.001", clinical_note="A" * 50)

        with pytest.raises(ValidationError):
            BaseTestCase(id="test 001", clinical_note="A" * 50)

    def test_icd_test_case_valid(self, sample_icd_test_case):
        """Test valid ICD test case."""
        tc = ICDTestCase(**sample_icd_test_case)

        assert tc.id == "icd_test_001"
        assert tc.specialty == Specialty.ENDOCRINOLOGY
        assert "E11.9" in tc.expected_icd_codes
        assert tc.complexity == Complexity.MODERATE

    def test_icd_test_case_code_validation(self):
        """Test ICD code format validation."""
        # Valid codes
        tc = ICDTestCase(
            id="test_001",
            clinical_note="A" * 50,
            expected_icd_codes=["E11.9", "I10", "J45.20"],
        )
        assert len(tc.expected_icd_codes) == 3

        # Invalid code format
        with pytest.raises(ValidationError):
            ICDTestCase(
                id="test_001",
                clinical_note="A" * 50,
                expected_icd_codes=["INVALID"],
            )

    def test_icd_test_case_primary_code_consistency(self):
        """Test primary code must be in expected codes."""
        with pytest.raises(ValidationError):
            ICDTestCase(
                id="test_001",
                clinical_note="A" * 50,
                expected_icd_codes=["E11.9"],
                primary_code="I10",  # Not in expected_icd_codes
            )

    def test_hcc_test_case_valid(self, sample_hcc_test_case):
        """Test valid HCC test case."""
        tc = HCCTestCase(**sample_hcc_test_case)

        assert "HCC85" in tc.expected_hccs
        assert tc.expected_raf_range == (0.3, 0.5)
        assert tc.patient_age == 65
        assert tc.patient_gender == "M"

    def test_hcc_test_case_raf_range_validation(self):
        """Test RAF range validation."""
        # Invalid: min > max
        with pytest.raises(ValidationError):
            HCCTestCase(
                id="test_001",
                clinical_note="A" * 50,
                expected_hccs=["HCC85"],
                expected_raf_range=(0.5, 0.3),  # Invalid
                patient_age=65,
                patient_gender="M",
            )

        # Invalid: negative RAF
        with pytest.raises(ValidationError):
            HCCTestCase(
                id="test_001",
                clinical_note="A" * 50,
                expected_hccs=["HCC85"],
                expected_raf_range=(-0.1, 0.5),  # Invalid
                patient_age=65,
                patient_gender="M",
            )

    def test_gap_test_case_valid(self, sample_gap_test_case):
        """Test valid gap test case."""
        tc = GapTestCase(**sample_gap_test_case)

        assert len(tc.expected_gaps) == 1
        assert tc.expected_gaps[0].gap_type == "unconfirmed_diagnosis"
        assert tc.expected_gaps[0].condition == "unstable angina"

    def test_gap_test_case_consistency_validation(self):
        """Test gap test case consistency."""
        # Invalid: no_gaps_expected=True but expected_gaps provided
        with pytest.raises(ValidationError):
            GapTestCase(
                id="test_001",
                clinical_note="A" * 50,
                expected_gaps=[ExpectedGap(gap_type="test", condition="test")],
                no_gaps_expected=True,
            )

    def test_query_test_case_valid(self):
        """Test valid query test case."""
        tc = QueryTestCase(
            id="query_001",
            clinical_note="A" * 50,
            gap=ExpectedGap(
                gap_type="unconfirmed_diagnosis",
                condition="diabetes",
            ),
            quality_criteria=QueryQualityCriteria(
                must_mention=["diabetes"],
                must_not_mention=["definitely", "obviously"],
                min_evidence_citations=2,
            ),
        )

        assert tc.gap.condition == "diabetes"
        assert tc.quality_criteria.min_evidence_citations == 2

    def test_em_test_case_valid(self, sample_em_test_case):
        """Test valid E/M test case."""
        tc = EMTestCase(**sample_em_test_case)

        assert tc.expected_code == "99215"
        assert tc.expected_level == 5
        assert tc.expected_mdm.problems == 4
        assert tc.encounter_type == "outpatient"

    def test_specialty_enum(self):
        """Test specialty enumeration."""
        assert Specialty.CARDIOLOGY.value == "cardiology"
        assert Specialty.ENDOCRINOLOGY.value == "endocrinology"

        # Test case assignment
        tc = BaseTestCase(
            id="test_001",
            clinical_note="A" * 50,
            specialty=Specialty.CARDIOLOGY,
        )
        assert tc.specialty == Specialty.CARDIOLOGY

    def test_complexity_enum(self):
        """Test complexity enumeration."""
        assert Complexity.LOW.value == "low"
        assert Complexity.HIGH.value == "high"


class TestCodeValidation:
    """Tests for code format validation helpers."""

    def test_icd10_code_patterns(self):
        """Test ICD-10 code pattern matching."""
        from nuvii_eval.datasets.schemas import validate_icd10_code

        # Valid codes
        valid_codes = [
            "A00",
            "E11.9",
            "I50.22",
            "Z99.89",
            "J45.20",
            "M54.5",
            "S72.001",
            "T36.0X1",
        ]
        for code in valid_codes:
            assert validate_icd10_code(code) == code

        # Invalid codes
        invalid_codes = [
            "111.9",  # Must start with letter
            "E1",  # Too short
            "E11.99999",  # Too many decimals
            "e11.9",  # Lowercase (if not allowed)
        ]
        for code in invalid_codes:
            with pytest.raises(ValueError):
                validate_icd10_code(code)

    def test_hcc_code_patterns(self):
        """Test HCC code pattern matching."""
        from nuvii_eval.datasets.schemas import validate_hcc_code

        # Valid codes
        valid_codes = ["HCC1", "HCC18", "HCC85", "HCC189"]
        for code in valid_codes:
            assert validate_hcc_code(code) == code

        # Invalid codes
        with pytest.raises(ValueError):
            validate_hcc_code("85")  # Missing HCC prefix

        with pytest.raises(ValueError):
            validate_hcc_code("HCC")  # Missing number

    def test_cpt_code_patterns(self):
        """Test CPT code pattern matching."""
        from nuvii_eval.datasets.schemas import validate_cpt_code

        # Valid codes
        valid_codes = ["99213", "99214", "43239", "27447"]
        for code in valid_codes:
            assert validate_cpt_code(code) == code

        # Invalid codes
        with pytest.raises(ValueError):
            validate_cpt_code("9921")  # Too short

        with pytest.raises(ValueError):
            validate_cpt_code("992134")  # Too long
