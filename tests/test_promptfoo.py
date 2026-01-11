"""
Unit tests for Promptfoo integration module.

Tests configuration generation, assertion builders, converters,
runner, and regression detection.
"""

from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest

from nuvii_eval.datasets.schemas import (
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
from nuvii_eval.promptfoo.assertions import (
    AssertionBuilder,
    EMAssertionBuilder,
    GapAssertionBuilder,
    HCCAssertionBuilder,
    ICDAssertionBuilder,
    QueryAssertionBuilder,
    get_assertion_builder,
)
from nuvii_eval.promptfoo.config import (
    ConfigGeneratorOptions,
    PromptfooAssertion,
    PromptfooConfig,
    PromptfooProvider,
    PromptfooTest,
    create_nuvii_provider,
    create_openai_provider,
    generate_em_config,
    generate_gap_config,
    generate_hcc_config,
    generate_icd_config,
    generate_promptfoo_config,
    generate_query_config,
)
from nuvii_eval.promptfoo.converter import (
    TestCaseConverter,
    convert_single_test,
    convert_test_suite,
)
from nuvii_eval.promptfoo.regression import (
    DetectorConfig,
    Regression,
    RegressionDetector,
    RegressionReport,
    RegressionSeverity,
    RegressionType,
    check_for_blockers,
    compare_results,
    get_regression_summary,
)
from nuvii_eval.promptfoo.runner import (
    AssertionResult,
    PromptfooResult,
    PromptfooRunner,
    RunConfig,
    TestResult,
    analyze_results,
    format_ci_report,
)


# =============================================================================
# Test Fixtures - Clinical Notes
# =============================================================================


SAMPLE_CLINICAL_NOTE = """
CHIEF COMPLAINT: Chest pain and shortness of breath.

HISTORY OF PRESENT ILLNESS:
65-year-old male with history of hypertension and type 2 diabetes mellitus
presents with intermittent chest pain for the past 3 days. Patient describes
the pain as pressure-like, radiating to left arm. Associated with dyspnea
on exertion. No fever, cough, or syncope.

PHYSICAL EXAMINATION:
Vitals: BP 150/90, HR 88, RR 18, SpO2 96% on RA
General: Alert, in mild distress
Cardiovascular: Regular rhythm, no murmurs, no JVD
Lungs: Clear bilaterally

ASSESSMENT AND PLAN:
1. Acute coronary syndrome - will order troponin, EKG, admit for observation
2. Uncontrolled hypertension - adjust medications
3. Type 2 diabetes mellitus - continue current regimen
"""


# =============================================================================
# Test Fixtures - Test Cases
# =============================================================================


@pytest.fixture
def icd_test_case():
    """Sample ICD test case."""
    return ICDTestCase(
        id="icd-test-001",
        clinical_note=SAMPLE_CLINICAL_NOTE,
        specialty=Specialty.CARDIOLOGY,
        complexity=Complexity.MODERATE,
        expected_icd_codes=["I20.9", "I10", "E11.9"],
        acceptable_icd_codes=["I20.0"],
        unacceptable_codes=["I21.0"],
        primary_code="I20.9",
        code_sequence_matters=True,
        tags=["cardiac", "acute"],
    )


@pytest.fixture
def hcc_test_case():
    """Sample HCC test case."""
    return HCCTestCase(
        id="hcc-test-001",
        clinical_note=SAMPLE_CLINICAL_NOTE,
        specialty=Specialty.CARDIOLOGY,
        complexity=Complexity.MODERATE,
        expected_hccs=["HCC85", "HCC19"],
        expected_raf_range=(1.0, 1.5),
        expected_opportunities=["HCC18"],
        patient_age=65,
        patient_gender="M",
        model_year="2024",
    )


@pytest.fixture
def gap_test_case():
    """Sample gap detection test case."""
    return GapTestCase(
        id="gap-test-001",
        clinical_note=SAMPLE_CLINICAL_NOTE,
        specialty=Specialty.CARDIOLOGY,
        complexity=Complexity.MODERATE,
        expected_gaps=[
            ExpectedGap(
                gap_type="missing_specificity",
                condition="diabetes mellitus",
                min_priority=2,
            ),
            ExpectedGap(
                gap_type="unconfirmed_diagnosis",
                condition="acute coronary syndrome",
                min_priority=1,
            ),
        ],
        false_positive_conditions=["hypertension"],
    )


@pytest.fixture
def query_test_case():
    """Sample query test case."""
    return QueryTestCase(
        id="query-test-001",
        clinical_note=SAMPLE_CLINICAL_NOTE,
        specialty=Specialty.CARDIOLOGY,
        complexity=Complexity.MODERATE,
        gap=ExpectedGap(
            gap_type="missing_specificity",
            condition="diabetes mellitus",
            min_priority=2,
        ),
        quality_criteria=QueryQualityCriteria(
            must_mention=["diabetes", "type"],
            must_not_mention=["definitely", "obviously"],
            expected_query_type="clarification",
            must_provide_options=True,
        ),
        reference_query="Could you please clarify if this patient has Type 1 or Type 2 diabetes?",
    )


@pytest.fixture
def em_test_case():
    """Sample E/M test case."""
    return EMTestCase(
        id="em-test-001",
        clinical_note=SAMPLE_CLINICAL_NOTE,
        specialty=Specialty.CARDIOLOGY,
        complexity=Complexity.MODERATE,
        encounter_type="outpatient",
        patient_type="established",
        expected_code="99214",
        expected_level=4,
        expected_mdm=ExpectedMDM(problems=3, data=3, risk=3),
        acceptable_codes=["99213"],
        documented_time=30,
    )


# =============================================================================
# Test: Configuration Models
# =============================================================================


class TestPromptfooProvider:
    """Tests for PromptfooProvider."""

    def test_basic_provider(self):
        """Test basic provider creation."""
        provider = PromptfooProvider(id="openai:gpt-4")
        result = provider.to_dict()

        assert result["id"] == "openai:gpt-4"
        assert "label" not in result
        assert "config" not in result

    def test_provider_with_config(self):
        """Test provider with configuration."""
        provider = PromptfooProvider(
            id="http",
            label="Custom API",
            config={"url": "http://example.com", "timeout": 30000},
        )
        result = provider.to_dict()

        assert result["id"] == "http"
        assert result["label"] == "Custom API"
        assert result["config"]["url"] == "http://example.com"


class TestPromptfooAssertion:
    """Tests for PromptfooAssertion."""

    def test_simple_assertion(self):
        """Test simple assertion."""
        assertion = PromptfooAssertion(type="contains", value="diabetes")
        result = assertion.to_dict()

        assert result["type"] == "contains"
        assert result["value"] == "diabetes"
        assert "weight" not in result  # Default weight excluded

    def test_weighted_assertion(self):
        """Test assertion with weight."""
        assertion = PromptfooAssertion(
            type="llm-rubric",
            value="Check accuracy",
            threshold=0.8,
            weight=2.0,
        )
        result = assertion.to_dict()

        assert result["type"] == "llm-rubric"
        assert result["threshold"] == 0.8
        assert result["weight"] == 2.0


class TestPromptfooTest:
    """Tests for PromptfooTest."""

    def test_basic_test(self):
        """Test basic test case."""
        test = PromptfooTest(
            description="Test case 1",
            vars={"clinical_note": "Sample note"},
        )
        result = test.to_dict()

        assert result["description"] == "Test case 1"
        assert result["vars"]["clinical_note"] == "Sample note"
        assert "assert" not in result

    def test_test_with_assertions(self):
        """Test case with assertions."""
        test = PromptfooTest(
            description="Test case 1",
            vars={"clinical_note": "Sample note"},
            assert_=[
                PromptfooAssertion(type="contains", value="diabetes"),
            ],
            metadata={"specialty": "cardiology"},
        )
        result = test.to_dict()

        assert len(result["assert"]) == 1
        assert result["metadata"]["specialty"] == "cardiology"


class TestPromptfooConfig:
    """Tests for PromptfooConfig."""

    def test_full_config(self):
        """Test full configuration generation."""
        config = PromptfooConfig(
            description="Test Suite",
            providers=[PromptfooProvider(id="openai:gpt-4")],
            prompts=["{{clinical_note}}"],
            tests=[
                PromptfooTest(
                    description="Test 1",
                    vars={"clinical_note": "Note"},
                )
            ],
            output_path="output.json",
            sharing=False,
        )
        result = config.to_dict()

        assert result["description"] == "Test Suite"
        assert len(result["providers"]) == 1
        assert len(result["prompts"]) == 1
        assert len(result["tests"]) == 1
        assert result["outputPath"] == "output.json"
        assert result["sharing"] is False

    def test_yaml_export(self):
        """Test YAML export."""
        config = PromptfooConfig(
            description="Test",
            providers=[PromptfooProvider(id="openai:gpt-4")],
            prompts=["{{input}}"],
            tests=[],
        )
        yaml_str = config.to_yaml()

        assert "description: Test" in yaml_str
        assert "providers:" in yaml_str


# =============================================================================
# Test: Provider Creation
# =============================================================================


class TestProviderCreation:
    """Tests for provider creation functions."""

    @patch("nuvii_eval.promptfoo.config.get_settings")
    def test_create_nuvii_provider(self, mock_settings):
        """Test Nuvii provider creation."""
        mock_settings.return_value.nuvii_api.base_url = "http://api.example.com"

        provider = create_nuvii_provider(endpoint="/api/v2/test")
        result = provider.to_dict()

        assert result["id"] == "http"
        assert result["label"] == "Nuvii CDI API"
        assert "/api/v2/test" in result["config"]["url"]

    def test_create_openai_provider(self):
        """Test OpenAI provider creation."""
        provider = create_openai_provider(model="gpt-4o", temperature=0.1)
        result = provider.to_dict()

        assert result["id"] == "openai:gpt-4o"
        assert result["config"]["temperature"] == 0.1


# =============================================================================
# Test: Assertion Builders
# =============================================================================


class TestICDAssertionBuilder:
    """Tests for ICD assertion builder."""

    def test_build_assertions(self, icd_test_case):
        """Test building ICD assertions."""
        builder = ICDAssertionBuilder()
        assertions = builder.build(icd_test_case)

        # Should have assertions for expected codes, unacceptable codes, regex, and rubrics
        assert len(assertions) >= 5

        # Check for expected code assertions
        contains_assertions = [a for a in assertions if a.type == "contains"]
        assert any(a.value == "I20.9" for a in contains_assertions)

        # Check for unacceptable code assertions
        not_contains_assertions = [a for a in assertions if a.type == "not-contains"]
        assert any(a.value == "I21.0" for a in not_contains_assertions)

        # Check for ICD format regex
        regex_assertions = [a for a in assertions if a.type == "regex"]
        assert len(regex_assertions) >= 1

    def test_primary_code_assertion(self, icd_test_case):
        """Test primary code assertion when sequence matters."""
        builder = ICDAssertionBuilder()
        assertions = builder.build(icd_test_case)

        # Should have LLM rubric for primary code sequence
        llm_rubrics = [a for a in assertions if a.type == "llm-rubric"]
        primary_rubric = [r for r in llm_rubrics if "I20.9" in r.value and "first" in r.value]
        assert len(primary_rubric) >= 1


class TestHCCAssertionBuilder:
    """Tests for HCC assertion builder."""

    def test_build_assertions(self, hcc_test_case):
        """Test building HCC assertions."""
        builder = HCCAssertionBuilder()
        assertions = builder.build(hcc_test_case)

        # Check for HCC code assertions
        contains_assertions = [a for a in assertions if a.type == "contains"]
        assert any(a.value == "HCC85" for a in contains_assertions)
        assert any(a.value == "HCC19" for a in contains_assertions)

        # Check for JavaScript RAF range assertion
        js_assertions = [a for a in assertions if a.type == "javascript"]
        assert len(js_assertions) >= 1

    def test_opportunity_assertions(self, hcc_test_case):
        """Test opportunity assertions."""
        builder = HCCAssertionBuilder()
        assertions = builder.build(hcc_test_case)

        contains_assertions = [a for a in assertions if a.type == "contains"]
        assert any(a.value == "HCC18" for a in contains_assertions)


class TestGapAssertionBuilder:
    """Tests for Gap assertion builder."""

    def test_build_assertions(self, gap_test_case):
        """Test building gap assertions."""
        builder = GapAssertionBuilder()
        assertions = builder.build(gap_test_case)

        # Check for gap condition assertions
        contains_assertions = [a for a in assertions if a.type == "contains"]
        assert any("diabetes" in a.value for a in contains_assertions)

        # Check for false positive assertions
        not_contains_assertions = [a for a in assertions if a.type == "not-contains"]
        assert any(a.value == "hypertension" for a in not_contains_assertions)

    def test_no_gaps_expected(self):
        """Test when no gaps are expected."""
        # Create a minimal test case where no gaps are expected
        # (need to bypass validation that requires expected_gaps)
        test_case = MagicMock()
        test_case.no_gaps_expected = True
        test_case.expected_gaps = []
        test_case.false_positive_conditions = []

        builder = GapAssertionBuilder()
        assertions = builder.build(test_case)

        # Should have LLM rubric for no gaps
        llm_rubrics = [a for a in assertions if a.type == "llm-rubric"]
        assert any("No documentation gaps" in r.value for r in llm_rubrics)


class TestQueryAssertionBuilder:
    """Tests for Query assertion builder."""

    def test_build_assertions(self, query_test_case):
        """Test building query assertions."""
        builder = QueryAssertionBuilder()
        assertions = builder.build(query_test_case)

        # Check for must_mention assertions
        contains_assertions = [a for a in assertions if a.type == "contains"]
        assert any("diabetes" in a.value for a in contains_assertions)

        # Check for must_not_mention assertions
        not_contains_assertions = [a for a in assertions if a.type == "not-contains"]
        assert any(a.value == "definitely" for a in not_contains_assertions)

        # Check for leading language patterns
        leading_patterns = ["please confirm", "do you agree", "obviously"]
        for pattern in leading_patterns:
            assert any(a.value == pattern for a in not_contains_assertions)

    def test_non_leading_language_rubric(self, query_test_case):
        """Test non-leading language assertion."""
        builder = QueryAssertionBuilder()
        assertions = builder.build(query_test_case)

        llm_rubrics = [a for a in assertions if a.type == "llm-rubric"]
        assert any("non-leading" in r.value for r in llm_rubrics)


class TestEMAssertionBuilder:
    """Tests for E/M assertion builder."""

    def test_build_assertions(self, em_test_case):
        """Test building E/M assertions."""
        builder = EMAssertionBuilder()
        assertions = builder.build(em_test_case)

        # Check for expected code assertion
        contains_assertions = [a for a in assertions if a.type == "contains"]
        assert any(a.value == "99214" for a in contains_assertions)

        # Check for E/M code format regex
        regex_assertions = [a for a in assertions if a.type == "regex"]
        assert any("99" in a.value for a in regex_assertions)

        # Check for MDM rubric
        llm_rubrics = [a for a in assertions if a.type == "llm-rubric"]
        assert any("MDM" in r.value for r in llm_rubrics)

    def test_no_upcoding_rubric(self, em_test_case):
        """Test upcoding prevention assertion."""
        builder = EMAssertionBuilder()
        assertions = builder.build(em_test_case)

        llm_rubrics = [a for a in assertions if a.type == "llm-rubric"]
        assert any("upcoding" in r.value for r in llm_rubrics)


class TestGetAssertionBuilder:
    """Tests for assertion builder factory."""

    def test_get_icd_builder(self):
        """Test getting ICD builder."""
        builder = get_assertion_builder("icd")
        assert isinstance(builder, ICDAssertionBuilder)

    def test_get_hcc_builder(self):
        """Test getting HCC builder."""
        builder = get_assertion_builder("hcc")
        assert isinstance(builder, HCCAssertionBuilder)

    def test_get_gap_builder(self):
        """Test getting Gap builder."""
        builder = get_assertion_builder("gap")
        assert isinstance(builder, GapAssertionBuilder)

    def test_get_query_builder(self):
        """Test getting Query builder."""
        builder = get_assertion_builder("query")
        assert isinstance(builder, QueryAssertionBuilder)

    def test_get_em_builder(self):
        """Test getting E/M builder."""
        builder = get_assertion_builder("em")
        assert isinstance(builder, EMAssertionBuilder)

    def test_unknown_task_type(self):
        """Test unknown task type raises error."""
        with pytest.raises(ValueError, match="Unknown task type"):
            get_assertion_builder("unknown")


# =============================================================================
# Test: Test Case Converter
# =============================================================================


class TestTestCaseConverter:
    """Tests for TestCaseConverter."""

    def test_convert_icd_case(self, icd_test_case):
        """Test converting ICD test case."""
        converter = TestCaseConverter("icd")
        result = converter.convert(icd_test_case)

        assert isinstance(result, PromptfooTest)
        assert "icd-test-001" in result.description
        assert result.vars["clinical_note"] == SAMPLE_CLINICAL_NOTE
        assert result.vars["test_case_id"] == "icd-test-001"
        assert result.vars["expected_codes"] == ["I20.9", "I10", "E11.9"]
        assert len(result.assert_) > 0
        assert result.metadata["task_type"] == "icd"
        assert result.metadata["specialty"] == "cardiology"

    def test_convert_hcc_case(self, hcc_test_case):
        """Test converting HCC test case."""
        converter = TestCaseConverter("hcc")
        result = converter.convert(hcc_test_case)

        assert result.vars["expected_hccs"] == ["HCC85", "HCC19"]
        assert result.vars["patient_age"] == 65
        assert result.vars["patient_gender"] == "M"

    def test_convert_gap_case(self, gap_test_case):
        """Test converting Gap test case."""
        converter = TestCaseConverter("gap")
        result = converter.convert(gap_test_case)

        assert len(result.vars["expected_gaps"]) == 2
        assert result.vars["false_positives"] == ["hypertension"]

    def test_convert_query_case(self, query_test_case):
        """Test converting Query test case."""
        converter = TestCaseConverter("query")
        result = converter.convert(query_test_case)

        assert result.vars["gap"]["condition"] == "diabetes mellitus"
        assert result.vars["quality_criteria"]["must_mention"] == ["diabetes", "type"]

    def test_convert_em_case(self, em_test_case):
        """Test converting E/M test case."""
        converter = TestCaseConverter("em")
        result = converter.convert(em_test_case)

        assert result.vars["encounter_type"] == "outpatient"
        assert result.vars["expected_code"] == "99214"
        assert result.vars["expected_level"] == 4

    def test_convert_batch(self, icd_test_case):
        """Test batch conversion."""
        converter = TestCaseConverter("icd")
        results = converter.convert_batch([icd_test_case, icd_test_case])

        assert len(results) == 2
        assert all(isinstance(r, PromptfooTest) for r in results)


class TestConvenienceFunctions:
    """Tests for convenience conversion functions."""

    def test_convert_single_test(self, icd_test_case):
        """Test convert_single_test function."""
        result = convert_single_test(icd_test_case, "icd")
        assert isinstance(result, PromptfooTest)

    def test_convert_test_suite(self, icd_test_case, hcc_test_case):
        """Test convert_test_suite function."""
        # Need to use same task type for all cases
        results = convert_test_suite([icd_test_case], "icd")
        assert len(results) == 1


# =============================================================================
# Test: Configuration Generation
# =============================================================================


class TestConfigGeneration:
    """Tests for configuration generation."""

    @patch("nuvii_eval.promptfoo.config.get_settings")
    def test_generate_promptfoo_config(self, mock_settings, icd_test_case):
        """Test full config generation."""
        mock_settings.return_value.nuvii_api.base_url = "http://api.example.com"

        tests = [convert_single_test(icd_test_case, "icd")]
        options = ConfigGeneratorOptions(
            description="Test Suite",
            include_baseline=True,
        )

        config = generate_promptfoo_config(tests, "icd", options)

        assert "ICD" in config.description
        assert len(config.providers) == 2  # Nuvii + OpenAI baseline
        assert len(config.tests) == 1
        assert "ICD-10" in config.prompts[0]

    @patch("nuvii_eval.promptfoo.config.get_settings")
    def test_generate_icd_config(self, mock_settings, icd_test_case):
        """Test ICD config generation."""
        mock_settings.return_value.nuvii_api.base_url = "http://api.example.com"

        tests = [convert_single_test(icd_test_case, "icd")]
        config = generate_icd_config(tests)

        assert "ICD" in config.description

    @patch("nuvii_eval.promptfoo.config.get_settings")
    def test_generate_hcc_config(self, mock_settings, hcc_test_case):
        """Test HCC config generation."""
        mock_settings.return_value.nuvii_api.base_url = "http://api.example.com"

        tests = [convert_single_test(hcc_test_case, "hcc")]
        config = generate_hcc_config(tests)

        assert "HCC" in config.description

    @patch("nuvii_eval.promptfoo.config.get_settings")
    def test_generate_gap_config(self, mock_settings, gap_test_case):
        """Test Gap config generation."""
        mock_settings.return_value.nuvii_api.base_url = "http://api.example.com"

        tests = [convert_single_test(gap_test_case, "gap")]
        config = generate_gap_config(tests)

        assert "GAP" in config.description

    @patch("nuvii_eval.promptfoo.config.get_settings")
    def test_generate_query_config(self, mock_settings, query_test_case):
        """Test Query config generation."""
        mock_settings.return_value.nuvii_api.base_url = "http://api.example.com"

        tests = [convert_single_test(query_test_case, "query")]
        config = generate_query_config(tests)

        assert "QUERY" in config.description

    @patch("nuvii_eval.promptfoo.config.get_settings")
    def test_generate_em_config(self, mock_settings, em_test_case):
        """Test E/M config generation."""
        mock_settings.return_value.nuvii_api.base_url = "http://api.example.com"

        tests = [convert_single_test(em_test_case, "em")]
        config = generate_em_config(tests)

        assert "EM" in config.description


# =============================================================================
# Test: Runner Models
# =============================================================================


class TestRunConfig:
    """Tests for RunConfig."""

    def test_to_args_basic(self):
        """Test basic argument generation."""
        config = RunConfig(config_path="promptfoo.yaml")
        args = config.to_args()

        assert "promptfoo" in args
        assert "eval" in args
        assert "--config" in args
        assert "promptfoo.yaml" in args

    def test_to_args_with_options(self):
        """Test argument generation with all options."""
        config = RunConfig(
            config_path="promptfoo.yaml",
            output_path="output.json",
            max_concurrency=10,
            env_file=".env",
            verbose=True,
            no_cache=True,
            filter_pattern="test-*",
            grader="openai:gpt-4",
        )
        args = config.to_args()

        assert "--env-file" in args
        assert "--verbose" in args
        assert "--no-cache" in args
        assert "--filter-pattern" in args
        assert "--grader" in args


class TestAssertionResult:
    """Tests for AssertionResult."""

    def test_from_dict(self):
        """Test creating from dictionary."""
        data = {
            "type": "contains",
            "pass": True,
            "score": 1.0,
            "reason": "Found match",
        }
        result = AssertionResult.from_dict(data)

        assert result.type == "contains"
        assert result.pass_ is True
        assert result.score == 1.0
        assert result.reason == "Found match"


class TestTestResult:
    """Tests for TestResult."""

    def test_from_dict(self):
        """Test creating from dictionary."""
        data = {
            "vars": {"test_case_id": "test-001"},
            "description": "Test case 1",
            "pass": True,
            "assertions": [
                {"type": "contains", "pass": True, "score": 1.0},
                {"type": "regex", "pass": False, "score": 0.0},
            ],
            "latencyMs": 150,
            "output": "Sample output",
        }
        result = TestResult.from_dict(data)

        assert result.test_id == "test-001"
        assert result.pass_ is True
        assert result.score == 0.5  # 1/2 assertions passed
        assert result.latency_ms == 150
        assert len(result.assertions) == 2

    def test_failed_assertions(self):
        """Test getting failed assertions."""
        data = {
            "vars": {"test_case_id": "test-001"},
            "description": "Test",
            "pass": False,
            "assertions": [
                {"type": "contains", "pass": True, "score": 1.0},
                {"type": "regex", "pass": False, "score": 0.0},
            ],
        }
        result = TestResult.from_dict(data)

        failed = result.failed_assertions
        assert len(failed) == 1
        assert failed[0].type == "regex"


class TestPromptfooResult:
    """Tests for PromptfooResult."""

    def test_properties(self):
        """Test result properties."""
        tests = [
            TestResult(
                test_id="test-001",
                description="Test 1",
                pass_=True,
                score=1.0,
                assertions=[],
                latency_ms=100,
            ),
            TestResult(
                test_id="test-002",
                description="Test 2",
                pass_=False,
                score=0.5,
                assertions=[],
                latency_ms=200,
            ),
            TestResult(
                test_id="test-003",
                description="Test 3",
                pass_=True,
                score=0.8,
                assertions=[],
                latency_ms=150,
            ),
        ]
        result = PromptfooResult(
            timestamp=datetime.now(),
            config_path="test.yaml",
            tests=tests,
        )

        assert result.total_tests == 3
        assert result.passed_tests == 2
        assert result.failed_tests == 1
        assert result.pass_rate == pytest.approx(66.67, rel=0.01)
        assert result.average_score == pytest.approx(0.767, rel=0.01)
        assert result.average_latency_ms == 150

    def test_get_failed_tests(self):
        """Test getting failed tests."""
        tests = [
            TestResult(test_id="t1", description="T1", pass_=True, score=1.0, assertions=[]),
            TestResult(test_id="t2", description="T2", pass_=False, score=0.0, assertions=[]),
        ]
        result = PromptfooResult(
            timestamp=datetime.now(),
            config_path="test.yaml",
            tests=tests,
        )

        failed = result.get_failed_tests()
        assert len(failed) == 1
        assert failed[0].test_id == "t2"

    def test_get_test_by_id(self):
        """Test getting test by ID."""
        tests = [
            TestResult(test_id="t1", description="T1", pass_=True, score=1.0, assertions=[]),
        ]
        result = PromptfooResult(
            timestamp=datetime.now(),
            config_path="test.yaml",
            tests=tests,
        )

        found = result.get_test_by_id("t1")
        assert found is not None
        assert found.test_id == "t1"

        not_found = result.get_test_by_id("nonexistent")
        assert not_found is None


# =============================================================================
# Test: Result Analysis
# =============================================================================


class TestAnalyzeResults:
    """Tests for analyze_results function."""

    def test_analyze_results(self):
        """Test result analysis."""
        tests = [
            TestResult(
                test_id="t1",
                description="T1",
                pass_=True,
                score=1.0,
                assertions=[
                    AssertionResult(type="contains", pass_=True, score=1.0),
                ],
            ),
            TestResult(
                test_id="t2",
                description="T2",
                pass_=False,
                score=0.5,
                assertions=[
                    AssertionResult(type="contains", pass_=True, score=1.0),
                    AssertionResult(type="regex", pass_=False, score=0.0, reason="No match"),
                ],
                error="Assertion failed",
            ),
        ]
        result = PromptfooResult(
            timestamp=datetime.now(),
            config_path="test.yaml",
            tests=tests,
        )

        analysis = analyze_results(result)

        assert "summary" in analysis
        assert "metrics" in analysis
        assert "failing_tests" in analysis
        assert "assertion_breakdown" in analysis

        # Check assertion breakdown
        assert analysis["assertion_breakdown"]["contains"]["passed"] == 2
        assert analysis["assertion_breakdown"]["regex"]["failed"] == 1

        # Check failing tests
        assert len(analysis["failing_tests"]) == 1
        assert analysis["failing_tests"][0]["test_id"] == "t2"


class TestFormatCIReport:
    """Tests for format_ci_report function."""

    def test_format_passing_report(self):
        """Test formatting passing report."""
        tests = [
            TestResult(test_id="t1", description="T1", pass_=True, score=1.0, assertions=[]),
        ]
        result = PromptfooResult(
            timestamp=datetime.now(),
            config_path="test.yaml",
            tests=tests,
        )

        report = format_ci_report(result)

        assert "PROMPTFOO EVALUATION REPORT" in report
        assert "1/1 passed" in report
        assert "STATUS: PASS" in report

    def test_format_failing_report(self):
        """Test formatting failing report."""
        tests = [
            TestResult(
                test_id="t1",
                description="Test 1",
                pass_=False,
                score=0.0,
                assertions=[
                    AssertionResult(type="contains", pass_=False, score=0.0, reason="Not found"),
                ],
            ),
        ]
        result = PromptfooResult(
            timestamp=datetime.now(),
            config_path="test.yaml",
            tests=tests,
        )

        report = format_ci_report(result)

        assert "FAILED TESTS" in report
        assert "STATUS: FAIL" in report


# =============================================================================
# Test: Regression Detection
# =============================================================================


class TestRegressionModels:
    """Tests for regression data models."""

    def test_regression_to_dict(self):
        """Test Regression to_dict."""
        regression = Regression(
            type=RegressionType.NEW_FAILURE,
            severity=RegressionSeverity.HIGH,
            test_id="test-001",
            description="Test now failing",
            baseline_value="PASS",
            current_value="FAIL",
            delta=-1.0,
        )
        result = regression.to_dict()

        assert result["type"] == "new_failure"
        assert result["severity"] == "high"
        assert result["test_id"] == "test-001"

    def test_regression_report_properties(self):
        """Test RegressionReport properties."""
        report = RegressionReport(
            timestamp=datetime.now(),
            baseline_timestamp=datetime.now(),
            regressions=[
                Regression(
                    type=RegressionType.NEW_FAILURE,
                    severity=RegressionSeverity.CRITICAL,
                    test_id="t1",
                    description="Test 1",
                    baseline_value="PASS",
                    current_value="FAIL",
                ),
                Regression(
                    type=RegressionType.SCORE_DROP,
                    severity=RegressionSeverity.MEDIUM,
                    test_id="t2",
                    description="Test 2",
                    baseline_value="1.0",
                    current_value="0.5",
                ),
            ],
            improvements=[],
        )

        assert report.has_regressions is True
        assert report.has_critical_regressions is True
        assert report.has_blocking_regressions is True
        assert report.regression_count == 2

    def test_get_regressions_by_severity(self):
        """Test filtering regressions by severity."""
        report = RegressionReport(
            timestamp=datetime.now(),
            baseline_timestamp=None,
            regressions=[
                Regression(
                    type=RegressionType.NEW_FAILURE,
                    severity=RegressionSeverity.HIGH,
                    test_id="t1",
                    description="Test 1",
                    baseline_value="PASS",
                    current_value="FAIL",
                ),
                Regression(
                    type=RegressionType.SCORE_DROP,
                    severity=RegressionSeverity.LOW,
                    test_id="t2",
                    description="Test 2",
                    baseline_value="1.0",
                    current_value="0.9",
                ),
            ],
            improvements=[],
        )

        high_regs = report.get_regressions_by_severity(RegressionSeverity.HIGH)
        assert len(high_regs) == 1

        low_regs = report.get_regressions_by_severity(RegressionSeverity.LOW)
        assert len(low_regs) == 1

    def test_format_report(self):
        """Test report formatting."""
        report = RegressionReport(
            timestamp=datetime.now(),
            baseline_timestamp=datetime.now(),
            regressions=[
                Regression(
                    type=RegressionType.NEW_FAILURE,
                    severity=RegressionSeverity.HIGH,
                    test_id="t1",
                    description="Test failed",
                    baseline_value="PASS",
                    current_value="FAIL",
                ),
            ],
            improvements=[],
        )

        formatted = report.format_report()

        assert "REGRESSION ANALYSIS REPORT" in formatted
        assert "Test failed" in formatted
        assert "BLOCKED" in formatted


class TestRegressionDetector:
    """Tests for RegressionDetector."""

    def test_detect_new_failure(self):
        """Test detection of new failures."""
        baseline = PromptfooResult(
            timestamp=datetime.now(),
            config_path="test.yaml",
            tests=[
                TestResult(test_id="t1", description="T1", pass_=True, score=1.0, assertions=[]),
            ],
        )
        current = PromptfooResult(
            timestamp=datetime.now(),
            config_path="test.yaml",
            tests=[
                TestResult(test_id="t1", description="T1", pass_=False, score=0.0, assertions=[]),
            ],
        )

        detector = RegressionDetector()
        report = detector.compare(baseline, current)

        assert report.has_regressions
        assert len(report.regressions) >= 1
        new_failures = report.get_regressions_by_type(RegressionType.NEW_FAILURE)
        assert len(new_failures) == 1

    def test_detect_score_drop(self):
        """Test detection of score drops."""
        baseline = PromptfooResult(
            timestamp=datetime.now(),
            config_path="test.yaml",
            tests=[
                TestResult(test_id="t1", description="T1", pass_=True, score=1.0, assertions=[]),
            ],
        )
        current = PromptfooResult(
            timestamp=datetime.now(),
            config_path="test.yaml",
            tests=[
                TestResult(test_id="t1", description="T1", pass_=True, score=0.5, assertions=[]),
            ],
        )

        detector = RegressionDetector()
        report = detector.compare(baseline, current)

        score_drops = report.get_regressions_by_type(RegressionType.SCORE_DROP)
        assert len(score_drops) >= 1

    def test_detect_pass_rate_drop(self):
        """Test detection of pass rate drops."""
        baseline = PromptfooResult(
            timestamp=datetime.now(),
            config_path="test.yaml",
            tests=[
                TestResult(test_id="t1", description="T1", pass_=True, score=1.0, assertions=[]),
                TestResult(test_id="t2", description="T2", pass_=True, score=1.0, assertions=[]),
            ],
        )
        current = PromptfooResult(
            timestamp=datetime.now(),
            config_path="test.yaml",
            tests=[
                TestResult(test_id="t1", description="T1", pass_=True, score=1.0, assertions=[]),
                TestResult(test_id="t2", description="T2", pass_=False, score=0.0, assertions=[]),
            ],
        )

        detector = RegressionDetector()
        report = detector.compare(baseline, current)

        pass_rate_drops = report.get_regressions_by_type(RegressionType.PASS_RATE_DROP)
        # 50% drop (100% -> 50%) exceeds 5% threshold
        assert len(pass_rate_drops) >= 1

    def test_detect_latency_increase(self):
        """Test detection of latency increases."""
        baseline = PromptfooResult(
            timestamp=datetime.now(),
            config_path="test.yaml",
            tests=[
                TestResult(test_id="t1", description="T1", pass_=True, score=1.0, assertions=[], latency_ms=100),
            ],
        )
        current = PromptfooResult(
            timestamp=datetime.now(),
            config_path="test.yaml",
            tests=[
                TestResult(test_id="t1", description="T1", pass_=True, score=1.0, assertions=[], latency_ms=200),
            ],
        )

        detector = RegressionDetector()
        report = detector.compare(baseline, current)

        latency_increases = report.get_regressions_by_type(RegressionType.LATENCY_INCREASE)
        # 100% increase exceeds 50% threshold
        assert len(latency_increases) >= 1

    def test_detect_improvements(self):
        """Test detection of improvements."""
        baseline = PromptfooResult(
            timestamp=datetime.now(),
            config_path="test.yaml",
            tests=[
                TestResult(test_id="t1", description="T1", pass_=True, score=0.5, assertions=[], latency_ms=200),
            ],
        )
        current = PromptfooResult(
            timestamp=datetime.now(),
            config_path="test.yaml",
            tests=[
                TestResult(test_id="t1", description="T1", pass_=True, score=1.0, assertions=[], latency_ms=100),
            ],
        )

        detector = RegressionDetector()
        report = detector.compare(baseline, current)

        assert report.improvement_count >= 1

    def test_custom_config(self):
        """Test detector with custom configuration."""
        config = DetectorConfig(
            score_drop_threshold=0.05,  # More sensitive
            latency_increase_threshold=0.2,  # More sensitive
            pass_rate_drop_threshold=2.0,  # More sensitive
        )
        detector = RegressionDetector(config)

        baseline = PromptfooResult(
            timestamp=datetime.now(),
            config_path="test.yaml",
            tests=[
                TestResult(test_id="t1", description="T1", pass_=True, score=1.0, assertions=[]),
            ],
        )
        current = PromptfooResult(
            timestamp=datetime.now(),
            config_path="test.yaml",
            tests=[
                TestResult(test_id="t1", description="T1", pass_=True, score=0.9, assertions=[]),
            ],
        )

        report = detector.compare(baseline, current)

        # 10% drop should trigger with 5% threshold
        score_drops = report.get_regressions_by_type(RegressionType.SCORE_DROP)
        assert len(score_drops) >= 1


class TestRegressionConvenienceFunctions:
    """Tests for regression convenience functions."""

    def test_compare_results(self):
        """Test compare_results function."""
        baseline = PromptfooResult(
            timestamp=datetime.now(),
            config_path="test.yaml",
            tests=[
                TestResult(test_id="t1", description="T1", pass_=True, score=1.0, assertions=[]),
            ],
        )
        current = PromptfooResult(
            timestamp=datetime.now(),
            config_path="test.yaml",
            tests=[
                TestResult(test_id="t1", description="T1", pass_=False, score=0.0, assertions=[]),
            ],
        )

        report = compare_results(baseline, current)
        assert report.has_regressions

    def test_check_for_blockers(self):
        """Test check_for_blockers function."""
        blocking_report = RegressionReport(
            timestamp=datetime.now(),
            baseline_timestamp=None,
            regressions=[
                Regression(
                    type=RegressionType.NEW_FAILURE,
                    severity=RegressionSeverity.CRITICAL,
                    test_id="t1",
                    description="Critical",
                    baseline_value="PASS",
                    current_value="FAIL",
                ),
            ],
            improvements=[],
        )
        assert check_for_blockers(blocking_report) is True

        non_blocking_report = RegressionReport(
            timestamp=datetime.now(),
            baseline_timestamp=None,
            regressions=[
                Regression(
                    type=RegressionType.SCORE_DROP,
                    severity=RegressionSeverity.LOW,
                    test_id="t1",
                    description="Low",
                    baseline_value="1.0",
                    current_value="0.95",
                ),
            ],
            improvements=[],
        )
        assert check_for_blockers(non_blocking_report) is False

    def test_get_regression_summary(self):
        """Test get_regression_summary function."""
        report = RegressionReport(
            timestamp=datetime.now(),
            baseline_timestamp=None,
            regressions=[
                Regression(
                    type=RegressionType.NEW_FAILURE,
                    severity=RegressionSeverity.HIGH,
                    test_id="t1",
                    description="High",
                    baseline_value="PASS",
                    current_value="FAIL",
                ),
                Regression(
                    type=RegressionType.SCORE_DROP,
                    severity=RegressionSeverity.MEDIUM,
                    test_id="t2",
                    description="Medium",
                    baseline_value="1.0",
                    current_value="0.5",
                ),
            ],
            improvements=[],
        )

        summary = get_regression_summary(report)
        assert "1 high" in summary
        assert "1 medium" in summary

    def test_get_regression_summary_no_regressions(self):
        """Test summary when no regressions."""
        report = RegressionReport(
            timestamp=datetime.now(),
            baseline_timestamp=None,
            regressions=[],
            improvements=[],
        )

        summary = get_regression_summary(report)
        assert "No regressions detected" in summary


# =============================================================================
# Test: PromptfooRunner
# =============================================================================


class TestPromptfooRunner:
    """Tests for PromptfooRunner."""

    def test_check_installation_success(self):
        """Test checking promptfoo installation."""
        runner = PromptfooRunner()

        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0)
            result = runner.check_installation()
            assert result is True

    def test_check_installation_failure(self):
        """Test checking when promptfoo not installed."""
        runner = PromptfooRunner()

        with patch("subprocess.run") as mock_run:
            mock_run.side_effect = FileNotFoundError()
            result = runner.check_installation()
            assert result is False

    def test_run_success(self, tmp_path):
        """Test successful run."""
        runner = PromptfooRunner(working_dir=tmp_path)

        # Create mock output file
        output_file = tmp_path / "output.json"
        output_file.write_text('{"results": [], "stats": {}}')

        config = RunConfig(
            config_path="promptfoo.yaml",
            output_path="output.json",
        )

        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stderr="")
            result = runner.run(config)

            assert isinstance(result, PromptfooResult)

    def test_run_failure(self, tmp_path):
        """Test run failure."""
        runner = PromptfooRunner(working_dir=tmp_path)

        config = RunConfig(config_path="promptfoo.yaml")

        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=1, stderr="Error occurred")

            with pytest.raises(RuntimeError, match="Promptfoo failed"):
                runner.run(config)

    def test_run_timeout(self, tmp_path):
        """Test run timeout."""
        import subprocess
        runner = PromptfooRunner(working_dir=tmp_path)

        config = RunConfig(config_path="promptfoo.yaml", timeout_seconds=1)

        with patch("subprocess.run") as mock_run:
            mock_run.side_effect = subprocess.TimeoutExpired(cmd="promptfoo", timeout=1)

            with pytest.raises(RuntimeError, match="timed out"):
                runner.run(config)

    def test_run_from_file(self, tmp_path):
        """Test run_from_file convenience method."""
        runner = PromptfooRunner(working_dir=tmp_path)

        # Create mock output file
        output_file = tmp_path / "promptfoo_output.json"
        output_file.write_text('{"results": [], "stats": {}}')

        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stderr="")
            result = runner.run_from_file("config.yaml", verbose=True)

            assert isinstance(result, PromptfooResult)
