"""
Test case converter for Promptfoo.

Converts CDI test cases to Promptfoo test format.
"""

from typing import Any

import structlog

from nuvii_eval.datasets.schemas import (
    BaseTestCase,
    EMTestCase,
    GapTestCase,
    HCCTestCase,
    ICDTestCase,
    QueryTestCase,
)
from nuvii_eval.promptfoo.assertions import get_assertion_builder
from nuvii_eval.promptfoo.config import PromptfooTest

logger = structlog.get_logger(__name__)


# =============================================================================
# Test Case Converter
# =============================================================================


class TestCaseConverter:
    """
    Converts CDI test cases to Promptfoo test format.

    Handles conversion of test case data and assertions.
    """

    def __init__(self, task_type: str):
        """
        Initialize the converter.

        Args:
            task_type: Type of CDI task (icd, hcc, gap, query, em)
        """
        self.task_type = task_type
        self.assertion_builder = get_assertion_builder(task_type)

    def convert(self, test_case: BaseTestCase) -> PromptfooTest:
        """
        Convert a single test case to Promptfoo format.

        Args:
            test_case: CDI test case to convert

        Returns:
            PromptfooTest configuration
        """
        # Build variables
        vars_data = self._build_vars(test_case)

        # Build assertions
        assertions = self.assertion_builder.build(test_case)

        # Build metadata
        metadata = self._build_metadata(test_case)

        return PromptfooTest(
            description=self._build_description(test_case),
            vars=vars_data,
            assert_=assertions,
            metadata=metadata,
        )

    def convert_batch(self, test_cases: list[BaseTestCase]) -> list[PromptfooTest]:
        """
        Convert multiple test cases.

        Args:
            test_cases: List of CDI test cases

        Returns:
            List of PromptfooTest configurations
        """
        return [self.convert(tc) for tc in test_cases]

    def _build_description(self, test_case: BaseTestCase) -> str:
        """Build a description for the test case."""
        base_desc = f"[{test_case.id}]"

        # Add specialty and complexity
        if hasattr(test_case, "specialty"):
            base_desc += f" {test_case.specialty.value}"
        if hasattr(test_case, "complexity"):
            base_desc += f" ({test_case.complexity.value})"

        return base_desc

    def _build_vars(self, test_case: BaseTestCase) -> dict[str, Any]:
        """Build variables dictionary for the test case."""
        vars_data: dict[str, Any] = {
            "clinical_note": test_case.clinical_note,
            "test_case_id": test_case.id,
        }

        # Add task-specific variables
        if isinstance(test_case, ICDTestCase):
            vars_data.update(self._build_icd_vars(test_case))
        elif isinstance(test_case, HCCTestCase):
            vars_data.update(self._build_hcc_vars(test_case))
        elif isinstance(test_case, GapTestCase):
            vars_data.update(self._build_gap_vars(test_case))
        elif isinstance(test_case, QueryTestCase):
            vars_data.update(self._build_query_vars(test_case))
        elif isinstance(test_case, EMTestCase):
            vars_data.update(self._build_em_vars(test_case))

        return vars_data

    def _build_icd_vars(self, test_case: ICDTestCase) -> dict[str, Any]:
        """Build ICD-specific variables."""
        return {
            "expected_codes": test_case.expected_icd_codes,
            "acceptable_codes": test_case.acceptable_icd_codes,
            "primary_code": test_case.primary_code,
            "options": {
                "max_codes": 10,
                "include_confidence": True,
            },
        }

    def _build_hcc_vars(self, test_case: HCCTestCase) -> dict[str, Any]:
        """Build HCC-specific variables."""
        return {
            "expected_hccs": test_case.expected_hccs,
            "expected_opportunities": test_case.expected_opportunities,
            "patient_age": test_case.patient_age,
            "patient_gender": test_case.patient_gender,
            "model_year": test_case.model_year,
            "options": {
                "include_opportunities": True,
                "include_raf": True,
            },
        }

    def _build_gap_vars(self, test_case: GapTestCase) -> dict[str, Any]:
        """Build gap detection variables."""
        expected_gaps = [
            {
                "condition": g.condition,
                "gap_type": g.gap_type,
                "min_priority": g.min_priority,
            }
            for g in test_case.expected_gaps
        ]
        return {
            "expected_gaps": expected_gaps,
            "false_positives": test_case.false_positive_conditions,
            "options": {
                "include_queries": True,
                "max_gaps": 10,
            },
        }

    def _build_query_vars(self, test_case: QueryTestCase) -> dict[str, Any]:
        """Build query-specific variables."""
        return {
            "gap": {
                "condition": test_case.gap.condition,
                "gap_type": test_case.gap.gap_type,
            },
            "quality_criteria": {
                "must_mention": test_case.quality_criteria.must_mention,
                "must_not_mention": test_case.quality_criteria.must_not_mention,
                "expected_query_type": test_case.quality_criteria.expected_query_type,
            },
            "reference_query": test_case.reference_query,
            "options": {
                "include_options": True,
                "include_evidence": True,
            },
        }

    def _build_em_vars(self, test_case: EMTestCase) -> dict[str, Any]:
        """Build E/M-specific variables."""
        return {
            "encounter_type": test_case.encounter_type,
            "patient_type": test_case.patient_type,
            "expected_code": test_case.expected_code,
            "expected_level": test_case.expected_level,
            "documented_time": test_case.documented_time,
            "options": {
                "include_mdm": True,
                "include_time_based": test_case.time_based_acceptable,
            },
        }

    def _build_metadata(self, test_case: BaseTestCase) -> dict[str, Any]:
        """Build metadata for the test case."""
        metadata = {
            "test_case_id": test_case.id,
            "task_type": self.task_type,
        }

        if hasattr(test_case, "specialty"):
            metadata["specialty"] = test_case.specialty.value
        if hasattr(test_case, "complexity"):
            metadata["complexity"] = test_case.complexity.value
        if test_case.tags:
            metadata["tags"] = test_case.tags
        if test_case.source:
            metadata["source"] = test_case.source

        return metadata


# =============================================================================
# Convenience Functions
# =============================================================================


def convert_single_test(
    test_case: BaseTestCase,
    task_type: str,
) -> PromptfooTest:
    """
    Convert a single test case to Promptfoo format.

    Args:
        test_case: CDI test case
        task_type: Type of CDI task

    Returns:
        PromptfooTest configuration
    """
    converter = TestCaseConverter(task_type)
    return converter.convert(test_case)


def convert_test_suite(
    test_cases: list[BaseTestCase],
    task_type: str,
) -> list[PromptfooTest]:
    """
    Convert a suite of test cases to Promptfoo format.

    Args:
        test_cases: List of CDI test cases
        task_type: Type of CDI task

    Returns:
        List of PromptfooTest configurations
    """
    converter = TestCaseConverter(task_type)
    return converter.convert_batch(test_cases)


def export_to_yaml(
    test_cases: list[BaseTestCase],
    task_type: str,
    output_path: str,
    description: str = "CDI Evaluation Suite",
) -> str:
    """
    Export test cases to a Promptfoo YAML file.

    Args:
        test_cases: List of CDI test cases
        task_type: Type of CDI task
        output_path: Path to write YAML file
        description: Description for the test suite

    Returns:
        Path to the generated file
    """
    from nuvii_eval.promptfoo.config import (
        ConfigGeneratorOptions,
        generate_promptfoo_config,
    )

    # Convert test cases
    promptfoo_tests = convert_test_suite(test_cases, task_type)

    # Generate configuration
    options = ConfigGeneratorOptions(
        description=description,
        output_path=f"{task_type}_results.json",
    )
    config = generate_promptfoo_config(promptfoo_tests, task_type, options)

    # Save to file
    config.save(output_path)

    return output_path
