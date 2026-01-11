"""
Assertion builders for Promptfoo CDI evaluation.

Provides builders to create Promptfoo assertions from CDI evaluation criteria.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

from nuvii_eval.datasets.schemas import (
    BaseTestCase,
    EMTestCase,
    GapTestCase,
    HCCTestCase,
    ICDTestCase,
    QueryTestCase,
)
from nuvii_eval.promptfoo.config import PromptfooAssertion


# =============================================================================
# Base Assertion Builder
# =============================================================================


class AssertionBuilder(ABC):
    """
    Abstract base class for assertion builders.

    Converts CDI test case expectations into Promptfoo assertions.
    """

    @abstractmethod
    def build(self, test_case: BaseTestCase) -> list[PromptfooAssertion]:
        """
        Build assertions for a test case.

        Args:
            test_case: CDI test case

        Returns:
            List of Promptfoo assertions
        """
        pass

    def _create_contains_assertion(
        self,
        value: str,
        weight: float = 1.0,
    ) -> PromptfooAssertion:
        """Create a 'contains' assertion."""
        return PromptfooAssertion(
            type="contains",
            value=value,
            weight=weight,
        )

    def _create_not_contains_assertion(
        self,
        value: str,
        weight: float = 1.0,
    ) -> PromptfooAssertion:
        """Create a 'not-contains' assertion."""
        return PromptfooAssertion(
            type="not-contains",
            value=value,
            weight=weight,
        )

    def _create_regex_assertion(
        self,
        pattern: str,
        weight: float = 1.0,
    ) -> PromptfooAssertion:
        """Create a regex match assertion."""
        return PromptfooAssertion(
            type="regex",
            value=pattern,
            weight=weight,
        )

    def _create_json_assertion(
        self,
        path: str,
        value: Any,
        weight: float = 1.0,
    ) -> PromptfooAssertion:
        """Create a JSON path assertion."""
        return PromptfooAssertion(
            type="equals",
            value=f"json.{path}={value}",
            weight=weight,
        )

    def _create_llm_rubric_assertion(
        self,
        rubric: str,
        threshold: float = 0.7,
        weight: float = 1.0,
    ) -> PromptfooAssertion:
        """Create an LLM-based rubric assertion."""
        return PromptfooAssertion(
            type="llm-rubric",
            value=rubric,
            threshold=threshold,
            weight=weight,
        )

    def _create_javascript_assertion(
        self,
        code: str,
        weight: float = 1.0,
    ) -> PromptfooAssertion:
        """Create a JavaScript evaluation assertion."""
        return PromptfooAssertion(
            type="javascript",
            value=code,
            weight=weight,
        )

    def _create_python_assertion(
        self,
        code: str,
        weight: float = 1.0,
    ) -> PromptfooAssertion:
        """Create a Python evaluation assertion."""
        return PromptfooAssertion(
            type="python",
            value=code,
            weight=weight,
        )


# =============================================================================
# ICD Assertion Builder
# =============================================================================


class ICDAssertionBuilder(AssertionBuilder):
    """
    Builds assertions for ICD-10 code evaluation.

    Assertions check:
    - Expected codes are present
    - Unacceptable codes are absent
    - Primary code is first (if applicable)
    - Code format is valid
    """

    def build(self, test_case: ICDTestCase) -> list[PromptfooAssertion]:
        """Build ICD-specific assertions."""
        assertions = []

        # Assert expected codes are present
        for code in test_case.expected_icd_codes:
            assertions.append(self._create_contains_assertion(
                value=code,
                weight=1.5,  # Higher weight for expected codes
            ))

        # Assert unacceptable codes are absent
        for code in test_case.unacceptable_codes:
            assertions.append(self._create_not_contains_assertion(
                value=code,
                weight=2.0,  # Heavy penalty for wrong codes
            ))

        # Assert valid ICD-10 code format in response
        assertions.append(self._create_regex_assertion(
            pattern=r"[A-TV-Z]\d{2}(\.\d{1,4})?",
            weight=0.5,
        ))

        # If primary code specified and sequence matters
        if test_case.code_sequence_matters and test_case.primary_code:
            # Primary code should appear first
            assertions.append(self._create_llm_rubric_assertion(
                rubric=f"The code {test_case.primary_code} should be listed first as the primary diagnosis",
                threshold=0.8,
                weight=1.0,
            ))

        # LLM rubric for overall accuracy
        assertions.append(self._create_llm_rubric_assertion(
            rubric=f"The suggested ICD-10 codes should accurately represent the diagnoses documented in the clinical note. Expected codes include: {', '.join(test_case.expected_icd_codes)}",
            threshold=0.7,
            weight=1.0,
        ))

        return assertions


# =============================================================================
# HCC Assertion Builder
# =============================================================================


class HCCAssertionBuilder(AssertionBuilder):
    """
    Builds assertions for HCC/RAF evaluation.

    Assertions check:
    - Expected HCCs are detected
    - RAF is within expected range
    - Opportunities are identified
    - Supersession rules are followed
    """

    def build(self, test_case: HCCTestCase) -> list[PromptfooAssertion]:
        """Build HCC-specific assertions."""
        assertions = []

        # Assert expected HCCs are present
        for hcc in test_case.expected_hccs:
            assertions.append(self._create_contains_assertion(
                value=hcc,
                weight=1.5,
            ))

        # Assert expected opportunities are identified
        for opp in test_case.expected_opportunities:
            assertions.append(self._create_contains_assertion(
                value=opp,
                weight=1.0,
            ))

        # Assert HCC code format
        assertions.append(self._create_regex_assertion(
            pattern=r"HCC\d{1,3}",
            weight=0.5,
        ))

        # RAF range check using JavaScript
        min_raf, max_raf = test_case.expected_raf_range
        raf_check_js = f"""
            const raf = parseFloat(output.match(/RAF[:\\s]+([\\d.]+)/i)?.[1] || '0');
            return raf >= {min_raf} && raf <= {max_raf};
        """
        assertions.append(self._create_javascript_assertion(
            code=raf_check_js.strip(),
            weight=1.0,
        ))

        # LLM rubric for comprehensive HCC analysis
        assertions.append(self._create_llm_rubric_assertion(
            rubric=f"The HCC analysis should identify the documented conditions and their corresponding HCC codes. Expected HCCs: {', '.join(test_case.expected_hccs)}. The RAF score should be between {min_raf} and {max_raf}.",
            threshold=0.7,
            weight=1.0,
        ))

        return assertions


# =============================================================================
# Gap Assertion Builder
# =============================================================================


class GapAssertionBuilder(AssertionBuilder):
    """
    Builds assertions for documentation gap evaluation.

    Assertions check:
    - Expected gaps are detected
    - False positive conditions are not flagged
    - Gap types are correct
    - Priority ordering is appropriate
    """

    def build(self, test_case: GapTestCase) -> list[PromptfooAssertion]:
        """Build gap detection assertions."""
        assertions = []

        # Handle no-gaps-expected case
        if test_case.no_gaps_expected:
            assertions.append(self._create_llm_rubric_assertion(
                rubric="No documentation gaps should be identified. The clinical documentation is complete.",
                threshold=0.8,
                weight=2.0,
            ))
            return assertions

        # Assert expected gaps are detected
        for gap in test_case.expected_gaps:
            # Check condition is mentioned
            assertions.append(self._create_contains_assertion(
                value=gap.condition,
                weight=1.5,
            ))

            # Check gap type is mentioned
            assertions.append(self._create_llm_rubric_assertion(
                rubric=f"A {gap.gap_type} gap should be identified for {gap.condition}",
                threshold=0.7,
                weight=1.0,
            ))

        # Assert false positives are not flagged
        for fp_condition in test_case.false_positive_conditions:
            assertions.append(self._create_not_contains_assertion(
                value=fp_condition,
                weight=2.0,  # Heavy penalty for false positives
            ))

        # LLM rubric for overall gap analysis
        gap_descriptions = [f"{g.condition} ({g.gap_type})" for g in test_case.expected_gaps]
        assertions.append(self._create_llm_rubric_assertion(
            rubric=f"The gap analysis should identify the following documentation gaps: {', '.join(gap_descriptions)}. Each gap should have supporting evidence from the clinical note.",
            threshold=0.7,
            weight=1.0,
        ))

        return assertions


# =============================================================================
# Query Assertion Builder
# =============================================================================


class QueryAssertionBuilder(AssertionBuilder):
    """
    Builds assertions for CDI query quality evaluation.

    Assertions check:
    - Required terms are mentioned
    - Forbidden terms are absent (no leading language)
    - Query is non-leading
    - Evidence is cited
    - Response options are provided
    """

    def build(self, test_case: QueryTestCase) -> list[PromptfooAssertion]:
        """Build query quality assertions."""
        assertions = []
        criteria = test_case.quality_criteria

        # Assert required terms are mentioned
        for term in criteria.must_mention:
            assertions.append(self._create_contains_assertion(
                value=term,
                weight=1.0,
            ))

        # Assert forbidden terms are absent
        for term in criteria.must_not_mention:
            assertions.append(self._create_not_contains_assertion(
                value=term,
                weight=2.0,
            ))

        # Check for leading language patterns
        leading_patterns = [
            "please confirm",
            "do you agree",
            "obviously",
            "clearly",
            "definitely has",
            "must have",
        ]
        for pattern in leading_patterns:
            assertions.append(self._create_not_contains_assertion(
                value=pattern,
                weight=1.5,
            ))

        # LLM rubric for non-leading language
        assertions.append(self._create_llm_rubric_assertion(
            rubric="The query should use non-leading language that does not assume a diagnosis. It should ask for clarification without suggesting the answer.",
            threshold=0.8,
            weight=2.0,
        ))

        # LLM rubric for evidence citation
        assertions.append(self._create_llm_rubric_assertion(
            rubric="The query should cite specific evidence from the clinical documentation to support the clarification request.",
            threshold=0.7,
            weight=1.0,
        ))

        # Check for response options if required
        if criteria.must_provide_options:
            assertions.append(self._create_llm_rubric_assertion(
                rubric="The query should provide response options for the provider to select from.",
                threshold=0.7,
                weight=0.5,
            ))

        # Check query type if expected
        if criteria.expected_query_type:
            assertions.append(self._create_llm_rubric_assertion(
                rubric=f"This should be a {criteria.expected_query_type} query appropriate for the identified gap.",
                threshold=0.7,
                weight=0.5,
            ))

        return assertions


# =============================================================================
# E/M Assertion Builder
# =============================================================================


class EMAssertionBuilder(AssertionBuilder):
    """
    Builds assertions for E/M level evaluation.

    Assertions check:
    - Correct E/M code is recommended
    - Level is within acceptable range
    - MDM components are appropriate
    - No upcoding
    """

    def build(self, test_case: EMTestCase) -> list[PromptfooAssertion]:
        """Build E/M level assertions."""
        assertions = []

        # Assert expected code is recommended
        assertions.append(self._create_contains_assertion(
            value=test_case.expected_code,
            weight=1.5,
        ))

        # Also accept acceptable alternatives
        for code in test_case.acceptable_codes:
            # These don't fail if missing, but pass if present
            pass  # Acceptable codes handled by overall rubric

        # Assert valid E/M code format
        assertions.append(self._create_regex_assertion(
            pattern=r"99\d{3}",
            weight=0.5,
        ))

        # Check for level mention
        assertions.append(self._create_regex_assertion(
            pattern=rf"level\s*{test_case.expected_level}",
            weight=0.5,
        ))

        # LLM rubric for MDM accuracy
        mdm = test_case.expected_mdm
        assertions.append(self._create_llm_rubric_assertion(
            rubric=f"The Medical Decision Making (MDM) analysis should reflect: Problems - {mdm.problems}/4, Data - {mdm.data}/4, Risk - {mdm.risk}/4. The overall MDM level should support a level {test_case.expected_level} visit.",
            threshold=0.7,
            weight=1.0,
        ))

        # LLM rubric for no upcoding
        assertions.append(self._create_llm_rubric_assertion(
            rubric="The recommended E/M level should be supported by the documentation and not represent upcoding beyond what is clinically justified.",
            threshold=0.8,
            weight=2.0,
        ))

        # Overall E/M accuracy rubric
        acceptable_codes = [test_case.expected_code] + test_case.acceptable_codes
        assertions.append(self._create_llm_rubric_assertion(
            rubric=f"The E/M level recommendation should be accurate for this {test_case.encounter_type} encounter. Acceptable codes: {', '.join(acceptable_codes)}",
            threshold=0.7,
            weight=1.0,
        ))

        return assertions


# =============================================================================
# Factory Function
# =============================================================================


def get_assertion_builder(task_type: str) -> AssertionBuilder:
    """
    Get the appropriate assertion builder for a task type.

    Args:
        task_type: Type of CDI task (icd, hcc, gap, query, em)

    Returns:
        Configured assertion builder

    Raises:
        ValueError: If task type is unknown
    """
    builders = {
        "icd": ICDAssertionBuilder,
        "hcc": HCCAssertionBuilder,
        "gap": GapAssertionBuilder,
        "query": QueryAssertionBuilder,
        "em": EMAssertionBuilder,
    }

    if task_type not in builders:
        raise ValueError(f"Unknown task type: {task_type}")

    return builders[task_type]()
