"""
Pydantic schemas for evaluation test cases.

Defines the structure of test case datasets used for evaluating
the Nuvii CDI Agent across different task types.
"""

import re
from enum import Enum
from typing import Annotated, Any, Literal

from pydantic import BaseModel, Field, field_validator, model_validator


class Specialty(str, Enum):
    """Medical specialty classification."""

    CARDIOLOGY = "cardiology"
    ENDOCRINOLOGY = "endocrinology"
    PULMONOLOGY = "pulmonology"
    NEPHROLOGY = "nephrology"
    ONCOLOGY = "oncology"
    NEUROLOGY = "neurology"
    GASTROENTEROLOGY = "gastroenterology"
    RHEUMATOLOGY = "rheumatology"
    INFECTIOUS_DISEASE = "infectious_disease"
    HEMATOLOGY = "hematology"
    GENERAL = "general"
    EMERGENCY = "emergency"
    PRIMARY_CARE = "primary_care"
    SURGERY = "surgery"
    ORTHOPEDICS = "orthopedics"
    PSYCHIATRY = "psychiatry"


class Complexity(str, Enum):
    """Test case complexity level."""

    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"


# ICD-10 code pattern: letter followed by 2 digits, optional decimal with 1-4 alphanumeric characters
# Supports: A00, E11.9, T36.0X1, S72.001A, Z99.89
# Range includes U (COVID codes) and excludes only specific reserved letters
ICD10_PATTERN = re.compile(r"^[A-Z]\d{2}(\.[A-Z0-9]{1,4})?$")

# HCC code pattern: HCC followed by 1-3 digits
HCC_PATTERN = re.compile(r"^HCC\d{1,3}$")

# CPT code pattern: 5 digits, optionally with modifier
CPT_PATTERN = re.compile(r"^\d{5}(-\d{2})?$")


def validate_icd10_code(code: str) -> str:
    """Validate ICD-10 code format."""
    if not ICD10_PATTERN.match(code):
        raise ValueError(f"Invalid ICD-10 format: {code}. Expected format: X00.0000")
    return code


def validate_hcc_code(code: str) -> str:
    """Validate HCC code format."""
    if not HCC_PATTERN.match(code):
        raise ValueError(f"Invalid HCC format: {code}. Expected format: HCC00")
    return code


def validate_cpt_code(code: str) -> str:
    """Validate CPT code format."""
    if not CPT_PATTERN.match(code):
        raise ValueError(f"Invalid CPT format: {code}. Expected format: 00000")
    return code


# =============================================================================
# Base Test Case
# =============================================================================


class BaseTestCase(BaseModel):
    """
    Base schema for all test cases.

    All test case types inherit from this base and add task-specific fields.
    """

    id: str = Field(
        ...,
        min_length=1,
        max_length=100,
        description="Unique test case identifier",
    )
    clinical_note: str = Field(
        ...,
        min_length=50,
        description="Clinical note text for evaluation",
    )
    specialty: Specialty = Field(
        default=Specialty.GENERAL,
        description="Medical specialty of the case",
    )
    complexity: Complexity = Field(
        default=Complexity.MODERATE,
        description="Complexity level of the case",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata for the test case",
    )
    tags: list[str] = Field(
        default_factory=list,
        description="Tags for filtering test cases",
    )
    source: str | None = Field(
        default=None,
        description="Source of the test case (e.g., 'synthetic', 'curated')",
    )

    @field_validator("id")
    @classmethod
    def validate_id_format(cls, v: str) -> str:
        """Ensure ID is alphanumeric with underscores/hyphens."""
        if not re.match(r"^[a-zA-Z0-9_-]+$", v):
            raise ValueError("ID must be alphanumeric with underscores/hyphens only")
        return v


# =============================================================================
# ICD Test Case
# =============================================================================


class ICDTestCase(BaseTestCase):
    """
    Test case for ICD-10 code suggestion evaluation.

    Evaluates the accuracy of ICD-10 code suggestions from the CDI agent.
    """

    expected_icd_codes: Annotated[list[str], Field(min_length=1)] = Field(
        ...,
        description="Primary expected ICD-10 codes (must be present)",
    )
    acceptable_icd_codes: list[str] = Field(
        default_factory=list,
        description="Alternative acceptable codes (also count as correct)",
    )
    unacceptable_codes: list[str] = Field(
        default_factory=list,
        description="Codes that should NOT be suggested (explicit false positives)",
    )
    primary_code: str | None = Field(
        default=None,
        description="The single primary diagnosis code (if applicable)",
    )
    code_sequence_matters: bool = Field(
        default=False,
        description="Whether the order of codes matters for evaluation",
    )

    @field_validator("expected_icd_codes", "acceptable_icd_codes", "unacceptable_codes")
    @classmethod
    def validate_icd_codes(cls, v: list[str]) -> list[str]:
        """Validate all ICD-10 codes in the list."""
        return [validate_icd10_code(code) for code in v]

    @field_validator("primary_code")
    @classmethod
    def validate_primary_code(cls, v: str | None) -> str | None:
        """Validate primary code if provided."""
        if v is not None:
            return validate_icd10_code(v)
        return v

    @model_validator(mode="after")
    def validate_code_consistency(self) -> "ICDTestCase":
        """Ensure primary code is in expected codes."""
        if self.primary_code and self.primary_code not in self.expected_icd_codes:
            raise ValueError("primary_code must be in expected_icd_codes")
        return self


# =============================================================================
# HCC Test Case
# =============================================================================


class HCCTestCase(BaseTestCase):
    """
    Test case for HCC/RAF evaluation.

    Evaluates HCC code detection and RAF score accuracy.
    """

    expected_hccs: Annotated[list[str], Field(min_length=1)] = Field(
        ...,
        description="Expected HCC codes to be captured",
    )
    expected_raf_range: tuple[float, float] = Field(
        ...,
        description="Expected RAF score range (min, max)",
    )
    expected_opportunities: list[str] = Field(
        default_factory=list,
        description="Expected HCC opportunities to be detected",
    )
    patient_age: int = Field(
        ...,
        ge=0,
        le=120,
        description="Patient age in years",
    )
    patient_gender: Literal["M", "F"] = Field(
        ...,
        description="Patient gender",
    )
    is_dual_eligible: bool = Field(
        default=False,
        description="Whether patient is dual Medicare/Medicaid eligible",
    )
    model_year: str = Field(
        default="2024",
        description="CMS-HCC model year",
    )

    @field_validator("expected_hccs", "expected_opportunities")
    @classmethod
    def validate_hcc_codes(cls, v: list[str]) -> list[str]:
        """Validate all HCC codes in the list."""
        return [validate_hcc_code(code) for code in v]

    @field_validator("expected_raf_range")
    @classmethod
    def validate_raf_range(cls, v: tuple[float, float]) -> tuple[float, float]:
        """Ensure RAF range is valid."""
        min_raf, max_raf = v
        if min_raf < 0:
            raise ValueError("RAF minimum cannot be negative")
        if max_raf < min_raf:
            raise ValueError("RAF maximum must be >= minimum")
        return v


# =============================================================================
# Gap Test Case
# =============================================================================


class ExpectedGap(BaseModel):
    """Expected gap in a test case."""

    gap_type: str = Field(
        ...,
        description="Type of gap expected",
        examples=["missing_specificity", "unconfirmed_diagnosis", "missing_laterality"],
    )
    condition: str = Field(
        ...,
        description="Clinical condition related to the gap",
    )
    min_priority: int = Field(
        default=3,
        ge=1,
        le=5,
        description="Minimum expected priority (1=highest)",
    )
    expected_icd_codes: list[str] = Field(
        default_factory=list,
        description="ICD codes that should be suggested for this gap",
    )

    @field_validator("expected_icd_codes")
    @classmethod
    def validate_icd_codes(cls, v: list[str]) -> list[str]:
        """Validate ICD codes if provided."""
        return [validate_icd10_code(code) for code in v]


class GapTestCase(BaseTestCase):
    """
    Test case for gap detection evaluation.

    Evaluates the accuracy of documentation gap detection.
    """

    expected_gaps: Annotated[list[ExpectedGap], Field(min_length=1)] = Field(
        ...,
        description="Gaps expected to be detected",
    )
    false_positive_conditions: list[str] = Field(
        default_factory=list,
        description="Conditions that should NOT be flagged as gaps",
    )
    no_gaps_expected: bool = Field(
        default=False,
        description="If True, no gaps should be detected",
    )

    @model_validator(mode="after")
    def validate_gaps_consistency(self) -> "GapTestCase":
        """Ensure consistency between gaps and no_gaps flag."""
        if self.no_gaps_expected and self.expected_gaps:
            raise ValueError("Cannot have expected_gaps when no_gaps_expected is True")
        return self


# =============================================================================
# Query Test Case
# =============================================================================


class QueryQualityCriteria(BaseModel):
    """Quality criteria for evaluating CDI queries."""

    must_mention: list[str] = Field(
        default_factory=list,
        description="Terms that must appear in the query",
    )
    must_not_mention: list[str] = Field(
        default_factory=list,
        description="Terms that must NOT appear (leading language, etc.)",
    )
    expected_query_type: str | None = Field(
        default=None,
        description="Expected query type (clarification, confirmation, etc.)",
    )
    min_evidence_citations: int = Field(
        default=1,
        ge=0,
        description="Minimum number of evidence citations required",
    )
    must_provide_options: bool = Field(
        default=True,
        description="Whether query must provide response options",
    )
    max_length: int | None = Field(
        default=None,
        ge=50,
        description="Maximum query length in characters",
    )


class QueryTestCase(BaseTestCase):
    """
    Test case for CDI query quality evaluation.

    Evaluates the quality of generated provider queries.
    """

    gap: ExpectedGap = Field(
        ...,
        description="The gap this query should address",
    )
    quality_criteria: QueryQualityCriteria = Field(
        ...,
        description="Quality criteria for evaluating the query",
    )
    reference_query: str | None = Field(
        default=None,
        description="Gold standard reference query for comparison",
    )
    context_requirements: list[str] = Field(
        default_factory=list,
        description="Required context elements in the query",
    )


# =============================================================================
# E/M Test Case
# =============================================================================


class ExpectedMDM(BaseModel):
    """Expected MDM component scores."""

    problems: int = Field(..., ge=1, le=4, description="Problems addressed")
    data: int = Field(..., ge=1, le=4, description="Data reviewed")
    risk: int = Field(..., ge=1, le=4, description="Risk level")


class EMTestCase(BaseTestCase):
    """
    Test case for E/M level evaluation.

    Evaluates the accuracy of E/M level determination.
    """

    encounter_type: Literal["outpatient", "inpatient", "observation", "ed", "telehealth"] = Field(
        ...,
        description="Type of encounter",
    )
    patient_type: Literal["new", "established"] = Field(
        default="established",
        description="Whether patient is new or established",
    )
    documented_time: int | None = Field(
        default=None,
        ge=0,
        description="Documented time in minutes (for time-based billing)",
    )
    expected_code: str = Field(
        ...,
        description="Expected CPT E/M code",
    )
    expected_level: int = Field(
        ...,
        ge=1,
        le=5,
        description="Expected E/M level (1-5)",
    )
    expected_mdm: ExpectedMDM = Field(
        ...,
        description="Expected MDM component scores",
    )
    acceptable_codes: list[str] = Field(
        default_factory=list,
        description="Alternative acceptable E/M codes",
    )
    time_based_acceptable: bool = Field(
        default=False,
        description="Whether time-based coding is acceptable",
    )

    @field_validator("expected_code", "acceptable_codes")
    @classmethod
    def validate_cpt_codes(cls, v: str | list[str]) -> str | list[str]:
        """Validate CPT codes."""
        if isinstance(v, list):
            return [validate_cpt_code(code) for code in v]
        return validate_cpt_code(v)


# =============================================================================
# CPT Test Case
# =============================================================================


class CPTTestCase(BaseTestCase):
    """
    Test case for CPT procedure code evaluation.

    Evaluates the accuracy of CPT code suggestions for procedures.
    """

    procedure_description: str = Field(
        ...,
        description="Description of the procedure performed",
    )
    expected_cpt_codes: Annotated[list[str], Field(min_length=1)] = Field(
        ...,
        description="Expected CPT codes",
    )
    acceptable_cpt_codes: list[str] = Field(
        default_factory=list,
        description="Alternative acceptable CPT codes",
    )
    expected_modifiers: list[str] = Field(
        default_factory=list,
        description="Expected modifiers (e.g., '25', '59')",
    )
    unacceptable_codes: list[str] = Field(
        default_factory=list,
        description="Codes that should NOT be suggested",
    )

    @field_validator("expected_cpt_codes", "acceptable_cpt_codes", "unacceptable_codes")
    @classmethod
    def validate_cpt_codes(cls, v: list[str]) -> list[str]:
        """Validate CPT codes."""
        return [validate_cpt_code(code) for code in v]


# =============================================================================
# Type mapping for loader
# =============================================================================

TEST_CASE_TYPES = {
    "icd": ICDTestCase,
    "hcc": HCCTestCase,
    "gap": GapTestCase,
    "query": QueryTestCase,
    "em": EMTestCase,
    "cpt": CPTTestCase,
}
