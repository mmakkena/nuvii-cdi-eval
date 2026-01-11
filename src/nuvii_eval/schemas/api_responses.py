"""
Pydantic schemas for Nuvii CDI Agent V2 API responses.

These schemas define the structure of responses from the Nuvii CDI API,
enabling validation and type safety throughout the evaluation framework.
"""

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class ConfidenceLevel(str, Enum):
    """Confidence level for predictions."""

    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


# =============================================================================
# ICD Coding Schemas
# =============================================================================


class ICDSuggestion(BaseModel):
    """Single ICD-10 code suggestion from the coding API."""

    icd10_code: str = Field(
        ...,
        description="ICD-10-CM code",
        examples=["E11.65", "I50.22"],
    )
    description: str = Field(
        ...,
        description="Human-readable description of the code",
    )
    confidence: ConfidenceLevel = Field(
        ...,
        description="Confidence level of the suggestion",
    )
    evidence_spans: list[str] = Field(
        default_factory=list,
        description="Text spans from the note supporting this code",
    )
    hcc_code: str | None = Field(
        default=None,
        description="Associated HCC code if applicable",
    )
    raf_weight: float | None = Field(
        default=None,
        ge=0.0,
        description="RAF weight for risk adjustment",
    )
    specificity_level: int | None = Field(
        default=None,
        ge=3,
        le=7,
        description="Code specificity level (3=category, 7=max specificity)",
    )


class CodingSuggestResponse(BaseModel):
    """Response from /api/v2/coding/suggest endpoint."""

    request_id: str = Field(..., description="Unique request identifier")
    suggested_codes: list[ICDSuggestion] = Field(
        default_factory=list,
        description="List of suggested ICD-10 codes",
    )
    processing_time_ms: int = Field(
        ...,
        ge=0,
        description="Processing time in milliseconds",
    )
    model_version: str = Field(
        ...,
        description="Version of the model used",
    )
    token_count: int | None = Field(
        default=None,
        description="Total tokens used",
    )


# =============================================================================
# Gap Detection Schemas
# =============================================================================


class GapCandidate(BaseModel):
    """Detected documentation gap."""

    gap_id: str = Field(..., description="Unique gap identifier")
    gap_type: str = Field(
        ...,
        description="Type of gap",
        examples=["missing_specificity", "unconfirmed_diagnosis", "missing_laterality"],
    )
    condition: str = Field(
        ...,
        description="Clinical condition related to the gap",
    )
    current_evidence: list[str] = Field(
        default_factory=list,
        description="Evidence spans currently in the note",
    )
    suggested_icd_codes: list[str] = Field(
        default_factory=list,
        description="ICD codes that could result from addressing this gap",
    )
    priority: int = Field(
        ...,
        ge=1,
        le=5,
        description="Priority level (1=highest, 5=lowest)",
    )
    confidence: ConfidenceLevel = Field(
        ...,
        description="Confidence in the gap detection",
    )
    clinical_indicators: list[str] = Field(
        default_factory=list,
        description="Clinical indicators supporting this gap",
    )
    potential_raf_impact: float | None = Field(
        default=None,
        ge=0.0,
        description="Potential RAF score impact if gap is addressed",
    )


class GapDetectionResponse(BaseModel):
    """Response from /api/v2/cdi/gaps endpoint."""

    request_id: str = Field(..., description="Unique request identifier")
    gaps: list[GapCandidate] = Field(
        default_factory=list,
        description="List of detected gaps",
    )
    facts_cache_key: str = Field(
        ...,
        description="Cache key for extracted facts (use in subsequent calls)",
    )
    processing_time_ms: int = Field(
        ...,
        ge=0,
        description="Processing time in milliseconds",
    )
    extracted_conditions: list[str] | None = Field(
        default=None,
        description="Conditions extracted from the note",
    )


# =============================================================================
# Query Generation Schemas
# =============================================================================


class ProviderQuery(BaseModel):
    """Generated CDI query for provider."""

    query_id: str = Field(..., description="Unique query identifier")
    gap_id: str = Field(..., description="Associated gap ID")
    query_text: str = Field(
        ...,
        min_length=10,
        description="The query text to present to the provider",
    )
    query_type: str = Field(
        ...,
        description="Type of query",
        examples=["clarification", "confirmation", "specificity", "clinical_validation"],
    )
    evidence_cited: list[str] = Field(
        default_factory=list,
        description="Evidence from the note cited in the query",
    )
    suggested_responses: list[str] = Field(
        default_factory=list,
        description="Suggested response options for the provider",
    )
    icd_impact: list[str] = Field(
        default_factory=list,
        description="ICD codes affected by query response",
    )
    compliance_score: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="ACDIS compliance score",
    )


class QueryGenerationResponse(BaseModel):
    """Response from /api/v2/cdi/queries endpoint."""

    request_id: str = Field(..., description="Unique request identifier")
    queries: list[ProviderQuery] = Field(
        default_factory=list,
        description="Generated queries",
    )
    processing_time_ms: int = Field(
        ...,
        ge=0,
        description="Processing time in milliseconds",
    )
    gaps_cache_key: str | None = Field(
        default=None,
        description="Cache key used for gap context",
    )


# =============================================================================
# E/M Analysis Schemas
# =============================================================================


class MDMComponent(BaseModel):
    """Medical Decision Making component scores."""

    problems: int = Field(
        ...,
        ge=1,
        le=4,
        description="Number and complexity of problems (1-4 points)",
    )
    data: int = Field(
        ...,
        ge=1,
        le=4,
        description="Amount and complexity of data reviewed (1-4 points)",
    )
    risk: int = Field(
        ...,
        ge=1,
        le=4,
        description="Risk of complications/morbidity/mortality (1-4 points)",
    )

    @property
    def mdm_level(self) -> int:
        """Calculate MDM level (2 of 3 elements determine level)."""
        scores = sorted([self.problems, self.data, self.risk], reverse=True)
        return scores[1]  # Second highest determines level


class EMAnalysisResult(BaseModel):
    """E/M level analysis result."""

    request_id: str = Field(..., description="Unique request identifier")
    recommended_code: str = Field(
        ...,
        description="Recommended CPT code",
        examples=["99213", "99214", "99215"],
    )
    recommended_level: int = Field(
        ...,
        ge=1,
        le=5,
        description="Recommended E/M level",
    )
    mdm_score: MDMComponent = Field(
        ...,
        description="Medical Decision Making component breakdown",
    )
    time_based_code: str | None = Field(
        default=None,
        description="Alternative code if time-based billing is used",
    )
    documented_time: int | None = Field(
        default=None,
        ge=0,
        description="Documented time in minutes",
    )
    justification: str = Field(
        ...,
        description="Justification for the recommended level",
    )
    upcoding_risk: bool = Field(
        default=False,
        description="Flag if there's risk of upcoding",
    )
    downcoding_risk: bool = Field(
        default=False,
        description="Flag if there's risk of downcoding",
    )
    supporting_elements: dict[str, Any] = Field(
        default_factory=dict,
        description="Supporting documentation elements",
    )
    processing_time_ms: int = Field(
        ...,
        ge=0,
        description="Processing time in milliseconds",
    )


# =============================================================================
# HCC/Risk Analysis Schemas
# =============================================================================


class HCCOpportunity(BaseModel):
    """HCC capture opportunity."""

    hcc_code: str = Field(
        ...,
        description="HCC code",
        examples=["HCC18", "HCC85"],
    )
    hcc_description: str = Field(
        ...,
        description="Description of the HCC category",
    )
    raf_weight: float = Field(
        ...,
        ge=0.0,
        description="RAF weight for this HCC",
    )
    supporting_evidence: list[str] = Field(
        default_factory=list,
        description="Evidence supporting this HCC opportunity",
    )
    suggested_icd_codes: list[str] = Field(
        default_factory=list,
        description="ICD codes that would capture this HCC",
    )
    confidence: ConfidenceLevel = Field(
        ...,
        description="Confidence in the opportunity",
    )
    requires_query: bool = Field(
        default=False,
        description="Whether a provider query is needed",
    )


class RiskAnalysisResult(BaseModel):
    """Response from /api/v2/risk/analyze endpoint."""

    request_id: str = Field(..., description="Unique request identifier")
    current_hccs: list[str] = Field(
        default_factory=list,
        description="Currently captured HCC codes",
    )
    current_raf: float = Field(
        ...,
        ge=0.0,
        description="Current RAF score",
    )
    opportunities: list[HCCOpportunity] = Field(
        default_factory=list,
        description="HCC capture opportunities",
    )
    projected_raf: float = Field(
        ...,
        ge=0.0,
        description="Projected RAF if all opportunities are captured",
    )
    raf_gap: float = Field(
        default=0.0,
        ge=0.0,
        description="Gap between current and projected RAF",
    )
    patient_risk_tier: str | None = Field(
        default=None,
        description="Patient risk tier classification",
    )
    processing_time_ms: int = Field(
        ...,
        ge=0,
        description="Processing time in milliseconds",
    )


# =============================================================================
# CPT Coding Schemas
# =============================================================================


class CPTSuggestion(BaseModel):
    """Single CPT code suggestion."""

    cpt_code: str = Field(
        ...,
        description="CPT procedure code",
        examples=["99213", "43239", "27447"],
    )
    description: str = Field(
        ...,
        description="Description of the procedure",
    )
    confidence: ConfidenceLevel = Field(
        ...,
        description="Confidence level",
    )
    modifiers: list[str] = Field(
        default_factory=list,
        description="Applicable modifiers",
    )
    evidence_spans: list[str] = Field(
        default_factory=list,
        description="Supporting evidence from documentation",
    )


class CPTSuggestResponse(BaseModel):
    """Response from CPT suggestion endpoint."""

    request_id: str = Field(..., description="Unique request identifier")
    suggested_codes: list[CPTSuggestion] = Field(
        default_factory=list,
        description="Suggested CPT codes",
    )
    processing_time_ms: int = Field(
        ...,
        ge=0,
        description="Processing time in milliseconds",
    )


# =============================================================================
# Facts Extraction Schemas
# =============================================================================


class ExtractedFact(BaseModel):
    """A clinical fact extracted from the note."""

    fact_type: str = Field(
        ...,
        description="Type of fact",
        examples=["diagnosis", "medication", "vital_sign", "lab_result"],
    )
    value: str = Field(..., description="The extracted value")
    evidence_span: str = Field(..., description="Source text span")
    confidence: ConfidenceLevel = Field(..., description="Extraction confidence")
    normalized_value: str | None = Field(
        default=None,
        description="Normalized/standardized value",
    )


class FactsExtractionResponse(BaseModel):
    """Response from facts extraction endpoint."""

    request_id: str = Field(..., description="Unique request identifier")
    facts: list[ExtractedFact] = Field(
        default_factory=list,
        description="Extracted facts",
    )
    cache_key: str = Field(..., description="Cache key for subsequent calls")
    processing_time_ms: int = Field(
        ...,
        ge=0,
        description="Processing time in milliseconds",
    )
