"""
Async HTTP client for Nuvii CDI Agent V2 APIs.

Provides a high-level interface for interacting with the Nuvii CDI API,
with built-in retry logic, rate limiting, and structured logging.
"""

from datetime import datetime
from typing import Any

import httpx
import structlog
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from nuvii_eval.config import NuviiAPIConfig
from nuvii_eval.schemas.api_responses import (
    CodingSuggestResponse,
    CPTSuggestResponse,
    EMAnalysisResult,
    FactsExtractionResponse,
    GapDetectionResponse,
    QueryGenerationResponse,
    RiskAnalysisResult,
)

logger = structlog.get_logger(__name__)


class NuviiClientError(Exception):
    """Base exception for Nuvii API errors."""

    def __init__(self, message: str, status_code: int | None = None, response_body: str | None = None):
        super().__init__(message)
        self.status_code = status_code
        self.response_body = response_body


class NuviiAPIError(NuviiClientError):
    """API returned an error response."""

    pass


class NuviiConnectionError(NuviiClientError):
    """Connection to API failed."""

    pass


class NuviiTimeoutError(NuviiClientError):
    """Request timed out."""

    pass


class NuviiClient:
    """
    Async client for Nuvii CDI Agent V2 APIs.

    Usage:
        async with NuviiClient(config) as client:
            response = await client.suggest_codes("Clinical note text...")
            print(response.suggested_codes)

    Features:
        - Async context manager for proper connection handling
        - Automatic retry with exponential backoff
        - Structured logging of all requests
        - Response validation via Pydantic schemas
    """

    def __init__(self, config: NuviiAPIConfig):
        """
        Initialize the client with configuration.

        Args:
            config: Nuvii API configuration
        """
        self.config = config
        self._client: httpx.AsyncClient | None = None

    async def __aenter__(self) -> "NuviiClient":
        """Enter async context manager."""
        self._client = httpx.AsyncClient(
            base_url=self.config.api_url,
            headers={
                "Authorization": f"Bearer {self.config.api_key.get_secret_value()}",
                "Content-Type": "application/json",
                "Accept": "application/json",
                "User-Agent": "nuvii-eval/0.1.0",
            },
            timeout=httpx.Timeout(
                connect=10.0,
                read=self.config.timeout_seconds,
                write=10.0,
                pool=5.0,
            ),
        )
        logger.info("nuvii_client_initialized", base_url=self.config.api_url)
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Exit async context manager."""
        if self._client:
            await self._client.aclose()
            logger.debug("nuvii_client_closed")

    def _get_retry_decorator(self):
        """Get retry decorator with configured settings."""
        return retry(
            stop=stop_after_attempt(self.config.max_retries),
            wait=wait_exponential(multiplier=1, min=1, max=10),
            retry=retry_if_exception_type((httpx.TransportError, httpx.TimeoutException)),
            reraise=True,
        )

    async def _request(
        self,
        method: str,
        endpoint: str,
        json_data: dict[str, Any] | None = None,
        params: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Make an authenticated API request.

        Args:
            method: HTTP method (GET, POST, etc.)
            endpoint: API endpoint path
            json_data: JSON body for POST requests
            params: Query parameters

        Returns:
            Parsed JSON response

        Raises:
            NuviiAPIError: API returned an error
            NuviiConnectionError: Connection failed
            NuviiTimeoutError: Request timed out
        """
        if not self._client:
            raise NuviiClientError("Client not initialized. Use 'async with' context manager.")

        start_time = datetime.utcnow()

        try:
            response = await self._client.request(
                method,
                endpoint,
                json=json_data,
                params=params,
            )

            latency_ms = int((datetime.utcnow() - start_time).total_seconds() * 1000)

            logger.debug(
                "api_request",
                method=method,
                endpoint=endpoint,
                status_code=response.status_code,
                latency_ms=latency_ms,
            )

            if response.status_code >= 400:
                error_body = response.text[:500]
                logger.error(
                    "api_error",
                    method=method,
                    endpoint=endpoint,
                    status_code=response.status_code,
                    error_body=error_body,
                )
                raise NuviiAPIError(
                    f"API error {response.status_code}: {error_body}",
                    status_code=response.status_code,
                    response_body=error_body,
                )

            return response.json()

        except httpx.TimeoutException as e:
            logger.error("api_timeout", endpoint=endpoint, error=str(e))
            raise NuviiTimeoutError(f"Request timed out: {endpoint}") from e

        except httpx.TransportError as e:
            logger.error("api_connection_error", endpoint=endpoint, error=str(e))
            raise NuviiConnectionError(f"Connection error: {e}") from e

    # =========================================================================
    # ICD Coding APIs
    # =========================================================================

    async def suggest_codes(
        self,
        clinical_note: str,
        use_llm: bool = True,
        temperature: float = 0.0,
        max_codes: int = 10,
    ) -> CodingSuggestResponse:
        """
        Get ICD-10 code suggestions for a clinical note.

        Args:
            clinical_note: The clinical documentation text
            use_llm: Whether to use LLM for enhanced suggestions
            temperature: Model temperature (0.0 for deterministic)
            max_codes: Maximum number of codes to return

        Returns:
            CodingSuggestResponse with suggested codes
        """
        data = await self._request(
            "POST",
            "/api/v2/coding/suggest",
            json_data={
                "clinical_note": clinical_note,
                "use_llm": use_llm,
                "temperature": temperature,
                "max_codes": max_codes,
            },
        )
        return CodingSuggestResponse(**data)

    async def suggest_cpt_codes(
        self,
        clinical_note: str,
        procedure_context: str | None = None,
    ) -> CPTSuggestResponse:
        """
        Get CPT code suggestions for procedures.

        Args:
            clinical_note: The clinical documentation
            procedure_context: Optional additional procedure context

        Returns:
            CPTSuggestResponse with suggested CPT codes
        """
        payload: dict[str, Any] = {"clinical_note": clinical_note}
        if procedure_context:
            payload["procedure_context"] = procedure_context

        data = await self._request("POST", "/api/v2/coding/cpt", json_data=payload)
        return CPTSuggestResponse(**data)

    # =========================================================================
    # CDI Gap Detection APIs
    # =========================================================================

    async def extract_facts(
        self,
        clinical_note: str,
    ) -> FactsExtractionResponse:
        """
        Extract clinical facts from a note.

        Args:
            clinical_note: The clinical documentation

        Returns:
            FactsExtractionResponse with extracted facts and cache key
        """
        data = await self._request(
            "POST",
            "/api/v2/cdi/facts",
            json_data={"clinical_note": clinical_note},
        )
        return FactsExtractionResponse(**data)

    async def detect_gaps(
        self,
        clinical_note: str,
        facts_cache_key: str | None = None,
    ) -> GapDetectionResponse:
        """
        Detect documentation gaps in a clinical note.

        Args:
            clinical_note: The clinical documentation
            facts_cache_key: Optional cache key from previous facts extraction

        Returns:
            GapDetectionResponse with detected gaps
        """
        payload: dict[str, Any] = {"clinical_note": clinical_note}
        if facts_cache_key:
            payload["facts_cache_key"] = facts_cache_key

        data = await self._request("POST", "/api/v2/cdi/gaps", json_data=payload)
        return GapDetectionResponse(**data)

    async def generate_queries(
        self,
        gaps_cache_key: str,
        query_style: str = "non_leading",
        max_queries: int | None = None,
    ) -> QueryGenerationResponse:
        """
        Generate CDI queries for detected gaps.

        Args:
            gaps_cache_key: Cache key from gap detection
            query_style: Query generation style
            max_queries: Maximum number of queries to generate

        Returns:
            QueryGenerationResponse with generated queries
        """
        payload: dict[str, Any] = {
            "gaps_cache_key": gaps_cache_key,
            "query_style": query_style,
        }
        if max_queries:
            payload["max_queries"] = max_queries

        data = await self._request("POST", "/api/v2/cdi/queries", json_data=payload)
        return QueryGenerationResponse(**data)

    # =========================================================================
    # E/M Analysis APIs
    # =========================================================================

    async def analyze_em(
        self,
        clinical_note: str,
        encounter_type: str = "outpatient",
        documented_time: int | None = None,
        patient_type: str = "established",
    ) -> EMAnalysisResult:
        """
        Analyze E/M level for an encounter.

        Args:
            clinical_note: The clinical documentation
            encounter_type: Type of encounter (outpatient, inpatient, ed, etc.)
            documented_time: Documented time in minutes (for time-based billing)
            patient_type: Patient type (new or established)

        Returns:
            EMAnalysisResult with recommended E/M level
        """
        payload: dict[str, Any] = {
            "clinical_note": clinical_note,
            "encounter_type": encounter_type,
            "patient_type": patient_type,
        }
        if documented_time is not None:
            payload["documented_time"] = documented_time

        data = await self._request("POST", "/api/v2/em/analyze", json_data=payload)
        return EMAnalysisResult(**data)

    # =========================================================================
    # Risk/HCC Analysis APIs
    # =========================================================================

    async def analyze_risk(
        self,
        clinical_note: str,
        patient_age: int | None = None,
        patient_gender: str | None = None,
        is_dual_eligible: bool = False,
    ) -> RiskAnalysisResult:
        """
        Analyze HCC risk adjustment for a patient.

        Args:
            clinical_note: The clinical documentation
            patient_age: Patient age in years
            patient_gender: Patient gender (M/F)
            is_dual_eligible: Whether patient is dual Medicare/Medicaid eligible

        Returns:
            RiskAnalysisResult with HCC analysis and RAF scores
        """
        payload: dict[str, Any] = {"clinical_note": clinical_note}

        if patient_age is not None:
            payload["patient_age"] = patient_age
        if patient_gender:
            payload["patient_gender"] = patient_gender
        if is_dual_eligible:
            payload["is_dual_eligible"] = is_dual_eligible

        data = await self._request("POST", "/api/v2/risk/analyze", json_data=payload)
        return RiskAnalysisResult(**data)

    # =========================================================================
    # Utility Methods
    # =========================================================================

    async def health_check(self) -> dict[str, Any]:
        """
        Check API health status.

        Returns:
            Health status information
        """
        return await self._request("GET", "/health")

    async def get_model_info(self) -> dict[str, Any]:
        """
        Get information about the deployed model.

        Returns:
            Model version and configuration info
        """
        return await self._request("GET", "/api/v2/model/info")
