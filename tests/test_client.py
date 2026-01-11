"""Tests for Nuvii API client."""

import pytest
import httpx
import respx

from nuvii_eval.client import (
    NuviiClient,
    NuviiClientError,
    NuviiAPIError,
    NuviiConnectionError,
    NuviiTimeoutError,
)
from nuvii_eval.config import NuviiAPIConfig


@pytest.fixture
def api_config() -> NuviiAPIConfig:
    """Create test API configuration."""
    return NuviiAPIConfig(
        api_url="http://test-api.local:8000",
        api_key="test_key_12345",
        timeout_seconds=5,
        max_retries=2,
    )


@pytest.fixture
def mock_coding_response() -> dict:
    """Mock coding API response."""
    return {
        "request_id": "req_123",
        "suggested_codes": [
            {
                "icd10_code": "E11.9",
                "description": "Type 2 diabetes mellitus without complications",
                "confidence": "high",
                "evidence_spans": ["diabetes mellitus"],
            }
        ],
        "processing_time_ms": 250,
        "model_version": "v2.1.0",
    }


@pytest.fixture
def mock_gaps_response() -> dict:
    """Mock gaps API response."""
    return {
        "request_id": "req_456",
        "gaps": [
            {
                "gap_id": "gap_001",
                "gap_type": "missing_specificity",
                "condition": "diabetes",
                "current_evidence": [],
                "suggested_icd_codes": ["E11.65"],
                "priority": 2,
                "confidence": "medium",
            }
        ],
        "facts_cache_key": "cache_123",
        "processing_time_ms": 300,
    }


class TestNuviiClientContext:
    """Tests for client context manager."""

    @respx.mock
    async def test_context_manager_creates_client(self, api_config):
        """Test that context manager properly initializes client."""
        async with NuviiClient(api_config) as client:
            assert client._client is not None
            assert isinstance(client._client, httpx.AsyncClient)

    @respx.mock
    async def test_context_manager_closes_client(self, api_config):
        """Test that context manager properly closes client."""
        async with NuviiClient(api_config) as client:
            internal_client = client._client

        # After exiting context, client should be closed
        assert internal_client.is_closed

    async def test_request_without_context_raises(self, api_config):
        """Test that making request without context raises error."""
        client = NuviiClient(api_config)

        with pytest.raises(NuviiClientError, match="not initialized"):
            await client._request("GET", "/test")


class TestNuviiClientCoding:
    """Tests for coding API methods."""

    @respx.mock
    async def test_suggest_codes_success(self, api_config, mock_coding_response):
        """Test successful code suggestion."""
        respx.post("http://test-api.local:8000/api/v2/coding/suggest").mock(
            return_value=httpx.Response(200, json=mock_coding_response)
        )

        async with NuviiClient(api_config) as client:
            response = await client.suggest_codes(
                clinical_note="Patient with diabetes mellitus",
                use_llm=True,
                temperature=0.0,
            )

        assert response.request_id == "req_123"
        assert len(response.suggested_codes) == 1
        assert response.suggested_codes[0].icd10_code == "E11.9"

    @respx.mock
    async def test_suggest_codes_api_error(self, api_config):
        """Test handling of API error response."""
        respx.post("http://test-api.local:8000/api/v2/coding/suggest").mock(
            return_value=httpx.Response(400, json={"error": "Invalid request"})
        )

        async with NuviiClient(api_config) as client:
            with pytest.raises(NuviiAPIError) as exc_info:
                await client.suggest_codes("Invalid note")

        assert exc_info.value.status_code == 400

    @respx.mock
    async def test_suggest_codes_server_error(self, api_config):
        """Test handling of server error."""
        respx.post("http://test-api.local:8000/api/v2/coding/suggest").mock(
            return_value=httpx.Response(500, text="Internal Server Error")
        )

        async with NuviiClient(api_config) as client:
            with pytest.raises(NuviiAPIError) as exc_info:
                await client.suggest_codes("Some note")

        assert exc_info.value.status_code == 500


class TestNuviiClientGaps:
    """Tests for gap detection API methods."""

    @respx.mock
    async def test_detect_gaps_success(self, api_config, mock_gaps_response):
        """Test successful gap detection."""
        respx.post("http://test-api.local:8000/api/v2/cdi/gaps").mock(
            return_value=httpx.Response(200, json=mock_gaps_response)
        )

        async with NuviiClient(api_config) as client:
            response = await client.detect_gaps(
                clinical_note="Patient with diabetes"
            )

        assert len(response.gaps) == 1
        assert response.gaps[0].gap_type == "missing_specificity"
        assert response.facts_cache_key == "cache_123"

    @respx.mock
    async def test_detect_gaps_with_cache_key(self, api_config, mock_gaps_response):
        """Test gap detection with pre-existing cache key."""
        route = respx.post("http://test-api.local:8000/api/v2/cdi/gaps").mock(
            return_value=httpx.Response(200, json=mock_gaps_response)
        )

        async with NuviiClient(api_config) as client:
            await client.detect_gaps(
                clinical_note="Patient with diabetes",
                facts_cache_key="existing_cache_key",
            )

        # Verify cache key was sent
        request = route.calls.last.request
        request_body = request.content.decode()
        assert "existing_cache_key" in request_body

    @respx.mock
    async def test_generate_queries_success(self, api_config):
        """Test successful query generation."""
        mock_response = {
            "request_id": "req_789",
            "queries": [
                {
                    "query_id": "query_001",
                    "gap_id": "gap_001",
                    "query_text": "Please clarify the type of diabetes complications.",
                    "query_type": "clarification",
                    "evidence_cited": ["diabetes mellitus"],
                    "suggested_responses": ["DKA", "Neuropathy", "None"],
                    "icd_impact": ["E11.65"],
                }
            ],
            "processing_time_ms": 200,
        }

        respx.post("http://test-api.local:8000/api/v2/cdi/queries").mock(
            return_value=httpx.Response(200, json=mock_response)
        )

        async with NuviiClient(api_config) as client:
            response = await client.generate_queries(
                gaps_cache_key="cache_123",
                query_style="non_leading",
            )

        assert len(response.queries) == 1
        assert response.queries[0].query_type == "clarification"


class TestNuviiClientEM:
    """Tests for E/M analysis API methods."""

    @respx.mock
    async def test_analyze_em_success(self, api_config):
        """Test successful E/M analysis."""
        mock_response = {
            "request_id": "req_em_001",
            "recommended_code": "99214",
            "recommended_level": 4,
            "mdm_score": {"problems": 3, "data": 3, "risk": 3},
            "justification": "Moderate complexity visit",
            "processing_time_ms": 150,
        }

        respx.post("http://test-api.local:8000/api/v2/em/analyze").mock(
            return_value=httpx.Response(200, json=mock_response)
        )

        async with NuviiClient(api_config) as client:
            response = await client.analyze_em(
                clinical_note="Complex patient encounter",
                encounter_type="outpatient",
                patient_type="established",
            )

        assert response.recommended_code == "99214"
        assert response.recommended_level == 4
        assert response.mdm_score.problems == 3


class TestNuviiClientRisk:
    """Tests for risk analysis API methods."""

    @respx.mock
    async def test_analyze_risk_success(self, api_config):
        """Test successful risk analysis."""
        mock_response = {
            "request_id": "req_risk_001",
            "current_hccs": ["HCC18", "HCC85"],
            "current_raf": 1.25,
            "opportunities": [
                {
                    "hcc_code": "HCC19",
                    "hcc_description": "Diabetes without complications",
                    "raf_weight": 0.105,
                    "supporting_evidence": ["diabetes"],
                    "suggested_icd_codes": ["E11.9"],
                    "confidence": "medium",
                }
            ],
            "projected_raf": 1.355,
            "processing_time_ms": 200,
        }

        respx.post("http://test-api.local:8000/api/v2/risk/analyze").mock(
            return_value=httpx.Response(200, json=mock_response)
        )

        async with NuviiClient(api_config) as client:
            response = await client.analyze_risk(
                clinical_note="Patient with multiple conditions",
                patient_age=72,
                patient_gender="F",
            )

        assert "HCC18" in response.current_hccs
        assert response.current_raf == 1.25
        assert len(response.opportunities) == 1


class TestNuviiClientErrorHandling:
    """Tests for error handling."""

    @respx.mock
    async def test_timeout_error(self, api_config):
        """Test timeout handling."""
        respx.post("http://test-api.local:8000/api/v2/coding/suggest").mock(
            side_effect=httpx.TimeoutException("Connection timed out")
        )

        async with NuviiClient(api_config) as client:
            with pytest.raises(NuviiTimeoutError):
                await client.suggest_codes("Test note")

    @respx.mock
    async def test_connection_error(self, api_config):
        """Test connection error handling."""
        respx.post("http://test-api.local:8000/api/v2/coding/suggest").mock(
            side_effect=httpx.ConnectError("Connection refused")
        )

        async with NuviiClient(api_config) as client:
            with pytest.raises(NuviiConnectionError):
                await client.suggest_codes("Test note")

    @respx.mock
    async def test_health_check(self, api_config):
        """Test health check endpoint."""
        respx.get("http://test-api.local:8000/health").mock(
            return_value=httpx.Response(200, json={"status": "healthy"})
        )

        async with NuviiClient(api_config) as client:
            result = await client.health_check()

        assert result["status"] == "healthy"
