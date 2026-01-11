"""
Pytest configuration and shared fixtures for Nuvii CDI Evaluation tests.
"""

import json
import os
from pathlib import Path
from typing import Generator

import pytest


@pytest.fixture
def temp_dir(tmp_path: Path) -> Path:
    """Provide a temporary directory for test files."""
    return tmp_path


@pytest.fixture
def sample_clinical_note() -> str:
    """Provide a sample clinical note for testing."""
    return """
    72-year-old female with a history of type 2 diabetes mellitus,
    currently managed on metformin 1000mg BID. Recent HbA1c was 7.2%.
    Patient denies any symptoms of hypoglycemia. No evidence of diabetic
    retinopathy on recent ophthalmologic exam. Microalbumin/creatinine
    ratio within normal limits. Blood pressure well controlled on lisinopril.

    Assessment:
    1. Type 2 diabetes mellitus, without complications, well controlled
    2. Essential hypertension, controlled

    Plan:
    - Continue current medications
    - Recheck HbA1c in 3 months
    - Annual diabetic eye exam scheduled
    """


@pytest.fixture
def sample_icd_test_case() -> dict:
    """Provide a sample ICD test case."""
    return {
        "id": "icd_test_001",
        "clinical_note": "72-year-old female with type 2 diabetes mellitus, "
        "currently on metformin 1000mg BID. HbA1c 7.2%. No complications.",
        "specialty": "endocrinology",
        "complexity": "moderate",
        "expected_icd_codes": ["E11.9"],
        "acceptable_icd_codes": ["E11.65"],
        "tags": ["diabetes", "chronic"],
    }


@pytest.fixture
def sample_hcc_test_case() -> dict:
    """Provide a sample HCC test case."""
    return {
        "id": "hcc_test_001",
        "clinical_note": "65-year-old male with congestive heart failure, "
        "LVEF 35%, NYHA Class III. On carvedilol, lisinopril, furosemide.",
        "specialty": "cardiology",
        "complexity": "high",
        "expected_hccs": ["HCC85"],
        "expected_raf_range": [0.3, 0.5],
        "patient_age": 65,
        "patient_gender": "M",
        "is_dual_eligible": False,
        "tags": ["heart_failure", "cardiac"],
    }


@pytest.fixture
def sample_gap_test_case() -> dict:
    """Provide a sample Gap test case."""
    return {
        "id": "gap_test_001",
        "clinical_note": "Patient presents with chest pain. History of CAD. "
        "Recent cath shows 70% LAD stenosis. Reports chest discomfort with exertion.",
        "specialty": "cardiology",
        "complexity": "moderate",
        "expected_gaps": [
            {
                "gap_type": "unconfirmed_diagnosis",
                "condition": "unstable angina",
                "min_priority": 2,
                "expected_icd_codes": ["I20.0"],
            }
        ],
        "tags": ["cardiac", "angina"],
    }


@pytest.fixture
def sample_em_test_case() -> dict:
    """Provide a sample E/M test case."""
    return {
        "id": "em_test_001",
        "clinical_note": "Established patient with multiple chronic conditions. "
        "Reviewed recent labs, imaging, and consulted with specialist. "
        "High complexity medical decision making required.",
        "specialty": "primary_care",
        "complexity": "high",
        "encounter_type": "outpatient",
        "patient_type": "established",
        "expected_code": "99215",
        "expected_level": 5,
        "expected_mdm": {"problems": 4, "data": 4, "risk": 4},
        "tags": ["high_complexity"],
    }


@pytest.fixture
def sample_jsonl_dataset(tmp_path: Path, sample_icd_test_case: dict) -> Path:
    """Create a sample JSONL dataset file."""
    file_path = tmp_path / "test_dataset.jsonl"

    test_cases = [
        sample_icd_test_case,
        {
            **sample_icd_test_case,
            "id": "icd_test_002",
            "expected_icd_codes": ["I10"],
            "acceptable_icd_codes": [],
            "tags": ["hypertension"],
        },
        {
            **sample_icd_test_case,
            "id": "icd_test_003",
            "specialty": "cardiology",
            "expected_icd_codes": ["I50.9"],
            "acceptable_icd_codes": ["I50.1", "I50.20"],
            "tags": ["heart_failure"],
        },
    ]

    with open(file_path, "w") as f:
        for tc in test_cases:
            f.write(json.dumps(tc) + "\n")

    return file_path


@pytest.fixture
def sample_api_response_coding() -> dict:
    """Provide a sample coding API response."""
    return {
        "request_id": "req_123",
        "suggested_codes": [
            {
                "icd10_code": "E11.9",
                "description": "Type 2 diabetes mellitus without complications",
                "confidence": "high",
                "evidence_spans": ["type 2 diabetes mellitus", "HbA1c 7.2%"],
            },
            {
                "icd10_code": "I10",
                "description": "Essential hypertension",
                "confidence": "high",
                "evidence_spans": ["hypertension", "lisinopril"],
            },
        ],
        "processing_time_ms": 250,
        "model_version": "v2.1.0",
    }


@pytest.fixture
def sample_api_response_gaps() -> dict:
    """Provide a sample gaps API response."""
    return {
        "request_id": "req_456",
        "gaps": [
            {
                "gap_id": "gap_001",
                "gap_type": "missing_specificity",
                "condition": "diabetes complications",
                "current_evidence": ["diabetes mellitus"],
                "suggested_icd_codes": ["E11.65", "E11.621"],
                "priority": 2,
                "confidence": "medium",
            }
        ],
        "facts_cache_key": "cache_abc123",
        "processing_time_ms": 300,
    }


@pytest.fixture
def env_vars() -> Generator[dict, None, None]:
    """Set up and tear down environment variables for testing."""
    original_env = os.environ.copy()

    test_env = {
        "NUVII_API_URL": "http://test-api.local:8000",
        "NUVII_API_KEY": "test_key_12345",
        "PHOENIX_ENABLED": "false",
        "EVAL_PHI_SAFE_MODE": "true",
    }

    os.environ.update(test_env)

    yield test_env

    # Restore original environment
    os.environ.clear()
    os.environ.update(original_env)
