"""
Dataset loading utilities for evaluation test cases.

Provides functionality to load, validate, and iterate over test case datasets
stored in JSONL format.
"""

import json
from collections.abc import Iterator
from pathlib import Path
from typing import TypeVar

import structlog

from nuvii_eval.datasets.schemas import (
    TEST_CASE_TYPES,
    BaseTestCase,
    CPTTestCase,
    EMTestCase,
    GapTestCase,
    HCCTestCase,
    ICDTestCase,
    QueryTestCase,
)

logger = structlog.get_logger(__name__)

T = TypeVar("T", bound=BaseTestCase)


class DatasetLoadError(Exception):
    """Error loading or validating a dataset."""

    def __init__(self, message: str, file_path: str | None = None, line_number: int | None = None):
        super().__init__(message)
        self.file_path = file_path
        self.line_number = line_number


class DatasetValidationError(DatasetLoadError):
    """Validation error in dataset."""

    pass


class DatasetLoader:
    """
    Loads and validates test case datasets from JSONL files.

    Usage:
        loader = DatasetLoader("./datasets")

        # Load entire dataset
        test_cases = loader.load_jsonl("golden/icd_test_cases.jsonl", "icd")

        # Stream large datasets
        for test_case in loader.iter_jsonl("golden/icd_test_cases.jsonl", "icd"):
            process(test_case)

        # Load a complete test suite
        suite = loader.load_suite("golden")
    """

    def __init__(self, base_path: Path | str = "./datasets"):
        """
        Initialize the dataset loader.

        Args:
            base_path: Base directory for dataset files
        """
        self.base_path = Path(base_path)

    def _resolve_path(self, file_path: Path | str) -> Path:
        """Resolve file path relative to base path or as absolute."""
        path = Path(file_path)
        if path.is_absolute():
            return path
        return self.base_path / path

    def _get_schema_class(self, schema_type: str) -> type[BaseTestCase]:
        """Get the schema class for a given type."""
        schema_class = TEST_CASE_TYPES.get(schema_type)
        if not schema_class:
            valid_types = ", ".join(TEST_CASE_TYPES.keys())
            raise DatasetLoadError(
                f"Unknown schema type: {schema_type}. Valid types: {valid_types}"
            )
        return schema_class

    def load_jsonl(
        self,
        file_path: Path | str,
        schema_type: str,
        limit: int | None = None,
        skip_invalid: bool = False,
    ) -> list[BaseTestCase]:
        """
        Load test cases from a JSONL file.

        Args:
            file_path: Path to JSONL file (relative to base_path or absolute)
            schema_type: Type of test cases ("icd", "hcc", "gap", "query", "em", "cpt")
            limit: Maximum number of cases to load (None for all)
            skip_invalid: If True, skip invalid records instead of raising

        Returns:
            List of validated test cases

        Raises:
            DatasetLoadError: If file not found or parsing fails
            DatasetValidationError: If validation fails and skip_invalid is False
        """
        path = self._resolve_path(file_path)
        schema_class = self._get_schema_class(schema_type)

        if not path.exists():
            raise DatasetLoadError(f"Dataset file not found: {path}", file_path=str(path))

        test_cases: list[BaseTestCase] = []
        errors: list[dict] = []

        with open(path, encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                if limit and len(test_cases) >= limit:
                    break

                line = line.strip()
                if not line:
                    continue

                try:
                    data = json.loads(line)
                    test_case = schema_class(**data)
                    test_cases.append(test_case)

                except json.JSONDecodeError as e:
                    error_info = {
                        "line": line_num,
                        "error": f"Invalid JSON: {e}",
                        "content": line[:100],
                    }
                    errors.append(error_info)

                    if not skip_invalid:
                        raise DatasetValidationError(
                            f"Invalid JSON at line {line_num}: {e}",
                            file_path=str(path),
                            line_number=line_num,
                        ) from e

                except Exception as e:
                    error_info = {
                        "line": line_num,
                        "error": str(e),
                        "content": line[:100],
                    }
                    errors.append(error_info)

                    if not skip_invalid:
                        raise DatasetValidationError(
                            f"Validation error at line {line_num}: {e}",
                            file_path=str(path),
                            line_number=line_num,
                        ) from e

        if errors:
            logger.warning(
                "dataset_load_errors",
                path=str(path),
                error_count=len(errors),
                first_errors=errors[:5],
            )

        logger.info(
            "dataset_loaded",
            path=str(path),
            schema_type=schema_type,
            loaded_count=len(test_cases),
            error_count=len(errors),
            limit=limit,
        )

        return test_cases

    def iter_jsonl(
        self,
        file_path: Path | str,
        schema_type: str,
        skip_invalid: bool = True,
    ) -> Iterator[BaseTestCase]:
        """
        Iterate over test cases lazily (memory-efficient for large datasets).

        Args:
            file_path: Path to JSONL file
            schema_type: Type of test cases
            skip_invalid: If True, skip invalid records silently

        Yields:
            Validated test cases
        """
        path = self._resolve_path(file_path)
        schema_class = self._get_schema_class(schema_type)

        if not path.exists():
            raise DatasetLoadError(f"Dataset file not found: {path}", file_path=str(path))

        with open(path, encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue

                try:
                    data = json.loads(line)
                    yield schema_class(**data)
                except Exception as e:
                    if not skip_invalid:
                        raise DatasetValidationError(
                            f"Error at line {line_num}: {e}",
                            file_path=str(path),
                            line_number=line_num,
                        ) from e
                    logger.debug(
                        "skipped_invalid_record",
                        line=line_num,
                        error=str(e),
                    )

    def load_suite(
        self,
        suite_name: str,
        limit_per_type: int | None = None,
    ) -> dict[str, list[BaseTestCase]]:
        """
        Load a complete test suite (all dataset types from a directory).

        Args:
            suite_name: Name of the suite directory (e.g., "golden", "regression")
            limit_per_type: Maximum cases per dataset type

        Returns:
            Dictionary mapping schema type to list of test cases
        """
        suite_path = self.base_path / suite_name

        if not suite_path.is_dir():
            raise DatasetLoadError(f"Suite directory not found: {suite_path}")

        suite: dict[str, list[BaseTestCase]] = {}

        for file_path in suite_path.glob("*.jsonl"):
            # Infer schema type from filename
            # Expected format: "icd_test_cases.jsonl" or "icd.jsonl"
            stem = file_path.stem
            schema_type = stem.split("_")[0]

            if schema_type in TEST_CASE_TYPES:
                try:
                    suite[schema_type] = self.load_jsonl(
                        file_path,
                        schema_type,
                        limit=limit_per_type,
                        skip_invalid=True,
                    )
                except DatasetLoadError as e:
                    logger.warning(
                        "failed_to_load_suite_file",
                        file=str(file_path),
                        error=str(e),
                    )

        logger.info(
            "suite_loaded",
            suite_name=suite_name,
            types_loaded=list(suite.keys()),
            total_cases=sum(len(cases) for cases in suite.values()),
        )

        return suite

    def validate_dataset(
        self,
        file_path: Path | str,
        schema_type: str,
    ) -> dict:
        """
        Validate a dataset and return statistics.

        Args:
            file_path: Path to JSONL file
            schema_type: Type of test cases

        Returns:
            Dictionary with validation statistics
        """
        path = self._resolve_path(file_path)

        valid_count = 0
        invalid_count = 0
        by_specialty: dict[str, int] = {}
        by_complexity: dict[str, int] = {}
        errors: list[dict] = []

        for line_num, tc in enumerate(self.iter_jsonl(file_path, schema_type, skip_invalid=True), 1):
            valid_count += 1
            by_specialty[tc.specialty.value] = by_specialty.get(tc.specialty.value, 0) + 1
            by_complexity[tc.complexity.value] = by_complexity.get(tc.complexity.value, 0) + 1

        # Count total lines to get invalid count
        with open(path, encoding="utf-8") as f:
            total_lines = sum(1 for line in f if line.strip())

        invalid_count = total_lines - valid_count

        return {
            "file": str(path),
            "schema_type": schema_type,
            "total_lines": total_lines,
            "valid_count": valid_count,
            "invalid_count": invalid_count,
            "validation_rate": valid_count / total_lines if total_lines > 0 else 0,
            "by_specialty": by_specialty,
            "by_complexity": by_complexity,
        }

    def get_test_case_by_id(
        self,
        file_path: Path | str,
        schema_type: str,
        test_case_id: str,
    ) -> BaseTestCase | None:
        """
        Find a specific test case by ID.

        Args:
            file_path: Path to JSONL file
            schema_type: Type of test cases
            test_case_id: ID to search for

        Returns:
            Test case if found, None otherwise
        """
        for tc in self.iter_jsonl(file_path, schema_type, skip_invalid=True):
            if tc.id == test_case_id:
                return tc
        return None

    def filter_by_tags(
        self,
        file_path: Path | str,
        schema_type: str,
        tags: list[str],
        match_all: bool = False,
    ) -> list[BaseTestCase]:
        """
        Filter test cases by tags.

        Args:
            file_path: Path to JSONL file
            schema_type: Type of test cases
            tags: Tags to filter by
            match_all: If True, test case must have all tags; if False, any tag

        Returns:
            Filtered list of test cases
        """
        results = []
        tag_set = set(tags)

        for tc in self.iter_jsonl(file_path, schema_type, skip_invalid=True):
            tc_tags = set(tc.tags)
            if match_all:
                if tag_set.issubset(tc_tags):
                    results.append(tc)
            else:
                if tag_set & tc_tags:
                    results.append(tc)

        return results

    def filter_by_specialty(
        self,
        file_path: Path | str,
        schema_type: str,
        specialties: list[str],
    ) -> list[BaseTestCase]:
        """
        Filter test cases by specialty.

        Args:
            file_path: Path to JSONL file
            schema_type: Type of test cases
            specialties: List of specialty values to include

        Returns:
            Filtered list of test cases
        """
        specialty_set = set(specialties)
        return [
            tc
            for tc in self.iter_jsonl(file_path, schema_type, skip_invalid=True)
            if tc.specialty.value in specialty_set
        ]


def create_sample_datasets(output_dir: Path | str = "./datasets/samples") -> None:
    """
    Create sample dataset files for testing.

    Args:
        output_dir: Directory to write sample files
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Sample ICD test case
    icd_sample = {
        "id": "icd_sample_001",
        "clinical_note": "72-year-old female with history of type 2 diabetes mellitus, "
        "currently on metformin 1000mg BID. HbA1c 7.2%. No evidence of "
        "diabetic retinopathy or nephropathy. Blood pressure controlled.",
        "specialty": "endocrinology",
        "complexity": "moderate",
        "expected_icd_codes": ["E11.9"],
        "acceptable_icd_codes": ["E11.65"],
        "tags": ["diabetes", "chronic"],
    }

    # Sample HCC test case
    hcc_sample = {
        "id": "hcc_sample_001",
        "clinical_note": "65-year-old male with congestive heart failure, "
        "LVEF 35%, NYHA Class III. History of prior MI. "
        "Currently on carvedilol, lisinopril, and furosemide.",
        "specialty": "cardiology",
        "complexity": "high",
        "expected_hccs": ["HCC85"],
        "expected_raf_range": [0.3, 0.5],
        "patient_age": 65,
        "patient_gender": "M",
        "is_dual_eligible": False,
        "tags": ["heart_failure", "cardiac"],
    }

    # Sample Gap test case
    gap_sample = {
        "id": "gap_sample_001",
        "clinical_note": "Patient presents with chest pain. History of coronary artery disease. "
        "Recent cardiac catheterization shows 70% stenosis of LAD. "
        "Patient reports intermittent chest discomfort with exertion.",
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

    # Write samples
    samples = [
        ("icd_samples.jsonl", [icd_sample]),
        ("hcc_samples.jsonl", [hcc_sample]),
        ("gap_samples.jsonl", [gap_sample]),
    ]

    for filename, data in samples:
        file_path = output_path / filename
        with open(file_path, "w", encoding="utf-8") as f:
            for item in data:
                f.write(json.dumps(item) + "\n")

    logger.info("sample_datasets_created", output_dir=str(output_path))
