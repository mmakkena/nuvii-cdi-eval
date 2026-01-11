"""Tests for dataset loading and validation."""

import json
from pathlib import Path

import pytest

from nuvii_eval.datasets.loader import (
    DatasetLoader,
    DatasetLoadError,
    DatasetValidationError,
    create_sample_datasets,
)
from nuvii_eval.datasets.schemas import ICDTestCase, HCCTestCase, GapTestCase


class TestDatasetLoader:
    """Tests for DatasetLoader class."""

    def test_init_with_string_path(self, tmp_path):
        """Test initialization with string path."""
        loader = DatasetLoader(str(tmp_path))
        assert loader.base_path == tmp_path

    def test_init_with_path_object(self, tmp_path):
        """Test initialization with Path object."""
        loader = DatasetLoader(tmp_path)
        assert loader.base_path == tmp_path

    def test_load_nonexistent_file_raises(self, tmp_path):
        """Test that loading nonexistent file raises error."""
        loader = DatasetLoader(tmp_path)

        with pytest.raises(DatasetLoadError, match="not found"):
            loader.load_jsonl("nonexistent.jsonl", "icd")

    def test_load_unknown_schema_type_raises(self, tmp_path):
        """Test that unknown schema type raises error."""
        # Create a dummy file
        (tmp_path / "test.jsonl").touch()

        loader = DatasetLoader(tmp_path)

        with pytest.raises(DatasetLoadError, match="Unknown schema type"):
            loader.load_jsonl("test.jsonl", "unknown_type")


class TestDatasetLoaderJSONL:
    """Tests for JSONL loading functionality."""

    def test_load_jsonl_success(self, tmp_path, sample_icd_test_case):
        """Test successful JSONL loading."""
        file_path = tmp_path / "test.jsonl"
        with open(file_path, "w") as f:
            f.write(json.dumps(sample_icd_test_case) + "\n")

        loader = DatasetLoader(tmp_path)
        test_cases = loader.load_jsonl("test.jsonl", "icd")

        assert len(test_cases) == 1
        assert isinstance(test_cases[0], ICDTestCase)
        assert test_cases[0].id == "icd_test_001"

    def test_load_jsonl_with_limit(self, sample_jsonl_dataset):
        """Test loading with limit parameter."""
        loader = DatasetLoader(sample_jsonl_dataset.parent)
        test_cases = loader.load_jsonl(sample_jsonl_dataset.name, "icd", limit=2)

        assert len(test_cases) == 2

    def test_load_jsonl_skips_empty_lines(self, tmp_path, sample_icd_test_case):
        """Test that empty lines are skipped."""
        file_path = tmp_path / "test.jsonl"
        with open(file_path, "w") as f:
            f.write(json.dumps(sample_icd_test_case) + "\n")
            f.write("\n")  # Empty line
            f.write("   \n")  # Whitespace only
            f.write(json.dumps({**sample_icd_test_case, "id": "icd_002"}) + "\n")

        loader = DatasetLoader(tmp_path)
        test_cases = loader.load_jsonl("test.jsonl", "icd")

        assert len(test_cases) == 2

    def test_load_jsonl_invalid_json_raises(self, tmp_path):
        """Test that invalid JSON raises error when not skipping."""
        file_path = tmp_path / "test.jsonl"
        with open(file_path, "w") as f:
            f.write("not valid json\n")

        loader = DatasetLoader(tmp_path)

        with pytest.raises(DatasetValidationError, match="Invalid JSON"):
            loader.load_jsonl("test.jsonl", "icd", skip_invalid=False)

    def test_load_jsonl_skip_invalid(self, tmp_path, sample_icd_test_case):
        """Test skipping invalid records."""
        file_path = tmp_path / "test.jsonl"
        with open(file_path, "w") as f:
            f.write(json.dumps(sample_icd_test_case) + "\n")
            f.write("invalid json\n")
            f.write(json.dumps({**sample_icd_test_case, "id": "icd_002"}) + "\n")

        loader = DatasetLoader(tmp_path)
        test_cases = loader.load_jsonl("test.jsonl", "icd", skip_invalid=True)

        assert len(test_cases) == 2

    def test_load_jsonl_validation_error_raises(self, tmp_path):
        """Test that validation errors raise when not skipping."""
        file_path = tmp_path / "test.jsonl"
        invalid_case = {
            "id": "test_001",
            "clinical_note": "A" * 50,
            "expected_icd_codes": ["INVALID_CODE"],  # Invalid format
        }
        with open(file_path, "w") as f:
            f.write(json.dumps(invalid_case) + "\n")

        loader = DatasetLoader(tmp_path)

        with pytest.raises(DatasetValidationError, match="Validation error"):
            loader.load_jsonl("test.jsonl", "icd", skip_invalid=False)


class TestDatasetLoaderIterator:
    """Tests for iterator functionality."""

    def test_iter_jsonl(self, sample_jsonl_dataset):
        """Test lazy iteration over dataset."""
        loader = DatasetLoader(sample_jsonl_dataset.parent)

        count = 0
        for tc in loader.iter_jsonl(sample_jsonl_dataset.name, "icd"):
            assert isinstance(tc, ICDTestCase)
            count += 1

        assert count == 3

    def test_iter_jsonl_skips_invalid_by_default(self, tmp_path, sample_icd_test_case):
        """Test that iterator skips invalid records by default."""
        file_path = tmp_path / "test.jsonl"
        with open(file_path, "w") as f:
            f.write(json.dumps(sample_icd_test_case) + "\n")
            f.write("invalid\n")

        loader = DatasetLoader(tmp_path)
        test_cases = list(loader.iter_jsonl("test.jsonl", "icd"))

        assert len(test_cases) == 1


class TestDatasetLoaderSuite:
    """Tests for suite loading functionality."""

    def test_load_suite(self, tmp_path, sample_icd_test_case, sample_hcc_test_case):
        """Test loading a complete test suite."""
        suite_dir = tmp_path / "golden"
        suite_dir.mkdir()

        # Create ICD dataset
        with open(suite_dir / "icd_test_cases.jsonl", "w") as f:
            f.write(json.dumps(sample_icd_test_case) + "\n")

        # Create HCC dataset
        with open(suite_dir / "hcc_test_cases.jsonl", "w") as f:
            f.write(json.dumps(sample_hcc_test_case) + "\n")

        loader = DatasetLoader(tmp_path)
        suite = loader.load_suite("golden")

        assert "icd" in suite
        assert "hcc" in suite
        assert len(suite["icd"]) == 1
        assert len(suite["hcc"]) == 1

    def test_load_suite_nonexistent_raises(self, tmp_path):
        """Test that loading nonexistent suite raises error."""
        loader = DatasetLoader(tmp_path)

        with pytest.raises(DatasetLoadError, match="not found"):
            loader.load_suite("nonexistent_suite")


class TestDatasetLoaderValidation:
    """Tests for dataset validation functionality."""

    def test_validate_dataset(self, sample_jsonl_dataset):
        """Test dataset validation statistics."""
        loader = DatasetLoader(sample_jsonl_dataset.parent)
        stats = loader.validate_dataset(sample_jsonl_dataset.name, "icd")

        assert stats["valid_count"] == 3
        assert stats["invalid_count"] == 0
        assert stats["validation_rate"] == 1.0
        assert "endocrinology" in stats["by_specialty"]
        assert "cardiology" in stats["by_specialty"]

    def test_validate_dataset_with_invalid(self, tmp_path, sample_icd_test_case):
        """Test validation with invalid records."""
        file_path = tmp_path / "test.jsonl"
        with open(file_path, "w") as f:
            f.write(json.dumps(sample_icd_test_case) + "\n")
            f.write("invalid json\n")

        loader = DatasetLoader(tmp_path)
        stats = loader.validate_dataset("test.jsonl", "icd")

        assert stats["valid_count"] == 1
        assert stats["invalid_count"] == 1
        assert stats["validation_rate"] == 0.5


class TestDatasetLoaderFiltering:
    """Tests for filtering functionality."""

    def test_get_test_case_by_id(self, sample_jsonl_dataset):
        """Test finding test case by ID."""
        loader = DatasetLoader(sample_jsonl_dataset.parent)

        tc = loader.get_test_case_by_id(
            sample_jsonl_dataset.name, "icd", "icd_test_001"
        )

        assert tc is not None
        assert tc.id == "icd_test_001"

    def test_get_test_case_by_id_not_found(self, sample_jsonl_dataset):
        """Test that None is returned for nonexistent ID."""
        loader = DatasetLoader(sample_jsonl_dataset.parent)

        tc = loader.get_test_case_by_id(
            sample_jsonl_dataset.name, "icd", "nonexistent_id"
        )

        assert tc is None

    def test_filter_by_tags(self, sample_jsonl_dataset):
        """Test filtering by tags."""
        loader = DatasetLoader(sample_jsonl_dataset.parent)

        # Filter for diabetes tag
        results = loader.filter_by_tags(
            sample_jsonl_dataset.name, "icd", ["diabetes"]
        )

        assert len(results) == 1
        assert "diabetes" in results[0].tags

    def test_filter_by_tags_match_all(self, tmp_path, sample_icd_test_case):
        """Test filtering requiring all tags."""
        file_path = tmp_path / "test.jsonl"
        with open(file_path, "w") as f:
            f.write(json.dumps({**sample_icd_test_case, "tags": ["diabetes", "chronic"]}) + "\n")
            f.write(json.dumps({**sample_icd_test_case, "id": "icd_002", "tags": ["diabetes"]}) + "\n")

        loader = DatasetLoader(tmp_path)

        # Require both tags
        results = loader.filter_by_tags(
            "test.jsonl", "icd", ["diabetes", "chronic"], match_all=True
        )

        assert len(results) == 1

    def test_filter_by_specialty(self, sample_jsonl_dataset):
        """Test filtering by specialty."""
        loader = DatasetLoader(sample_jsonl_dataset.parent)

        results = loader.filter_by_specialty(
            sample_jsonl_dataset.name, "icd", ["cardiology"]
        )

        assert len(results) == 1
        assert results[0].specialty.value == "cardiology"


class TestCreateSampleDatasets:
    """Tests for sample dataset creation."""

    def test_create_sample_datasets(self, tmp_path):
        """Test sample dataset creation."""
        create_sample_datasets(tmp_path / "samples")

        # Check files were created
        assert (tmp_path / "samples" / "icd_samples.jsonl").exists()
        assert (tmp_path / "samples" / "hcc_samples.jsonl").exists()
        assert (tmp_path / "samples" / "gap_samples.jsonl").exists()

        # Verify content is valid
        loader = DatasetLoader(tmp_path / "samples")
        icd_cases = loader.load_jsonl("icd_samples.jsonl", "icd")
        assert len(icd_cases) == 1
        assert isinstance(icd_cases[0], ICDTestCase)
