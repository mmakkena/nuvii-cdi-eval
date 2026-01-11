"""
Dataset validation utilities.

Provides validation for test case datasets.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import structlog
import yaml
from pydantic import ValidationError

from nuvii_eval.datasets.schemas import TEST_CASE_TYPES

logger = structlog.get_logger(__name__)


# =============================================================================
# Validation Results
# =============================================================================


@dataclass
class ValidationResult:
    """Result of validating a dataset."""

    is_valid: bool
    test_count: int
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    fixable_issues: list[dict[str, Any]] = field(default_factory=list)


# =============================================================================
# Dataset Validator
# =============================================================================


class DatasetValidator:
    """
    Validates test case datasets.

    Checks schema conformance and data quality.
    """

    def __init__(self, strict: bool = False):
        """
        Initialize the validator.

        Args:
            strict: Enable strict validation mode
        """
        self.strict = strict

    def validate(self, path: str) -> ValidationResult:
        """
        Validate a dataset file.

        Args:
            path: Path to dataset file

        Returns:
            ValidationResult with errors and warnings
        """
        errors = []
        warnings = []
        fixable_issues = []
        test_count = 0

        try:
            # Load file
            file_path = Path(path)
            with open(file_path) as f:
                if file_path.suffix in [".yaml", ".yml"]:
                    data = yaml.safe_load(f)
                else:
                    data = json.load(f)

            # Get test cases
            test_cases = data.get("test_cases", data) if isinstance(data, dict) else data

            if not isinstance(test_cases, list):
                errors.append("Dataset must be a list of test cases or contain 'test_cases' key")
                return ValidationResult(
                    is_valid=False,
                    test_count=0,
                    errors=errors,
                )

            test_count = len(test_cases)

            if test_count == 0:
                warnings.append("Dataset is empty")

            # Validate each test case
            seen_ids = set()
            for i, tc_data in enumerate(test_cases):
                tc_errors, tc_warnings, tc_fixes = self._validate_test_case(
                    tc_data, i, seen_ids
                )
                errors.extend(tc_errors)
                warnings.extend(tc_warnings)
                fixable_issues.extend(tc_fixes)

        except json.JSONDecodeError as e:
            errors.append(f"Invalid JSON: {e}")
        except yaml.YAMLError as e:
            errors.append(f"Invalid YAML: {e}")
        except FileNotFoundError:
            errors.append(f"File not found: {path}")
        except Exception as e:
            errors.append(f"Validation error: {e}")

        return ValidationResult(
            is_valid=len(errors) == 0,
            test_count=test_count,
            errors=errors,
            warnings=warnings,
            fixable_issues=fixable_issues,
        )

    def _validate_test_case(
        self,
        data: dict[str, Any],
        index: int,
        seen_ids: set,
    ) -> tuple[list[str], list[str], list[dict]]:
        """Validate a single test case."""
        errors = []
        warnings = []
        fixable_issues = []

        # Check required fields
        if "id" not in data:
            errors.append(f"Test case {index}: missing 'id' field")
            return errors, warnings, fixable_issues

        test_id = data["id"]

        # Check for duplicate IDs
        if test_id in seen_ids:
            errors.append(f"Duplicate test case ID: {test_id}")
        seen_ids.add(test_id)

        # Check clinical note
        if "clinical_note" not in data:
            errors.append(f"Test case {test_id}: missing 'clinical_note' field")
        elif len(data.get("clinical_note", "")) < 50:
            if self.strict:
                errors.append(f"Test case {test_id}: clinical_note too short (< 50 chars)")
            else:
                warnings.append(f"Test case {test_id}: clinical_note is short")

        # Determine task type and validate schema
        task_type = self._infer_task_type(data)
        if task_type and task_type in TEST_CASE_TYPES:
            try:
                TEST_CASE_TYPES[task_type](**data)
            except ValidationError as e:
                for error in e.errors():
                    loc = ".".join(str(l) for l in error["loc"])
                    msg = error["msg"]
                    errors.append(f"Test case {test_id}: {loc} - {msg}")

                    # Check if fixable
                    if error["type"] == "missing":
                        fixable_issues.append({
                            "test_id": test_id,
                            "field": loc,
                            "issue": "missing_field",
                        })

        # Check for common issues
        if "tags" in data and not isinstance(data["tags"], list):
            warnings.append(f"Test case {test_id}: 'tags' should be a list")
            fixable_issues.append({
                "test_id": test_id,
                "field": "tags",
                "issue": "should_be_list",
            })

        return errors, warnings, fixable_issues

    def _infer_task_type(self, data: dict[str, Any]) -> str | None:
        """Infer task type from test case data."""
        if "expected_icd_codes" in data:
            return "icd"
        elif "expected_hccs" in data:
            return "hcc"
        elif "expected_gaps" in data:
            return "gap"
        elif "quality_criteria" in data:
            return "query"
        elif "expected_code" in data and "expected_level" in data:
            return "em"
        elif "expected_cpt_codes" in data:
            return "cpt"
        return None

    def fix(self, path: str, issues: list[dict[str, Any]]) -> None:
        """
        Attempt to fix issues in a dataset file.

        Args:
            path: Path to dataset file
            issues: List of fixable issues
        """
        file_path = Path(path)

        # Load data
        with open(file_path) as f:
            if file_path.suffix in [".yaml", ".yml"]:
                data = yaml.safe_load(f)
            else:
                data = json.load(f)

        # Get test cases
        test_cases = data.get("test_cases", data) if isinstance(data, dict) else data
        test_case_map = {tc["id"]: tc for tc in test_cases if "id" in tc}

        # Apply fixes
        for issue in issues:
            test_id = issue["test_id"]
            if test_id not in test_case_map:
                continue

            tc = test_case_map[test_id]

            if issue["issue"] == "should_be_list":
                field = issue["field"]
                if field in tc and not isinstance(tc[field], list):
                    tc[field] = [tc[field]] if tc[field] else []

        # Save updated data
        with open(file_path, "w") as f:
            if file_path.suffix in [".yaml", ".yml"]:
                yaml.dump(data, f, default_flow_style=False, allow_unicode=True)
            else:
                json.dump(data, f, indent=2)

        logger.info("dataset_fixed", path=path, fixes=len(issues))
