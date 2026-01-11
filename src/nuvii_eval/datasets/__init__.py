"""Dataset schemas and loaders for evaluation test cases."""

from nuvii_eval.datasets.loader import DatasetLoader, DatasetLoadError, load_dataset
from nuvii_eval.datasets.schemas import (
    BaseTestCase,
    Complexity,
    CPTTestCase,
    EMTestCase,
    ExpectedGap,
    GapTestCase,
    HCCTestCase,
    ICDTestCase,
    QueryQualityCriteria,
    QueryTestCase,
    Specialty,
)
from nuvii_eval.datasets.validator import DatasetValidator, ValidationResult

__all__ = [
    # Loader
    "DatasetLoader",
    "DatasetLoadError",
    "load_dataset",
    # Validator
    "DatasetValidator",
    "ValidationResult",
    # Schemas
    "BaseTestCase",
    "ICDTestCase",
    "HCCTestCase",
    "GapTestCase",
    "QueryTestCase",
    "EMTestCase",
    "CPTTestCase",
    "ExpectedGap",
    "QueryQualityCriteria",
    "Specialty",
    "Complexity",
]
