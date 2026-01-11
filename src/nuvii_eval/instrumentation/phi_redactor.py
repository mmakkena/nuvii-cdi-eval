"""
PHI (Protected Health Information) redaction utilities.

Provides rule-based and pattern-based redaction of sensitive health information
from text data to ensure HIPAA compliance in logs, traces, and artifacts.

Note: This is a rule-based implementation. For production use with real PHI,
consider augmenting with a trained NER model for higher accuracy.
"""

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable

import structlog

logger = structlog.get_logger(__name__)


class PHICategory(str, Enum):
    """Categories of PHI as defined by HIPAA."""

    NAME = "name"
    DATE = "date"
    PHONE = "phone"
    EMAIL = "email"
    SSN = "ssn"
    MRN = "mrn"
    ACCOUNT = "account"
    LICENSE = "license"
    VEHICLE = "vehicle"
    DEVICE = "device"
    URL = "url"
    IP = "ip"
    BIOMETRIC = "biometric"
    PHOTO = "photo"
    ADDRESS = "address"
    AGE = "age"
    LOCATION = "location"


@dataclass
class RedactionPattern:
    """
    Pattern definition for PHI redaction.

    Attributes:
        name: Unique identifier for the pattern
        pattern: Regular expression pattern
        replacement: Replacement text (can include group references)
        category: PHI category this pattern detects
        flags: Regex flags (default: 0)
        priority: Higher priority patterns are applied first
    """

    name: str
    pattern: str
    replacement: str
    category: PHICategory
    flags: int = 0
    priority: int = 0


@dataclass
class RedactionResult:
    """Result of a redaction operation."""

    original_length: int
    redacted_length: int
    redaction_count: int
    categories_found: set[PHICategory] = field(default_factory=set)
    pattern_matches: dict[str, int] = field(default_factory=dict)


class PHIRedactor:
    """
    Redacts Protected Health Information from text.

    Supports multiple redaction strategies:
    - Pattern-based: Regex patterns for structured data (SSN, MRN, dates)
    - Contextual: Patterns that consider surrounding context
    - Custom: User-defined patterns

    Usage:
        redactor = PHIRedactor()
        safe_text = redactor.redact("Patient John Smith, SSN: 123-45-6789")
        # Returns: "Patient [NAME], SSN: [SSN]"

        # Get redaction statistics
        result = redactor.redact_with_stats("Patient DOB: 01/15/1990")
        print(result.categories_found)  # {PHICategory.DATE}
    """

    # Default patterns for common PHI types
    DEFAULT_PATTERNS: list[RedactionPattern] = [
        # SSN patterns
        RedactionPattern(
            name="ssn_dashed",
            pattern=r"\b\d{3}-\d{2}-\d{4}\b",
            replacement="[SSN]",
            category=PHICategory.SSN,
            priority=10,
        ),
        RedactionPattern(
            name="ssn_no_dash",
            pattern=r"\b(?<!\d)\d{9}(?!\d)\b",
            replacement="[SSN]",
            category=PHICategory.SSN,
            priority=10,
        ),
        # MRN patterns (various formats)
        RedactionPattern(
            name="mrn_labeled",
            pattern=r"\bMRN[:\s#]*(\d{6,12})\b",
            replacement="MRN: [MRN]",
            category=PHICategory.MRN,
            flags=re.IGNORECASE,
            priority=10,
        ),
        RedactionPattern(
            name="mrn_alphanumeric",
            pattern=r"\b[A-Z]{2,3}\d{6,10}\b",
            replacement="[MRN]",
            category=PHICategory.MRN,
            priority=8,
        ),
        # Date patterns
        RedactionPattern(
            name="date_slash_mdy",
            pattern=r"\b(0?[1-9]|1[0-2])/(0?[1-9]|[12]\d|3[01])/(\d{2}|\d{4})\b",
            replacement="[DATE]",
            category=PHICategory.DATE,
            priority=5,
        ),
        RedactionPattern(
            name="date_slash_dmy",
            pattern=r"\b(0?[1-9]|[12]\d|3[01])/(0?[1-9]|1[0-2])/(\d{2}|\d{4})\b",
            replacement="[DATE]",
            category=PHICategory.DATE,
            priority=5,
        ),
        RedactionPattern(
            name="date_dash",
            pattern=r"\b\d{1,2}-\d{1,2}-\d{2,4}\b",
            replacement="[DATE]",
            category=PHICategory.DATE,
            priority=5,
        ),
        RedactionPattern(
            name="date_iso",
            pattern=r"\b\d{4}-\d{2}-\d{2}\b",
            replacement="[DATE]",
            category=PHICategory.DATE,
            priority=5,
        ),
        RedactionPattern(
            name="date_written",
            pattern=r"\b(?:January|February|March|April|May|June|July|August|September|October|November|December|Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\.?\s+\d{1,2},?\s+\d{4}\b",
            replacement="[DATE]",
            category=PHICategory.DATE,
            flags=re.IGNORECASE,
            priority=5,
        ),
        RedactionPattern(
            name="dob_labeled",
            pattern=r"\b(?:DOB|Date of Birth|Birth Date)[:\s]*[\d/\-]+\b",
            replacement="DOB: [DATE]",
            category=PHICategory.DATE,
            flags=re.IGNORECASE,
            priority=10,
        ),
        # Phone patterns
        RedactionPattern(
            name="phone_standard",
            pattern=r"\b\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b",
            replacement="[PHONE]",
            category=PHICategory.PHONE,
            priority=5,
        ),
        RedactionPattern(
            name="phone_international",
            pattern=r"\b\+1[-.\s]?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b",
            replacement="[PHONE]",
            category=PHICategory.PHONE,
            priority=5,
        ),
        # Email pattern
        RedactionPattern(
            name="email",
            pattern=r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
            replacement="[EMAIL]",
            category=PHICategory.EMAIL,
            priority=5,
        ),
        # Address patterns
        RedactionPattern(
            name="zip_code",
            pattern=r"\b\d{5}(?:-\d{4})?\b",
            replacement="[ZIP]",
            category=PHICategory.ADDRESS,
            priority=3,
        ),
        RedactionPattern(
            name="street_address",
            pattern=r"\b\d+\s+(?:[A-Z][a-z]+\s+){1,3}(?:Street|St|Avenue|Ave|Road|Rd|Boulevard|Blvd|Drive|Dr|Lane|Ln|Way|Court|Ct|Circle|Cir|Place|Pl)\.?\b",
            replacement="[ADDRESS]",
            category=PHICategory.ADDRESS,
            flags=re.IGNORECASE,
            priority=5,
        ),
        # Age with context
        RedactionPattern(
            name="age_years_old",
            pattern=r"\b(\d{1,3})[\s-]?(?:year[s]?[\s-]?old|y/?o|yo)\b",
            replacement="[AGE]-year-old",
            category=PHICategory.AGE,
            flags=re.IGNORECASE,
            priority=5,
        ),
        RedactionPattern(
            name="age_labeled",
            pattern=r"\b(?:age[d]?)[:\s]*(\d{1,3})\b",
            replacement="age: [AGE]",
            category=PHICategory.AGE,
            flags=re.IGNORECASE,
            priority=5,
        ),
        # Account/ID numbers
        RedactionPattern(
            name="account_number",
            pattern=r"\b(?:Account|Acct|Member)[\s#:]*(\d{8,16})\b",
            replacement="[ACCOUNT]",
            category=PHICategory.ACCOUNT,
            flags=re.IGNORECASE,
            priority=5,
        ),
        # IP addresses
        RedactionPattern(
            name="ip_address",
            pattern=r"\b(?:\d{1,3}\.){3}\d{1,3}\b",
            replacement="[IP]",
            category=PHICategory.IP,
            priority=3,
        ),
    ]

    # Name patterns (more conservative - higher false positive risk)
    NAME_PATTERNS: list[RedactionPattern] = [
        RedactionPattern(
            name="name_titled",
            pattern=r"\b(?:Dr|Mr|Mrs|Ms|Miss|Prof)\.?\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?\b",
            replacement="[NAME]",
            category=PHICategory.NAME,
            priority=8,
        ),
        RedactionPattern(
            name="name_patient_label",
            pattern=r"\b(?:Patient|Pt)[:]\s*([A-Z][a-z]+\s+[A-Z][a-z]+)\b",
            replacement="Patient: [NAME]",
            category=PHICategory.NAME,
            priority=10,
        ),
        RedactionPattern(
            name="name_field",
            pattern=r"\b(?:Name|Patient Name)[:\s]+([A-Z][a-z]+(?:\s+[A-Z]\.?)?\s+[A-Z][a-z]+)\b",
            replacement="Name: [NAME]",
            category=PHICategory.NAME,
            flags=re.IGNORECASE,
            priority=10,
        ),
    ]

    def __init__(
        self,
        include_names: bool = True,
        include_ages: bool = True,
        custom_patterns: list[RedactionPattern] | None = None,
        excluded_categories: set[PHICategory] | None = None,
        clinical_term_whitelist: set[str] | None = None,
    ):
        """
        Initialize the PHI redactor.

        Args:
            include_names: Include name detection patterns (higher false positive risk)
            include_ages: Include age detection patterns
            custom_patterns: Additional custom patterns to include
            excluded_categories: PHI categories to exclude from redaction
            clinical_term_whitelist: Terms to preserve (not redact)
        """
        self.excluded_categories = excluded_categories or set()
        self.clinical_term_whitelist = clinical_term_whitelist or self._default_whitelist()

        # Build pattern list
        patterns = list(self.DEFAULT_PATTERNS)

        if include_names:
            patterns.extend(self.NAME_PATTERNS)

        if not include_ages:
            patterns = [p for p in patterns if p.category != PHICategory.AGE]

        if custom_patterns:
            patterns.extend(custom_patterns)

        # Filter excluded categories
        patterns = [p for p in patterns if p.category not in self.excluded_categories]

        # Sort by priority (higher first)
        patterns.sort(key=lambda p: -p.priority)

        # Compile patterns - store tuple of (compiled_regex, replacement, name, category, priority)
        self._patterns = [
            (
                re.compile(p.pattern, p.flags),
                p.replacement,
                p.name,
                p.category,
                p.priority,
            )
            for p in patterns
        ]

        logger.debug(
            "phi_redactor_initialized",
            pattern_count=len(self._patterns),
            include_names=include_names,
        )

    def _default_whitelist(self) -> set[str]:
        """Default clinical terms that should not be redacted."""
        return {
            # Medical terms that might match name patterns
            "diabetes",
            "mellitus",
            "hypertension",
            "pneumonia",
            "cancer",
            "carcinoma",
            "melanoma",
            "syndrome",
            "disease",
            "disorder",
            "failure",
            "chronic",
            "acute",
            # Drug names that might match patterns
            "aspirin",
            "metformin",
            "insulin",
            "lisinopril",
            "metoprolol",
        }

    def redact(self, text: str) -> str:
        """
        Redact PHI from text.

        Args:
            text: Input text potentially containing PHI

        Returns:
            Text with PHI replaced by category placeholders
        """
        if not text:
            return text

        result = text

        for compiled, replacement, name, category, _priority in self._patterns:
            result = compiled.sub(replacement, result)

        return result

    def redact_with_stats(self, text: str) -> tuple[str, RedactionResult]:
        """
        Redact PHI and return statistics about what was redacted.

        Args:
            text: Input text potentially containing PHI

        Returns:
            Tuple of (redacted_text, RedactionResult)
        """
        if not text:
            return text, RedactionResult(
                original_length=0,
                redacted_length=0,
                redaction_count=0,
            )

        result = text
        categories_found: set[PHICategory] = set()
        pattern_matches: dict[str, int] = {}
        total_redactions = 0

        for compiled, replacement, name, category, _priority in self._patterns:
            matches = compiled.findall(result)
            if matches:
                match_count = len(matches) if isinstance(matches[0], str) else len(matches)
                pattern_matches[name] = match_count
                categories_found.add(category)
                total_redactions += match_count
                result = compiled.sub(replacement, result)

        stats = RedactionResult(
            original_length=len(text),
            redacted_length=len(result),
            redaction_count=total_redactions,
            categories_found=categories_found,
            pattern_matches=pattern_matches,
        )

        return result, stats

    def redact_dict(
        self,
        data: dict,
        keys_to_redact: set[str] | None = None,
        recursive: bool = True,
    ) -> dict:
        """
        Redact PHI from dictionary values.

        Args:
            data: Dictionary potentially containing PHI in values
            keys_to_redact: Specific keys to redact (None = auto-detect)
            recursive: Whether to recursively process nested dicts

        Returns:
            Dictionary with PHI redacted from specified keys
        """
        # Default keys that commonly contain PHI
        default_keys = {
            "clinical_note",
            "note",
            "text",
            "content",
            "evidence",
            "evidence_spans",
            "query_text",
            "message",
            "description",
            "patient_name",
            "name",
            "address",
            "ssn",
            "mrn",
            "dob",
        }

        keys_to_check = keys_to_redact or default_keys
        redacted = {}

        for key, value in data.items():
            key_lower = key.lower()

            if key_lower in keys_to_check or any(k in key_lower for k in keys_to_check):
                if isinstance(value, str):
                    redacted[key] = self.redact(value)
                elif isinstance(value, list):
                    redacted[key] = [
                        self.redact(v) if isinstance(v, str) else v for v in value
                    ]
                elif isinstance(value, dict) and recursive:
                    redacted[key] = self.redact_dict(value, keys_to_redact, recursive)
                else:
                    redacted[key] = value
            elif isinstance(value, dict) and recursive:
                redacted[key] = self.redact_dict(value, keys_to_redact, recursive)
            else:
                redacted[key] = value

        return redacted

    def get_redaction_report(self, text: str) -> dict:
        """
        Generate a detailed redaction report for text.

        Args:
            text: Input text to analyze

        Returns:
            Dictionary containing redaction analysis
        """
        _, stats = self.redact_with_stats(text)

        return {
            "original_length": stats.original_length,
            "redacted_length": stats.redacted_length,
            "total_redactions": stats.redaction_count,
            "categories_found": [c.value for c in stats.categories_found],
            "pattern_matches": stats.pattern_matches,
            "redaction_density": (
                stats.redaction_count / (stats.original_length / 100)
                if stats.original_length > 0
                else 0
            ),
        }

    def is_safe(self, text: str) -> bool:
        """
        Check if text appears to be free of PHI.

        Args:
            text: Text to check

        Returns:
            True if no PHI patterns detected
        """
        _, stats = self.redact_with_stats(text)
        return stats.redaction_count == 0

    def add_pattern(self, pattern: RedactionPattern) -> None:
        """
        Add a custom pattern at runtime.

        Args:
            pattern: Pattern to add
        """
        if pattern.category not in self.excluded_categories:
            # Insert based on priority (higher priority first)
            compiled = (
                re.compile(pattern.pattern, pattern.flags),
                pattern.replacement,
                pattern.name,
                pattern.category,
                pattern.priority,
            )

            # Find insertion point - patterns are sorted by priority (highest first)
            for i, (_, _, _, _, existing_priority) in enumerate(self._patterns):
                if pattern.priority > existing_priority:
                    self._patterns.insert(i, compiled)
                    return

            self._patterns.append(compiled)


class RedactionAuditLog:
    """
    Audit logger for PHI redaction operations.

    Tracks what was redacted without storing the actual PHI.
    """

    def __init__(self):
        self.entries: list[dict] = []

    def log_redaction(
        self,
        source: str,
        stats: RedactionResult,
        context: str | None = None,
    ) -> None:
        """Log a redaction operation."""
        self.entries.append(
            {
                "timestamp": datetime.utcnow().isoformat(),
                "source": source,
                "redaction_count": stats.redaction_count,
                "categories": [c.value for c in stats.categories_found],
                "patterns": stats.pattern_matches,
                "context": context,
            }
        )

    def get_summary(self) -> dict:
        """Get summary of all redaction operations."""
        if not self.entries:
            return {"total_operations": 0}

        total_redactions = sum(e["redaction_count"] for e in self.entries)
        all_categories: set[str] = set()
        for e in self.entries:
            all_categories.update(e["categories"])

        return {
            "total_operations": len(self.entries),
            "total_redactions": total_redactions,
            "categories_found": list(all_categories),
            "average_redactions_per_op": total_redactions / len(self.entries),
        }


# Import datetime for audit log
from datetime import datetime
