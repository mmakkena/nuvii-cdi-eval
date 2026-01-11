"""Tests for PHI redaction utilities."""

import pytest

from nuvii_eval.instrumentation.phi_redactor import (
    PHICategory,
    PHIRedactor,
    RedactionAuditLog,
    RedactionPattern,
)


class TestPHIRedactorBasic:
    """Basic PHI redaction tests."""

    def test_redact_ssn_dashed(self):
        """Test SSN redaction with dashes."""
        redactor = PHIRedactor()
        text = "Patient SSN: 123-45-6789"

        result = redactor.redact(text)

        assert "123-45-6789" not in result
        assert "[SSN]" in result

    def test_redact_ssn_no_dash(self):
        """Test SSN redaction without dashes."""
        redactor = PHIRedactor()
        text = "SSN number is 123456789 for this patient"

        result = redactor.redact(text)

        assert "123456789" not in result
        assert "[SSN]" in result

    def test_redact_phone_number(self):
        """Test phone number redaction."""
        redactor = PHIRedactor()

        # Various formats
        test_cases = [
            "Call 555-123-4567",
            "Phone: (555) 123-4567",
            "Contact: 555.123.4567",
            "Tel: 5551234567",
        ]

        for text in test_cases:
            result = redactor.redact(text)
            assert "[PHONE]" in result, f"Failed for: {text}"

    def test_redact_email(self):
        """Test email redaction."""
        redactor = PHIRedactor()
        text = "Contact patient at john.doe@email.com for follow-up"

        result = redactor.redact(text)

        assert "john.doe@email.com" not in result
        assert "[EMAIL]" in result

    def test_redact_empty_string(self):
        """Test redaction of empty string."""
        redactor = PHIRedactor()

        assert redactor.redact("") == ""
        assert redactor.redact(None) is None


class TestPHIRedactorDates:
    """Date redaction tests."""

    def test_redact_date_slash_format(self):
        """Test date redaction with slash format."""
        redactor = PHIRedactor()

        test_cases = [
            "DOB: 01/15/1990",
            "Date: 1/5/90",
            "Born on 12/31/2000",
        ]

        for text in test_cases:
            result = redactor.redact(text)
            assert "[DATE]" in result, f"Failed for: {text}"

    def test_redact_date_dash_format(self):
        """Test date redaction with dash format."""
        redactor = PHIRedactor()
        text = "Patient DOB: 1990-01-15"

        result = redactor.redact(text)

        assert "1990-01-15" not in result
        assert "[DATE]" in result

    def test_redact_date_written_format(self):
        """Test date redaction with written format."""
        redactor = PHIRedactor()

        test_cases = [
            "January 15, 1990",
            "Jan 15, 1990",
            "December 31 2000",
        ]

        for text in test_cases:
            result = redactor.redact(text)
            assert "[DATE]" in result, f"Failed for: {text}"

    def test_redact_dob_labeled(self):
        """Test labeled DOB redaction."""
        redactor = PHIRedactor()
        text = "DOB: 01/15/1990, MRN: 12345678"

        result = redactor.redact(text)

        assert "01/15/1990" not in result


class TestPHIRedactorMRN:
    """MRN (Medical Record Number) redaction tests."""

    def test_redact_mrn_labeled(self):
        """Test labeled MRN redaction."""
        redactor = PHIRedactor()

        test_cases = [
            "MRN: 12345678",
            "MRN#12345678",
            "mrn 123456789",
        ]

        for text in test_cases:
            result = redactor.redact(text)
            # Should not contain the number
            assert not any(c.isdigit() and len(c) >= 6 for c in result.split())

    def test_redact_mrn_alphanumeric(self):
        """Test alphanumeric MRN redaction."""
        redactor = PHIRedactor()
        text = "Patient ID: ABC12345678"

        result = redactor.redact(text)

        assert "ABC12345678" not in result
        assert "[MRN]" in result


class TestPHIRedactorNames:
    """Name redaction tests."""

    def test_redact_titled_name(self):
        """Test name with title redaction."""
        redactor = PHIRedactor(include_names=True)

        test_cases = [
            "Dr. John Smith",
            "Mr. James Wilson",
            "Mrs. Mary Johnson",
            "Ms. Jane Doe",
        ]

        for text in test_cases:
            result = redactor.redact(text)
            assert "[NAME]" in result, f"Failed for: {text}"

    def test_redact_patient_labeled_name(self):
        """Test patient-labeled name redaction."""
        redactor = PHIRedactor(include_names=True)
        text = "Patient: John Smith presents with chest pain"

        result = redactor.redact(text)

        assert "John Smith" not in result

    def test_no_names_when_disabled(self):
        """Test that names are not redacted when disabled."""
        redactor = PHIRedactor(include_names=False)
        text = "Dr. John Smith prescribed medication"

        result = redactor.redact(text)

        # Name should still be present (titled names won't be caught)
        assert "John Smith" in result


class TestPHIRedactorAddress:
    """Address redaction tests."""

    def test_redact_zip_code(self):
        """Test ZIP code redaction."""
        redactor = PHIRedactor()

        test_cases = [
            "ZIP: 12345",
            "Zip code 12345-6789",
        ]

        for text in test_cases:
            result = redactor.redact(text)
            assert "[ZIP]" in result, f"Failed for: {text}"

    def test_redact_street_address(self):
        """Test street address redaction."""
        redactor = PHIRedactor()

        test_cases = [
            "123 Main Street",
            "456 Oak Avenue",
            "789 First Rd",
        ]

        for text in test_cases:
            result = redactor.redact(text)
            assert "[ADDRESS]" in result, f"Failed for: {text}"


class TestPHIRedactorAge:
    """Age redaction tests."""

    def test_redact_age_years_old(self):
        """Test age with 'years old' redaction."""
        redactor = PHIRedactor(include_ages=True)

        test_cases = [
            "72 year old female",
            "72-year-old male",
            "45 y/o patient",
            "65 yo presents",
        ]

        for text in test_cases:
            result = redactor.redact(text)
            assert "[AGE]" in result, f"Failed for: {text}"

    def test_redact_age_labeled(self):
        """Test labeled age redaction."""
        redactor = PHIRedactor(include_ages=True)
        text = "Age: 72, presenting with chest pain"

        result = redactor.redact(text)

        assert "[AGE]" in result

    def test_no_age_when_disabled(self):
        """Test that ages are not redacted when disabled."""
        redactor = PHIRedactor(include_ages=False)
        text = "72 year old female with diabetes"

        result = redactor.redact(text)

        assert "72" in result


class TestPHIRedactorWithStats:
    """Tests for redaction with statistics."""

    def test_redact_with_stats_counts(self):
        """Test that stats correctly count redactions."""
        redactor = PHIRedactor()
        text = "SSN: 123-45-6789, Phone: 555-123-4567, Email: test@test.com"

        result, stats = redactor.redact_with_stats(text)

        assert stats.redaction_count >= 3
        assert PHICategory.SSN in stats.categories_found
        assert PHICategory.PHONE in stats.categories_found
        assert PHICategory.EMAIL in stats.categories_found

    def test_redact_with_stats_lengths(self):
        """Test that stats track length changes."""
        redactor = PHIRedactor()
        text = "Patient SSN: 123-45-6789"

        result, stats = redactor.redact_with_stats(text)

        assert stats.original_length == len(text)
        assert stats.redacted_length == len(result)
        assert stats.redacted_length != stats.original_length

    def test_redact_with_stats_no_phi(self):
        """Test stats when no PHI is found."""
        redactor = PHIRedactor()
        text = "Patient presents with chest pain and shortness of breath"

        result, stats = redactor.redact_with_stats(text)

        assert stats.redaction_count == 0
        assert len(stats.categories_found) == 0
        assert result == text


class TestPHIRedactorDict:
    """Tests for dictionary redaction."""

    def test_redact_dict_basic(self):
        """Test basic dictionary redaction."""
        redactor = PHIRedactor()
        data = {
            "clinical_note": "Patient SSN: 123-45-6789",
            "id": "test_001",
        }

        result = redactor.redact_dict(data)

        assert "[SSN]" in result["clinical_note"]
        assert result["id"] == "test_001"

    def test_redact_dict_nested(self):
        """Test nested dictionary redaction."""
        redactor = PHIRedactor()
        data = {
            "patient": {
                "note": "DOB: 01/15/1990",
                "mrn": "MRN: 12345678",
            },
            "metadata": {"source": "test"},
        }

        result = redactor.redact_dict(data, recursive=True)

        assert "[DATE]" in result["patient"]["note"]
        assert result["metadata"]["source"] == "test"

    def test_redact_dict_list_values(self):
        """Test dictionary with list values."""
        redactor = PHIRedactor()
        data = {
            "evidence_spans": [
                "Patient DOB: 01/15/1990",
                "SSN: 123-45-6789",
            ]
        }

        result = redactor.redact_dict(data)

        assert "[DATE]" in result["evidence_spans"][0]
        assert "[SSN]" in result["evidence_spans"][1]

    def test_redact_dict_custom_keys(self):
        """Test redaction with custom key set."""
        redactor = PHIRedactor()
        data = {
            "custom_field": "SSN: 123-45-6789",
            "other_field": "SSN: 987-65-4321",
        }

        result = redactor.redact_dict(data, keys_to_redact={"custom_field"})

        assert "[SSN]" in result["custom_field"]
        # other_field should not be redacted (not in keys_to_redact)
        assert "987-65-4321" in result["other_field"]


class TestPHIRedactorCustomPatterns:
    """Tests for custom pattern support."""

    def test_add_custom_pattern(self):
        """Test adding a custom pattern."""
        redactor = PHIRedactor()

        custom = RedactionPattern(
            name="custom_id",
            pattern=r"\bCUST-\d{6}\b",
            replacement="[CUSTOM_ID]",
            category=PHICategory.ACCOUNT,
            priority=10,
        )
        redactor.add_pattern(custom)

        text = "Customer ID: CUST-123456"
        result = redactor.redact(text)

        assert "[CUSTOM_ID]" in result

    def test_custom_patterns_in_init(self):
        """Test providing custom patterns at initialization."""
        custom = RedactionPattern(
            name="policy_number",
            pattern=r"\bPOL-\d{8}\b",
            replacement="[POLICY]",
            category=PHICategory.ACCOUNT,
        )

        redactor = PHIRedactor(custom_patterns=[custom])
        text = "Policy: POL-12345678"
        result = redactor.redact(text)

        assert "[POLICY]" in result


class TestPHIRedactorExclusions:
    """Tests for category exclusions."""

    def test_exclude_category(self):
        """Test excluding a category from redaction."""
        redactor = PHIRedactor(excluded_categories={PHICategory.DATE})
        text = "DOB: 01/15/1990, SSN: 123-45-6789"

        result = redactor.redact(text)

        # Date should NOT be redacted
        assert "01/15/1990" in result or "1990" in result
        # SSN should still be redacted
        assert "[SSN]" in result

    def test_exclude_multiple_categories(self):
        """Test excluding multiple categories."""
        redactor = PHIRedactor(
            excluded_categories={PHICategory.DATE, PHICategory.AGE}
        )
        text = "72 year old, DOB: 01/15/1990, SSN: 123-45-6789"

        result = redactor.redact(text)

        # Date and age should NOT be redacted
        assert "72" in result
        # SSN should still be redacted
        assert "[SSN]" in result


class TestPHIRedactorSafetyChecks:
    """Tests for safety checking functionality."""

    def test_is_safe_true(self):
        """Test is_safe returns True for clean text."""
        redactor = PHIRedactor()
        text = "Patient presents with chest pain and shortness of breath"

        assert redactor.is_safe(text) is True

    def test_is_safe_false(self):
        """Test is_safe returns False for text with PHI."""
        redactor = PHIRedactor()
        text = "Patient SSN: 123-45-6789"

        assert redactor.is_safe(text) is False

    def test_get_redaction_report(self):
        """Test getting a detailed redaction report."""
        redactor = PHIRedactor()
        text = "SSN: 123-45-6789, Phone: 555-123-4567"

        report = redactor.get_redaction_report(text)

        assert report["total_redactions"] >= 2
        assert "ssn" in [c.lower() for c in report["categories_found"]]
        assert "phone" in [c.lower() for c in report["categories_found"]]
        assert report["original_length"] == len(text)


class TestRedactionAuditLog:
    """Tests for the redaction audit log."""

    def test_log_redaction(self):
        """Test logging a redaction operation."""
        audit_log = RedactionAuditLog()
        redactor = PHIRedactor()

        text = "SSN: 123-45-6789"
        _, stats = redactor.redact_with_stats(text)

        audit_log.log_redaction("test_source", stats, "test_context")

        assert len(audit_log.entries) == 1
        assert audit_log.entries[0]["source"] == "test_source"
        assert audit_log.entries[0]["redaction_count"] == stats.redaction_count

    def test_get_summary(self):
        """Test getting audit log summary."""
        audit_log = RedactionAuditLog()
        redactor = PHIRedactor()

        # Log multiple redactions
        for i in range(3):
            _, stats = redactor.redact_with_stats(f"SSN: 123-45-678{i}")
            audit_log.log_redaction(f"source_{i}", stats)

        summary = audit_log.get_summary()

        assert summary["total_operations"] == 3
        assert summary["total_redactions"] >= 3
        assert "ssn" in [c.lower() for c in summary["categories_found"]]

    def test_empty_audit_log_summary(self):
        """Test summary of empty audit log."""
        audit_log = RedactionAuditLog()

        summary = audit_log.get_summary()

        assert summary["total_operations"] == 0


class TestPHIRedactorClinicalContext:
    """Tests for clinical context handling."""

    def test_preserves_medical_terms(self):
        """Test that medical terms are not incorrectly redacted."""
        redactor = PHIRedactor()

        # These should NOT be redacted
        medical_texts = [
            "Patient has diabetes mellitus",
            "Diagnosis: hypertension",
            "Prescribed metformin 500mg",
            "History of pneumonia",
        ]

        for text in medical_texts:
            result = redactor.redact(text)
            # Text should be largely unchanged (no PHI markers)
            assert "[NAME]" not in result, f"Incorrectly redacted: {text}"

    def test_clinical_note_realistic(self):
        """Test redaction of a realistic clinical note."""
        redactor = PHIRedactor()

        note = """
        Patient: John Smith (MRN: 12345678)
        DOB: 01/15/1950
        72 year old male with history of type 2 diabetes mellitus,
        currently on metformin 1000mg BID. HbA1c 7.2%.
        Contact: 555-123-4567, john.smith@email.com
        """

        result = redactor.redact(note)

        # PHI should be redacted
        assert "12345678" not in result
        assert "01/15/1950" not in result
        assert "555-123-4567" not in result
        assert "john.smith@email.com" not in result

        # Medical content should be preserved
        assert "diabetes mellitus" in result
        assert "metformin" in result
        assert "HbA1c" in result
