"""
ICD-10 code suggestion evaluator.

Evaluates the accuracy of ICD-10 code suggestions from the Nuvii CDI Agent,
including top-N accuracy, precision, recall, and hierarchical scoring.
"""

from nuvii_eval.datasets.schemas import ICDTestCase
from nuvii_eval.evaluators.base import (
    BaseEvaluator,
    EvalResult,
    EvalScore,
    f1_score,
    normalize_code,
    precision,
    recall,
    top_n_hit,
)
from nuvii_eval.schemas.api_responses import CodingSuggestResponse


class ICDEvaluator(BaseEvaluator[ICDTestCase, CodingSuggestResponse]):
    """
    Evaluates ICD-10 code suggestion accuracy.

    Metrics:
        - top_1_accuracy: Primary expected code is rank 1
        - top_3_accuracy: Primary expected code in top 3
        - top_5_accuracy: Primary expected code in top 5
        - acceptable_recall: Recall over expected + acceptable codes
        - precision: Suggested codes that are correct
        - f1_score: Harmonic mean of precision and recall
        - specificity_score: Credit for maximum specificity codes
        - hierarchy_score: Partial credit for parent/child codes
        - false_positive_penalty: Penalty for unacceptable codes

    Configuration:
        - hierarchy_credit: Credit for hierarchically related codes (default: 0.5)
        - truncated_match: Allow truncated code matching (default: False)
        - max_codes_to_evaluate: Maximum codes to consider (default: 10)
    """

    evaluator_type = "icd"
    pass_threshold = 0.75

    def _setup(self) -> None:
        """Initialize evaluator configuration."""
        self.hierarchy_credit = self.config.get("hierarchy_credit", 0.5)
        self.truncated_match = self.config.get("truncated_match", False)
        self.max_codes = self.config.get("max_codes_to_evaluate", 10)

    def evaluate(
        self,
        test_case: ICDTestCase,
        response: CodingSuggestResponse,
    ) -> EvalResult:
        """
        Evaluate ICD-10 code suggestions against expected codes.

        Args:
            test_case: Test case with expected ICD codes
            response: API response with suggested codes

        Returns:
            EvalResult with ICD-specific metrics
        """
        # Extract predicted codes (limit to max_codes)
        predicted_codes = [
            normalize_code(s.icd10_code)
            for s in response.suggested_codes[: self.max_codes]
        ]

        # Normalize expected codes
        expected = {normalize_code(c) for c in test_case.expected_icd_codes}
        acceptable = {normalize_code(c) for c in test_case.acceptable_icd_codes}
        unacceptable = {normalize_code(c) for c in test_case.unacceptable_codes}

        # All correct codes (expected + acceptable alternatives)
        all_correct = expected | acceptable

        # Calculate predicted set
        predicted_set = set(predicted_codes)

        # Calculate metrics
        prec = precision(predicted_set, all_correct)
        rec = recall(predicted_set, all_correct)
        f1 = f1_score(prec, rec)

        scores = [
            # Top-N accuracy (higher weight for top-1)
            EvalScore(
                name="top_1_accuracy",
                value=1.0 if top_n_hit(predicted_codes, expected, 1) else 0.0,
                weight=1.5,
                details={
                    "predicted_top_1": predicted_codes[0] if predicted_codes else None,
                    "expected_primary": list(expected)[:3],
                },
            ),
            EvalScore(
                name="top_3_accuracy",
                value=1.0 if top_n_hit(predicted_codes, expected, 3) else 0.0,
                weight=1.2,
                details={"predicted_top_3": predicted_codes[:3]},
            ),
            EvalScore(
                name="top_5_accuracy",
                value=1.0 if top_n_hit(predicted_codes, expected, 5) else 0.0,
                weight=1.0,
            ),
            # Precision and Recall
            EvalScore(
                name="precision",
                value=prec,
                weight=0.8,
                details={"correct_predictions": list(predicted_set & all_correct)},
            ),
            EvalScore(
                name="recall",
                value=rec,
                weight=1.0,
                details={
                    "found": list(predicted_set & all_correct),
                    "missed": list(all_correct - predicted_set),
                },
            ),
            EvalScore(
                name="f1_score",
                value=f1,
                weight=1.0,
            ),
            # Specificity score
            EvalScore(
                name="specificity_score",
                value=self._specificity_score(predicted_codes, expected),
                weight=0.5,
            ),
            # Hierarchy score (partial credit for related codes)
            EvalScore(
                name="hierarchy_score",
                value=self._hierarchy_score(predicted_codes, expected),
                weight=0.3,
            ),
            # False positive penalty
            EvalScore(
                name="false_positive_penalty",
                value=self._false_positive_penalty(predicted_set, unacceptable),
                weight=1.0,
                details={
                    "false_positives": list(predicted_set & unacceptable),
                },
            ),
        ]

        # Check sequence if required
        if test_case.code_sequence_matters and test_case.primary_code:
            sequence_correct = (
                predicted_codes[0] == normalize_code(test_case.primary_code)
                if predicted_codes
                else False
            )
            scores.append(
                EvalScore(
                    name="sequence_accuracy",
                    value=1.0 if sequence_correct else 0.0,
                    weight=0.5,
                )
            )

        return self._create_result(
            test_case,
            scores,
            details={
                "predicted_codes": predicted_codes,
                "expected_codes": list(expected),
                "acceptable_codes": list(acceptable),
                "model_version": response.model_version,
            },
            latency_ms=response.processing_time_ms,
        )

    def _specificity_score(
        self,
        predicted: list[str],
        expected: set[str],
    ) -> float:
        """
        Score for code specificity.

        More specific codes (more digits after decimal) get higher scores.
        E11.65 (6 chars) is more specific than E11.6 (5 chars) or E11 (3 chars).

        Args:
            predicted: Ordered list of predicted codes
            expected: Set of expected codes

        Returns:
            Specificity score (0.0 to 1.0)
        """
        if not predicted or not expected:
            return 0.0

        # Calculate max expected specificity
        max_expected_len = max(len(c.replace(".", "")) for c in expected)

        # Find best matching prediction
        for code in predicted:
            norm_code = normalize_code(code)

            # Direct match
            if norm_code in expected:
                code_len = len(norm_code.replace(".", ""))
                return code_len / max_expected_len

            # Check for hierarchical match
            for exp_code in expected:
                if self._is_hierarchically_related(norm_code, exp_code):
                    code_len = len(norm_code.replace(".", ""))
                    # Partial credit for related codes
                    return (code_len / max_expected_len) * self.hierarchy_credit

        return 0.0

    def _hierarchy_score(
        self,
        predicted: list[str],
        expected: set[str],
    ) -> float:
        """
        Partial credit for hierarchically related codes.

        If expected is E11.65 and predicted is E11.6 or E11, give partial credit
        based on how close the codes are in the hierarchy.

        Args:
            predicted: Ordered list of predicted codes
            expected: Set of expected codes

        Returns:
            Hierarchy score (0.0 to 1.0)
        """
        if not predicted or not expected:
            return 0.0

        # Direct match = full credit
        predicted_set = set(normalize_code(c) for c in predicted)
        if predicted_set & expected:
            return 1.0

        best_score = 0.0

        for pred in predicted:
            pred_norm = normalize_code(pred)
            pred_base = self._get_code_category(pred_norm)

            for exp in expected:
                exp_norm = normalize_code(exp)
                exp_base = self._get_code_category(exp_norm)

                # Same subcategory (e.g., E11.6x)
                if len(pred_norm) >= 4 and len(exp_norm) >= 4:
                    if pred_norm[:5] == exp_norm[:5]:  # E11.6 matches E11.65
                        best_score = max(best_score, self.hierarchy_credit)
                        continue

                # Same category (e.g., E11.x)
                if pred_base == exp_base:
                    best_score = max(best_score, self.hierarchy_credit * 0.7)
                    continue

                # Same chapter (e.g., E0x-E14)
                if pred_norm[0] == exp_norm[0]:
                    best_score = max(best_score, self.hierarchy_credit * 0.3)

        return best_score

    def _false_positive_penalty(
        self,
        predicted: set[str],
        unacceptable: set[str],
    ) -> float:
        """
        Calculate penalty for unacceptable predictions.

        Returns 1.0 (no penalty) if no false positives,
        decreasing score as false positives increase.

        Args:
            predicted: Set of predicted codes
            unacceptable: Set of explicitly unacceptable codes

        Returns:
            Score from 0.0 (all false positives) to 1.0 (no false positives)
        """
        if not predicted or not unacceptable:
            return 1.0

        false_positives = predicted & unacceptable
        if not false_positives:
            return 1.0

        # Penalty proportional to false positive rate
        fp_rate = len(false_positives) / len(predicted)
        return 1.0 - fp_rate

    def _is_hierarchically_related(self, code1: str, code2: str) -> bool:
        """Check if two codes are hierarchically related."""
        # One is a prefix of the other
        c1, c2 = normalize_code(code1), normalize_code(code2)

        # Remove decimal for comparison
        c1_base = c1.replace(".", "")
        c2_base = c2.replace(".", "")

        return c1_base.startswith(c2_base) or c2_base.startswith(c1_base)

    def _get_code_category(self, code: str) -> str:
        """
        Get the category portion of an ICD-10 code.

        E11.65 -> E11
        I50.22 -> I50

        Args:
            code: ICD-10 code

        Returns:
            Category string (first 3 characters)
        """
        norm = normalize_code(code)
        return norm[:3] if len(norm) >= 3 else norm


class ICDEvaluatorStrict(ICDEvaluator):
    """
    Strict ICD evaluator that requires exact matches.

    No partial credit for hierarchically related codes.
    Higher pass threshold.
    """

    evaluator_type = "icd_strict"
    pass_threshold = 0.85

    def _setup(self) -> None:
        """Initialize with strict settings."""
        self.hierarchy_credit = 0.0  # No partial credit
        self.truncated_match = False
        self.max_codes = self.config.get("max_codes_to_evaluate", 10)


class ICDEvaluatorLenient(ICDEvaluator):
    """
    Lenient ICD evaluator with more partial credit.

    Higher hierarchy credit and lower pass threshold.
    Useful for initial development/debugging.
    """

    evaluator_type = "icd_lenient"
    pass_threshold = 0.6

    def _setup(self) -> None:
        """Initialize with lenient settings."""
        self.hierarchy_credit = 0.8  # More partial credit
        self.truncated_match = True
        self.max_codes = self.config.get("max_codes_to_evaluate", 15)
