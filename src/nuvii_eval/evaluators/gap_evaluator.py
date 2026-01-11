"""
Documentation gap detection evaluator.

Evaluates the accuracy of CDI gap detection from the Nuvii CDI Agent,
including gap identification, classification, and prioritization.
"""

from nuvii_eval.datasets.schemas import ExpectedGap, GapTestCase
from nuvii_eval.evaluators.base import (
    BaseEvaluator,
    EvalResult,
    EvalScore,
    f1_score,
    jaccard_similarity,
)
from nuvii_eval.schemas.api_responses import GapCandidate, GapDetectionResponse


def normalize_condition(condition: str) -> str:
    """Normalize a condition string for comparison."""
    return condition.lower().strip()


def tokenize_condition(condition: str) -> set[str]:
    """Tokenize a condition into a set of words."""
    # Remove common stop words and normalize
    stop_words = {"the", "a", "an", "of", "with", "for", "and", "or", "in", "to"}
    words = normalize_condition(condition).split()
    return {w for w in words if w not in stop_words and len(w) > 2}


class GapEvaluator(BaseEvaluator[GapTestCase, GapDetectionResponse]):
    """
    Evaluates documentation gap detection accuracy.

    Metrics:
        - precision: Detected gaps that are true gaps
        - recall: Expected gaps that were detected
        - f1_score: Harmonic mean of precision and recall
        - gap_type_accuracy: Correct gap type classification
        - priority_accuracy: Priority ranking accuracy
        - condition_match_score: Quality of condition matching
        - false_positive_penalty: Penalty for explicit false positives

    Configuration:
        - condition_similarity_threshold: Min similarity for condition match (default: 0.4)
        - strict_type_matching: Require exact gap type match (default: False)
        - priority_tolerance: Allowed priority deviation (default: 1)
    """

    evaluator_type = "gap"
    pass_threshold = 0.7

    def _setup(self) -> None:
        """Initialize evaluator configuration."""
        self.condition_threshold = self.config.get("condition_similarity_threshold", 0.4)
        self.strict_type_matching = self.config.get("strict_type_matching", False)
        self.priority_tolerance = self.config.get("priority_tolerance", 1)

    def evaluate(
        self,
        test_case: GapTestCase,
        response: GapDetectionResponse,
    ) -> EvalResult:
        """
        Evaluate gap detection against expected gaps.

        Args:
            test_case: Test case with expected gaps
            response: API response with detected gaps

        Returns:
            EvalResult with gap detection metrics
        """
        predicted_gaps = response.gaps
        expected_gaps = test_case.expected_gaps
        false_positive_conditions = set(test_case.false_positive_conditions)

        # Handle no-gaps-expected case
        if test_case.no_gaps_expected:
            no_gaps_detected = len(predicted_gaps) == 0
            return self._create_result(
                test_case,
                scores=[
                    EvalScore(
                        name="no_gaps_correct",
                        value=1.0 if no_gaps_detected else 0.0,
                        weight=2.0,
                        details={"gaps_detected": len(predicted_gaps)},
                    )
                ],
                details={"expected_no_gaps": True, "gaps_found": len(predicted_gaps)},
                custom_pass_check=no_gaps_detected,
                latency_ms=response.processing_time_ms,
            )

        # Match predicted to expected gaps
        matches = self._match_gaps(predicted_gaps, expected_gaps)

        # Count metrics
        tp = len(matches)
        fp = len(predicted_gaps) - tp
        fn = len(expected_gaps) - tp

        # Calculate precision, recall, F1
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 1.0
        f1 = f1_score(prec, rec)

        # Gap type accuracy
        type_accuracy = self._gap_type_accuracy(matches)

        # Priority accuracy
        priority_accuracy = self._priority_accuracy(matches)

        # Condition match quality
        condition_score = self._condition_match_score(matches)

        # False positive check
        explicit_fps = self._count_explicit_false_positives(
            predicted_gaps, false_positive_conditions
        )
        fp_penalty = 1.0 - (explicit_fps / max(len(predicted_gaps), 1))

        scores = [
            EvalScore(
                name="precision",
                value=prec,
                weight=1.0,
                details={"true_positives": tp, "false_positives": fp},
            ),
            EvalScore(
                name="recall",
                value=rec,
                weight=1.2,  # Missing gaps is worse than extra gaps
                details={"true_positives": tp, "false_negatives": fn},
            ),
            EvalScore(
                name="f1_score",
                value=f1,
                weight=1.0,
            ),
            EvalScore(
                name="gap_type_accuracy",
                value=type_accuracy,
                weight=0.8,
            ),
            EvalScore(
                name="priority_accuracy",
                value=priority_accuracy,
                weight=0.5,
            ),
            EvalScore(
                name="condition_match_score",
                value=condition_score,
                weight=0.6,
            ),
            EvalScore(
                name="false_positive_penalty",
                value=fp_penalty,
                weight=1.0,
                details={"explicit_false_positives": explicit_fps},
            ),
        ]

        return self._create_result(
            test_case,
            scores,
            details={
                "true_positives": tp,
                "false_positives": fp,
                "false_negatives": fn,
                "matches": [
                    {"predicted": m[0].condition, "expected": m[1].condition}
                    for m in matches
                ],
                "unmatched_predictions": [
                    g.condition for g in predicted_gaps
                    if g not in [m[0] for m in matches]
                ],
                "missed_gaps": [
                    g.condition for g in expected_gaps
                    if g not in [m[1] for m in matches]
                ],
            },
            latency_ms=response.processing_time_ms,
        )

    def _match_gaps(
        self,
        predicted: list[GapCandidate],
        expected: list[ExpectedGap],
    ) -> list[tuple[GapCandidate, ExpectedGap]]:
        """
        Match predicted gaps to expected gaps.

        Uses condition similarity and optional gap type matching.
        Each expected gap can only be matched once.

        Args:
            predicted: List of predicted gaps
            expected: List of expected gaps

        Returns:
            List of (predicted, expected) matches
        """
        matches: list[tuple[GapCandidate, ExpectedGap]] = []
        used_expected: set[int] = set()

        # Sort predicted by priority (higher priority first)
        sorted_predicted = sorted(predicted, key=lambda g: g.priority)

        for pred in sorted_predicted:
            best_match: tuple[int, ExpectedGap, float] | None = None
            pred_tokens = tokenize_condition(pred.condition)

            for i, exp in enumerate(expected):
                if i in used_expected:
                    continue

                exp_tokens = tokenize_condition(exp.condition)

                # Calculate similarity
                similarity = self._condition_similarity(
                    pred.condition, exp.condition, pred_tokens, exp_tokens
                )

                if similarity >= self.condition_threshold:
                    # Check type matching if strict
                    if self.strict_type_matching and pred.gap_type != exp.gap_type:
                        continue

                    if best_match is None or similarity > best_match[2]:
                        best_match = (i, exp, similarity)

            if best_match is not None:
                matches.append((pred, best_match[1]))
                used_expected.add(best_match[0])

        return matches

    def _condition_similarity(
        self,
        cond1: str,
        cond2: str,
        tokens1: set[str] | None = None,
        tokens2: set[str] | None = None,
    ) -> float:
        """
        Calculate similarity between two condition strings.

        Uses a combination of:
        - Exact substring matching
        - Jaccard similarity of tokens

        Args:
            cond1: First condition
            cond2: Second condition
            tokens1: Pre-tokenized first condition (optional)
            tokens2: Pre-tokenized second condition (optional)

        Returns:
            Similarity score (0.0 to 1.0)
        """
        norm1 = normalize_condition(cond1)
        norm2 = normalize_condition(cond2)

        # Exact match
        if norm1 == norm2:
            return 1.0

        # Substring match
        if norm1 in norm2 or norm2 in norm1:
            shorter = min(len(norm1), len(norm2))
            longer = max(len(norm1), len(norm2))
            return shorter / longer

        # Token-based similarity
        t1 = tokens1 if tokens1 is not None else tokenize_condition(cond1)
        t2 = tokens2 if tokens2 is not None else tokenize_condition(cond2)

        return jaccard_similarity(t1, t2)

    def _gap_type_accuracy(
        self,
        matches: list[tuple[GapCandidate, ExpectedGap]],
    ) -> float:
        """
        Calculate gap type classification accuracy.

        Args:
            matches: List of (predicted, expected) matches

        Returns:
            Accuracy score (0.0 to 1.0)
        """
        if not matches:
            return 0.0

        correct = sum(
            1 for pred, exp in matches
            if pred.gap_type.lower() == exp.gap_type.lower()
        )

        return correct / len(matches)

    def _priority_accuracy(
        self,
        matches: list[tuple[GapCandidate, ExpectedGap]],
    ) -> float:
        """
        Calculate priority ranking accuracy.

        A match is correct if predicted priority is at least as high
        (lower number) as expected minimum priority.

        Args:
            matches: List of (predicted, expected) matches

        Returns:
            Accuracy score (0.0 to 1.0)
        """
        if not matches:
            return 0.0

        correct = 0
        for pred, exp in matches:
            # Lower priority number = higher priority
            if pred.priority <= exp.min_priority + self.priority_tolerance:
                correct += 1

        return correct / len(matches)

    def _condition_match_score(
        self,
        matches: list[tuple[GapCandidate, ExpectedGap]],
    ) -> float:
        """
        Calculate average condition match quality.

        Args:
            matches: List of (predicted, expected) matches

        Returns:
            Average similarity score (0.0 to 1.0)
        """
        if not matches:
            return 0.0

        total_similarity = sum(
            self._condition_similarity(pred.condition, exp.condition)
            for pred, exp in matches
        )

        return total_similarity / len(matches)

    def _count_explicit_false_positives(
        self,
        predicted: list[GapCandidate],
        false_positive_conditions: set[str],
    ) -> int:
        """
        Count gaps that match explicit false positive conditions.

        Args:
            predicted: List of predicted gaps
            false_positive_conditions: Set of conditions that should NOT be flagged

        Returns:
            Count of explicit false positives
        """
        if not false_positive_conditions:
            return 0

        count = 0
        fp_normalized = {normalize_condition(c) for c in false_positive_conditions}

        for pred in predicted:
            pred_norm = normalize_condition(pred.condition)

            # Check for exact or partial match
            for fp in fp_normalized:
                if fp in pred_norm or pred_norm in fp:
                    count += 1
                    break

        return count


class GapEvaluatorStrict(GapEvaluator):
    """
    Strict gap evaluator requiring exact type matching.

    Higher similarity threshold and strict gap type matching.
    """

    evaluator_type = "gap_strict"
    pass_threshold = 0.8

    def _setup(self) -> None:
        """Initialize with strict settings."""
        self.condition_threshold = 0.6
        self.strict_type_matching = True
        self.priority_tolerance = 0
