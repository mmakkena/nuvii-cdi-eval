"""
HCC (Hierarchical Condition Category) evaluator.

Evaluates the accuracy of HCC detection and RAF (Risk Adjustment Factor)
scoring from the Nuvii CDI Agent risk analysis.
"""

from nuvii_eval.datasets.schemas import HCCTestCase
from nuvii_eval.evaluators.base import (
    BaseEvaluator,
    EvalResult,
    EvalScore,
    f1_score,
    normalize_code,
    precision,
    recall,
)
from nuvii_eval.schemas.api_responses import RiskAnalysisResult


# =============================================================================
# HCC Hierarchy and Supersession Rules
# =============================================================================

# CMS-HCC Model V24 supersession rules (simplified subset)
# Key: Superior HCC, Value: List of inferior HCCs it supersedes
HCC_SUPERSESSIONS: dict[str, list[str]] = {
    # Diabetes hierarchy
    "HCC17": ["HCC18", "HCC19"],  # Diabetes with acute complications
    "HCC18": ["HCC19"],  # Diabetes with chronic complications
    # Cancer hierarchy
    "HCC8": ["HCC9", "HCC10", "HCC11", "HCC12"],  # Metastatic cancer
    "HCC9": ["HCC10", "HCC11", "HCC12"],  # Lung cancer
    "HCC10": ["HCC11", "HCC12"],  # Lymphoma
    "HCC11": ["HCC12"],  # Colorectal cancer
    # Heart failure hierarchy
    "HCC85": ["HCC86", "HCC87", "HCC88"],  # CHF
    "HCC86": ["HCC87", "HCC88"],
    "HCC87": ["HCC88"],
    # COPD hierarchy
    "HCC111": ["HCC112"],  # COPD
    # Renal hierarchy
    "HCC134": ["HCC135", "HCC136", "HCC137", "HCC138"],  # Dialysis
    "HCC135": ["HCC136", "HCC137", "HCC138"],  # Stage 5 CKD
    "HCC136": ["HCC137", "HCC138"],  # Stage 4 CKD
    "HCC137": ["HCC138"],  # Stage 3 CKD
    # Stroke hierarchy
    "HCC99": ["HCC100"],  # Stroke with complications
    # Vascular hierarchy
    "HCC106": ["HCC107", "HCC108"],  # Vascular disease
    "HCC107": ["HCC108"],
}

# HCC category groups for similarity scoring
HCC_GROUPS: dict[str, list[str]] = {
    "diabetes": ["HCC17", "HCC18", "HCC19"],
    "cancer": ["HCC8", "HCC9", "HCC10", "HCC11", "HCC12"],
    "heart_failure": ["HCC85", "HCC86", "HCC87", "HCC88"],
    "copd": ["HCC111", "HCC112"],
    "renal": ["HCC134", "HCC135", "HCC136", "HCC137", "HCC138"],
    "stroke": ["HCC99", "HCC100"],
    "vascular": ["HCC106", "HCC107", "HCC108"],
}


def get_hcc_group(hcc: str) -> str | None:
    """Get the clinical group for an HCC code."""
    hcc_norm = normalize_code(hcc)
    for group, codes in HCC_GROUPS.items():
        if hcc_norm in codes:
            return group
    return None


def get_superseded_hccs(hcc: str) -> set[str]:
    """Get all HCCs superseded by the given HCC."""
    return set(HCC_SUPERSESSIONS.get(normalize_code(hcc), []))


class HCCEvaluator(BaseEvaluator[HCCTestCase, RiskAnalysisResult]):
    """
    Evaluates HCC detection and RAF scoring accuracy.

    Metrics:
        - hcc_precision: Predicted HCCs that are correct
        - hcc_recall: Expected HCCs that were detected
        - hcc_f1: F1 score for HCC detection
        - raf_accuracy: RAF score within expected range
        - raf_error: Absolute error in RAF score
        - opportunity_recall: Expected opportunities detected
        - supersession_accuracy: Correct handling of HCC hierarchies

    Configuration:
        - raf_tolerance: Acceptable RAF deviation (default: 0.1)
        - strict_supersession: Enforce supersession rules (default: True)
    """

    evaluator_type = "hcc"
    pass_threshold = 0.75

    def _setup(self) -> None:
        """Initialize evaluator configuration."""
        self.raf_tolerance = self.config.get("raf_tolerance", 0.1)
        self.strict_supersession = self.config.get("strict_supersession", True)

    def evaluate(
        self,
        test_case: HCCTestCase,
        response: RiskAnalysisResult,
    ) -> EvalResult:
        """
        Evaluate HCC detection and RAF scoring.

        Args:
            test_case: Test case with expected HCCs and RAF range
            response: API response with detected HCCs and RAF

        Returns:
            EvalResult with HCC-specific metrics
        """
        # Normalize codes
        predicted_hccs = {normalize_code(h) for h in response.current_hccs}
        expected_hccs = {normalize_code(h) for h in test_case.expected_hccs}

        predicted_opps = {normalize_code(o.hcc_code) for o in response.opportunities}
        expected_opps = {normalize_code(o) for o in test_case.expected_opportunities}

        # Calculate precision, recall, F1
        prec = precision(predicted_hccs, expected_hccs)
        rec = recall(predicted_hccs, expected_hccs)
        f1 = f1_score(prec, rec)

        # RAF accuracy
        raf_in_range = self._raf_in_range(
            response.current_raf, test_case.expected_raf_range
        )
        raf_accuracy = self._raf_accuracy_score(
            response.current_raf, test_case.expected_raf_range
        )
        raf_error = self._raf_error(response.current_raf, test_case.expected_raf_range)

        # Opportunity detection
        opp_recall = recall(predicted_opps, expected_opps) if expected_opps else 1.0

        # Supersession accuracy
        super_accuracy = self._supersession_accuracy(predicted_hccs)

        # Group accuracy (are we in the right clinical category?)
        group_accuracy = self._group_accuracy(predicted_hccs, expected_hccs)

        scores = [
            EvalScore(
                name="hcc_precision",
                value=prec,
                weight=1.0,
                details={"correct": list(predicted_hccs & expected_hccs)},
            ),
            EvalScore(
                name="hcc_recall",
                value=rec,
                weight=1.2,  # Recall is more important for risk adjustment
                details={
                    "found": list(predicted_hccs & expected_hccs),
                    "missed": list(expected_hccs - predicted_hccs),
                },
            ),
            EvalScore(
                name="hcc_f1",
                value=f1,
                weight=1.0,
            ),
            EvalScore(
                name="raf_accuracy",
                value=raf_accuracy,
                weight=1.0,
                details={
                    "predicted_raf": response.current_raf,
                    "expected_range": test_case.expected_raf_range,
                    "in_range": raf_in_range,
                },
            ),
            EvalScore(
                name="raf_error",
                value=max(0.0, 1.0 - raf_error),  # Convert error to score
                weight=0.5,
                details={"absolute_error": raf_error},
            ),
            EvalScore(
                name="opportunity_recall",
                value=opp_recall,
                weight=0.8,
                details={
                    "found_opportunities": list(predicted_opps & expected_opps),
                    "missed_opportunities": list(expected_opps - predicted_opps),
                },
            ),
            EvalScore(
                name="supersession_accuracy",
                value=super_accuracy,
                weight=0.5 if self.strict_supersession else 0.2,
            ),
            EvalScore(
                name="group_accuracy",
                value=group_accuracy,
                weight=0.3,
            ),
        ]

        return self._create_result(
            test_case,
            scores,
            details={
                "predicted_hccs": list(predicted_hccs),
                "expected_hccs": list(expected_hccs),
                "predicted_raf": response.current_raf,
                "projected_raf": response.projected_raf,
                "raf_gap": response.raf_gap,
                "opportunity_count": len(response.opportunities),
            },
            latency_ms=response.processing_time_ms,
        )

    def _raf_in_range(
        self,
        predicted_raf: float,
        expected_range: tuple[float, float],
    ) -> bool:
        """Check if RAF is within expected range."""
        min_raf, max_raf = expected_range
        return min_raf <= predicted_raf <= max_raf

    def _raf_accuracy_score(
        self,
        predicted_raf: float,
        expected_range: tuple[float, float],
    ) -> float:
        """
        Calculate RAF accuracy score.

        Returns 1.0 if in range, decreasing score based on distance from range.

        Args:
            predicted_raf: Predicted RAF score
            expected_range: (min, max) expected RAF range

        Returns:
            Score from 0.0 to 1.0
        """
        min_raf, max_raf = expected_range

        if min_raf <= predicted_raf <= max_raf:
            return 1.0

        # Calculate distance from range
        if predicted_raf < min_raf:
            distance = min_raf - predicted_raf
        else:
            distance = predicted_raf - max_raf

        # Use range size or tolerance as reference
        range_size = max_raf - min_raf
        tolerance = max(range_size, self.raf_tolerance)

        # Score decays with distance
        return max(0.0, 1.0 - (distance / tolerance))

    def _raf_error(
        self,
        predicted_raf: float,
        expected_range: tuple[float, float],
    ) -> float:
        """Calculate absolute RAF error from expected range."""
        min_raf, max_raf = expected_range

        if min_raf <= predicted_raf <= max_raf:
            return 0.0

        if predicted_raf < min_raf:
            return min_raf - predicted_raf
        else:
            return predicted_raf - max_raf

    def _supersession_accuracy(self, predicted_hccs: set[str]) -> float:
        """
        Check that HCC supersession rules are correctly applied.

        If a superior HCC is present, inferior HCCs should NOT be present.

        Args:
            predicted_hccs: Set of predicted HCC codes

        Returns:
            Score from 0.0 (many violations) to 1.0 (no violations)
        """
        violations = 0
        checks = 0

        for superior, inferiors in HCC_SUPERSESSIONS.items():
            if superior in predicted_hccs:
                checks += 1
                for inferior in inferiors:
                    if inferior in predicted_hccs:
                        violations += 1

        if checks == 0:
            return 1.0  # No supersession rules applicable

        # Score based on violation rate
        # Multiple violations from same superior count separately
        violation_rate = violations / (checks * 2)  # Normalize
        return max(0.0, 1.0 - violation_rate)

    def _group_accuracy(
        self,
        predicted_hccs: set[str],
        expected_hccs: set[str],
    ) -> float:
        """
        Check if predictions are in the correct clinical groups.

        Even if the specific HCC is wrong, being in the right group
        (e.g., diabetes-related) indicates partial understanding.

        Args:
            predicted_hccs: Set of predicted HCC codes
            expected_hccs: Set of expected HCC codes

        Returns:
            Score from 0.0 to 1.0
        """
        if not expected_hccs:
            return 1.0

        expected_groups = {get_hcc_group(h) for h in expected_hccs}
        expected_groups.discard(None)

        if not expected_groups:
            return 1.0  # No grouped HCCs expected

        predicted_groups = {get_hcc_group(h) for h in predicted_hccs}
        predicted_groups.discard(None)

        # Score based on group overlap
        correct_groups = expected_groups & predicted_groups
        return len(correct_groups) / len(expected_groups)


class HCCEvaluatorV28(HCCEvaluator):
    """
    HCC Evaluator for CMS-HCC Model V28.

    V28 has different supersession rules and HCC mappings.
    Override this class with V28-specific rules when available.
    """

    evaluator_type = "hcc_v28"

    def _setup(self) -> None:
        """Initialize with V28-specific settings."""
        super()._setup()
        # V28-specific configuration would go here
        # For now, uses same rules as base evaluator
