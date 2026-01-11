"""
E/M (Evaluation and Management) level evaluator.

Evaluates the accuracy of E/M level determination from the Nuvii CDI Agent,
including CPT code selection, MDM scoring, and up/downcoding detection.
"""

from nuvii_eval.datasets.schemas import EMTestCase
from nuvii_eval.evaluators.base import BaseEvaluator, EvalResult, EvalScore
from nuvii_eval.schemas.api_responses import EMAnalysisResult


# =============================================================================
# E/M Code Reference Data
# =============================================================================

# E/M code to level mapping
EM_CODE_LEVELS: dict[str, int] = {
    # Office/Outpatient - New Patient
    "99201": 1,
    "99202": 2,
    "99203": 3,
    "99204": 4,
    "99205": 5,
    # Office/Outpatient - Established Patient
    "99211": 1,
    "99212": 2,
    "99213": 3,
    "99214": 4,
    "99215": 5,
    # Hospital Inpatient - Initial
    "99221": 1,
    "99222": 2,
    "99223": 3,
    # Hospital Inpatient - Subsequent
    "99231": 1,
    "99232": 2,
    "99233": 3,
    # Hospital Observation
    "99218": 1,
    "99219": 2,
    "99220": 3,
    # Emergency Department
    "99281": 1,
    "99282": 2,
    "99283": 3,
    "99284": 4,
    "99285": 5,
    # Nursing Facility
    "99304": 1,
    "99305": 2,
    "99306": 3,
    "99307": 1,
    "99308": 2,
    "99309": 3,
    "99310": 4,
    # Domiciliary/Home
    "99324": 1,
    "99325": 2,
    "99326": 3,
    "99327": 4,
    "99328": 5,
    "99334": 1,
    "99335": 2,
    "99336": 3,
    "99337": 4,
}

# Code families (for determining if codes are in same family)
EM_CODE_FAMILIES: dict[str, list[str]] = {
    "office_new": ["99201", "99202", "99203", "99204", "99205"],
    "office_established": ["99211", "99212", "99213", "99214", "99215"],
    "inpatient_initial": ["99221", "99222", "99223"],
    "inpatient_subsequent": ["99231", "99232", "99233"],
    "observation": ["99218", "99219", "99220"],
    "emergency": ["99281", "99282", "99283", "99284", "99285"],
}

# MDM level descriptions
MDM_LEVEL_NAMES: dict[int, str] = {
    1: "Straightforward",
    2: "Low",
    3: "Moderate",
    4: "High",
}


def get_code_level(code: str) -> int | None:
    """Get the E/M level for a CPT code."""
    return EM_CODE_LEVELS.get(code.strip())


def get_code_family(code: str) -> str | None:
    """Get the family (category) for an E/M code."""
    code = code.strip()
    for family, codes in EM_CODE_FAMILIES.items():
        if code in codes:
            return family
    return None


def codes_in_same_family(code1: str, code2: str) -> bool:
    """Check if two codes are in the same E/M family."""
    family1 = get_code_family(code1)
    family2 = get_code_family(code2)
    return family1 is not None and family1 == family2


class EMEvaluator(BaseEvaluator[EMTestCase, EMAnalysisResult]):
    """
    Evaluates E/M level determination accuracy.

    Metrics:
        - exact_match: Exact CPT code match
        - within_one_level: Within 1 E/M level
        - level_accuracy: Granular level accuracy score
        - mdm_accuracy: MDM component accuracy
        - upcoding_penalty: Penalty for inappropriate upcoding
        - downcoding_penalty: Penalty for inappropriate downcoding
        - family_match: Correct E/M code family

    Configuration:
        - strict_family_match: Require same code family (default: True)
        - upcoding_weight: Weight for upcoding penalty (default: 1.5)
        - allow_time_based: Accept time-based alternatives (default: True)
    """

    evaluator_type = "em"
    pass_threshold = 0.75

    def _setup(self) -> None:
        """Initialize evaluator configuration."""
        self.strict_family_match = self.config.get("strict_family_match", True)
        self.upcoding_weight = self.config.get("upcoding_weight", 1.5)
        self.allow_time_based = self.config.get("allow_time_based", True)

    def evaluate(
        self,
        test_case: EMTestCase,
        response: EMAnalysisResult,
    ) -> EvalResult:
        """
        Evaluate E/M level determination.

        Args:
            test_case: Test case with expected E/M level
            response: API response with recommended E/M

        Returns:
            EvalResult with E/M-specific metrics
        """
        predicted_code = response.recommended_code
        predicted_level = response.recommended_level
        expected_code = test_case.expected_code
        expected_level = test_case.expected_level

        # Calculate level difference
        level_diff = predicted_level - expected_level

        # Check exact match
        exact_match = predicted_code == expected_code

        # Check acceptable codes
        is_acceptable = predicted_code in ([expected_code] + test_case.acceptable_codes)

        # Check time-based alternative
        time_based_ok = (
            self.allow_time_based
            and test_case.time_based_acceptable
            and response.time_based_code is not None
        )

        # Check family match
        family_match = codes_in_same_family(predicted_code, expected_code)

        # MDM accuracy
        mdm_accuracy = self._mdm_accuracy(
            response.mdm_score, test_case.expected_mdm
        )

        scores = [
            EvalScore(
                name="exact_match",
                value=1.0 if exact_match else 0.0,
                weight=1.5,
                details={
                    "predicted": predicted_code,
                    "expected": expected_code,
                },
            ),
            EvalScore(
                name="within_one_level",
                value=1.0 if abs(level_diff) <= 1 else 0.0,
                weight=1.2,
                details={
                    "level_difference": level_diff,
                },
            ),
            EvalScore(
                name="level_accuracy",
                value=self._level_accuracy_score(predicted_level, expected_level),
                weight=1.0,
                details={
                    "predicted_level": predicted_level,
                    "expected_level": expected_level,
                },
            ),
            EvalScore(
                name="mdm_accuracy",
                value=mdm_accuracy,
                weight=0.8,
                details={
                    "predicted_mdm": {
                        "problems": response.mdm_score.problems,
                        "data": response.mdm_score.data,
                        "risk": response.mdm_score.risk,
                    },
                    "expected_mdm": test_case.expected_mdm,
                },
            ),
            EvalScore(
                name="acceptable_code",
                value=1.0 if is_acceptable else 0.0,
                weight=0.5,
                details={"acceptable_codes": test_case.acceptable_codes},
            ),
            EvalScore(
                name="family_match",
                value=1.0 if family_match else 0.5,
                weight=0.3 if self.strict_family_match else 0.1,
            ),
            # Penalties for coding errors
            EvalScore(
                name="upcoding_penalty",
                value=self._upcoding_penalty(level_diff, response.upcoding_risk),
                weight=self.upcoding_weight,
            ),
            EvalScore(
                name="downcoding_penalty",
                value=self._downcoding_penalty(level_diff, response.downcoding_risk),
                weight=0.8,
            ),
        ]

        # Add time-based score if applicable
        if test_case.documented_time and response.time_based_code:
            scores.append(
                EvalScore(
                    name="time_based_option",
                    value=1.0 if time_based_ok else 0.5,
                    weight=0.3,
                    details={"time_based_code": response.time_based_code},
                )
            )

        return self._create_result(
            test_case,
            scores,
            details={
                "predicted_code": predicted_code,
                "expected_code": expected_code,
                "level_difference": level_diff,
                "upcoding_risk": response.upcoding_risk,
                "downcoding_risk": response.downcoding_risk,
                "justification": response.justification[:200] if response.justification else None,
            },
            latency_ms=response.processing_time_ms,
        )

    def _level_accuracy_score(self, predicted: int, expected: int) -> float:
        """
        Calculate granular level accuracy score.

        Args:
            predicted: Predicted E/M level
            expected: Expected E/M level

        Returns:
            Score from 0.0 to 1.0
        """
        diff = abs(predicted - expected)

        if diff == 0:
            return 1.0
        elif diff == 1:
            return 0.7
        elif diff == 2:
            return 0.3
        else:
            return 0.0

    def _mdm_accuracy(
        self,
        predicted_mdm,
        expected_mdm,
    ) -> float:
        """
        Calculate MDM component accuracy.

        Compares each MDM component (problems, data, risk).

        Args:
            predicted_mdm: Predicted MDM scores (MDMComponent or similar)
            expected_mdm: Expected MDM scores (ExpectedMDM, dict, or similar)

        Returns:
            Accuracy score (0.0 to 1.0)
        """
        components = ["problems", "data", "risk"]
        total_score = 0.0

        for comp in components:
            pred_val = getattr(predicted_mdm, comp, 0)
            # Handle both dict and Pydantic model
            if hasattr(expected_mdm, comp):
                exp_val = getattr(expected_mdm, comp, 0)
            elif hasattr(expected_mdm, "get"):
                exp_val = expected_mdm.get(comp, 0)
            else:
                exp_val = 0

            if pred_val == exp_val:
                total_score += 1.0
            elif abs(pred_val - exp_val) == 1:
                total_score += 0.5
            # Else 0 for difference > 1

        return total_score / len(components)

    def _upcoding_penalty(self, level_diff: int, flagged_risk: bool) -> float:
        """
        Calculate penalty for upcoding.

        Upcoding (coding higher than supported) is a compliance risk.

        Args:
            level_diff: Predicted level - expected level (positive = upcoding)
            flagged_risk: Whether the system flagged upcoding risk

        Returns:
            Score from 0.0 (severe upcoding) to 1.0 (no upcoding)
        """
        if level_diff <= 0:
            # No upcoding
            return 1.0

        if level_diff == 1:
            # Minor upcoding
            return 0.7 if not flagged_risk else 0.8

        if level_diff == 2:
            # Moderate upcoding
            return 0.3

        # Severe upcoding (3+ levels)
        return 0.0

    def _downcoding_penalty(self, level_diff: int, flagged_risk: bool) -> float:
        """
        Calculate penalty for downcoding.

        Downcoding (coding lower than supported) leaves revenue on table
        but is less serious than upcoding.

        Args:
            level_diff: Predicted level - expected level (negative = downcoding)
            flagged_risk: Whether the system flagged downcoding risk

        Returns:
            Score from 0.0 (severe downcoding) to 1.0 (no downcoding)
        """
        if level_diff >= 0:
            # No downcoding
            return 1.0

        if level_diff == -1:
            # Minor downcoding
            return 0.8 if not flagged_risk else 0.9

        if level_diff == -2:
            # Moderate downcoding
            return 0.5

        # Severe downcoding (3+ levels)
        return 0.2


class EMEvaluatorStrict(EMEvaluator):
    """
    Strict E/M evaluator with zero tolerance for upcoding.

    Any upcoding by more than 1 level fails.
    """

    evaluator_type = "em_strict"
    pass_threshold = 0.85

    def _setup(self) -> None:
        """Initialize with strict settings."""
        super()._setup()
        self.upcoding_weight = 2.0  # Heavy penalty
        self.strict_family_match = True

    def _upcoding_penalty(self, level_diff: int, flagged_risk: bool) -> float:
        """Stricter upcoding penalty."""
        if level_diff <= 0:
            return 1.0
        if level_diff == 1:
            return 0.5
        return 0.0  # Fail for 2+ level upcoding


class EMEvaluatorLenient(EMEvaluator):
    """
    Lenient E/M evaluator for development.

    More forgiving on level differences.
    """

    evaluator_type = "em_lenient"
    pass_threshold = 0.6

    def _setup(self) -> None:
        """Initialize with lenient settings."""
        super()._setup()
        self.upcoding_weight = 0.8
        self.strict_family_match = False
