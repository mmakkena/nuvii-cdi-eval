"""
Evaluators for CDI agent outputs.

This module provides evaluators for different CDI task types:
- ICD-10 code suggestions
- HCC detection and RAF scoring
- Documentation gap detection
- Provider query quality
- E/M level determination
"""

from nuvii_eval.evaluators.base import (
    BaseEvaluator,
    EvalResult,
    EvalScore,
    f1_score,
    jaccard_similarity,
    normalize_code,
    precision,
    recall,
    top_n_hit,
)
from nuvii_eval.evaluators.em_evaluator import (
    EM_CODE_FAMILIES,
    EM_CODE_LEVELS,
    EMEvaluator,
    EMEvaluatorLenient,
    EMEvaluatorStrict,
    codes_in_same_family,
    get_code_family,
    get_code_level,
)
from nuvii_eval.evaluators.gap_evaluator import (
    GapEvaluator,
    GapEvaluatorStrict,
    normalize_condition,
    tokenize_condition,
)
from nuvii_eval.evaluators.hcc_evaluator import (
    HCC_GROUPS,
    HCC_SUPERSESSIONS,
    HCCEvaluator,
    HCCEvaluatorV28,
    get_hcc_group,
    get_superseded_hccs,
)
from nuvii_eval.evaluators.icd_evaluator import (
    ICDEvaluator,
    ICDEvaluatorLenient,
    ICDEvaluatorStrict,
)
from nuvii_eval.evaluators.query_evaluator import (
    COMPLIANT_PATTERNS,
    LEADING_PATTERNS,
    QueryEvaluator,
    QueryEvaluatorLenient,
    QueryEvaluatorStrict,
)

# Registry of evaluator types
EVALUATOR_REGISTRY: dict[str, type[BaseEvaluator]] = {
    # ICD evaluators
    "icd": ICDEvaluator,
    "icd_strict": ICDEvaluatorStrict,
    "icd_lenient": ICDEvaluatorLenient,
    # HCC evaluators
    "hcc": HCCEvaluator,
    "hcc_v28": HCCEvaluatorV28,
    # Gap evaluators
    "gap": GapEvaluator,
    "gap_strict": GapEvaluatorStrict,
    # Query evaluators
    "query": QueryEvaluator,
    "query_strict": QueryEvaluatorStrict,
    "query_lenient": QueryEvaluatorLenient,
    # E/M evaluators
    "em": EMEvaluator,
    "em_strict": EMEvaluatorStrict,
    "em_lenient": EMEvaluatorLenient,
}


def get_evaluator(evaluator_type: str, **config) -> BaseEvaluator:
    """
    Get an evaluator instance by type.

    Args:
        evaluator_type: Type of evaluator (e.g., "icd", "hcc_strict")
        **config: Configuration options for the evaluator

    Returns:
        Configured evaluator instance

    Raises:
        ValueError: If evaluator type is not registered
    """
    if evaluator_type not in EVALUATOR_REGISTRY:
        available = ", ".join(sorted(EVALUATOR_REGISTRY.keys()))
        raise ValueError(
            f"Unknown evaluator type: {evaluator_type}. Available: {available}"
        )

    evaluator_class = EVALUATOR_REGISTRY[evaluator_type]
    return evaluator_class(config=config)


__all__ = [
    # Base classes and utilities
    "BaseEvaluator",
    "EvalResult",
    "EvalScore",
    "precision",
    "recall",
    "f1_score",
    "top_n_hit",
    "jaccard_similarity",
    "normalize_code",
    # ICD evaluators
    "ICDEvaluator",
    "ICDEvaluatorStrict",
    "ICDEvaluatorLenient",
    # HCC evaluators
    "HCCEvaluator",
    "HCCEvaluatorV28",
    "HCC_SUPERSESSIONS",
    "HCC_GROUPS",
    "get_hcc_group",
    "get_superseded_hccs",
    # Gap evaluators
    "GapEvaluator",
    "GapEvaluatorStrict",
    "normalize_condition",
    "tokenize_condition",
    # Query evaluators
    "QueryEvaluator",
    "QueryEvaluatorStrict",
    "QueryEvaluatorLenient",
    "LEADING_PATTERNS",
    "COMPLIANT_PATTERNS",
    # E/M evaluators
    "EMEvaluator",
    "EMEvaluatorStrict",
    "EMEvaluatorLenient",
    "EM_CODE_LEVELS",
    "EM_CODE_FAMILIES",
    "get_code_level",
    "get_code_family",
    "codes_in_same_family",
    # Registry and factory
    "EVALUATOR_REGISTRY",
    "get_evaluator",
]
