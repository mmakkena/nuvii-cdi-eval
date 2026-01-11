"""
RAGAS (RAG Assessment) integration for CDI evaluation.

This module provides RAGAS-based evaluation metrics for assessing
the quality of RAG-generated responses in the CDI context.

Key metrics:
- Faithfulness: How factually consistent is the response with the context?
- Answer Relevancy: How relevant is the response to the question?
- Context Precision: How precise is the retrieved context?
- Context Recall: Does the context contain necessary information?
"""

from nuvii_eval.ragas.metrics import (
    RAGASConfig,
    RAGASMetricResult,
    RAGASEvaluator,
    FaithfulnessEvaluator,
    AnswerRelevancyEvaluator,
    ContextPrecisionEvaluator,
    ContextRecallEvaluator,
)
from nuvii_eval.ragas.adapters import (
    CDIRAGASAdapter,
    create_ragas_dataset,
    convert_to_ragas_format,
)

__all__ = [
    # Config and results
    "RAGASConfig",
    "RAGASMetricResult",
    # Evaluators
    "RAGASEvaluator",
    "FaithfulnessEvaluator",
    "AnswerRelevancyEvaluator",
    "ContextPrecisionEvaluator",
    "ContextRecallEvaluator",
    # Adapters
    "CDIRAGASAdapter",
    "create_ragas_dataset",
    "convert_to_ragas_format",
]
