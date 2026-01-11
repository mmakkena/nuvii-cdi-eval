"""
Instrumentation for tracing, PHI safety, and observability.

This module provides:
- Phoenix tracing integration for evaluation debugging and analysis
- PHI (Protected Health Information) redaction for HIPAA compliance
- LangChain callback handlers for agent tracing
"""

from nuvii_eval.instrumentation.phi_redactor import (
    PHICategory,
    PHIRedactor,
    RedactionAuditLog,
    RedactionPattern,
    RedactionResult,
)
from nuvii_eval.instrumentation.phoenix_tracer import (
    PhoenixTracer,
    SpanContext,
    get_tracer,
)

__all__ = [
    # Phoenix tracing
    "PhoenixTracer",
    "SpanContext",
    "get_tracer",
    # PHI redaction
    "PHIRedactor",
    "PHICategory",
    "RedactionPattern",
    "RedactionResult",
    "RedactionAuditLog",
]

# Conditionally export callbacks if LangChain is available
try:
    from nuvii_eval.instrumentation.callbacks import (
        MetricsCallbackHandler,
        TracingCallbackHandler,
    )

    __all__.extend(["TracingCallbackHandler", "MetricsCallbackHandler"])
except ImportError:
    pass
