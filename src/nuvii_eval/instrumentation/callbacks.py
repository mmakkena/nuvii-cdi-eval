"""
LangChain callback handlers for tracing CDI agent operations.

Provides callbacks that integrate with Phoenix tracing to capture
retrieval, LLM calls, and agent execution for analysis.
"""

from datetime import datetime
from typing import Any
from uuid import UUID

import structlog

from nuvii_eval.instrumentation.phoenix_tracer import PhoenixTracer, SpanContext
from nuvii_eval.instrumentation.phi_redactor import PHIRedactor

logger = structlog.get_logger(__name__)


# Check if LangChain is available
try:
    from langchain_core.callbacks import BaseCallbackHandler
    from langchain_core.outputs import LLMResult

    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False
    BaseCallbackHandler = object  # type: ignore
    LLMResult = None  # type: ignore


class TracingCallbackHandler(BaseCallbackHandler):
    """
    LangChain callback handler for Phoenix tracing.

    Captures:
    - LLM calls with prompts, completions, and token usage
    - Retrieval operations with chunks and scores
    - Chain execution with inputs and outputs
    - Agent actions and tool calls

    Usage:
        from langchain_openai import ChatOpenAI
        from nuvii_eval.instrumentation.callbacks import TracingCallbackHandler

        handler = TracingCallbackHandler(tracer, redactor)
        llm = ChatOpenAI(callbacks=[handler])
    """

    def __init__(
        self,
        tracer: PhoenixTracer | None = None,
        redactor: PHIRedactor | None = None,
        capture_inputs: bool = False,
        capture_outputs: bool = True,
    ):
        """
        Initialize the callback handler.

        Args:
            tracer: Phoenix tracer instance
            redactor: PHI redactor for safe logging
            capture_inputs: Whether to capture input text (disable for PHI safety)
            capture_outputs: Whether to capture output text
        """
        if not LANGCHAIN_AVAILABLE:
            logger.warning(
                "langchain_not_available",
                hint="Install with: pip install langchain langchain-core",
            )

        self.tracer = tracer
        self.redactor = redactor or PHIRedactor()
        self.capture_inputs = capture_inputs
        self.capture_outputs = capture_outputs

        # Track active spans
        self._active_spans: dict[UUID, SpanContext] = {}
        self._run_metadata: dict[UUID, dict] = {}

    def _safe_text(self, text: str | None) -> str:
        """Safely redact text if redactor is available."""
        if text is None:
            return ""
        if self.redactor:
            return self.redactor.redact(text)
        return "[REDACTED]"

    # =========================================================================
    # LLM Callbacks
    # =========================================================================

    def on_llm_start(
        self,
        serialized: dict[str, Any],
        prompts: list[str],
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        """Called when LLM starts generating."""
        self._run_metadata[run_id] = {
            "start_time": datetime.utcnow(),
            "model": serialized.get("name", "unknown"),
            "prompt_count": len(prompts),
        }

        if self.tracer and self.tracer.is_initialized():
            with self.tracer.trace_api_call("llm", "GENERATE") as ctx:
                ctx.set_attribute("llm.model", serialized.get("name", "unknown"))
                ctx.set_attribute("llm.prompt_count", len(prompts))

                if self.capture_inputs:
                    # Redact and truncate prompts
                    safe_prompts = [self._safe_text(p)[:500] for p in prompts[:3]]
                    ctx.set_attribute("llm.prompts", safe_prompts)

                if tags:
                    ctx.set_attribute("llm.tags", tags)

                self._active_spans[run_id] = ctx

        logger.debug(
            "llm_start",
            run_id=str(run_id),
            model=serialized.get("name"),
            prompt_count=len(prompts),
        )

    def on_llm_end(
        self,
        response: "LLMResult",
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        """Called when LLM finishes generating."""
        metadata = self._run_metadata.pop(run_id, {})
        start_time = metadata.get("start_time")

        latency_ms = 0
        if start_time:
            latency_ms = int((datetime.utcnow() - start_time).total_seconds() * 1000)

        # Extract token usage if available
        token_usage = {}
        if response.llm_output:
            token_usage = response.llm_output.get("token_usage", {})

        if run_id in self._active_spans:
            ctx = self._active_spans.pop(run_id)
            ctx.set_attribute("llm.latency_ms", latency_ms)

            if token_usage:
                ctx.set_attribute("llm.tokens.prompt", token_usage.get("prompt_tokens", 0))
                ctx.set_attribute("llm.tokens.completion", token_usage.get("completion_tokens", 0))
                ctx.set_attribute("llm.tokens.total", token_usage.get("total_tokens", 0))

            if self.capture_outputs and response.generations:
                # Get first generation text
                for gen_list in response.generations[:1]:
                    if gen_list:
                        output_text = gen_list[0].text[:500]
                        if self.redactor:
                            output_text = self.redactor.redact(output_text)
                        ctx.set_attribute("llm.output_preview", output_text)

        logger.debug(
            "llm_end",
            run_id=str(run_id),
            latency_ms=latency_ms,
            tokens=token_usage.get("total_tokens"),
        )

    def on_llm_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        """Called when LLM errors."""
        self._run_metadata.pop(run_id, None)

        if run_id in self._active_spans:
            ctx = self._active_spans.pop(run_id)
            ctx.set_attribute("llm.error", str(error)[:500])
            ctx.record_exception(error)

        logger.error("llm_error", run_id=str(run_id), error=str(error))

    # =========================================================================
    # Chain Callbacks
    # =========================================================================

    def on_chain_start(
        self,
        serialized: dict[str, Any],
        inputs: dict[str, Any],
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        """Called when chain starts running."""
        chain_name = serialized.get("name", serialized.get("id", ["unknown"])[-1])

        self._run_metadata[run_id] = {
            "start_time": datetime.utcnow(),
            "chain_name": chain_name,
        }

        logger.debug(
            "chain_start",
            run_id=str(run_id),
            chain=chain_name,
        )

    def on_chain_end(
        self,
        outputs: dict[str, Any],
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        """Called when chain finishes."""
        metadata = self._run_metadata.pop(run_id, {})
        start_time = metadata.get("start_time")

        latency_ms = 0
        if start_time:
            latency_ms = int((datetime.utcnow() - start_time).total_seconds() * 1000)

        logger.debug(
            "chain_end",
            run_id=str(run_id),
            chain=metadata.get("chain_name"),
            latency_ms=latency_ms,
        )

    def on_chain_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        """Called when chain errors."""
        self._run_metadata.pop(run_id, None)
        logger.error("chain_error", run_id=str(run_id), error=str(error))

    # =========================================================================
    # Retrieval Callbacks
    # =========================================================================

    def on_retriever_start(
        self,
        serialized: dict[str, Any],
        query: str,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        """Called when retriever starts."""
        self._run_metadata[run_id] = {
            "start_time": datetime.utcnow(),
            "query": query if self.capture_inputs else "[REDACTED]",
        }

        logger.debug(
            "retriever_start",
            run_id=str(run_id),
        )

    def on_retriever_end(
        self,
        documents: list,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        """Called when retriever finishes."""
        metadata = self._run_metadata.pop(run_id, {})
        start_time = metadata.get("start_time")

        latency_ms = 0
        if start_time:
            latency_ms = int((datetime.utcnow() - start_time).total_seconds() * 1000)

        # Log retrieval to tracer if available
        if self.tracer and self.tracer.is_initialized():
            # Note: Would need an active span context here
            # This is a simplified version
            pass

        logger.debug(
            "retriever_end",
            run_id=str(run_id),
            document_count=len(documents),
            latency_ms=latency_ms,
        )

    def on_retriever_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        """Called when retriever errors."""
        self._run_metadata.pop(run_id, None)
        logger.error("retriever_error", run_id=str(run_id), error=str(error))

    # =========================================================================
    # Tool Callbacks
    # =========================================================================

    def on_tool_start(
        self,
        serialized: dict[str, Any],
        input_str: str,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        """Called when tool starts."""
        tool_name = serialized.get("name", "unknown")

        self._run_metadata[run_id] = {
            "start_time": datetime.utcnow(),
            "tool_name": tool_name,
        }

        logger.debug(
            "tool_start",
            run_id=str(run_id),
            tool=tool_name,
        )

    def on_tool_end(
        self,
        output: str,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        """Called when tool finishes."""
        metadata = self._run_metadata.pop(run_id, {})

        logger.debug(
            "tool_end",
            run_id=str(run_id),
            tool=metadata.get("tool_name"),
        )

    def on_tool_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        """Called when tool errors."""
        self._run_metadata.pop(run_id, None)
        logger.error("tool_error", run_id=str(run_id), error=str(error))


class MetricsCallbackHandler(BaseCallbackHandler):
    """
    Callback handler for collecting evaluation metrics.

    Aggregates metrics across LLM calls, retrievals, and chains
    for performance analysis.
    """

    def __init__(self):
        """Initialize the metrics handler."""
        if not LANGCHAIN_AVAILABLE:
            logger.warning("langchain_not_available")

        self.metrics = {
            "llm_calls": 0,
            "llm_total_tokens": 0,
            "llm_total_latency_ms": 0,
            "retrieval_calls": 0,
            "retrieval_total_docs": 0,
            "retrieval_total_latency_ms": 0,
            "chain_calls": 0,
            "errors": 0,
        }
        self._run_starts: dict[UUID, datetime] = {}

    def on_llm_start(self, *args, run_id: UUID, **kwargs) -> None:
        self._run_starts[run_id] = datetime.utcnow()
        self.metrics["llm_calls"] += 1

    def on_llm_end(self, response: "LLMResult", *, run_id: UUID, **kwargs) -> None:
        if run_id in self._run_starts:
            latency = (datetime.utcnow() - self._run_starts.pop(run_id)).total_seconds() * 1000
            self.metrics["llm_total_latency_ms"] += latency

        if response.llm_output:
            token_usage = response.llm_output.get("token_usage", {})
            self.metrics["llm_total_tokens"] += token_usage.get("total_tokens", 0)

    def on_llm_error(self, *args, run_id: UUID, **kwargs) -> None:
        self._run_starts.pop(run_id, None)
        self.metrics["errors"] += 1

    def on_retriever_start(self, *args, run_id: UUID, **kwargs) -> None:
        self._run_starts[run_id] = datetime.utcnow()
        self.metrics["retrieval_calls"] += 1

    def on_retriever_end(self, documents: list, *, run_id: UUID, **kwargs) -> None:
        if run_id in self._run_starts:
            latency = (datetime.utcnow() - self._run_starts.pop(run_id)).total_seconds() * 1000
            self.metrics["retrieval_total_latency_ms"] += latency
        self.metrics["retrieval_total_docs"] += len(documents)

    def on_retriever_error(self, *args, run_id: UUID, **kwargs) -> None:
        self._run_starts.pop(run_id, None)
        self.metrics["errors"] += 1

    def on_chain_start(self, *args, run_id: UUID, **kwargs) -> None:
        self.metrics["chain_calls"] += 1

    def on_chain_error(self, *args, **kwargs) -> None:
        self.metrics["errors"] += 1

    def get_metrics(self) -> dict:
        """Get aggregated metrics."""
        metrics = dict(self.metrics)

        # Calculate averages
        if metrics["llm_calls"] > 0:
            metrics["llm_avg_latency_ms"] = metrics["llm_total_latency_ms"] / metrics["llm_calls"]
            metrics["llm_avg_tokens"] = metrics["llm_total_tokens"] / metrics["llm_calls"]

        if metrics["retrieval_calls"] > 0:
            metrics["retrieval_avg_latency_ms"] = (
                metrics["retrieval_total_latency_ms"] / metrics["retrieval_calls"]
            )
            metrics["retrieval_avg_docs"] = (
                metrics["retrieval_total_docs"] / metrics["retrieval_calls"]
            )

        return metrics

    def reset(self) -> None:
        """Reset all metrics."""
        for key in self.metrics:
            self.metrics[key] = 0
        self._run_starts.clear()
