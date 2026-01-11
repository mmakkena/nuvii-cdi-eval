"""
Phoenix tracing integration for evaluation runs.

Provides OpenTelemetry-based tracing for debugging, analysis, and auditability
of evaluation runs against the Nuvii CDI Agent.
"""

from contextlib import contextmanager
from datetime import datetime
from typing import Any, Generator

import structlog

from nuvii_eval.config import EvalConfig, PhoenixConfig

logger = structlog.get_logger(__name__)

# Track initialization state
_phoenix_initialized = False
_tracer = None


def _lazy_import_phoenix():
    """Lazily import Phoenix to avoid startup overhead when disabled."""
    try:
        import phoenix as px
        from opentelemetry import trace
        from opentelemetry.trace import Status, StatusCode, SpanKind

        return px, trace, Status, StatusCode, SpanKind
    except ImportError as e:
        logger.warning("phoenix_import_failed", error=str(e))
        return None, None, None, None, None


class SpanContext:
    """
    Context object for a tracing span.

    Provides a simple interface for logging attributes and events
    to the current span without exposing OpenTelemetry internals.
    """

    def __init__(
        self,
        span: Any | None = None,
        span_id: str | None = None,
        trace_id: str | None = None,
    ):
        self._span = span
        self.span_id = span_id
        self.trace_id = trace_id

    def set_attribute(self, key: str, value: Any) -> None:
        """Set an attribute on the span."""
        if self._span is not None:
            try:
                # Convert complex types to string for OTel compatibility
                if isinstance(value, (dict, list)):
                    value = str(value)[:2000]  # Truncate long values
                self._span.set_attribute(key, value)
            except Exception as e:
                logger.debug("span_attribute_error", key=key, error=str(e))

    def add_event(self, name: str, attributes: dict[str, Any] | None = None) -> None:
        """Add an event to the span."""
        if self._span is not None:
            try:
                self._span.add_event(name, attributes=attributes or {})
            except Exception as e:
                logger.debug("span_event_error", name=name, error=str(e))

    def record_exception(self, exception: Exception) -> None:
        """Record an exception on the span."""
        if self._span is not None:
            try:
                self._span.record_exception(exception)
            except Exception as e:
                logger.debug("span_exception_error", error=str(e))

    @property
    def is_recording(self) -> bool:
        """Check if the span is actively recording."""
        return self._span is not None and self._span.is_recording()


class PhoenixTracer:
    """
    Phoenix integration for evaluation tracing.

    Provides:
    - Automatic span creation for evaluation runs
    - API call logging with latency and token metrics
    - Evaluation result recording
    - PHI-safe attribute handling

    Usage:
        tracer = PhoenixTracer(phoenix_config, eval_config)
        tracer.initialize()

        with tracer.trace_evaluation("test_001", "icd", run_config) as ctx:
            # Do evaluation
            ctx.set_attribute("custom_metric", 0.95)
    """

    def __init__(self, phoenix_config: PhoenixConfig, eval_config: EvalConfig):
        """
        Initialize the Phoenix tracer.

        Args:
            phoenix_config: Phoenix configuration
            eval_config: Evaluation configuration (for PHI safety settings)
        """
        self.phoenix_config = phoenix_config
        self.eval_config = eval_config
        self._initialized = False
        self._tracer = None
        self._px = None
        self._trace = None
        self._Status = None
        self._StatusCode = None
        self._SpanKind = None

    def initialize(self) -> bool:
        """
        Initialize Phoenix connection and OpenTelemetry tracer.

        Returns:
            True if initialization succeeded, False otherwise
        """
        global _phoenix_initialized, _tracer

        if not self.phoenix_config.enabled:
            logger.info("phoenix_disabled", reason="config")
            return False

        if self._initialized:
            return True

        # Lazy import
        self._px, self._trace, self._Status, self._StatusCode, self._SpanKind = (
            _lazy_import_phoenix()
        )

        if self._px is None:
            logger.warning(
                "phoenix_unavailable",
                reason="import_failed",
                hint="Install with: pip install arize-phoenix",
            )
            return False

        try:
            # Launch local Phoenix if endpoint is localhost
            if "localhost" in self.phoenix_config.endpoint:
                try:
                    self._px.launch_app()
                    logger.info("phoenix_app_launched")
                except Exception as e:
                    # May already be running
                    logger.debug("phoenix_launch_skipped", reason=str(e))

            # Register OpenTelemetry tracer with Phoenix
            from phoenix.otel import register

            tracer_provider = register(
                project_name=self.phoenix_config.project_name,
                endpoint=self.phoenix_config.endpoint,
            )

            self._tracer = self._trace.get_tracer(
                "nuvii_eval",
                schema_url="https://opentelemetry.io/schemas/1.21.0",
            )

            self._initialized = True
            _phoenix_initialized = True
            _tracer = self._tracer

            logger.info(
                "phoenix_initialized",
                endpoint=self.phoenix_config.endpoint,
                project=self.phoenix_config.project_name,
            )

            return True

        except Exception as e:
            logger.error("phoenix_init_failed", error=str(e))
            self._initialized = False
            return False

    def is_initialized(self) -> bool:
        """Check if tracer is initialized and ready."""
        return self._initialized

    @contextmanager
    def trace_evaluation(
        self,
        test_case_id: str,
        evaluator_type: str,
        run_config: dict[str, Any],
    ) -> Generator[SpanContext, None, None]:
        """
        Context manager for tracing a single evaluation.

        Args:
            test_case_id: Unique identifier for the test case
            evaluator_type: Type of evaluator (icd, hcc, gap, etc.)
            run_config: Configuration for this run

        Yields:
            SpanContext for adding attributes and events
        """
        if not self._initialized or self._tracer is None:
            # Return a no-op context
            yield SpanContext()
            return

        span_name = f"eval.{evaluator_type}.{test_case_id}"

        with self._tracer.start_as_current_span(
            name=span_name,
            kind=self._SpanKind.INTERNAL,
        ) as span:
            # Extract span context
            span_ctx = span.get_span_context()
            ctx = SpanContext(
                span=span,
                span_id=format(span_ctx.span_id, "016x") if span_ctx else None,
                trace_id=format(span_ctx.trace_id, "032x") if span_ctx else None,
            )

            # Set standard attributes
            ctx.set_attribute("eval.test_case_id", test_case_id)
            ctx.set_attribute("eval.evaluator_type", evaluator_type)
            ctx.set_attribute("eval.timestamp", datetime.utcnow().isoformat())
            ctx.set_attribute("eval.phi_safe_mode", self.eval_config.phi_safe_mode)

            # Set run config (non-sensitive parts)
            for key, value in run_config.items():
                if key not in ("api_key", "credentials", "token"):
                    ctx.set_attribute(f"config.{key}", value)

            try:
                yield ctx
                span.set_status(self._Status(self._StatusCode.OK))

            except Exception as e:
                span.set_status(self._Status(self._StatusCode.ERROR, str(e)))
                span.record_exception(e)
                raise

    @contextmanager
    def trace_api_call(
        self,
        endpoint: str,
        method: str = "POST",
    ) -> Generator[SpanContext, None, None]:
        """
        Context manager for tracing an API call.

        Args:
            endpoint: API endpoint being called
            method: HTTP method

        Yields:
            SpanContext for the API call span
        """
        if not self._initialized or self._tracer is None:
            yield SpanContext()
            return

        span_name = f"api.{method.lower()}.{endpoint.replace('/', '.')}"

        with self._tracer.start_as_current_span(
            name=span_name,
            kind=self._SpanKind.CLIENT,
        ) as span:
            span_ctx = span.get_span_context()
            ctx = SpanContext(
                span=span,
                span_id=format(span_ctx.span_id, "016x") if span_ctx else None,
                trace_id=format(span_ctx.trace_id, "032x") if span_ctx else None,
            )

            ctx.set_attribute("http.method", method)
            ctx.set_attribute("http.url", endpoint)
            ctx.set_attribute("api.timestamp", datetime.utcnow().isoformat())

            start_time = datetime.utcnow()

            try:
                yield ctx

                latency_ms = int((datetime.utcnow() - start_time).total_seconds() * 1000)
                ctx.set_attribute("api.latency_ms", latency_ms)
                span.set_status(self._Status(self._StatusCode.OK))

            except Exception as e:
                latency_ms = int((datetime.utcnow() - start_time).total_seconds() * 1000)
                ctx.set_attribute("api.latency_ms", latency_ms)
                ctx.set_attribute("api.error", str(e))
                span.set_status(self._Status(self._StatusCode.ERROR, str(e)))
                span.record_exception(e)
                raise

    def log_api_call(
        self,
        ctx: SpanContext,
        endpoint: str,
        request_payload: dict[str, Any] | None,
        response: dict[str, Any] | None,
        latency_ms: int,
        token_count: int | None = None,
        status_code: int = 200,
    ) -> None:
        """
        Log API call details to a span context.

        Args:
            ctx: The span context to log to
            endpoint: API endpoint
            request_payload: Request body (will be redacted if PHI safe mode)
            response: Response body
            latency_ms: Request latency in milliseconds
            token_count: Token count if available
            status_code: HTTP status code
        """
        ctx.set_attribute(f"api.{endpoint}.latency_ms", latency_ms)
        ctx.set_attribute(f"api.{endpoint}.status_code", status_code)
        ctx.set_attribute(f"api.{endpoint}.success", 200 <= status_code < 300)

        if token_count is not None:
            ctx.set_attribute(f"api.{endpoint}.tokens", token_count)

        # Only log payloads if not in PHI safe mode
        if self.phoenix_config.collect_inputs and not self.eval_config.phi_safe_mode:
            if request_payload:
                ctx.set_attribute(f"api.{endpoint}.request", str(request_payload)[:1000])

        if self.phoenix_config.collect_outputs:
            if response:
                # Always safe to log response structure (no PHI in API responses)
                ctx.set_attribute(f"api.{endpoint}.response_keys", list(response.keys()))

    def log_eval_result(
        self,
        ctx: SpanContext,
        scores: dict[str, float],
        passed: bool,
        details: dict[str, Any] | None = None,
    ) -> None:
        """
        Log evaluation results to a span context.

        Args:
            ctx: The span context to log to
            scores: Dictionary of metric names to scores
            passed: Whether the evaluation passed
            details: Additional details
        """
        ctx.set_attribute("eval.passed", passed)

        for metric_name, score in scores.items():
            ctx.set_attribute(f"eval.score.{metric_name}", score)

        if scores:
            ctx.set_attribute(
                "eval.score.composite",
                sum(scores.values()) / len(scores),
            )

        if details and not self.eval_config.phi_safe_mode:
            ctx.set_attribute("eval.details", str(details)[:2000])

    def log_retrieval(
        self,
        ctx: SpanContext,
        query: str,
        chunks: list[dict[str, Any]],
        scores: list[float],
    ) -> None:
        """
        Log retrieval results for RAG analysis.

        Args:
            ctx: The span context to log to
            query: The query used for retrieval
            chunks: Retrieved chunks
            scores: Relevance scores for each chunk
        """
        ctx.set_attribute("retrieval.chunk_count", len(chunks))
        ctx.set_attribute("retrieval.scores", scores[:10])  # Top 10

        if scores:
            ctx.set_attribute("retrieval.max_score", max(scores))
            ctx.set_attribute("retrieval.mean_score", sum(scores) / len(scores))

        if not self.eval_config.phi_safe_mode:
            ctx.set_attribute("retrieval.query", query[:500])
            # Log chunk previews
            previews = [c.get("text", "")[:100] for c in chunks[:5]]
            ctx.set_attribute("retrieval.chunk_previews", previews)

    def create_run_span(
        self,
        run_id: str,
        dataset_path: str,
        evaluator_types: list[str],
        config: dict[str, Any],
    ) -> SpanContext:
        """
        Create a parent span for an entire evaluation run.

        Args:
            run_id: Unique run identifier
            dataset_path: Path to the dataset being evaluated
            evaluator_types: List of evaluator types being run
            config: Run configuration

        Returns:
            SpanContext for the run span
        """
        if not self._initialized or self._tracer is None:
            return SpanContext()

        span = self._tracer.start_span(
            name=f"eval.run.{run_id}",
            kind=self._SpanKind.INTERNAL,
        )

        span_ctx = span.get_span_context()
        ctx = SpanContext(
            span=span,
            span_id=format(span_ctx.span_id, "016x") if span_ctx else None,
            trace_id=format(span_ctx.trace_id, "032x") if span_ctx else None,
        )

        ctx.set_attribute("run.id", run_id)
        ctx.set_attribute("run.dataset", dataset_path)
        ctx.set_attribute("run.evaluators", evaluator_types)
        ctx.set_attribute("run.started_at", datetime.utcnow().isoformat())

        return ctx

    def end_run_span(
        self,
        ctx: SpanContext,
        total_cases: int,
        passed_count: int,
        failed_count: int,
        error_count: int,
    ) -> None:
        """
        End a run span with summary statistics.

        Args:
            ctx: The run span context
            total_cases: Total number of test cases
            passed_count: Number of passed cases
            failed_count: Number of failed cases
            error_count: Number of cases with errors
        """
        ctx.set_attribute("run.total_cases", total_cases)
        ctx.set_attribute("run.passed", passed_count)
        ctx.set_attribute("run.failed", failed_count)
        ctx.set_attribute("run.errors", error_count)
        ctx.set_attribute("run.pass_rate", passed_count / total_cases if total_cases > 0 else 0)
        ctx.set_attribute("run.ended_at", datetime.utcnow().isoformat())

        if ctx._span is not None:
            ctx._span.end()


def get_tracer() -> PhoenixTracer | None:
    """Get the global tracer instance if initialized."""
    if _phoenix_initialized and _tracer is not None:
        # Return a wrapper - actual global access would need more setup
        return None
    return None
