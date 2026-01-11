"""Tests for Phoenix tracing integration."""

import pytest
from unittest.mock import MagicMock, patch

from nuvii_eval.config import EvalConfig, PhoenixConfig
from nuvii_eval.instrumentation.phoenix_tracer import (
    PhoenixTracer,
    SpanContext,
)


class TestSpanContext:
    """Tests for SpanContext class."""

    def test_span_context_no_span(self):
        """Test SpanContext with no underlying span."""
        ctx = SpanContext()

        # Should not raise
        ctx.set_attribute("key", "value")
        ctx.add_event("event")
        ctx.record_exception(Exception("test"))

        assert ctx.span_id is None
        assert ctx.trace_id is None
        assert not ctx.is_recording

    def test_span_context_with_mock_span(self):
        """Test SpanContext with a mock span."""
        mock_span = MagicMock()
        mock_span.is_recording.return_value = True

        ctx = SpanContext(
            span=mock_span,
            span_id="abc123",
            trace_id="def456",
        )

        ctx.set_attribute("test_key", "test_value")

        mock_span.set_attribute.assert_called_once_with("test_key", "test_value")
        assert ctx.span_id == "abc123"
        assert ctx.trace_id == "def456"
        assert ctx.is_recording

    def test_span_context_truncates_long_values(self):
        """Test that long values are truncated."""
        mock_span = MagicMock()
        ctx = SpanContext(span=mock_span)

        long_dict = {"key": "x" * 3000}
        ctx.set_attribute("long_value", long_dict)

        # Should have been called with truncated value
        call_args = mock_span.set_attribute.call_args
        assert len(str(call_args[0][1])) <= 2000

    def test_span_context_add_event(self):
        """Test adding events to span."""
        mock_span = MagicMock()
        ctx = SpanContext(span=mock_span)

        ctx.add_event("test_event", {"attr": "value"})

        mock_span.add_event.assert_called_once_with(
            "test_event",
            attributes={"attr": "value"},
        )

    def test_span_context_record_exception(self):
        """Test recording exceptions."""
        mock_span = MagicMock()
        ctx = SpanContext(span=mock_span)

        exc = ValueError("test error")
        ctx.record_exception(exc)

        mock_span.record_exception.assert_called_once_with(exc)


class TestPhoenixTracerInit:
    """Tests for PhoenixTracer initialization."""

    def test_tracer_disabled_by_config(self):
        """Test that tracer respects disabled config."""
        phoenix_config = PhoenixConfig(enabled=False)
        eval_config = EvalConfig()

        tracer = PhoenixTracer(phoenix_config, eval_config)
        result = tracer.initialize()

        assert result is False
        assert not tracer.is_initialized()

    def test_tracer_not_initialized_by_default(self):
        """Test that tracer is not initialized by default."""
        phoenix_config = PhoenixConfig(enabled=True)
        eval_config = EvalConfig()

        tracer = PhoenixTracer(phoenix_config, eval_config)

        assert not tracer.is_initialized()

    @patch("nuvii_eval.instrumentation.phoenix_tracer._lazy_import_phoenix")
    def test_tracer_handles_import_failure(self, mock_import):
        """Test graceful handling of Phoenix import failure."""
        mock_import.return_value = (None, None, None, None, None)

        phoenix_config = PhoenixConfig(enabled=True)
        eval_config = EvalConfig()

        tracer = PhoenixTracer(phoenix_config, eval_config)
        result = tracer.initialize()

        assert result is False
        assert not tracer.is_initialized()


class TestPhoenixTracerContextManagers:
    """Tests for tracer context managers."""

    def test_trace_evaluation_when_not_initialized(self):
        """Test trace_evaluation returns no-op when not initialized."""
        phoenix_config = PhoenixConfig(enabled=False)
        eval_config = EvalConfig()

        tracer = PhoenixTracer(phoenix_config, eval_config)

        with tracer.trace_evaluation("test_001", "icd", {}) as ctx:
            assert not ctx.is_recording
            # Should not raise
            ctx.set_attribute("key", "value")

    def test_trace_api_call_when_not_initialized(self):
        """Test trace_api_call returns no-op when not initialized."""
        phoenix_config = PhoenixConfig(enabled=False)
        eval_config = EvalConfig()

        tracer = PhoenixTracer(phoenix_config, eval_config)

        with tracer.trace_api_call("/api/test", "POST") as ctx:
            assert not ctx.is_recording

    @patch("nuvii_eval.instrumentation.phoenix_tracer._lazy_import_phoenix")
    def test_trace_evaluation_with_mocked_otel(self, mock_import):
        """Test trace_evaluation with mocked OpenTelemetry."""
        # Set up mocks
        mock_px = MagicMock()
        mock_trace = MagicMock()
        mock_Status = MagicMock()
        mock_StatusCode = MagicMock()
        mock_SpanKind = MagicMock()

        mock_import.return_value = (
            mock_px,
            mock_trace,
            mock_Status,
            mock_StatusCode,
            mock_SpanKind,
        )

        # Mock the tracer
        mock_tracer = MagicMock()
        mock_span = MagicMock()
        mock_span_context = MagicMock()
        mock_span_context.span_id = 12345
        mock_span_context.trace_id = 67890
        mock_span.get_span_context.return_value = mock_span_context

        mock_tracer.start_as_current_span.return_value.__enter__ = MagicMock(
            return_value=mock_span
        )
        mock_tracer.start_as_current_span.return_value.__exit__ = MagicMock(
            return_value=None
        )
        mock_trace.get_tracer.return_value = mock_tracer

        # Mock register
        with patch("nuvii_eval.instrumentation.phoenix_tracer.register"):
            phoenix_config = PhoenixConfig(enabled=True, endpoint="http://remote:6006")
            eval_config = EvalConfig()

            tracer = PhoenixTracer(phoenix_config, eval_config)
            tracer.initialize()

            with tracer.trace_evaluation("test_001", "icd", {"model": "v1"}) as ctx:
                ctx.set_attribute("custom_metric", 0.95)

            # Verify span was created
            mock_tracer.start_as_current_span.assert_called()


class TestPhoenixTracerLogging:
    """Tests for tracer logging methods."""

    def test_log_api_call(self):
        """Test logging API call to span context."""
        mock_span = MagicMock()
        ctx = SpanContext(span=mock_span)

        phoenix_config = PhoenixConfig(enabled=True, collect_inputs=False)
        eval_config = EvalConfig(phi_safe_mode=True)

        tracer = PhoenixTracer(phoenix_config, eval_config)

        tracer.log_api_call(
            ctx=ctx,
            endpoint="suggest_codes",
            request_payload={"clinical_note": "test note"},
            response={"codes": ["E11.9"]},
            latency_ms=250,
            token_count=100,
            status_code=200,
        )

        # Should have logged metrics
        assert mock_span.set_attribute.call_count >= 3

    def test_log_api_call_phi_safe_no_inputs(self):
        """Test that inputs are not logged in PHI safe mode."""
        mock_span = MagicMock()
        ctx = SpanContext(span=mock_span)

        phoenix_config = PhoenixConfig(enabled=True, collect_inputs=True)
        eval_config = EvalConfig(phi_safe_mode=True)

        tracer = PhoenixTracer(phoenix_config, eval_config)

        tracer.log_api_call(
            ctx=ctx,
            endpoint="suggest_codes",
            request_payload={"clinical_note": "sensitive data"},
            response={"codes": []},
            latency_ms=100,
        )

        # Verify request was NOT logged (PHI safe mode)
        call_args_list = [call[0] for call in mock_span.set_attribute.call_args_list]
        request_logged = any("request" in str(args) for args in call_args_list)
        assert not request_logged

    def test_log_eval_result(self):
        """Test logging evaluation results."""
        mock_span = MagicMock()
        ctx = SpanContext(span=mock_span)

        phoenix_config = PhoenixConfig(enabled=True)
        eval_config = EvalConfig()

        tracer = PhoenixTracer(phoenix_config, eval_config)

        tracer.log_eval_result(
            ctx=ctx,
            scores={"accuracy": 0.95, "f1": 0.90},
            passed=True,
            details={"predictions": ["E11.9"]},
        )

        # Should have logged scores and passed status
        call_args_list = [call[0] for call in mock_span.set_attribute.call_args_list]

        # Check that passed was logged
        passed_logged = any(
            args[0] == "eval.passed" and args[1] is True
            for args in call_args_list
        )
        assert passed_logged

    def test_log_retrieval(self):
        """Test logging retrieval results."""
        mock_span = MagicMock()
        ctx = SpanContext(span=mock_span)

        phoenix_config = PhoenixConfig(enabled=True)
        eval_config = EvalConfig(phi_safe_mode=False)

        tracer = PhoenixTracer(phoenix_config, eval_config)

        tracer.log_retrieval(
            ctx=ctx,
            query="diabetes diagnosis",
            chunks=[
                {"text": "Patient has diabetes", "score": 0.95},
                {"text": "Blood sugar elevated", "score": 0.85},
            ],
            scores=[0.95, 0.85],
        )

        # Should have logged chunk count and scores
        call_args_list = [call[0] for call in mock_span.set_attribute.call_args_list]

        chunk_count_logged = any(
            args[0] == "retrieval.chunk_count"
            for args in call_args_list
        )
        assert chunk_count_logged

    def test_log_retrieval_phi_safe(self):
        """Test retrieval logging respects PHI safe mode."""
        mock_span = MagicMock()
        ctx = SpanContext(span=mock_span)

        phoenix_config = PhoenixConfig(enabled=True)
        eval_config = EvalConfig(phi_safe_mode=True)

        tracer = PhoenixTracer(phoenix_config, eval_config)

        tracer.log_retrieval(
            ctx=ctx,
            query="sensitive query",
            chunks=[{"text": "sensitive content"}],
            scores=[0.9],
        )

        # Query and chunk previews should NOT be logged
        call_args_list = [call[0] for call in mock_span.set_attribute.call_args_list]

        query_logged = any("query" in str(args) for args in call_args_list)
        previews_logged = any("previews" in str(args) for args in call_args_list)

        assert not query_logged
        assert not previews_logged


class TestPhoenixTracerRunSpan:
    """Tests for run-level span management."""

    def test_create_run_span_when_not_initialized(self):
        """Test creating run span when tracer not initialized."""
        phoenix_config = PhoenixConfig(enabled=False)
        eval_config = EvalConfig()

        tracer = PhoenixTracer(phoenix_config, eval_config)

        ctx = tracer.create_run_span(
            run_id="run_001",
            dataset_path="datasets/test.jsonl",
            evaluator_types=["icd", "hcc"],
            config={},
        )

        assert not ctx.is_recording

    def test_end_run_span(self):
        """Test ending a run span with stats."""
        mock_span = MagicMock()
        ctx = SpanContext(span=mock_span)

        phoenix_config = PhoenixConfig(enabled=True)
        eval_config = EvalConfig()

        tracer = PhoenixTracer(phoenix_config, eval_config)

        tracer.end_run_span(
            ctx=ctx,
            total_cases=100,
            passed_count=85,
            failed_count=10,
            error_count=5,
        )

        # Should have logged all stats
        call_args_list = [call[0] for call in mock_span.set_attribute.call_args_list]

        total_logged = any(args[0] == "run.total_cases" for args in call_args_list)
        passed_logged = any(args[0] == "run.passed" for args in call_args_list)
        pass_rate_logged = any(args[0] == "run.pass_rate" for args in call_args_list)

        assert total_logged
        assert passed_logged
        assert pass_rate_logged

        # Should have called end()
        mock_span.end.assert_called_once()


class TestPhoenixTracerIntegration:
    """Integration-style tests for Phoenix tracer."""

    def test_full_evaluation_flow_disabled(self):
        """Test full evaluation flow with tracing disabled."""
        phoenix_config = PhoenixConfig(enabled=False)
        eval_config = EvalConfig()

        tracer = PhoenixTracer(phoenix_config, eval_config)
        tracer.initialize()

        # Simulate evaluation flow
        with tracer.trace_evaluation("test_001", "icd", {"model": "v1"}) as ctx:
            # Simulate API call
            with tracer.trace_api_call("/api/v2/coding/suggest") as api_ctx:
                tracer.log_api_call(
                    api_ctx,
                    "suggest_codes",
                    {"note": "test"},
                    {"codes": []},
                    100,
                )

            # Log results
            tracer.log_eval_result(
                ctx,
                scores={"accuracy": 0.95},
                passed=True,
            )

        # Should complete without error
        assert True

    def test_config_attributes_filtered(self):
        """Test that sensitive config values are filtered."""
        mock_span = MagicMock()

        # Create a mock for start_as_current_span
        ctx_manager = MagicMock()
        ctx_manager.__enter__ = MagicMock(return_value=mock_span)
        ctx_manager.__exit__ = MagicMock(return_value=None)

        mock_span_context = MagicMock()
        mock_span_context.span_id = 12345
        mock_span_context.trace_id = 67890
        mock_span.get_span_context.return_value = mock_span_context

        phoenix_config = PhoenixConfig(enabled=True)
        eval_config = EvalConfig()

        tracer = PhoenixTracer(phoenix_config, eval_config)
        tracer._initialized = True
        tracer._tracer = MagicMock()
        tracer._tracer.start_as_current_span.return_value = ctx_manager
        tracer._SpanKind = MagicMock()
        tracer._Status = MagicMock()
        tracer._StatusCode = MagicMock()

        # Run with sensitive config
        run_config = {
            "model": "v1",
            "api_key": "secret_key_12345",
            "credentials": {"user": "admin"},
        }

        with tracer.trace_evaluation("test_001", "icd", run_config):
            pass

        # Check that api_key was NOT logged
        call_args_list = [str(call) for call in mock_span.set_attribute.call_args_list]
        api_key_logged = any("secret_key" in str(args) for args in call_args_list)

        assert not api_key_logged
