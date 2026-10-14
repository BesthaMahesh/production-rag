"""OpenTelemetry instrumentation, local trace collection, cost estimation, and audit logging.
Provides genuine OpenTelemetry spans and an in-memory collector for UI inspection.
"""
from __future__ import annotations

import json
import logging
import os
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Generator

LOGGER = logging.getLogger("production_rag.telemetry")
DATA_DIR = Path(os.getenv("RAG_DATA_DIR", "data"))

# In-memory record of recent span events for UI observability
_LIVE_SPANS_BUFFER: list[dict[str, Any]] = []
_MAX_BUFFER_SIZE = 500

# OpenTelemetry initialization
_TRACER = None
_IN_MEMORY_EXPORTER = None
_OTEL_AVAILABLE = False
_OTLP_EXPORTER_ACTIVE = False


def init_telemetry(service_name: str = "production-rag-platform", otlp_endpoint: str = "") -> bool:
    """Initialize genuine OpenTelemetry TracerProvider with InMemory exporter and optional OTLP."""
    global _TRACER, _IN_MEMORY_EXPORTER, _OTEL_AVAILABLE, _OTLP_EXPORTER_ACTIVE
    try:
        from opentelemetry import trace
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import SimpleSpanProcessor
        from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

        resource = Resource.create({"service.name": service_name})
        provider = TracerProvider(resource=resource)
        
        # Local in-memory exporter for UI inspection
        _IN_MEMORY_EXPORTER = InMemorySpanExporter()
        provider.add_span_processor(SimpleSpanProcessor(_IN_MEMORY_EXPORTER))

        # Optional OTLP gRPC exporter
        if otlp_endpoint and otlp_endpoint.strip():
            try:
                from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
                otlp_exporter = OTLPSpanExporter(endpoint=otlp_endpoint.strip(), insecure=True)
                provider.add_span_processor(SimpleSpanProcessor(otlp_exporter))
                _OTLP_EXPORTER_ACTIVE = True
                LOGGER.info("Configured OTLP gRPC export to %s", otlp_endpoint)
            except Exception as otlp_err:
                LOGGER.warning("Could not initialize OTLP exporter for %s: %s", otlp_endpoint, otlp_err)
                _OTLP_EXPORTER_ACTIVE = False
        else:
            _OTLP_EXPORTER_ACTIVE = False

        trace.set_tracer_provider(provider)
        _TRACER = trace.get_tracer(service_name)
        _OTEL_AVAILABLE = True
        LOGGER.info("OpenTelemetry successfully initialized for service '%s'", service_name)
        return True
    except Exception as exc:
        LOGGER.warning("OpenTelemetry initialization failed: %s. Falling back to local logging.", exc)
        _OTEL_AVAILABLE = False
        return False


# Auto-initialize with default settings on module load
init_telemetry()


@contextmanager
def trace_stage(stage_name: str, attributes: dict[str, Any] | None = None) -> Generator[Any, None, None]:
    """Context manager to instrument any RAG stage as an OpenTelemetry span and buffer record."""
    started_perf = time.perf_counter()
    started_utc = datetime.now(timezone.utc).isoformat()
    attrs = dict(attributes or {})
    span_obj = None
    span_ctx = None

    if _OTEL_AVAILABLE and _TRACER is not None:
        try:
            span_ctx = _TRACER.start_as_current_span(stage_name)
            span_obj = span_ctx.__enter__()
            for k, v in attrs.items():
                if isinstance(v, (str, int, float, bool)):
                    span_obj.set_attribute(k, v)
                else:
                    span_obj.set_attribute(k, str(v))
        except Exception as e:
            LOGGER.debug("Error starting span: %s", e)

    status = "OK"
    error_msg = None
    try:
        yield span_obj
    except Exception as exc:
        status = "ERROR"
        error_msg = str(exc)
        if span_obj:
            try:
                span_obj.record_exception(exc)
                span_obj.set_attribute("error", True)
                span_obj.set_attribute("error.message", error_msg)
            except Exception:
                pass
        raise
    finally:
        elapsed_ms = round((time.perf_counter() - started_perf) * 1000, 2)
        if span_obj and span_ctx:
            try:
                span_obj.set_attribute("duration_ms", elapsed_ms)
                span_obj.set_attribute("status", status)
                span_ctx.__exit__(None, None, None)
            except Exception:
                pass

        # Record span event in live buffer
        record = {
            "timestamp": started_utc,
            "stage": stage_name,
            "duration_ms": elapsed_ms,
            "status": status,
            "error": error_msg,
            "attributes": attrs,
        }
        _LIVE_SPANS_BUFFER.append(record)
        if len(_LIVE_SPANS_BUFFER) > _MAX_BUFFER_SIZE:
            _LIVE_SPANS_BUFFER.pop(0)


def get_live_spans() -> list[dict[str, Any]]:
    """Return in-memory buffer of recent instrumented spans."""
    return list(_LIVE_SPANS_BUFFER)


def get_otel_status() -> dict[str, Any]:
    """Return current OpenTelemetry status information."""
    return {
        "otel_sdk_available": _OTEL_AVAILABLE,
        "otlp_grpc_exporter_active": _OTLP_EXPORTER_ACTIVE,
        "buffered_spans_count": len(_LIVE_SPANS_BUFFER),
        "in_memory_exporter_ready": _IN_MEMORY_EXPORTER is not None,
    }


def append_jsonl(filename: str, payload: dict[str, Any]) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    record = {"timestamp_utc": datetime.now(timezone.utc).isoformat(), **payload}
    with (DATA_DIR / filename).open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")


def estimate_cost(
    prompt_tokens: int | None,
    completion_tokens: int | None,
    input_price_per_million: float,
    output_price_per_million: float,
) -> dict[str, Any]:
    """Price rates are user-configured USD per 1M tokens; zero rates do not imply provider is always free."""
    if prompt_tokens is None or completion_tokens is None:
        return {"estimated_cost_usd": None, "cost_status": "usage_unavailable"}
    cost = (prompt_tokens * input_price_per_million + completion_tokens * output_price_per_million) / 1_000_000
    return {"estimated_cost_usd": round(cost, 8), "cost_status": "estimated_from_reported_usage"}


def log_run(payload: dict[str, Any]) -> None:
    with trace_stage("rag.log_run", {"run_id": payload.get("run_id", "unknown")}):
        append_jsonl("runs.jsonl", payload)


def log_feedback(payload: dict[str, Any]) -> None:
    with trace_stage("rag.log_feedback", {"run_id": payload.get("run_id", "unknown")}):
        append_jsonl("feedback.jsonl", payload)


def read_jsonl_logs(filename: str, max_records: int = 100) -> list[dict[str, Any]]:
    """Read recent records from a local JSONL file."""
    path = DATA_DIR / filename
    if not path.exists():
        return []
    records = []
    try:
        with path.open("r", encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        return records[-max_records:]
    except Exception as exc:
        LOGGER.warning("Could not read %s: %s", path, exc)
        return []
