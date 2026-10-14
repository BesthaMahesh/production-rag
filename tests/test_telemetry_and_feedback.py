from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from telemetry import (
    estimate_cost,
    get_live_spans,
    get_otel_status,
    log_feedback,
    log_run,
    read_jsonl_logs,
    trace_stage,
)


def test_otel_initialization_and_live_spans():
    status = get_otel_status()
    assert status["otel_sdk_available"] is True

    # Instrument a test stage
    with trace_stage("test.stage", {"test_attr": 42}) as span:
        pass

    spans = get_live_spans()
    assert any(s["stage"] == "test.stage" for s in spans)


def test_estimate_cost():
    # Prompt: 1000 tokens, Completion: 500 tokens, Input rate: $1.00/1M, Output rate: $2.00/1M
    cost = estimate_cost(1000, 500, 1.0, 2.0)
    # Expected: (1000*1 + 500*2) / 1,000,000 = 2000 / 1,000,000 = 0.002
    assert cost["estimated_cost_usd"] == 0.002
    assert cost["cost_status"] == "estimated_from_reported_usage"

    # Missing tokens
    cost_unavail = estimate_cost(None, None, 1.0, 2.0)
    assert cost_unavail["estimated_cost_usd"] is None
    assert cost_unavail["cost_status"] == "usage_unavailable"


def test_feedback_logging_and_reading():
    run_id = "test-run-12345"
    log_feedback({
        "run_id": run_id,
        "query": "What is the policy?",
        "rating": "thumbs_up",
        "comment": "Accurate answer",
    })

    records = read_jsonl_logs("feedback.jsonl", max_records=50)
    assert any(r.get("run_id") == run_id for r in records)
