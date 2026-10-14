from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from llm_client import detect_abstention, extract_citations
from rag_system import ProductionRAG, classify_query


def test_classify_query_categories():
    c_unans = classify_query("What is the password expiration period for corporate accounts?")
    assert c_unans["category"] == "unanswerable"

    c_comp = classify_query("Compare the access-review requirements in IT Security and Support Playbook")
    assert c_comp["category"] == "comparison"

    c_fact = classify_query("What is the maximum paid sick leave per calendar year?")
    assert c_fact["category"] == "factual_lookup"


def test_detect_abstention_and_citations():
    text_abstained = "The provided documents do not contain information regarding this question."
    assert detect_abstention(text_abstained) is True

    text_answered = "Full-time employees receive 18 days of annual leave [SOURCE 1]."
    assert detect_abstention(text_answered) is False
    assert extract_citations(text_answered) == ["1"]

    multi_cite = "According to [SOURCE 1] and [SOURCE 2], approval is required."
    assert extract_citations(multi_cite) == ["1", "2"]


def test_dense_and_bm25_retrieval_and_rrf():
    rag = ProductionRAG(vector_backend="numpy", use_reranker=False)
    with tempfile.NamedTemporaryFile("w+", suffix=".txt", delete=False, encoding="utf-8") as f:
        f.write("Employees may take up to 12 days of paid sick leave per calendar year. Medical certificates are required after 2 days.")
        temp_path = f.name

    try:
        rag.ingest_paths([temp_path], reset_index=True)
        ret = rag.retrieve("sick leave calendar year", dense_k=5, sparse_k=5, hybrid_k=5, rerank_k=3)
        assert len(ret["final_results"]) > 0
        top = ret["final_results"][0]
        assert "sick leave" in top["text"].lower()
        assert "[SOURCE 1]" in ret["context"]
        assert ret["metrics"]["retrieval_latency_ms"] > 0
    finally:
        Path(temp_path).unlink(missing_ok=True)


def test_empty_query_validation():
    rag = ProductionRAG(vector_backend="numpy")
    with pytest.raises(ValueError, match="empty"):
        rag.retrieve("   ")
