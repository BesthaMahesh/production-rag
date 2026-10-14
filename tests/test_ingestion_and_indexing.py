from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from rag_system import (
    Chunk,
    compute_content_hash,
    chunk_text,
    chunk_text_hierarchical,
    read_pdf,
    read_text_file,
    stable_id,
    ProductionRAG,
)


def test_tokenize_case_insensitive():
    from rag_system import tokenize
    assert tokenize("Maternity Leave") == tokenize("maternity leave")


def test_stable_id_deterministic():
    assert stable_id("doc1", "p1") == stable_id("doc1", "p1")
    assert stable_id("doc1", "p1") != stable_id("p1", "doc1")


def test_compute_content_hash():
    h1 = compute_content_hash("Sample chunk text")
    h2 = compute_content_hash("Sample chunk text")
    h3 = compute_content_hash("Different text")
    assert h1 == h2
    assert h1 != h3


def test_chunk_text_bounds_and_empty():
    assert chunk_text("   ") == []
    text = ("Sentence number one about HR policy. " * 50)
    chunks = chunk_text(text, max_chars=300, overlap_chars=50)
    assert chunks
    assert all(len(c) <= 300 for c in chunks)
    assert all(c.strip() for c in chunks)


def test_chunk_text_hierarchical():
    text = "Section A covers leave. Section B covers travel. Section C covers remote work."
    h_chunks = chunk_text_hierarchical(text, max_chars=100, overlap_chars=20)
    assert h_chunks
    for rec in h_chunks:
        assert "text" in rec
        assert "parent_text" in rec
        assert "parent_id" in rec
        assert rec["parent_id"] is not None


def test_pdf_extraction_page_preservation():
    pdf_path = Path("RAG_Document_01_HR_Policy_Handbook.pdf")
    if pdf_path.exists():
        pages, is_scanned = read_pdf(pdf_path)
        assert len(pages) == 2
        assert pages[0]["page"] == 1
        assert pages[1]["page"] == 2
        assert "annual leave" in pages[0]["text"].lower()
        assert not is_scanned


def test_repeat_ingestion_prevents_duplicate_chunks():
    rag = ProductionRAG(vector_backend="numpy")
    with tempfile.NamedTemporaryFile("w+", suffix=".txt", delete=False, encoding="utf-8") as f:
        f.write("Policy line 1. Policy line 2. Policy line 3.")
        temp_name = f.name

    try:
        res1 = rag.ingest_paths([temp_name], reset_index=True)
        count1 = res1["chunks"]
        assert count1 > 0

        # Ingest same path again without reset
        res2 = rag.ingest_paths([temp_name], reset_index=False)
        assert res2["new_chunks_added"] == 0
        assert len(rag.chunks) == count1
    finally:
        Path(temp_name).unlink(missing_ok=True)
