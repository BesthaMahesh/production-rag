from pathlib import Path
import tempfile

from rag_system import chunk_text, tokenize, stable_id


def test_tokenize_case_insensitive():
    assert tokenize("Maternity Leave") == tokenize("maternity leave")


def test_stable_id_deterministic():
    assert stable_id("a", "b") == stable_id("a", "b")
    assert stable_id("a", "b") != stable_id("b", "a")


def test_chunk_text_nonempty_and_bounds():
    text = ("This is a sentence about policy. " * 100)
    chunks = chunk_text(text, max_chars=300, overlap_chars=40)
    assert chunks
    assert all(len(c) <= 300 for c in chunks)
    assert all(c.strip() for c in chunks)


def test_chunk_text_empty():
    assert chunk_text("   ") == []
