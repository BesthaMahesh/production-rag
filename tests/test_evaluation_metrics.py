from __future__ import annotations

import pytest

from evaluation import (
    compute_ndcg_at_k,
    check_evidence_match,
    lexical_overlap,
)


def test_lexical_overlap():
    assert lexical_overlap("maternity leave is available", "maternity leave") == 1.0
    assert lexical_overlap("some completely different answer", "maternity leave") == 0.0
    assert lexical_overlap("some answer", "") == 0.0


def test_check_evidence_match():
    ans = "Employees receive 18 working days of annual leave."
    assert check_evidence_match(ans, "18 working days") == 1.0
    assert check_evidence_match(ans, "not specified") == 1.0
    assert check_evidence_match(ans, "unlimited PTO; 30 days") == 0.0


def test_compute_ndcg_at_k():
    # Rank 1 is relevant -> optimal nDCG = 1.0
    assert compute_ndcg_at_k([1], k=5) == 1.0
    # No relevant ranks -> nDCG = 0.0
    assert compute_ndcg_at_k([], k=5) == 0.0
    # Rank 2 is relevant -> DCG = 1 / log2(3) = 0.6309
    ndcg_r2 = compute_ndcg_at_k([2], k=5)
    assert 0.6 < ndcg_r2 < 0.65
    # Relevant item outside top k
    assert compute_ndcg_at_k([10], k=5) == 0.0
