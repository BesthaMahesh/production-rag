from evaluation import lexical_overlap

def test_lexical_overlap():
    assert lexical_overlap("maternity leave is available", "maternity leave") == 1.0

def test_empty_reference():
    assert lexical_overlap("some answer", "") == 0.0
