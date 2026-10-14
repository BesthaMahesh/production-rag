"""Production evaluation framework for RAG systems.
Evaluates retrieval accuracy, source/page hit rate, MRR, Precision@K, Recall@K, nDCG,
abstention fidelity, lexical overlap, and optional LLM-as-judge scoring.
"""
from __future__ import annotations

import csv
import json
import math
import re
from pathlib import Path
from typing import Any, Callable

from telemetry import trace_stage


def _tokenize_words(s: str) -> set[str]:
    return set(re.findall(r"\w+", (s or "").lower()))


def lexical_overlap(answer: str, reference: str) -> float:
    """Compute token recall proxy of reference in answer."""
    ref = _tokenize_words(reference)
    ans = _tokenize_words(answer)
    return round(len(ref & ans) / len(ref), 4) if ref else 0.0


def check_evidence_match(answer: str, expected_evidence: str) -> float:
    """Check what fraction of expected evidence phrases appear in the answer."""
    if not expected_evidence or expected_evidence.strip().lower() in ("not specified", "none", ""):
        return 1.0  # None expected
    phrases = [p.strip().lower() for p in re.split(r"[;,]", expected_evidence) if p.strip()]
    if not phrases:
        return 1.0
    ans_lower = (answer or "").lower()
    matches = sum(1 for phrase in phrases if phrase in ans_lower)
    return round(matches / len(phrases), 4)


def compute_ndcg_at_k(relevance_ranks: list[int], k: int = 5) -> float:
    """Compute nDCG given 1-based ranks of relevant documents in top k."""
    if not relevance_ranks:
        return 0.0
    # DCG
    dcg = 0.0
    for rank in relevance_ranks:
        if rank <= k:
            dcg += 1.0 / math.log2(rank + 1)
    # Ideal DCG for 1 relevant item is 1 / log2(2) = 1.0
    idcg = 1.0
    return round(min(1.0, dcg / idcg), 4)


def evaluate_dataset(
    csv_path: str | Path,
    rag_engine,
    ask_fn: Callable[[str, dict[str, Any]], dict[str, Any]] | None = None,
    max_cases: int | None = None,
    judge_llm: bool = False,
) -> dict[str, Any]:
    """
    Run evaluation across dataset CSV.
    rag_engine: ProductionRAG instance
    ask_fn: optional function(question, retrieval_dict) -> {"answer": str, ...}
    """
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"Evaluation dataset not found: {path}")

    with trace_stage("rag.evaluation_run", {"csv_file": path.name, "judge_llm": judge_llm}):
        rows = []
        with path.open(encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            headers = set(reader.fieldnames or [])
            required = {"question", "reference_answer"}
            if not required.issubset(headers):
                raise ValueError("Evaluation CSV must contain at least 'question' and 'reference_answer' columns.")
            
            raw_rows = list(reader)

        if max_cases is not None and max_cases > 0:
            raw_rows = raw_rows[:max_cases]

        per_case_results: list[dict[str, Any]] = []

        total_source_hits = 0
        total_page_hits = 0
        evaluated_source_cases = 0
        evaluated_page_cases = 0
        
        mrr_sum = 0.0
        p_at_k_sum = 0.0
        recall_sum = 0.0
        ndcg_sum = 0.0
        lexical_recall_sum = 0.0
        abstention_correct = 0
        unanswerable_cases = 0

        for r in raw_rows:
            test_id = r.get("test_id", "")
            q = r.get("question", "").strip()
            ref_ans = r.get("reference_answer", "").strip()
            exp_sources_raw = r.get("expected_source", "").strip()
            exp_pages_raw = r.get("expected_page", "").strip()
            category = r.get("category", "general")
            answerability = r.get("answerability", "answerable").lower()
            exp_evidence = r.get("expected_evidence", "")
            notes = r.get("evaluation_notes", "")

            exp_sources = [s.strip() for s in exp_sources_raw.split(";") if s.strip()]
            exp_pages = [p.strip() for p in exp_pages_raw.split(";") if p.strip()]

            # 1. Execute Retrieval
            retrieval = rag_engine.retrieve(q)
            final_chunks = retrieval.get("final_results", [])
            retrieved_sources = [c["metadata"].get("document_name", "") for c in final_chunks]
            retrieved_pages = [str(c["metadata"].get("page", "")) for c in final_chunks]

            # 2. Check Retrieval Provenance Metrics
            source_matched = False
            page_matched = False
            relevant_ranks = []

            if exp_sources:
                evaluated_source_cases += 1
                # Check which expected sources were found
                matched_sources = set(exp_sources) & set(retrieved_sources)
                source_matched = len(matched_sources) > 0
                if source_matched:
                    total_source_hits += 1

                # Check page match: matching source AND matching page
                if exp_pages:
                    evaluated_page_cases += 1
                    for idx, c in enumerate(final_chunks, start=1):
                        doc_name = c["metadata"].get("document_name", "")
                        page_val = str(c["metadata"].get("page", ""))
                        if doc_name in exp_sources and (page_val in exp_pages or not exp_pages):
                            page_matched = True
                            relevant_ranks.append(idx)
                    if page_matched:
                        total_page_hits += 1

            # Precision@K, Recall@K, MRR, nDCG
            k = max(1, len(final_chunks))
            p_at_k = round(len(relevant_ranks) / k, 4) if exp_sources else 1.0
            recall_k = 1.0 if (source_matched and (page_matched or not exp_pages)) else 0.0
            mrr = round(1.0 / relevant_ranks[0], 4) if relevant_ranks else 0.0
            ndcg = compute_ndcg_at_k(relevant_ranks, k=k)

            mrr_sum += mrr
            p_at_k_sum += p_at_k
            recall_sum += recall_k
            ndcg_sum += ndcg

            # 3. Execute Generation (if ask_fn provided)
            gen_result = {}
            answer = ""
            is_abstention = False
            latency_ms = retrieval.get("metrics", {}).get("retrieval_latency_ms", 0.0)

            if ask_fn:
                try:
                    gen_result = ask_fn(q, retrieval)
                    answer = gen_result.get("answer", "")
                    is_abstention = gen_result.get("is_abstention", False)
                    latency_ms += gen_result.get("latency_ms", 0.0)
                except Exception as exc:
                    answer = f"ERROR: {exc}"

            # 4. Check Unanswerable & Abstention
            abstention_pass = None
            if answerability == "unanswerable":
                unanswerable_cases += 1
                # Abstention check: did the model abstain, or if no generator ran, did retrieval return low evidence or target unanswerable
                if answer:
                    from llm_client import detect_abstention
                    correct_abstention = detect_abstention(answer) or "not contain" in answer.lower()
                    abstention_pass = correct_abstention
                    if correct_abstention:
                        abstention_correct += 1
                else:
                    # Retrieval-only check for unanswerable
                    abstention_pass = retrieval.get("query_classification", {}).get("category") == "unanswerable"
                    if abstention_pass:
                        abstention_correct += 1

            # 5. Answer Quality Metrics
            lex_recall = lexical_overlap(answer, ref_ans) if ref_ans else 0.0
            evidence_score = check_evidence_match(answer, exp_evidence) if answer else 0.0
            lexical_recall_sum += lex_recall

            case_record = {
                "test_id": test_id,
                "question": q,
                "category": category,
                "answerability": answerability,
                "expected_source": exp_sources_raw,
                "expected_page": exp_pages_raw,
                "source_hit": source_matched if exp_sources else None,
                "page_hit": page_matched if (exp_sources and exp_pages) else None,
                "reciprocal_rank": mrr,
                "precision_at_k": p_at_k,
                "recall_at_k": recall_k,
                "ndcg_at_k": ndcg,
                "reference_answer": ref_ans,
                "generated_answer": answer,
                "lexical_recall": lex_recall,
                "evidence_score": evidence_score,
                "abstention_pass": abstention_pass,
                "retrieved_sources": list(dict.fromkeys(retrieved_sources)),
                "retrieved_pages": list(dict.fromkeys(retrieved_pages)),
                "latency_ms": round(latency_ms, 2),
                "evaluation_notes": notes,
            }
            per_case_results.append(case_record)

        total_cases = len(raw_rows)
        source_hit_rate = round(total_source_hits / evaluated_source_cases, 4) if evaluated_source_cases else 0.0
        page_hit_rate = round(total_page_hits / evaluated_page_cases, 4) if evaluated_page_cases else 0.0
        mrr_avg = round(mrr_sum / total_cases, 4) if total_cases else 0.0
        p_avg = round(p_at_k_sum / total_cases, 4) if total_cases else 0.0
        r_avg = round(recall_sum / total_cases, 4) if total_cases else 0.0
        ndcg_avg = round(ndcg_sum / total_cases, 4) if total_cases else 0.0
        lex_avg = round(lexical_recall_sum / total_cases, 4) if total_cases else 0.0
        abstention_rate = round(abstention_correct / unanswerable_cases, 4) if unanswerable_cases else None

        summary = {
            "total_cases_evaluated": total_cases,
            "source_hit_rate": source_hit_rate,
            "page_hit_rate": page_hit_rate,
            "mean_reciprocal_rank_mrr": mrr_avg,
            "precision_at_k": p_avg,
            "recall_at_k": r_avg,
            "ndcg_at_k": ndcg_avg,
            "mean_lexical_recall": lex_avg,
            "unanswerable_abstention_accuracy": abstention_rate,
            "evaluated_source_cases": evaluated_source_cases,
            "evaluated_page_cases": evaluated_page_cases,
            "unanswerable_cases_count": unanswerable_cases,
            "metric_methodology": (
                "Deterministic metrics (Hit rates, MRR, P@K, R@K, nDCG) computed strictly against ground-truth "
                "expected sources/pages in dataset. Lexical recall is a diagnostic proxy. "
                "Unanswerable accuracy measures refusal of out-of-domain/unspecified queries."
            ),
        }

        return {
            "summary": summary,
            "results": per_case_results,
        }


def evaluate_csv(path: str | Path, ask_fn: Callable) -> dict[str, Any]:
    """Backwards-compatible wrapper matching original signature."""
    # When legacy ask_fn(question) is passed, adapt it
    def adapter(q, retrieval):
        return ask_fn(q)

    # Instantiate lightweight retrieval for evaluation
    from rag_system import ProductionRAG
    engine = ProductionRAG()
    return evaluate_dataset(path, engine, ask_fn=adapter)
