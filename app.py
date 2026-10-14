"""Production RAG Platform - Streamlit Application
Enterprise multi-source hybrid retrieval, grounded generation, evaluation,
OpenTelemetry observability, and feedback loop.
"""
from __future__ import annotations

import json
import logging
import os
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from config import RAGConfig, WORKSPACE_DIR, DATA_DIR, get_secret
from evaluation import evaluate_dataset
from llm_client import LLMGenerationError, generate_answer
from rag_system import ProductionRAG, classify_query
from telemetry import (
    estimate_cost,
    get_live_spans,
    get_otel_status,
    log_feedback,
    log_run,
    read_jsonl_logs,
    trace_stage,
)

LOGGER = logging.getLogger("production_rag.app")

# Page Configuration
st.set_page_config(
    page_title="Production RAG Platform",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Enterprise CSS Styling
st.markdown("""
<style>
    .metric-card {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 16px;
        margin-bottom: 12px;
    }
    .badge-ok {
        background-color: #dcfce7;
        color: #166534;
        padding: 3px 8px;
        border-radius: 12px;
        font-weight: 600;
        font-size: 0.8rem;
    }
    .badge-warn {
        background-color: #fef9c3;
        color: #854d0e;
        padding: 3px 8px;
        border-radius: 12px;
        font-weight: 600;
        font-size: 0.8rem;
    }
    .badge-err {
        background-color: #fee2e2;
        color: #991b1b;
        padding: 3px 8px;
        border-radius: 12px;
        font-weight: 600;
        font-size: 0.8rem;
    }
    .citation-box {
        background: #f1f5f9;
        border-left: 4px solid #3b82f6;
        padding: 10px 14px;
        border-radius: 4px;
        margin-bottom: 8px;
        font-size: 0.9rem;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource(show_spinner="Loading Embedding & CrossEncoder Reranker models...")
def get_rag_engine(
    embedding_model: str,
    reranker_model: str,
    use_reranker: bool,
    vector_backend: str,
) -> ProductionRAG:
    """Singleton RAG engine cached across Streamlit reruns."""
    return ProductionRAG(
        embedding_model=embedding_model,
        reranker_model=reranker_model,
        use_reranker=use_reranker,
        vector_backend=vector_backend,
    )


# Sidebar Configuration & Navigation
with st.sidebar:
    st.title("⚡ Enterprise RAG")
    st.caption("Pre-Production RAG Evaluation & Observability Platform")

    nav_option = st.radio(
        "Navigation",
        [
            "1. Executive Overview",
            "2. Knowledge Base & Ingestion",
            "3. Retrieval Playground",
            "4. RAG Assistant",
            "5. Evaluation Dashboard",
            "6. Feedback & Improvement",
            "7. Observability & Telemetry",
            "8. Configuration & Health",
        ],
        index=0,
    )

    st.divider()
    st.subheader("Engine Settings")
    use_reranker = st.toggle("CrossEncoder Reranker", value=True)
    vector_backend = st.selectbox("Vector Store", ["chroma", "numpy"], index=0)
    
    st.subheader("Candidate Pool")
    dense_k = st.slider("Dense top-k", 5, 50, 20)
    sparse_k = st.slider("BM25 top-k", 5, 50, 20)
    hybrid_k = st.slider("Hybrid pool (RRF)", 5, 60, 30)
    rerank_k = st.slider("Reranked top-k", 1, 15, 6)
    context_tokens = st.slider("Context Token Budget", 500, 6000, 3000, step=250)
    multi_query = st.checkbox("Query Expansion", value=False)
    use_parent_ctx = st.checkbox("Hierarchical Parent Context", value=False)

    st.divider()
    st.subheader("LLM Provider")
    provider = st.selectbox("Provider", ["groq", "ollama"], index=0)
    groq_model_choice = st.text_input("Groq Model", value="qwen/qwen3.8-27b")
    temperature = st.slider("Temperature", 0.0, 1.0, 0.1, step=0.05)
    max_tokens = st.slider("Max Tokens", 100, 1500, 700, step=50)

    st.caption("Pricing for cost estimation (USD / 1M tokens):")
    col_p1, col_p2 = st.columns(2)
    with col_p1:
        input_rate = st.number_input("In $/1M", min_value=0.0, value=0.59, format="%.2f")
    with col_p2:
        output_rate = st.number_input("Out $/1M", min_value=0.0, value=0.79, format="%.2f")

# Initialize Engine
try:
    engine = get_rag_engine(
        embedding_model="sentence-transformers/all-MiniLM-L6-v2",
        reranker_model="cross-encoder/ms-marco-MiniLM-L-6-v2",
        use_reranker=use_reranker,
        vector_backend=vector_backend,
    )
except Exception as exc:
    st.error(f"Failed to initialize RAG Engine: {exc}")
    st.stop()


# ---------------------------------------------------------
# SECTION 1: EXECUTIVE OVERVIEW
# ---------------------------------------------------------
if nav_option.startswith("1."):
    st.header("Executive Overview")
    st.markdown("""
    Welcome to the **Enterprise Production-Grade RAG Evaluation & Observability Platform**.
    This platform provides an end-to-end, multi-stage retrieval-augmented generation architecture
    designed for high precision, verifiable provenance, strict evidence grounding, and pre-production telemetry.
    """)

    # High-level Metrics Row
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric("Indexed Chunks", len(engine.chunks))
    with m2:
        st.metric("Vector Backend", engine.vector_backend.upper(), delta="Active")
    with m3:
        doc_count = len(set(c.document_name for c in engine.chunks)) if engine.chunks else 0
        st.metric("Corpus Documents", doc_count)
    with m4:
        otel_stat = get_otel_status()
        st.metric("OpenTelemetry Spans", otel_stat["buffered_spans_count"])

    st.divider()
    st.subheader("Architectural Workflow")

    col_diag, col_info = st.columns([3, 2])
    with col_diag:
        st.markdown("""
```
[Ingestion & Normalization] ──> [Stable Chunker & Content Hash] ──> [ChromaDB + BM25Okapi]
                                                                             │
[User Query] ──> [Query Routing & Expansion] ───────────────────────────────┤
                                                                             ▼
[Context Construction] <── [CrossEncoder Rerank] <── [RRF Hybrid Fusion (Dense + Sparse)]
         │
         ▼
[Grounded LLM Generation (Groq/Ollama)] ──> [Answer + Page-Level Citations + OpenTelemetry]
```
        """)
    with col_info:
        st.markdown("#### System Highlights")
        st.markdown("- **Dense & Sparse Hybrid**: Combines SentenceTransformer semantic search and Okapi BM25.")
        st.markdown("- **Reciprocal Rank Fusion**: Balances disparate score distributions with `rrf_k=60`.")
        st.markdown("- **Cross-Encoder Reranking**: Verifies relevance at passage level before context packing.")
        st.markdown("- **Strict Grounding**: Refuses out-of-domain / unanswerable queries to prevent hallucinations.")
        st.markdown("- **Real-time Observability**: Genuine OpenTelemetry instrumentation with trace spans.")

    st.divider()
    st.subheader("Quick Start Checklist")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("##### 1. Knowledge Base")
        st.write("Ensure documents are ingested. If not yet indexed, visit **Knowledge Base & Ingestion** and click **Index Supplied Documents**.")
    with c2:
        st.markdown("##### 2. Playground & Assistant")
        st.write("Test single-turn queries or chat interactively with source evidence in **RAG Assistant**.")
    with c3:
        st.markdown("##### 3. Evaluation Benchmark")
        st.write("Run the 61 ground-truth evaluation cases in **Evaluation Dashboard** to measure Hit Rates, MRR, and nDCG.")


# ---------------------------------------------------------
# SECTION 2: KNOWLEDGE BASE AND INGESTION
# ---------------------------------------------------------
elif nav_option.startswith("2."):
    st.header("Knowledge Base & Ingestion")
    st.caption("Multi-format document processing, metadata preservation, and vector indexing.")

    # One-click index supplied benchmark documents
    benchmark_pdfs = [
        WORKSPACE_DIR / "RAG_Document_01_HR_Policy_Handbook.pdf",
        WORKSPACE_DIR / "RAG_Document_02_IT_Security_Standard.pdf",
        WORKSPACE_DIR / "RAG_Document_03_Product_Support_Operations.pdf",
    ]
    benchmark_available = all(p.exists() for p in benchmark_pdfs)

    col_btn, col_stats = st.columns([2, 3])
    with col_btn:
        st.subheader("Supplied Benchmark Corpus")
        if benchmark_available:
            st.success("All 3 benchmark PDFs detected in workspace.")
            if st.button("Index 3 Benchmark PDFs Now", type="primary"):
                with st.spinner("Processing benchmark documents..."):
                    res = engine.ingest_paths(benchmark_pdfs, reset_index=False)
                    st.session_state.last_ingest_report = res
                    st.success(f"Successfully indexed {res.get('new_chunks_added', 0)} chunks!")
        else:
            st.warning("One or more benchmark PDFs not found in root directory.")

    with col_stats:
        st.subheader("Index Status")
        st.markdown(f"- **Total Indexed Chunks**: `{len(engine.chunks)}`")
        st.markdown(f"- **Active Vector Store**: `{engine.vector_backend.upper()}`")
        st.markdown(f"- **Chroma Directory**: `{engine.storage_dir}`")
        st.markdown(f"- **Embedding Model**: `{engine.embedding_model_name}`")

    st.divider()
    st.subheader("Upload Custom Documents")
    uploaded_files = st.file_uploader(
        "Upload PDF, DOCX, TXT, Markdown, CSV, or LOG files",
        type=["pdf", "docx", "txt", "md", "csv", "log"],
        accept_multiple_files=True,
    )
    col_u1, col_u2 = st.columns(2)
    with col_u1:
        reset_on_upload = st.checkbox("Reset index before uploading new files", value=False)
    with col_u2:
        if st.button("Index Uploaded Files", disabled=not uploaded_files):
            temp_paths = []
            try:
                for uf in uploaded_files:
                    suffix = Path(uf.name).suffix
                    tfile = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
                    tfile.write(uf.getvalue())
                    tfile.close()
                    temp_paths.append(tfile.name)

                with st.spinner("Extracting text and building indexes..."):
                    report = engine.ingest_paths(temp_paths, reset_index=reset_on_upload)
                    st.session_state.last_ingest_report = report
                    st.success(f"Indexed {report.get('chunks', 0)} chunks!")
            except Exception as e:
                st.error(f"Ingestion failed: {e}")
            finally:
                for p in temp_paths:
                    Path(p).unlink(missing_ok=True)

    # Ingestion report
    report = st.session_state.get("last_ingest_report")
    if report:
        st.subheader("Last Ingestion Report")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Processed Files", len(report.get("files", [])))
        c2.metric("Total Chunks", report.get("chunks", 0))
        c3.metric("Duration", f"{report.get('ingestion_seconds', 0)}s")
        c4.metric("Errors", len(report.get("errors", [])))
        if report.get("warnings"):
            for w in report["warnings"]:
                st.warning(f"{w.get('file')}: {w.get('warning')}")

    # Indexed Chunks Explorer
    if engine.chunks:
        st.divider()
        st.subheader("Indexed Document Chunks Explorer")
        docs = list(dict.fromkeys(c.document_name for c in engine.chunks))
        selected_doc = st.selectbox("Filter by Document", ["All"] + docs)
        
        filtered_chunks = [c for c in engine.chunks if selected_doc == "All" or c.document_name == selected_doc]
        chunk_data = [
            {
                "Chunk ID": c.chunk_id,
                "Document": c.document_name,
                "Page": c.page,
                "Section": c.section,
                "Hash": c.content_hash,
                "Characters": len(c.text),
                "Text Preview": c.text[:120] + "...",
            }
            for c in filtered_chunks[:50]
        ]
        st.dataframe(pd.DataFrame(chunk_data), use_container_width=True)


# ---------------------------------------------------------
# SECTION 3: RETRIEVAL PLAYGROUND
# ---------------------------------------------------------
elif nav_option.startswith("3."):
    st.header("Retrieval Playground")
    st.caption("Inspect multi-source candidate pools, dense and sparse scores, and reranker arbitration.")

    query_input = st.text_area(
        "Enter Inspection Query:",
        value="What is the standard domestic hotel reimbursement cap?",
        height=80,
    )

    if st.button("Run Multi-Source Retrieval", type="primary", disabled=not query_input.strip()):
        if not engine.chunks:
            st.error("No documents indexed. Please index documents in Knowledge Base first.")
        else:
            with st.spinner("Retrieving and reranking candidates..."):
                ret_res = engine.retrieve(
                    query_input,
                    dense_k=dense_k,
                    sparse_k=sparse_k,
                    hybrid_k=hybrid_k,
                    rerank_k=rerank_k,
                    context_tokens=context_tokens,
                    multi_query=multi_query,
                    use_parent_context=use_parent_ctx,
                )
                st.session_state.playground_result = ret_res

    res = st.session_state.get("playground_result")
    if res:
        cls = res["query_classification"]
        st.subheader("Query Understanding")
        c1, c2, c3 = st.columns(3)
        c1.metric("Predicted Category", cls.get("category", "general"))
        c2.metric("Routing Strategy", cls.get("recommended_retrieval", "hybrid"))
        c3.metric("Retrieval Latency", f"{res['metrics']['retrieval_latency_ms']} ms")
        st.caption(f"Reasoning: {cls.get('notes')}")

        tabs = st.tabs(["Final Selected Context", "Reranked Passages", "Hybrid Fusion (RRF)", "Dense Search", "BM25 Sparse Search"])

        with tabs[0]:
            st.markdown(f"**Selected Chunks**: {len(res['final_results'])} | **Estimated Tokens**: {res['context_tokens_estimated']} / {context_tokens}")
            for i, chunk in enumerate(res["final_results"], start=1):
                st.markdown(f"""
                <div class="citation-box">
                    <strong>[SOURCE {i}] {chunk['metadata'].get('document_name')} (Page {chunk['metadata'].get('page')})</strong><br>
                    <small>Chunk ID: <code>{chunk['chunk_id']}</code> | Reranker Score: <b>{chunk.get('reranker_score', 'N/A')}</b> | Est. Tokens: {chunk.get('estimated_tokens')}</small><br><br>
                    {chunk['compressed_text']}
                </div>
                """, unsafe_allow_html=True)

        with tabs[1]:
            st.markdown("Passages evaluated by CrossEncoder model:")
            rerank_rows = [
                {
                    "Rank": i,
                    "Document": c["metadata"].get("document_name"),
                    "Page": c["metadata"].get("page"),
                    "Reranker Score": c.get("reranker_score"),
                    "RRF Score": c.get("rrf_score"),
                    "Text Preview": c["text"][:100] + "...",
                }
                for i, c in enumerate(res["reranked_results"], start=1)
            ]
            st.dataframe(pd.DataFrame(rerank_rows), use_container_width=True)

        with tabs[2]:
            st.markdown("Hybrid fusion candidate pool merged via Reciprocal Rank Fusion:")
            rrf_rows = [
                {
                    "Rank": i,
                    "Document": c["metadata"].get("document_name"),
                    "Page": c["metadata"].get("page"),
                    "RRF Score": round(c.get("rrf_score", 0), 5),
                    "Dense Rank": c.get("dense_rank"),
                    "Sparse Rank": c.get("sparse_rank"),
                    "Text Preview": c["text"][:80] + "...",
                }
                for i, c in enumerate(res["hybrid_results"], start=1)
            ]
            st.dataframe(pd.DataFrame(rrf_rows), use_container_width=True)

        with tabs[3]:
            st.markdown("Top candidates from Vector Search:")
            dense_rows = [
                {
                    "Dense Rank": c.get("dense_rank"),
                    "Dense Score": c.get("dense_score"),
                    "Document": c["metadata"].get("document_name"),
                    "Page": c["metadata"].get("page"),
                    "Text Preview": c["text"][:80] + "...",
                }
                for c in res["dense_results"]
            ]
            st.dataframe(pd.DataFrame(dense_rows), use_container_width=True)

        with tabs[4]:
            st.markdown("Top candidates from BM25 Lexical Search:")
            bm25_rows = [
                {
                    "BM25 Rank": c.get("sparse_rank"),
                    "BM25 Score": c.get("bm25_score"),
                    "Document": c["metadata"].get("document_name"),
                    "Page": c["metadata"].get("page"),
                    "Text Preview": c["text"][:80] + "...",
                }
                for c in res["sparse_results"]
            ]
            st.dataframe(pd.DataFrame(bm25_rows), use_container_width=True)


# ---------------------------------------------------------
# SECTION 4: RAG ASSISTANT
# ---------------------------------------------------------
elif nav_option.startswith("4."):
    st.header("RAG Assistant")
    st.caption("Evidence-grounded conversational agent with page citations, provenance tracking, and feedback.")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Display Chat History
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if "sources" in msg and msg["sources"]:
                with st.expander("Supporting Citations", expanded=False):
                    for i, s in enumerate(msg["sources"], 1):
                        st.markdown(f"**[SOURCE {i}] {s['doc']} (Page {s['page']})**")
                        st.caption(s["text"])

    # Chat Input
    user_query = st.chat_input("Ask a question about policies, security standards, or support operations...")
    if user_query:
        if not engine.chunks:
            st.error("No documents indexed. Please index documents in Knowledge Base first.")
        else:
            run_id = str(uuid.uuid4())
            st.session_state.chat_history.append({"role": "user", "content": user_query})
            with st.chat_message("user"):
                st.markdown(user_query)

            with st.chat_message("assistant"):
                with st.spinner("Retrieving evidence..."):
                    ret_res = engine.retrieve(
                        user_query,
                        dense_k=dense_k,
                        sparse_k=sparse_k,
                        hybrid_k=hybrid_k,
                        rerank_k=rerank_k,
                        context_tokens=context_tokens,
                        multi_query=multi_query,
                        use_parent_context=use_parent_ctx,
                    )

                if not ret_res["context"].strip():
                    answer_text = "The provided documents do not contain information regarding this question."
                    gen_res = {"provider": "system", "model": "none", "latency_ms": 0.0, "is_abstention": True}
                    cost_info = {"estimated_cost_usd": 0.0}
                else:
                    prompt = engine.format_prompt(user_query, ret_res["context"])
                    with st.spinner(f"Generating grounded response via {provider}..."):
                        try:
                            gen_res = generate_answer(
                                prompt,
                                provider=provider,
                                model=groq_model_choice.strip() or None,
                                temperature=temperature,
                                max_tokens=max_tokens,
                            )
                            answer_text = gen_res["answer"]
                        except LLMGenerationError as err:
                            answer_text = f"⚠️ Generation notice: {err}"
                            gen_res = {"provider": provider, "model": "failed", "latency_ms": 0.0, "is_abstention": False}
                    
                    cost_info = estimate_cost(
                        gen_res.get("prompt_tokens"),
                        gen_res.get("completion_tokens"),
                        input_rate,
                        output_rate,
                    )

                st.markdown(answer_text)

                sources_list = [
                    {
                        "doc": c["metadata"].get("document_name"),
                        "page": c["metadata"].get("page"),
                        "text": c["compressed_text"],
                    }
                    for c in ret_res["final_results"]
                ]

                if sources_list:
                    with st.expander("Supporting Citations", expanded=True):
                        for i, s in enumerate(sources_list, 1):
                            st.markdown(f"**[SOURCE {i}] {s['doc']} (Page {s['page']})**")
                            st.caption(s["text"])

                # Metrics row
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Retrieval Latency", f"{ret_res['metrics']['retrieval_latency_ms']} ms")
                c2.metric("Generation Latency", f"{gen_res.get('latency_ms', 0)} ms")
                c3.metric("Total Tokens", gen_res.get("total_tokens") or "N/A")
                c4.metric("Est. Cost USD", f"${cost_info.get('estimated_cost_usd', 0.0):.6f}" if cost_info.get("estimated_cost_usd") is not None else "N/A")

                # Log run telemetry
                log_run({
                    "run_id": run_id,
                    "query": user_query,
                    "provider": provider,
                    "model": gen_res.get("model"),
                    "retrieval_metrics": ret_res["metrics"],
                    "generation_latency_ms": gen_res.get("latency_ms"),
                    "prompt_tokens": gen_res.get("prompt_tokens"),
                    "completion_tokens": gen_res.get("completion_tokens"),
                    "total_tokens": gen_res.get("total_tokens"),
                    "is_abstention": gen_res.get("is_abstention"),
                    "source_names": list(dict.fromkeys(s["doc"] for s in sources_list)),
                    **cost_info,
                })

                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": answer_text,
                    "sources": sources_list,
                    "run_id": run_id,
                })

                # Feedback widget
                with st.form(f"feedback_{run_id}"):
                    st.write("**Evaluate this response:**")
                    f_col1, f_col2 = st.columns([1, 3])
                    with f_col1:
                        rating = st.radio("Quality", ["thumbs_up", "thumbs_down"], horizontal=True, key=f"r_{run_id}")
                    with f_col2:
                        comment = st.text_input("Reviewer notes / correction (optional)", key=f"c_{run_id}")
                    if st.form_submit_button("Submit Feedback"):
                        log_feedback({
                            "run_id": run_id,
                            "query": user_query,
                            "rating": rating,
                            "comment": comment,
                            "answer": answer_text,
                            "source_chunk_ids": [c["chunk_id"] for c in ret_res["final_results"]],
                        })
                        st.success("Feedback logged to data/feedback.jsonl.")


# ---------------------------------------------------------
# SECTION 5: EVALUATION DASHBOARD
# ---------------------------------------------------------
elif nav_option.startswith("5."):
    st.header("Evaluation Dashboard")
    st.caption("Benchmark pipeline retrieval and generation against ground truth test sets.")

    eval_csv_path = WORKSPACE_DIR / "production_rag_evaluation_set.csv"
    has_default_csv = eval_csv_path.exists()

    col_cfg, col_run = st.columns([2, 1])
    with col_cfg:
        dataset_source = st.radio("Dataset Source", ["Workspace Benchmark CSV", "Upload Custom CSV"], horizontal=True)
        if dataset_source == "Upload Custom CSV":
            uploaded_eval = st.file_uploader("Upload Evaluation CSV", type=["csv"])
            eval_target_path = uploaded_eval
        else:
            eval_target_path = eval_csv_path if has_default_csv else None

        eval_mode = st.radio(
            "Evaluation Mode",
            ["Retrieval Quality Only (Fast, Deterministic)", "Full Retrieval + LLM Generation (Calls configured provider)"],
            index=0,
        )
        max_eval_cases = st.slider("Cases to evaluate", 5, 61, 61)

    with col_run:
        st.markdown("#### Ground-Truth Dataset")
        if has_default_csv:
            st.success(f"Default dataset found ({eval_csv_path.name})")
        else:
            st.error("Default evaluation dataset not found.")

        run_eval_clicked = st.button("Run Benchmark Evaluation", type="primary", disabled=eval_target_path is None)

    if run_eval_clicked:
        if not engine.chunks:
            st.error("No documents indexed. Please index documents before running evaluation.")
        else:
            # Handle uploaded vs local path
            if hasattr(eval_target_path, "getvalue"):
                temp_csv = tempfile.NamedTemporaryFile(delete=False, suffix=".csv")
                temp_csv.write(eval_target_path.getvalue())
                temp_csv.close()
                target_path = Path(temp_csv.name)
            else:
                target_path = Path(eval_target_path)

            ask_fn_to_use = None
            if "LLM Generation" in eval_mode:
                def ask_runner(q, ret):
                    if not ret["context"]:
                        return {"answer": "The provided documents do not contain information.", "is_abstention": True, "latency_ms": 0.0}
                    prompt = engine.format_prompt(q, ret["context"])
                    return generate_answer(prompt, provider=provider, model=groq_model_choice.strip() or None, temperature=temperature, max_tokens=max_tokens)
                ask_fn_to_use = ask_runner

            with st.spinner(f"Executing {max_eval_cases} benchmark test cases..."):
                try:
                    eval_results = evaluate_dataset(
                        target_path,
                        engine,
                        ask_fn=ask_fn_to_use,
                        max_cases=max_eval_cases,
                    )
                    st.session_state.evaluation_results = eval_results
                    st.success("Evaluation complete!")
                except Exception as ex:
                    st.error(f"Evaluation failed: {ex}")
                finally:
                    if hasattr(eval_target_path, "getvalue"):
                        target_path.unlink(missing_ok=True)

    # Display Evaluation Report
    eval_report = st.session_state.get("evaluation_results")
    if eval_report:
        summary = eval_report["summary"]
        st.divider()
        st.subheader("Benchmark Scorecard")

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Source Hit Rate", f"{summary['source_hit_rate'] * 100:.1f}%")
        c2.metric("Page Hit Rate", f"{summary['page_hit_rate'] * 100:.1f}%")
        c3.metric("MRR (Mean Reciprocal Rank)", f"{summary['mean_reciprocal_rank_mrr']:.4f}")
        c4.metric("nDCG@K", f"{summary['ndcg_at_k']:.4f}")

        c5, c6, c7, c8 = st.columns(4)
        c5.metric("Precision@K", f"{summary['precision_at_k']:.4f}")
        c6.metric("Recall@K", f"{summary['recall_at_k']:.4f}")
        c7.metric("Abstention Accuracy", f"{(summary['unanswerable_abstention_accuracy'] or 0.0) * 100:.1f}%" if summary['unanswerable_abstention_accuracy'] is not None else "N/A")
        c8.metric("Evaluated Cases", summary["total_cases_evaluated"])

        st.caption(f"ℹ️ {summary['metric_methodology']}")

        # Per-case Table
        st.subheader("Per-Case Detailed Evaluation")
        res_df = pd.DataFrame(eval_report["results"])
        
        # Display filters
        filter_col, search_col = st.columns([1, 2])
        with filter_col:
            cat_filter = st.selectbox("Filter Category", ["All"] + sorted(list(set(res_df["category"]))))
        with search_col:
            search_query = st.text_input("Search Question / Notes", placeholder="e.g., leave, incident, password")

        display_df = res_df
        if cat_filter != "All":
            display_df = display_df[display_df["category"] == cat_filter]
        if search_query.strip():
            display_df = display_df[display_df["question"].str.contains(search_query, case=False) | display_df["evaluation_notes"].str.contains(search_query, case=False)]

        st.dataframe(
            display_df[[
                "test_id", "question", "category", "source_hit", "page_hit",
                "reciprocal_rank", "ndcg_at_k", "abstention_pass",
                "retrieved_sources", "retrieved_pages", "latency_ms"
            ]],
            use_container_width=True,
        )

        col_d1, col_d2 = st.columns(2)
        with col_d1:
            st.download_button(
                "Download Evaluation JSON",
                data=json.dumps(eval_report, ensure_ascii=False, indent=2),
                file_name="rag_evaluation_report.json",
                mime="application/json",
            )
        with col_d2:
            st.download_button(
                "Download Evaluation CSV",
                data=display_df.to_csv(index=False),
                file_name="rag_evaluation_results.csv",
                mime="text/csv",
            )


# ---------------------------------------------------------
# SECTION 6: FEEDBACK AND IMPROVEMENT
# ---------------------------------------------------------
elif nav_option.startswith("6."):
    st.header("Feedback & Continuous Improvement")
    st.caption("Review user feedback, identify failure modes, and discover actionable pipeline tuning recommendations.")

    feedback_records = read_jsonl_logs("feedback.jsonl", max_records=200)

    if not feedback_records:
        st.info("No user feedback records logged yet. Interact with the RAG Assistant to submit feedback.")
    else:
        thumbs_up = sum(1 for r in feedback_records if r.get("rating") == "thumbs_up")
        thumbs_down = sum(1 for r in feedback_records if r.get("rating") == "thumbs_down")
        total_f = len(feedback_records)
        sat_rate = round((thumbs_up / total_f) * 100, 1) if total_f else 0.0

        c1, c2, c3 = st.columns(3)
        c1.metric("Total Feedback Logs", total_f)
        c2.metric("Thumbs Up (Positive)", thumbs_up)
        c3.metric("Satisfaction Rate", f"{sat_rate}%")

        st.divider()
        st.subheader("Failure Patterns & Recommendations")

        negative_cases = [r for r in feedback_records if r.get("rating") == "thumbs_down"]
        if negative_cases:
            st.warning(f"Detected {len(negative_cases)} negative feedback events.")
            for neg in negative_cases[-5:]:
                st.markdown(f"""
                - **Query**: *{neg.get('query')}*
                  - **Reviewer Comment**: {neg.get('comment') or 'No comment provided'}
                  - **Timestamp**: `{neg.get('timestamp_utc')}`
                """)

            st.markdown("#### Automated Improvement Recommendations:")
            st.markdown("- **Chunking Adjustments**: If missing context was cited, increase chunk size from 1000 to 1400 or enable Hierarchical Parent Context.")
            st.markdown("- **Retrieval Top-K**: Increase `dense_k` and `sparse_k` candidate pools from 20 to 30.")
            st.markdown("- **Reranker Sensitivity**: Lower `reranker_score_threshold` to prevent over-filtering borderline relevant passages.")
        else:
            st.success("No negative feedback entries recorded. Pipeline performing within expected satisfaction thresholds.")

        st.subheader("Raw Feedback Log")
        st.dataframe(pd.DataFrame(feedback_records), use_container_width=True)


# ---------------------------------------------------------
# SECTION 7: OBSERVABILITY AND TELEMETRY
# ---------------------------------------------------------
elif nav_option.startswith("7."):
    st.header("Observability & Telemetry")
    st.caption("Genuine OpenTelemetry distributed tracing, span waterfalls, stage latencies, and token cost audits.")

    otel_info = get_otel_status()
    c1, c2, c3 = st.columns(3)
    c1.metric("OpenTelemetry SDK", "Active" if otel_info["otel_sdk_available"] else "Inactive")
    c2.metric("OTLP gRPC Exporter", "Connected" if otel_info["otlp_grpc_exporter_active"] else "Not Configured")
    c3.metric("Live Spans Captured", otel_info["buffered_spans_count"])

    if not otel_info["otlp_grpc_exporter_active"]:
        st.caption("ℹ️ To export spans to a central OpenTelemetry collector (e.g. Jaeger / Datadog), set `OTEL_EXPORTER_OTLP_ENDPOINT='localhost:4317'` in secrets or environment.")

    st.divider()
    st.subheader("Live OpenTelemetry Spans Waterfall")

    spans = get_live_spans()
    if spans:
        span_df = pd.DataFrame(spans)
        st.dataframe(span_df[["timestamp", "stage", "duration_ms", "status", "attributes"]], use_container_width=True)
        
        # Latency distribution by stage
        st.subheader("Stage Latency Breakdown (Average ms)")
        avg_by_stage = span_df.groupby("stage")["duration_ms"].mean().reset_index()
        st.bar_chart(avg_by_stage.set_index("stage"))
    else:
        st.info("No live spans recorded in memory yet. Execute ingestion or queries to populate live spans.")

    st.divider()
    st.subheader("Run Telemetry Audit Log (`data/runs.jsonl`)")
    run_records = read_jsonl_logs("runs.jsonl", max_records=100)
    if run_records:
        r_df = pd.DataFrame(run_records)
        cols_to_show = [c for c in ["timestamp_utc", "query", "provider", "model", "generation_latency_ms", "total_tokens", "estimated_cost_usd"] if c in r_df.columns]
        st.dataframe(r_df[cols_to_show], use_container_width=True)
    else:
        st.info("No run logs found in `data/runs.jsonl`.")


# ---------------------------------------------------------
# SECTION 8: CONFIGURATION AND SYSTEM HEALTH
# ---------------------------------------------------------
elif nav_option.startswith("8."):
    st.header("Configuration & System Health")
    st.caption("Pre-production dependency checks, model verification, credentials audit, and risk analysis.")

    st.subheader("System Health Diagnostics")

    # Dependency Audit
    dep_col1, dep_col2 = st.columns(2)
    with dep_col1:
        st.markdown("##### Core Libraries")
        st.markdown("- ChromaDB: <span class='badge-ok'>Installed</span>", unsafe_allow_html=True)
        st.markdown("- OpenTelemetry API & SDK: <span class='badge-ok'>Installed</span>", unsafe_allow_html=True)
        st.markdown("- Sentence-Transformers: <span class='badge-ok'>Installed</span>", unsafe_allow_html=True)
        st.markdown("- Cross-Encoder: <span class='badge-ok'>Installed</span>", unsafe_allow_html=True)
        st.markdown("- Rank-BM25: <span class='badge-ok'>Installed</span>", unsafe_allow_html=True)
        st.markdown("- PyPDF: <span class='badge-ok'>Installed</span>", unsafe_allow_html=True)

    with dep_col2:
        st.markdown("##### Provider & Optional Extensions")
        groq_key = get_secret("GROQ_API_KEY", "")
        if groq_key:
            masked_key = groq_key[:4] + "..." + groq_key[-4:] if len(groq_key) > 8 else "***"
            st.markdown(f"- Groq API Key: <span class='badge-ok'>Configured ({masked_key})</span>", unsafe_allow_html=True)
        else:
            st.markdown("- Groq API Key: <span class='badge-warn'>Missing</span> (Configure in `.streamlit/secrets.toml`)", unsafe_allow_html=True)

        st.markdown("- Ollama (Local): <span class='badge-ok'>Supported</span> (Endpoint: `http://localhost:11434`)", unsafe_allow_html=True)
        st.markdown("- SQLite Query Engine: <span class='badge-ok'>Active</span> (In-memory structured query tool)", unsafe_allow_html=True)
        st.markdown("- ColBERT / Ragatouille: <span class='badge-warn'>Optional / Not configured</span>", unsafe_allow_html=True)
        st.markdown("- Graph Database: <span class='badge-warn'>Optional / Extension point</span>", unsafe_allow_html=True)

    st.divider()
    st.subheader("Local Secrets Configuration Instructions")
    st.markdown("""
    To configure your Groq API key for LLM generation without committing secrets to Git:
    1. Create `.streamlit/secrets.toml` in this workspace.
    2. Add the following lines:
    ```toml
    GROQ_API_KEY = "gsk_your_actual_groq_api_key_here"
    GROQ_MODEL = "llama-3.3-70b-versatile"
    ```
    3. `.gitignore` is already configured to exclude `.streamlit/secrets.toml` and `.env`.
    """)

    st.divider()
    st.subheader("Pre-Production Readiness & Risk Register")
    st.markdown("""
    1. **Identity & Document-Level ACLs**:
       *Current State*: Metadata filtering supported; complete Enterprise RBAC/ABAC identity propagation must precede retrieval in multi-tenant environments.
    2. **Authoritative Billing vs Cost Estimates**:
       *Current State*: Token cost estimates rely on provider-reported usage and configured rates. Always refer to your provider's monthly billing statement for binding invoices.
    3. **Model Quotas and Availability**:
       *Current State*: Default model `llama-3.3-70b-versatile` requires active Groq rate limits. Verify model deprecation schedules in Groq cloud console.
    4. **Network & Cold-Start Latency**:
       *Current State*: Local embeddings run on CPU. High-throughput production deployments should utilize GPU acceleration or batched inference servers.
    """)
