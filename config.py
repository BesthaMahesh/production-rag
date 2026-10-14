"""Centralized configuration for Production RAG System.
Loads defaults from environment variables and Streamlit secrets.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

WORKSPACE_DIR = Path(__file__).resolve().parent
DATA_DIR = WORKSPACE_DIR / "data"
CHROMA_DIR = DATA_DIR / "chromadb"
INDEX_DIR = DATA_DIR / "index"


def get_secret(key: str, default: Any = "") -> Any:
    """Safe lookup in streamlit.secrets then os.environ."""
    try:
        import streamlit as st
        if hasattr(st, "secrets") and key in st.secrets:
            return st.secrets[key]
    except Exception:
        pass
    return os.environ.get(key, default)


@dataclass
class RAGConfig:
    # Embedding & Reranker models
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    use_reranker: bool = True
    vector_backend: str = "chroma"  # "chroma" or "numpy"

    # Chunking
    max_chars: int = 1000
    overlap_chars: int = 150
    hierarchical_parent_chars: int = 2500

    # Retrieval parameters
    dense_k: int = 20
    sparse_k: int = 20
    hybrid_k: int = 30
    rerank_k: int = 8
    context_tokens: int = 3000
    rrf_k: int = 60
    multi_query: bool = False
    reranker_score_threshold: float = -10.0

    # Generation provider & models
    llm_provider: str = "groq"
    groq_api_key: str = ""
    groq_model: str = "qwen/qwen3.8-27b"
    ollama_url: str = "http://localhost:11434"
    ollama_model: str = "qwen2.5:3b"
    temperature: float = 0.1
    max_tokens: int = 700

    # Pricing rates (USD per 1M tokens)
    input_price_per_m: float = 0.59
    output_price_per_m: float = 0.79

    # Observability & OTel
    otel_service_name: str = "production-rag-platform"
    otel_endpoint: str = ""  # e.g., "localhost:4317" if OTLP gRPC collector is available
    enable_otel: bool = True

    @classmethod
    def from_environment(cls) -> "RAGConfig":
        return cls(
            embedding_model=os.getenv("RAG_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"),
            reranker_model=os.getenv("RAG_RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2"),
            vector_backend=os.getenv("RAG_VECTOR_BACKEND", "chroma"),
            groq_api_key=str(get_secret("GROQ_API_KEY", "")),
            groq_model=str(get_secret("GROQ_MODEL", "llama-3.3-70b-versatile")),
            otel_endpoint=str(get_secret("OTEL_EXPORTER_OTLP_ENDPOINT", os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", ""))),
            enable_otel=os.getenv("ENABLE_OTEL", "true").lower() in ("true", "1", "yes"),
        )
