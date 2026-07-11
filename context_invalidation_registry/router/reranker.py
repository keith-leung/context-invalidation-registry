"""
Reranker step — Cohere Rerank or BGE Reranker.

Retrieve -> rerank -> assemble is the standard 2026 RAG pipeline (SPEC §3 D3).
"""
from __future__ import annotations

from typing import List, Optional

from context_invalidation_registry.models import RerankResult


class Reranker:
    """Unified reranker interface."""

    def __init__(self, provider: str, model: Optional[str] = None,
                 api_key: Optional[str] = None, base_url: Optional[str] = None):
        self.provider = provider
        self.model = model
        self.api_key = api_key
        self.base_url = base_url
        self._client = None

    def rerank(self, query: str, documents: List[dict], top_n: int = 5) -> RerankResult:
        if self.provider == "mock":
            return self._mock_rerank(query, documents, top_n)
        if self.provider == "cohere":
            return self._cohere_rerank(query, documents, top_n)
        if self.provider == "bge":
            return self._bge_rerank(query, documents, top_n)
        raise ValueError(f"Unsupported reranker: {self.provider}")

    def _mock_rerank(self, query: str, documents: List[dict], top_n: int) -> RerankResult:
        # Deterministic mock: preserve original order, assign descending scores
        scores = [1.0 - (0.1 * i) for i in range(len(documents))]
        order = list(range(min(top_n, len(documents))))
        return RerankResult(
            query=query,
            documents=documents[:top_n],
            scores=scores[:top_n],
            reranked_order=order,
        )

    def _cohere_rerank(self, query: str, documents: List[dict], top_n: int) -> RerankResult:
        if self._client is None:
            import cohere
            self._client = cohere.Client(api_key=self.api_key)
        texts = [d.get("text", "") or d.get("content", "") for d in documents]
        resp = self._client.rerank(
            model=self.model or "rerank-english-v3.0",
            query=query,
            documents=texts,
            top_n=top_n,
        )
        order = [r.index for r in resp.results]
        scores = [r.relevance_score for r in resp.results]
        return RerankResult(
            query=query,
            documents=[documents[i] for i in order],
            scores=scores,
            reranked_order=order,
        )

    def _bge_rerank(self, query: str, documents: List[dict], top_n: int) -> RerankResult:
        # BGE Reranker via a local API (e.g., FlagEmbedding rerank server)
        if self._client is None:
            from openai import OpenAI
            self._client = OpenAI(base_url=self.base_url or "http://localhost:8000/v1", api_key=self.api_key or "none")
        texts = [d.get("text", "") or d.get("content", "") for d in documents]
        resp = self._client.embeddings.create(
            model=self.model or "BAAI/bge-reranker-v2-m3",
            input=[query + "\n" + t for t in texts],
        )
        scores = [d.embedding[0] if d.embedding else 0.0 for d in resp.data]
        paired = sorted(zip(scores, range(len(documents))), key=lambda x: x[0], reverse=True)
        order = [i for _, i in paired[:top_n]]
        top_scores = [scores[i] for i in order]
        return RerankResult(
            query=query,
            documents=[documents[i] for i in order],
            scores=top_scores,
            reranked_order=order,
        )


def rerank_documents(query: str, documents: List[dict], *, reranker: Optional[Reranker] = None, top_n: int = 5) -> RerankResult:
    if reranker is None:
        reranker = Reranker(provider="mock")
    return reranker.rerank(query, documents, top_n=top_n)
