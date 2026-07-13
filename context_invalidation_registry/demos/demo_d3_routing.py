"""
Demo D3 — staleness-aware three-tier routing + reranking.

A high-similarity stale case is downgraded to stale_context (not served Fast).
A fresh high-similarity case hits Fast normally.
"""
from __future__ import annotations

from typing import Any, Dict, List

from context_invalidation_registry.config import AppConfig
from context_invalidation_registry.llm.embedding import EmbeddingProvider, mock_embedding
from context_invalidation_registry.models import Case
from context_invalidation_registry.registry.event_store import JsonEventStore
from context_invalidation_registry.registry.staleness import check_case_staleness
from context_invalidation_registry.router.classifier import classify_intent, RoutePath
from context_invalidation_registry.router.reranker import Reranker


def _load_config() -> AppConfig:
    return AppConfig.load()


def demo_d3_routing() -> Dict[str, Any]:
    config = _load_config()
    store = JsonEventStore(config.seed_events_path)
    embedding = EmbeddingProvider(provider=config.embedding.provider, dimensions=config.embedding.dimensions)

    # Build two synthetic cases: one pre-event stale, one fresh.
    # We embed the same text for both so similarity is artificially high.
    query_text = "food marketing compliance strategy"
    query_emb = mock_embedding(query_text)

    stale_case = Case(
        case_id="STALE-001",
        industry="food_local_services",
        region="CN",
        data_collected_at="2024-06-15",
        text=query_text,
        metadata={"embedding": query_emb},
    )
    fresh_case = Case(
        case_id="FRESH-001",
        industry="food_local_services",
        region="CN",
        data_collected_at="2026-01-15",
        text=query_text,
        metadata={"embedding": query_emb},
    )
    cases = [stale_case, fresh_case]

    # Step 1: classify with real EventStore (stale-aware downgrade happens inside)
    route = classify_intent(
        query_text=query_text,
        query_embedding=query_emb,
        cases=cases,
        embedding_provider=embedding,
        event_store=store,
        region="CN",
    )

    # Step 2: reranker demo (mock)
    reranker = Reranker(provider="mock")
    rerank_result = reranker.rerank(query_text, route.matched_cases)

    return {
        "route_path": route.path,
        "best_similarity": round(route.best_similarity, 4),
        "matched_case_ids": route.matched_case_ids,
        "stale_flags_count": len(route.stale_flags),
        "reranked_order": rerank_result.reranked_order,
        "reranked_scores": [round(s, 4) for s in rerank_result.scores],
        "passed": route.path == RoutePath.STALE_CONTEXT,
    }
