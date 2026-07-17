"""D3 — staleness-aware three-tier routing.

The router MUST downgrade a high-similarity but stale case to
`stale_context` (not serve it from Fast). A fresh high-similarity case
MUST hit Fast.
"""
from __future__ import annotations

import pytest

from context_invalidation_registry.llm.embedding import EmbeddingProvider, mock_embedding
from context_invalidation_registry.models import Case, CriticalEvent
from context_invalidation_registry.registry.event_store import JsonEventStore
from context_invalidation_registry.router.classifier import classify_intent, RoutePath


@pytest.fixture
def store_with_food_event(tmp_path):
    event = CriticalEvent(
        event_id="CE-D3",
        event_name="D3 food event",
        event_date="2025-09-10",
        affected_industries=["food_local_services"],
        affected_regions=["CN"],
        impact_description="test",
        invalidated_strategies=[],
        new_required_strategies=[],
    )
    store = JsonEventStore(tmp_path / "events.json")
    store.register_event(event)
    return store


def _make_case(case_id: str, industry: str, region: str, date: str, query_emb):
    return Case(
        case_id=case_id,
        industry=industry,
        region=region,
        data_collected_at=date,
        metadata={"embedding": query_emb},
    )


def test_stale_high_similarity_case_downgrades_to_stale_context(
    store_with_food_event,
):
    query = "food marketing compliance"
    query_emb = mock_embedding(query)
    embedding = EmbeddingProvider(provider="mock", dimensions=1536)

    stale_case = _make_case(
        "S1", "food_local_services", "CN", "2024-06-15", query_emb
    )
    decision = classify_intent(
        query_text=query,
        query_embedding=query_emb,
        cases=[stale_case],
        embedding_provider=embedding,
        event_store=store_with_food_event,
        region="CN",
    )
    assert decision.path == RoutePath.STALE_CONTEXT
    assert len(decision.stale_flags) == 1


def test_fresh_high_similarity_case_hits_fast(store_with_food_event):
    query = "food marketing compliance"
    query_emb = mock_embedding(query)
    embedding = EmbeddingProvider(provider="mock", dimensions=1536)

    fresh_case = _make_case(
        "F1", "food_local_services", "CN", "2026-01-15", query_emb
    )
    decision = classify_intent(
        query_text=query,
        query_embedding=query_emb,
        cases=[fresh_case],
        embedding_provider=embedding,
        event_store=store_with_food_event,
        region="CN",
    )
    assert decision.path == RoutePath.FAST
    assert decision.stale_flags == []


def test_region_mismatch_prevents_downgrade(store_with_food_event):
    query = "us food strategy"
    query_emb = mock_embedding(query)
    embedding = EmbeddingProvider(provider="mock", dimensions=1536)

    us_case = _make_case(
        "US1", "food_local_services", "US", "2024-06-15", query_emb
    )
    decision = classify_intent(
        query_text=query,
        query_embedding=query_emb,
        cases=[us_case],
        embedding_provider=embedding,
        event_store=store_with_food_event,
        region="US",
    )
    # Event affects CN only → US case is fresh → hits Fast on similarity
    assert decision.path == RoutePath.FAST


def test_empty_cases_returns_full_route(store_with_food_event):
    query_emb = mock_embedding("something novel")
    embedding = EmbeddingProvider(provider="mock", dimensions=1536)

    decision = classify_intent(
        query_text="something novel",
        query_embedding=query_emb,
        cases=[],
        embedding_provider=embedding,
        event_store=store_with_food_event,
        region="CN",
    )
    assert decision.path == RoutePath.FULL
    assert decision.best_similarity == 0.0
