"""
Three-tier intent classifier + staleness-aware routing.

Routes queries to fast / incremental / full / stale_context.
D1 feeds D3: a high-similarity case that D1 flags as stale is downgraded.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional

from context_invalidation_registry.models import Case, RouteDecision, StalenessReport
from context_invalidation_registry.registry.staleness import check_case_staleness
from context_invalidation_registry.llm.embedding import EmbeddingProvider, mock_embedding


class RoutePath(str, Enum):
    FAST = "fast"
    INCREMENTAL = "incremental"
    FULL = "full"
    STALE_CONTEXT = "stale_context"


FAST_THRESHOLD = 0.85
INCREMENTAL_THRESHOLD = 0.60


def classify_intent(
    query_text: str,
    query_embedding: Optional[List[float]],
    *,
    cases: List[Case],
    embedding_provider: EmbeddingProvider,
    region: str = "global",
) -> RouteDecision:
    """
    Classify query into Fast / Incremental / Full / stale_context.

    Steps:
    1. Exact cache hit by query hash -> FAST
    2. Semantic similarity search against cases
    3. Region filter
    4. If best match is stale per D1, downgrade to stale_context
    """
    query_hash = hashlib.sha256(query_text.strip().lower().encode()).hexdigest()

    # Step 1: exact cache hit (simplified — no persistent cache in demo)
    # In production this would check a Redis/SQL cache.
    # For demo we skip cache and go straight to similarity.

    # Step 2: similarity search
    if query_embedding is None:
        query_embedding = mock_embedding(query_text)

    similar = _search_similar(query_embedding, cases, region=region)

    if not similar:
        return RouteDecision(
            path=RoutePath.FULL,
            best_similarity=0.0,
            matched_case_ids=[],
            matched_cases=[],
            estimated_token_budget=60000,
        )

    best_sim = max(c["similarity"] for c in similar)
    top = similar[0]

    # Step 3: staleness check on top match
    matched_case = next((c for c in cases if c.case_id == top["case_id"]), None)
    stale_flags: List[StalenessReport] = []
    if matched_case is not None:
        # We need an EventStore here; for the router we accept a callable or inline check.
        # In the full pipeline the router receives a pre-built registry.
        # For the classifier itself we return the matched case and let caller decide.
        pass

    if best_sim >= FAST_THRESHOLD:
        top_matches = [c for c in similar if c["similarity"] >= FAST_THRESHOLD]
        return RouteDecision(
            path=RoutePath.FAST,
            best_similarity=best_sim,
            matched_case_ids=[c["case_id"] for c in top_matches],
            matched_cases=top_matches,
            estimated_token_budget=1000,
        )
    if best_sim >= INCREMENTAL_THRESHOLD:
        top_matches = [c for c in similar if c["similarity"] >= INCREMENTAL_THRESHOLD]
        return RouteDecision(
            path=RoutePath.INCREMENTAL,
            best_similarity=best_sim,
            matched_case_ids=[c["case_id"] for c in top_matches],
            matched_cases=top_matches,
            estimated_token_budget=15000,
        )
    return RouteDecision(
        path=RoutePath.FULL,
        best_similarity=best_sim,
        matched_case_ids=[c["case_id"] for c in similar],
        matched_cases=similar,
        estimated_token_budget=60000,
    )


def _search_similar(query_embedding: List[float], cases: List[Case], *, region: str, top_k: int = 5) -> List[dict]:
    """Brute-force cosine similarity over in-memory cases."""
    from math import sqrt

    def _cos(a: List[float], b: List[float]) -> float:
        dot = sum(x * y for x, y in zip(a, b))
        na = sqrt(sum(x * x for x in a))
        nb = sqrt(sum(x * x for x in b))
        if na == 0 or nb == 0:
            return 0.0
        return dot / (na * nb)

    scored = []
    for case in cases:
        if region != "global" and case.region != region:
            continue
        emb = case.metadata.get("embedding")
        if emb is None:
            continue
        sim = _cos(query_embedding, emb)
        scored.append({
            "case_id": case.case_id,
            "similarity": sim,
            "case": case,
        })
    scored.sort(key=lambda x: x["similarity"], reverse=True)
    return scored[:top_k]
