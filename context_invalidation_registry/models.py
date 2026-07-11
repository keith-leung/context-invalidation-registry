"""
Data models for the Critical Events Registry.

All types are frozen/shape-locked per SPEC §7.2.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
from typing import List, Optional


@dataclass(frozen=True)
class CriticalEvent:
    """8-field schema — LOCKED per SPEC §3 D1."""
    event_id: str
    event_name: str
    event_date: str  # ISO-8601 date string
    affected_industries: List[str]
    affected_regions: List[str]
    impact_description: str
    invalidated_strategies: List[str]
    new_required_strategies: List[str]

    def parsed_date(self) -> Optional[date]:
        try:
            return date.fromisoformat(self.event_date)
        except (ValueError, TypeError):
            return None


@dataclass(frozen=True)
class Case:
    """A retrieved context item checked for staleness."""
    case_id: str
    industry: str
    region: str
    data_collected_at: Optional[str] = None
    created_at: Optional[str] = None
    text: Optional[str] = None
    metadata: dict = field(default_factory=dict)


@dataclass(frozen=True)
class StalenessReport:
    """Result of staleness check."""
    is_stale: bool
    reason: Optional[str] = None
    event_id: Optional[str] = None
    event_date: Optional[str] = None
    invalidated_strategies: Optional[List[str]] = None


@dataclass(frozen=True)
class InvalidationSignal:
    """D→A contract payload — FROZEN per SPEC §7.2."""
    source_event_id: str
    predicate: dict  # {industries, regions, vintage_before}
    reason: str
    emitted_at: str

    def to_dict(self) -> dict:
        return {
            "source_event_id": self.source_event_id,
            "predicate": self.predicate,
            "reason": self.reason,
            "emitted_at": self.emitted_at,
        }


@dataclass(frozen=True)
class RouteDecision:
    """Result of three-tier routing."""
    path: str  # fast | incremental | full | stale_context
    best_similarity: float
    matched_case_ids: List[str]
    matched_cases: List[dict]
    cache_hit: bool = False
    cached_result: Optional[dict] = None
    estimated_token_budget: int = 0
    stale_flags: List[StalenessReport] = field(default_factory=list)


@dataclass(frozen=True)
class GuardrailVerdict:
    """Result of input/output guardrail check."""
    passed: bool
    severity: str  # block | warn | pass
    violations: List[dict] = field(default_factory=list)
    sanitized_input: Optional[str] = None
    needs_disclaimer: bool = False


@dataclass(frozen=True)
class RerankResult:
    """Result of reranking step."""
    query: str
    documents: List[dict]
    scores: List[float]
    reranked_order: List[int]
