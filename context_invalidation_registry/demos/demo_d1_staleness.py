"""
Demo D1 — Critical Events Registry staleness logic.

Tests the three branches:
1. matching+pre-event -> stale
2. non-matching -> fresh
3. matching+no-vintage+recent-event -> stale (conservative)
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

from context_invalidation_registry.config import AppConfig
from context_invalidation_registry.models import Case
from context_invalidation_registry.registry.event_store import JsonEventStore
from context_invalidation_registry.registry.staleness import check_case_staleness


def _load_config() -> AppConfig:
    return AppConfig.load()


def demo_d1_staleness() -> Dict[str, any]:
    config = _load_config()
    store = JsonEventStore(config.seed_events_path)

    results = {}

    # Branch 1: matching industry+region, pre-event vintage -> stale
    case_pre = Case(
        case_id="CASE-001",
        industry="food_local_services",
        region="CN",
        data_collected_at="2024-06-15",
        text="A 2024 food marketing case from China.",
    )
    r1 = check_case_staleness(case_pre, store)
    results["branch_1_pre_event_stale"] = {
        "case_id": case_pre.case_id,
        "is_stale": r1.is_stale,
        "reason": r1.reason,
        "event_id": r1.event_id,
    }

    # Branch 2: non-matching industry -> fresh
    case_nonmatch = Case(
        case_id="CASE-002",
        industry="fitness_equipment",
        region="CN",
        data_collected_at="2024-06-15",
        text="A fitness equipment case.",
    )
    r2 = check_case_staleness(case_nonmatch, store)
    results["branch_2_non_match"] = {
        "case_id": case_nonmatch.case_id,
        "is_stale": r2.is_stale,
    }

    # Branch 3: matching industry+region, NO vintage, recent event -> stale (conservative)
    case_novintage = Case(
        case_id="CASE-003",
        industry="food_local_services",
        region="CN",
        text="A food case with no date.",
    )
    r3 = check_case_staleness(case_novintage, store)
    results["branch_3_no_vintage_recent"] = {
        "case_id": case_novintage.case_id,
        "is_stale": r3.is_stale,
        "reason": r3.reason,
    }

    passed = (
        r1.is_stale is True
        and r2.is_stale is False
        and r3.is_stale is True
    )
    results["passed"] = passed
    return results
