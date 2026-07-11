"""
Demo D2 — D→A invalidation signal emission + demo subscriber.

Registers a new event -> emit_invalidation produces InvalidationSignal ->
demo subscriber receives it -> signal predicate matches A's CheckpointEntry shape.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List

from context_invalidation_registry.config import AppConfig
from context_invalidation_registry.models import (
    Case,
    CriticalEvent,
    InvalidationSignal,
    StalenessReport,
)
from context_invalidation_registry.registry.event_store import JsonEventStore
from context_invalidation_registry.signals.bus import InvalidationBus
from context_invalidation_registry.signals.emitter import emit_invalidation


def _load_config() -> AppConfig:
    return AppConfig.load()


def demo_d2_signal() -> Dict[str, Any]:
    config = _load_config()
    store = JsonEventStore(config.seed_events_path)
    bus = InvalidationBus()

    received: List[InvalidationSignal] = []
    bus.subscribe(received.append)

    event = CriticalEvent(
        event_id="CE099",
        event_name="Demo Regulatory Shift",
        event_date="2025-08-01",
        affected_industries=["food_local_services", "fmcg_retail"],
        affected_regions=["CN"],
        impact_description="A demo regulatory shift invalidates pre-August 2025 food marketing strategies.",
        invalidated_strategies=["Legacy compliance messaging"],
        new_required_strategies=["Proactive transparency disclosures"],
    )
    store.register_event(event)

    signal = emit_invalidation(event, bus)

    # Contract shape check: predicate keys match A's CheckpointEntry fields (spec A §7.2)
    expected_keys = {"industries", "regions", "vintage_before"}
    predicate_keys = set(signal.predicate.keys())
    contract_valid = expected_keys == predicate_keys

    return {
        "signal": signal.to_dict(),
        "received_count": len(received),
        "contract_shape_valid": contract_valid,
        "passed": len(received) == 1 and contract_valid,
    }
