"""D2 — InvalidationSignal shape (D→A contract, FROZEN per SPEC §7.2).

If any of these assertions fail, A's `tombstone_items_matching` cannot
consume the signal. The contract is the portfolio's headline cross-repo
interface.
"""
from __future__ import annotations

from datetime import datetime

import pytest

from context_invalidation_registry.models import CriticalEvent
from context_invalidation_registry.signals.bus import InvalidationBus
from context_invalidation_registry.signals.emitter import emit_invalidation


@pytest.fixture
def sample_event() -> CriticalEvent:
    return CriticalEvent(
        event_id="CE-D2",
        event_name="D2 test event",
        event_date="2025-09-10",
        affected_industries=["food_local_services", "fmcg_retail"],
        affected_regions=["CN"],
        impact_description="test impact",
        invalidated_strategies=["strategy-a"],
        new_required_strategies=["strategy-b"],
    )


def test_emit_produces_signal_with_expected_predicate_keys(sample_event):
    bus = InvalidationBus()
    signal = emit_invalidation(sample_event, bus)
    # FROZEN contract keys
    assert set(signal.predicate.keys()) == {"industries", "regions", "vintage_before"}


def test_signal_predicate_industries_match_event(sample_event):
    bus = InvalidationBus()
    signal = emit_invalidation(sample_event, bus)
    assert signal.predicate["industries"] == ["food_local_services", "fmcg_retail"]


def test_signal_predicate_regions_match_event(sample_event):
    bus = InvalidationBus()
    signal = emit_invalidation(sample_event, bus)
    assert signal.predicate["regions"] == ["CN"]


def test_signal_predicate_vintage_before_matches_event_date(sample_event):
    bus = InvalidationBus()
    signal = emit_invalidation(sample_event, bus)
    assert signal.predicate["vintage_before"] == "2025-09-10"


def test_signal_source_event_id_matches(sample_event):
    bus = InvalidationBus()
    signal = emit_invalidation(sample_event, bus)
    assert signal.source_event_id == "CE-D2"


def test_signal_emitted_at_is_iso8601(sample_event):
    bus = InvalidationBus()
    signal = emit_invalidation(sample_event, bus)
    # Must parse as ISO 8601
    datetime.fromisoformat(signal.emitted_at)


def test_bus_delivers_signal_to_subscriber(sample_event):
    bus = InvalidationBus()
    received = []
    bus.subscribe(received.append)
    signal = emit_invalidation(sample_event, bus)
    assert len(received) == 1
    assert received[0] is signal


def test_bus_delivers_to_multiple_subscribers(sample_event):
    bus = InvalidationBus()
    handler_a: list = []
    handler_b: list = []
    bus.subscribe(handler_a.append)
    bus.subscribe(handler_b.append)
    emit_invalidation(sample_event, bus)
    assert len(handler_a) == 1
    assert len(handler_b) == 1


def test_signal_to_dict_round_trips_predicate(sample_event):
    bus = InvalidationBus()
    signal = emit_invalidation(sample_event, bus)
    d = signal.to_dict()
    assert d["predicate"] == signal.predicate
    assert d["source_event_id"] == signal.source_event_id
    assert d["reason"] == signal.reason
