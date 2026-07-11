"""
Emit invalidation signals from CriticalEvents to downstream consumers (notably repo A).

The InvalidationSignal shape is FROZEN per SPEC §7.2 / §3 D2.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import TYPE_CHECKING

from context_invalidation_registry.models import CriticalEvent, InvalidationSignal

if TYPE_CHECKING:
    from context_invalidation_registry.signals.bus import InvalidationBus


def emit_invalidation(event: CriticalEvent, bus: "InvalidationBus") -> InvalidationSignal:
    """
    Translate a CriticalEvent's footprint into the predicate shape A expects.

    A's `tombstone_items_matching` (spec A §7.2) consumes this predicate directly.
    """
    signal = InvalidationSignal(
        source_event_id=event.event_id,
        predicate={
            "industries": list(event.affected_industries),
            "regions": list(event.affected_regions),
            "vintage_before": event.event_date,
        },
        reason=f"{event.event_name}: {event.impact_description}",
        emitted_at=datetime.now(timezone.utc).isoformat(),
    )
    bus.publish(signal)
    return signal
