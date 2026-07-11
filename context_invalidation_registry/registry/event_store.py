"""
EventStore protocol + JsonEventStore implementation.

Storage-agnostic registry: pure Python + JSON (SPEC §3 D1).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import List, Protocol

from context_invalidation_registry.models import CriticalEvent


class EventStore(Protocol):
    def load_events(self) -> List[CriticalEvent]: ...
    def register_event(self, event: CriticalEvent) -> None: ...


class JsonEventStore:
    """File-backed JSON event store — the default demo implementation."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            self._write([])

    def load_events(self) -> List[CriticalEvent]:
        with open(self.path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        return [self._deserialize(r) for r in raw]

    def register_event(self, event: CriticalEvent) -> None:
        events = self.load_events()
        events = [e for e in events if e.event_id != event.event_id]
        events.append(event)
        self._write(events)

    def _write(self, events: List[CriticalEvent]) -> None:
        raw = [self._serialize(e) for e in events]
        tmp = self.path.with_suffix(".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(raw, f, ensure_ascii=False, indent=2)
        tmp.replace(self.path)

    @staticmethod
    def _serialize(event: CriticalEvent) -> dict:
        return {
            "event_id": event.event_id,
            "event_name": event.event_name,
            "event_date": event.event_date,
            "affected_industries": event.affected_industries,
            "affected_regions": event.affected_regions,
            "impact_description": event.impact_description,
            "invalidated_strategies": event.invalidated_strategies,
            "new_required_strategies": event.new_required_strategies,
        }

    @staticmethod
    def _deserialize(raw: dict) -> CriticalEvent:
        return CriticalEvent(
            event_id=raw["event_id"],
            event_name=raw["event_name"],
            event_date=raw["event_date"],
            affected_industries=raw.get("affected_industries", []),
            affected_regions=raw.get("affected_regions", []),
            impact_description=raw.get("impact_description", ""),
            invalidated_strategies=raw.get("invalidated_strategies", []),
            new_required_strategies=raw.get("new_required_strategies", []),
        )
