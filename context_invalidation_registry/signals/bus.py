"""
InvalidationBus — in-process event bus for invalidation signals.

In production this could be extended to a real pub/sub (Kafka, Redis, etc.).
"""
from __future__ import annotations

from typing import Callable, List

from context_invalidation_registry.models import InvalidationSignal


class InvalidationBus:
    """Simple in-process pub/sub for InvalidationSignal."""

    def __init__(self):
        self._subscribers: List[Callable[[InvalidationSignal], None]] = []

    def subscribe(self, handler: Callable[[InvalidationSignal], None]) -> None:
        self._subscribers.append(handler)

    def publish(self, signal: InvalidationSignal) -> None:
        for handler in self._subscribers:
            handler(signal)
