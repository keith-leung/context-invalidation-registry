"""Signals package — D→A invalidation contract."""
from __future__ import annotations

from .bus import InvalidationBus
from .emitter import emit_invalidation

__all__ = ["InvalidationBus", "emit_invalidation"]
