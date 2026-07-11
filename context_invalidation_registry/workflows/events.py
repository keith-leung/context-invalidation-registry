"""
Workflow events — explicit event/step shape for LlamaIndex Workflows.

The StaleContextEvent flows between steps, making the event-driven shape visible.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from context_invalidation_registry.models import (
    Case,
    GuardrailVerdict,
    RouteDecision,
    StalenessReport,
)


@dataclass
class StaleContextEvent:
    """Emitted when a retrieved case is flagged stale by the registry."""
    event: StalenessReport
    case: Case
    emitted_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


@dataclass
class WorkflowResult:
    """Final result of the context assembly workflow."""
    query: str
    route: RouteDecision
    output: str
    input_verdict: GuardrailVerdict
    output_verdict: GuardrailVerdict
    stale_events: List[StaleContextEvent] = field(default_factory=list)
    trace: List[Dict[str, Any]] = field(default_factory=list)
