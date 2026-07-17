"""D5 — LlamaIndex-shaped ContextAssemblyWorkflow.

The workflow must emit `StaleContextEvent` when a stale case is
retrieved, and route to `stale_context` in the assembly step.
"""
from __future__ import annotations

import pytest

from context_invalidation_registry.llm.embedding import EmbeddingProvider, mock_embedding
from context_invalidation_registry.models import Case, CriticalEvent
from context_invalidation_registry.registry.event_store import JsonEventStore
from context_invalidation_registry.workflows.context_assembly_workflow import (
    ContextAssemblyWorkflow,
)
from context_invalidation_registry.workflows.events import StaleContextEvent


@pytest.fixture
def workflow_with_stale_case(tmp_path):
    event = CriticalEvent(
        event_id="CE-D5",
        event_name="D5 event",
        event_date="2025-09-10",
        affected_industries=["food_local_services"],
        affected_regions=["CN"],
        impact_description="test",
        invalidated_strategies=["old"],
        new_required_strategies=["new"],
    )
    store = JsonEventStore(tmp_path / "events.json")
    store.register_event(event)

    embedding = EmbeddingProvider(provider="mock", dimensions=1536)
    query = "food strategy"
    emb = mock_embedding(query)
    cases = [
        Case(
            case_id="STALE",
            industry="food_local_services",
            region="CN",
            data_collected_at="2024-06-15",
            text=query,
            metadata={"embedding": emb},
        ),
    ]
    workflow = ContextAssemblyWorkflow(
        cases=cases,
        event_store=store,
        embedding_provider=embedding,
    )
    return workflow, query


def test_workflow_emits_stale_context_event(workflow_with_stale_case):
    workflow, query = workflow_with_stale_case
    result = workflow.run(query, region="CN")
    assert len(result.stale_events) == 1
    assert isinstance(result.stale_events[0], StaleContextEvent)


def test_workflow_routes_stale_case_to_stale_context(workflow_with_stale_case):
    workflow, query = workflow_with_stale_case
    result = workflow.run(query, region="CN")
    assert result.route.path == "stale_context"


def test_workflow_trace_captures_all_steps(workflow_with_stale_case):
    workflow, query = workflow_with_stale_case
    result = workflow.run(query, region="CN")
    steps = [t["step"] for t in result.trace]
    # Must at least include the guard, route, and output steps
    assert "input_guard" in steps
    assert "route" in steps
    assert "output_guard" in steps


def test_workflow_output_surfaces_invalidation(workflow_with_stale_case):
    workflow, query = workflow_with_stale_case
    result = workflow.run(query, region="CN")
    # Stale-context branch synthesizes a warning surfacing invalidation
    assert "STALE" in result.output.upper() or "invalid" in result.output.lower()


def test_workflow_returns_workflow_result_shape(workflow_with_stale_case):
    workflow, query = workflow_with_stale_case
    result = workflow.run(query, region="CN")
    # All fields present per SPEC
    assert result.query == query
    assert result.route is not None
    assert isinstance(result.output, str)
    assert result.input_verdict is not None
    assert result.output_verdict is not None
    assert isinstance(result.trace, list)
