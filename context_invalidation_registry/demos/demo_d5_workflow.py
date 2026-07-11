"""
Demo D5 — LlamaIndex Workflows integration.

Query flows through the workflow with explicit StaleContextEvent between steps.
"""
from __future__ import annotations

from typing import Any, Dict, List

from context_invalidation_registry.config import AppConfig
from context_invalidation_registry.llm.embedding import EmbeddingProvider, mock_embedding
from context_invalidation_registry.models import Case
from context_invalidation_registry.registry.event_store import JsonEventStore
from context_invalidation_registry.workflows.context_assembly_workflow import ContextAssemblyWorkflow
from context_invalidation_registry.workflows.events import StaleContextEvent


def _load_config() -> AppConfig:
    return AppConfig.load()


def demo_d5_workflow() -> Dict[str, Any]:
    config = _load_config()
    store = JsonEventStore(config.seed_events_path)
    embedding = EmbeddingProvider(provider=config.embedding.provider, dimensions=config.embedding.dimensions)

    cases = [
        Case(
            case_id="WKF-001",
            industry="food_local_services",
            region="CN",
            data_collected_at="2024-06-15",
            text="2024 Chinese food marketing case relying on national standards compliance as trust signal.",
            metadata={"embedding": mock_embedding("2024 Chinese food marketing case relying on national standards compliance")},
        ),
        Case(
            case_id="WKF-002",
            industry="food_local_services",
            region="CN",
            data_collected_at="2026-01-10",
            text="2026 food marketing case with modern transparency practices.",
            metadata={"embedding": mock_embedding("2026 food marketing case with modern transparency practices")},
        ),
    ]

    workflow = ContextAssemblyWorkflow(
        cases=cases,
        event_store=store,
        embedding_provider=embedding,
        reranker=None,
        input_guard=None,
        output_guard=None,
    )
    result = workflow.run("food marketing compliance strategy", region="CN")

    stale_event_ids = [se.event.event_id for se in result.stale_events]
    trace_steps = [t.get("step") for t in result.trace]

    return {
        "query": result.query,
        "route": result.route.path,
        "output_preview": result.output[:300],
        "stale_event_ids": stale_event_ids,
        "trace_steps": trace_steps,
        "passed": result.route.path == "stale_context" and len(result.stale_events) > 0,
    }
