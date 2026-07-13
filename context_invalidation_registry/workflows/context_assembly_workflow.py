"""
ContextAssemblyWorkflow — LlamaIndex Workflows event/step shape.

Pipeline: query -> route -> registry-check -> retrieve -> rerank -> assemble -> guardrail -> synthesize.

StaleContextEvent is emitted by the registry-check step and consumed by assembly.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from context_invalidation_registry.models import (
    Case,
    GuardrailVerdict,
    RouteDecision,
    StalenessReport,
)
from context_invalidation_registry.registry.staleness import check_case_staleness
from context_invalidation_registry.router.classifier import classify_intent, RoutePath
from context_invalidation_registry.router.reranker import rerank_documents
from context_invalidation_registry.workflows.events import StaleContextEvent, WorkflowResult


class ContextAssemblyWorkflow:
    """
    LlamaIndex Workflows-style retrieval-assembly workflow.

    In v1 we model the step/event shape manually so it runs without the full
    LlamaIndex Workflows runtime if that dependency is heavy.
    """

    def __init__(
        self,
        *,
        cases: List[Case],
        event_store,
        embedding_provider,
        reranker=None,
        input_guard=None,
        output_guard=None,
        llm_client=None,
        synthesize_fn=None,
    ):
        self.cases = cases
        self.event_store = event_store
        self.embedding_provider = embedding_provider
        self.reranker = reranker
        self.input_guard = input_guard
        self.output_guard = output_guard
        self.llm_client = llm_client
        self.synthesize_fn = synthesize_fn

    def run(self, query: str, region: str = "global") -> WorkflowResult:
        trace: List[Dict[str, Any]] = []
        stale_events: List[StaleContextEvent] = []

        # Step 1: input guardrail
        input_verdict = GuardrailVerdict(passed=True, severity="pass")
        if self.input_guard is not None:
            input_verdict = self.input_guard.validate(query)
        trace.append({"step": "input_guard", "verdict": input_verdict.__dict__})
        if not input_verdict.passed:
            return WorkflowResult(
                query=query,
                route=RouteDecision(path="full", best_similarity=0.0, matched_case_ids=[], matched_cases=[], estimated_token_budget=0),
                output="Input blocked by guardrails.",
                input_verdict=input_verdict,
                output_verdict=GuardrailVerdict(passed=True, severity="pass"),
                stale_events=stale_events,
                trace=trace,
            )

        # Step 2: route with real EventStore (stale-aware downgrade happens inside)
        query_emb = self.embedding_provider.embed(query)
        route = classify_intent(
            query_text=query,
            query_embedding=query_emb,
            cases=self.cases,
            embedding_provider=self.embedding_provider,
            event_store=self.event_store,
            region=region,
        )
        trace.append({"step": "route", "route": route.path, "best_similarity": route.best_similarity})

        # Step 3: registry-check + staleness-aware downgrade
        checked_cases: List[Case] = []
        for c in self.cases:
            report = check_case_staleness(c, self.event_store)
            if report.is_stale:
                stale_events.append(StaleContextEvent(event=report, case=c))
                if route.path == RoutePath.FAST:
                    route = RouteDecision(
                        path=RoutePath.STALE_CONTEXT,
                        best_similarity=route.best_similarity,
                        matched_case_ids=route.matched_case_ids,
                        matched_cases=route.matched_cases,
                        estimated_token_budget=route.estimated_token_budget,
                        stale_flags=route.stale_flags + [report],
                    )
                    trace.append({"step": "staleness_downgrade", "new_path": RoutePath.STALE_CONTEXT})
            else:
                checked_cases.append(c)

        # Step 4: retrieve (matched_cases already from classifier)
        retrieved = route.matched_cases if route.matched_cases else []

        # Step 5: rerank
        if retrieved and self.reranker is not None:
            rerank_result = rerank_documents(query, retrieved, reranker=self.reranker)
            retrieved = rerank_result.documents
            trace.append({"step": "rerank", "order": rerank_result.reranked_order, "scores": rerank_result.scores})

        # Step 6: assemble
        if route.path == RoutePath.STALE_CONTEXT:
            output = self._synthesize_stale_context(query, stale_events, retrieved)
        else:
            output = self._synthesize_normal(query, retrieved)

        # Step 7: output guardrail
        output_verdict = GuardrailVerdict(passed=True, severity="pass")
        if self.output_guard is not None:
            output_verdict = self.output_guard.validate(output)
            if not output_verdict.passed:
                output = self.output_guard.append_disclaimer(output)
                output_verdict = GuardrailVerdict(passed=True, severity="warn", needs_disclaimer=True)
        trace.append({"step": "output_guard", "verdict": output_verdict.__dict__})

        return WorkflowResult(
            query=query,
            route=route,
            output=output,
            input_verdict=input_verdict,
            output_verdict=output_verdict,
            stale_events=stale_events,
            trace=trace,
        )

    def _synthesize_normal(self, query: str, contexts: List[dict]) -> str:
        if self.synthesize_fn:
            return self.synthesize_fn(query, contexts)
        if self.llm_client:
            ctx = "\n".join(c.get("text", "") for c in contexts[:3])
            prompt = f"Query: {query}\n\nContext:\n{ctx}\n\nAnswer concisely."
            return self.llm_client.chat([{"role": "user", "content": prompt}])
        return f"Retrieved {len(contexts)} contexts for query: {query}"

    def _synthesize_stale_context(self, query: str, stale_events: List[StaleContextEvent], contexts: List[dict]) -> str:
        lines = [f"Query: {query}", "", "**STALE CONTEXT WARNING**"]
        for se in stale_events:
            lines.append(f"- Event {se.event.event_id}: {se.event.reason}")
            lines.append(f"  Invalidated strategies: {', '.join(se.event.invalidated_strategies or [])}")
        lines.append("")
        if contexts:
            lines.append("Non-stale contexts:")
            for c in contexts[:3]:
                lines.append(f"- {c.get('text', '')[:200]}")
        return "\n".join(lines)
