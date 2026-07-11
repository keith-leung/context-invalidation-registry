# context-invalidation-registry

**Event-driven semantic invalidation for agentic retrieval.**

A Critical Events Registry that detects when retrieved context becomes *dangerous* after the fact — a 2024 marketing case is fine when retrieved, but a 2025 scandal or regulatory event makes recommending it actively harmful. D is the event source that emits invalidation signals to downstream tombstone executors (notably repo A's checkpoint tombstoner), and gates the agent's I/O boundary with defense-in-depth guardrails.

## Problem

A RAG agent retrieves a marketing case from 2024 and recommends it in 2026. Last September a scandal made that strategy actively illegal or brand-destroying. How does the agent know? And what happens to the checkpoints that already baked in the now-poisoned recommendation?

There are three concrete failure modes this repo fixes:

1. **Stale-context-as-dangerous.** Ordinary staleness (old data) degrades quality; event-driven staleness (a regulatory or scandal event invalidated a strategy) makes the recommendation actively harmful. This repo's Critical Events Registry encodes exactly this: semantic invalidation, not just TTL expiry.
2. **No propagation.** Even if you detect the staleness, the agent has already retrieved the case, reasoned over it, and baked the recommendation into checkpoints or messages. Detecting staleness at retrieval time is necessary but insufficient. D connects to A: D is the event source, A is the tombstone executor.
3. **No I/O boundary guardrails.** Prompt-only safety is the athlete-as-referee problem in safety form. This repo structurally enforces safety at the I/O boundary with a layered guardrail stack.

## Architecture

```
query -> route -> retrieve -> rerank -> staleness-check -> assemble -> guardrail -> synthesize
```

### D1 — Critical Events Registry (the named IP)

Storage-agnostic registry (pure Python + JSON) with an 8-field event schema:

- `event_id`, `event_name`, `event_date`
- `affected_industries`, `affected_regions`
- `impact_description`
- `invalidated_strategies`, `new_required_strategies`

Staleness logic has three branches:

1. **Match + pre-event:** industry and region overlap, and case vintage is before the event date → stale.
2. **Non-match:** no overlap → fresh.
3. **No vintage + recent event:** when the case has no date, the conservative rule flags it if the event is less than two years old.

This is more than TTL — it is *event-driven semantic invalidation*: a datum was fine when retrieved; an event last week made recommending it actively dangerous.

### D2 — D→A invalidation signal

When D registers a new event, it emits an `InvalidationSignal` carrying a predicate that matches A's `tombstone_items_matching` contract (spec A §7.2). A demo subscriber stands in for A and records the signals, proving the contract is consumable.

### D3 — Staleness-aware three-tier router + reranking

The router classifies queries into Fast / Incremental / Full paths based on vector similarity. D's twist: the router consults D1's registry during routing. A high-similarity case that D1 flags as stale is downgraded to a `stale_context` branch that surfaces the invalidation rather than recommending the poisoned case.

After retrieval and before assembly, the pipeline includes a **reranker** step. The implementation supports **Cohere Rerank** (managed API) or **BGE Reranker** (self-hosted). This completes the standard 2026 RAG pipeline shape: retrieve → rerank → assemble.

### D4 — Defense-in-depth guardrails

Two-stage I/O boundary guardrails:

- **Input guard** BEFORE routing/LLM (block/warn/pass severity).
- **Output guard** AFTER synthesis BEFORE user (Constitutional-AI principles borrowed for policy design + hard blocklist + forced disclaimer).

The locked defense-in-depth stack is **NeMo Guardrails** (NVIDIA; policy/orchestration layer via Colang DSL) paired with **Llama Guard 4** (Meta; LLM-based safety classifier, multimodal). If NeMo fails to install at implementation time, the system falls back to Guardrails AI + Llama Guard 4 and documents the substitution honestly. Optional named comparators include ProtectAI LLM Guard, Guardrails AI, Bedrock Guardrails, Azure Prompt Shields, and OpenAI Moderation.

### D5 — LlamaIndex Workflows

LlamaIndex Workflows gives an explicit event/step model for multi-step retrieval (query → retrieve → rerank → assemble → guardrail-check → synthesize), where each step can emit events the next consumes. This fits D's event-driven nature (D1 events, D2 invalidation signals) better than a linear chain. A `StaleContextEvent` flows between the registry-check step and the assembly step, making the event-driven shape visible in the orchestration.

## Why event-driven staleness is not cache invalidation

Ordinary staleness (TTL, last-modified) is a quality problem: old data is less accurate. Event-driven staleness is a safety problem: a datum was correct when retrieved, but a subsequent event (regulatory change, scandal, safety recall) made it actively harmful to use. The registry doesn't say "this data is old"; it says "this strategy was valid on date X, but event Y on date Z invalidated it." Tombstone propagation through the checkpoint graph is the downstream consequence: A's tombstone executor removes or quarantines checkpoints that consumed the poisoned context.

This is the original architectural contribution of this repo: bridging database-tombstone semantics with agent state lineage through **event-driven semantic invalidation**, the **Critical Events Registry**, and **tombstone propagation**.

## Frameworks and landscape

### Retrieval frameworks (2026 agentic-RAG big three)

- **LangGraph** — control plane + event bus for orchestration glue.
- **LlamaIndex Workflows** + **LlamaParse** — retrieval/assembly event-step engine + document parsing.
- **Haystack 2.x** — peer alternative for broader keyword coverage.

### Retrieval-shape terms

This repo uses **agentic retrieval** (not naive RAG), **query routing** (the three-tier classifier), **multi-hop retrieval**, **self-RAG / self-correction**, **corrective RAG**, and **selective context assembly** (this repo's own descriptive framing for the source-verification step).

### Reranking

**Reranking** via **reranker** completes the standard 2026 RAG pipeline: retrieve → rerank → assemble. The implementation supports **Cohere Rerank** (managed) or **BGE Reranker** (self-hosted).

### GraphRAG (cite, don't build in v1)

D's Critical Events Registry invalidates flat-doc chunks. **GraphRAG** extends this to entity-relationship graphs where an event can invalidate a subgraph. Mentioned as the natural extension, not implemented in v1.

### Data ingestion landscape (cite)

D's demo corpus uses **LlamaParse**; the broader 2026 ingestion landscape includes **Firecrawl** (web-to-markdown extraction), **Unstructured** (universal document parser), and **Docling** (IBM's document parser).

### Vector DB products

The registry's vector store could plug into **Pinecone**, **Qdrant**, **Milvus**, or **pgvector** in production. The demo uses an in-memory store.

### Embeddings

Real-embedding options:

- **Voyage AI** `voyage-3-large` — headline quality pick. Re-verification note (2026-07-08): MongoDB-ownership and model-level hybrid-sparse+dense sub-claims were not cleanly primary-source-confirmed against current Voyage/MongoDB docs; these claims are dropped from this repo's documentation. Domain-variant claims are retained where Voyage's own docs describe them. If you need the most conservative documented option, use **OpenAI** `text-embedding-3-large`.
- **OpenAI** `text-embedding-3-large` — most-deployed alternative with Matryoshka dimension truncation.
- **Cohere** `embed-v4` — natively returns int8/uint8/binary embeddings.
- Local **BGE-M3**, **E5-Mistral**, **Nomic Embed** — zero-API-cost self-hosted options.
- `mock_embedding` (deterministic bag-of-ngrams) — [CI-MOCK] test fallback.

### Guardrail stack

**NeMo Guardrails** (NVIDIA; orchestration/policy layer via Colang DSL — NOT a classifier) paired with **Llama Guard 4** (Meta; LLM-based safety classifier, multimodal). This is the 2026 standard layered architecture, not a single-tool choice. NeMo Guardrails is installed and detected at runtime; full Colang DSL rail configuration is stubbed for v1 and can be extended. Llama Guard 4 is referenced as an API-callable classifier; the pip package availability is pending verification — the framework wrapper falls back to the hand-rolled mechanism with honest annotation if the classifier is unavailable. Optional comparators: ProtectAI LLM Guard, Guardrails AI, Bedrock Guardrails, Azure Prompt Shields, OpenAI Moderation. **I/O boundary guardrails** enforce input screening before the LLM and output compliance before the user, with block / warn / pass severity.

### Constitutional-AI principles

Anthropic's Constitutional AI is a training-time / model-alignment concept, NOT a runtime I/O-guardrail mechanism. D borrows the principle (output checked against a principle list) for policy design — the distinction is explicit.

## Setup

```bash
# Create the dedicated conda environment (mandatory per SPEC §6)
conda create -n context-invalidation-registry python=3.11 -y
conda activate context-invalidation-registry

# Install dependencies
pip install -r requirements.txt
```

## Configuration

- `config.yaml` — canonical run config (gitignored, never committed).
- `config.ci.yaml` — CI / test config (committed, `mode: mock`).
- `config.example.yaml` — template (committed, keys blank).

Mode switching is strictly via the `mode:` field in these files, NOT environment variables. The runner picks config via `--config <path>`.

## Usage

```bash
# Run all demos (smoke test)
python -m context_invalidation_registry.run --all

# Run a single demo
python -m context_invalidation_registry.run --demo d1
python -m context_invalidation_registry.run --demo d2
python -m context_invalidation_registry.run --demo d3
python -m context_invalidation_registry.run --demo d4
python -m context_invalidation_registry.run --demo d5

# Use a specific config
python -m context_invalidation_registry.run --all --config config.ci.yaml
```

## Demos

| Demo | What it proves | Mode |
|------|---------------|------|
| D1 | Three staleness branches (pre-event stale, non-match fresh, no-vintage conservative stale) | [CI-MOCK] |
| D2 | `InvalidationSignal` emitted + received by demo subscriber; predicate shape matches A's `tombstone_items_matching` contract | [CI-MOCK] |
| D3 | Stale high-similarity case downgraded to `stale_context`; fresh high-similarity case hits Fast | [REAL] + [CI-MOCK] |
| D4 | Input block/warn/pass branches; output blocklist + disclaimer append; blocklists from config | [REAL] + [CI-MOCK] |
| D5 | Workflow completes with explicit `StaleContextEvent` flowing between steps | [REAL] |

## Seed data

`data/critical_events.demo.json` is a desensitized version of the original seed data. Real entity names have been replaced with synthetic aliases; event type, 8-field schema, and staleness logic are preserved. The original reference data stays in `reference/` (private input) and does not enter the public repo.

## Cross-repo contracts

- **D → A (FROZEN):** `InvalidationSignal.predicate` matches A's `CheckpointEntry` fields so A's `tombstone_items_matching` consumes it directly. D emits; A acts. This is the portfolio's headline cross-repo interface.
- **D ↔ C (complement, no coupling):** C5 redacts credentials in runtime internal channels; D4 guards the I/O boundary. Cite, don't share code.

## License

MIT
