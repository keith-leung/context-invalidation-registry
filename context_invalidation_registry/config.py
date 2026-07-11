"""
Configuration loader.

Reads YAML config; NEVER falls back to env vars for mock/real switching.
Mode switching is strictly via --config flag (SPEC §6).
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

import yaml


@dataclass(frozen=True)
class ProviderConfig:
    base_url: str
    api_key: str
    tiers: Dict[str, Dict[str, str]]


@dataclass(frozen=True)
class JudgeConfig:
    provider: str
    tier: str


@dataclass(frozen=True)
class EmbeddingConfig:
    provider: str  # "mock" | "voyage" | "openai" | "cohere"
    model: Optional[str] = None
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    dimensions: int = 1536


@dataclass(frozen=True)
class RerankerConfig:
    provider: str  # "cohere" | "bge" | "mock"
    model: Optional[str] = None
    api_key: Optional[str] = None
    base_url: Optional[str] = None


@dataclass(frozen=True)
class GuardrailsConfig:
    enabled: bool = True
    nemoguardrails: bool = False
    llamaguard: bool = False
    fallback_reason: Optional[str] = None
    blocklists: Dict[str, list] = field(default_factory=dict)


@dataclass(frozen=True)
class AppConfig:
    mode: str  # "real" | "mock"
    default_provider: str
    providers: Dict[str, ProviderConfig]
    judge: JudgeConfig
    embedding: EmbeddingConfig
    reranker: RerankerConfig
    guardrails: GuardrailsConfig
    data_dir: Path
    seed_events_path: Path

    @classmethod
    def load(cls, path: Optional[str] = None) -> AppConfig:
        if path is None:
            path = os.environ.get("CONTEXT_INVALIDATION_CONFIG", "config.yaml")
        resolved = Path(path)
        if not resolved.exists():
            raise FileNotFoundError(f"Config not found: {resolved}")
        with open(resolved, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}

        mode = raw.get("mode", "real")
        default_provider = raw.get("default_provider", "mimo")
        providers_raw = raw.get("providers", {})
        providers = {
            name: ProviderConfig(
                base_url=p["base_url"],
                api_key=p.get("api_key", ""),
                tiers=p.get("tiers", {}),
            )
            for name, p in providers_raw.items()
        }

        judge_raw = raw.get("judge", {})
        judge = JudgeConfig(
            provider=judge_raw.get("provider", "minimax"),
            tier=judge_raw.get("tier", "medium"),
        )

        emb_raw = raw.get("embedding", {})
        embedding = EmbeddingConfig(
            provider=emb_raw.get("provider", "mock"),
            model=emb_raw.get("model"),
            api_key=emb_raw.get("api_key"),
            base_url=emb_raw.get("base_url"),
            dimensions=emb_raw.get("dimensions", 1536),
        )

        rerank_raw = raw.get("reranker", {})
        reranker = RerankerConfig(
            provider=rerank_raw.get("provider", "mock"),
            model=rerank_raw.get("model"),
            api_key=rerank_raw.get("api_key"),
            base_url=rerank_raw.get("base_url"),
        )

        guard_raw = raw.get("guardrails", {})
        guardrails = GuardrailsConfig(
            enabled=guard_raw.get("enabled", True),
            nemoguardrails=guard_raw.get("nemoguardrails", False),
            llamaguard=guard_raw.get("llamaguard", False),
            fallback_reason=guard_raw.get("fallback_reason"),
            blocklists=guard_raw.get("blocklists", {}),
        )

        data_dir = Path(raw.get("data_dir", "data"))
        seed_events_path = data_dir / raw.get("seed_events_file", "critical_events.demo.json")

        return cls(
            mode=mode,
            default_provider=default_provider,
            providers=providers,
            judge=judge,
            embedding=embedding,
            reranker=reranker,
            guardrails=guardrails,
            data_dir=data_dir,
            seed_events_path=seed_events_path,
        )
