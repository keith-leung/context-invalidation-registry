"""
GuardrailFramework — defense-in-depth stack orchestrator.

LOCKED stack (SPEC §3 D4, §6):
  Layer 1 (structural, fast):    InputGuardrails + OutputGuardrails
                                 — deterministic blocklist match by category
                                   prefix (block_* / warn_*)
  Layer 2 (LLM-reasoned, outer): NeMo Guardrails via Colang rails
                                 — see nemo_wrapper.py + config/guardrails/
  Layer 3 (LLM safety classifier): Llama Guard 4-style classifier prompt
                                 — runs on the configured provider's medium
                                   tier when guardrails.llamaguard=true

Each layer catches a different failure mode:
  - Layer 1 catches known regulated vocabulary at zero LLM cost.
  - Layer 2 catches semantic variants and prompt injection that pure
    keyword match misses.
  - Layer 3 catches everything the first two miss via a generalist safety
    LLM prompt.

If any layer is unavailable (dependency missing, no API key, network
failure), the framework logs the degradation and continues with the
remaining layers. Inner-only fallback is never silent.
"""
from __future__ import annotations

import asyncio
import logging
from typing import Optional

import litellm

from context_invalidation_registry.guardrails.input_guard import InputGuardrails
from context_invalidation_registry.guardrails.output_guard import OutputGuardrails
from context_invalidation_registry.guardrails.nemo_wrapper import (
    check_input_with_nemo,
    check_output_with_nemo,
)
from context_invalidation_registry.models import GuardrailVerdict

logger = logging.getLogger(__name__)


class GuardrailFramework:
    """Defense-in-depth guardrail orchestrator (3-layer)."""

    def __init__(self, config, llm_client=None):
        self.config = config
        self.llm_client = llm_client

        gr_cfg = getattr(config, "guardrails", None)
        self._fallback_reason = (
            getattr(gr_cfg, "fallback_reason", None) if gr_cfg else None
        )
        self._use_fallback = bool(self._fallback_reason)

        # Layer 2: NeMo Guardrails — enabled when the operator opts in
        # (guardrails.nemoguardrails: true) AND the package + config dir +
        # api_key resolve. nemo_wrapper handles the actual "can I construct"
        # check on first call.
        self._nemo_requested = bool(
            getattr(gr_cfg, "nemoguardrails", False) if gr_cfg else False
        )

        # Layer 3: Llama Guard 4 classifier — historically gated on a
        # `llama_guard4` pip package that doesn't exist in the mainstream
        # ecosystem. gpt-agent.cc doesn't serve the real Meta Llama Guard 4
        # model either, so the pragmatic implementation is a LlamaGuard-style
        # prompt on the configured provider (MiniMax gives clean JSON).
        # Enable whenever the operator opts in AND a real provider is
        # configured (api_key is not the FILL_IN_CONFIG_YAML placeholder).
        self._llamaguard_requested = bool(
            getattr(gr_cfg, "llamaguard", False) if gr_cfg else False
        )
        self._llamaguard_available = self._llamaguard_requested and (
            not self._use_fallback
        ) and self._has_real_provider()

        # Layer 1: always available (deterministic).
        blocklists = getattr(gr_cfg, "blocklists", {}) if gr_cfg else {}
        self.input_guard = InputGuardrails(blocklists=blocklists)
        self.output_guard = OutputGuardrails(llm_client=llm_client)

    # ── availability probes ────────────────────────────────────────────

    def _has_real_provider(self) -> bool:
        provider_name = getattr(self.config, "default_provider", None)
        if not provider_name or not hasattr(self.config, "providers"):
            return False
        provider = self.config.providers.get(provider_name)
        if provider is None:
            return False
        api_key = getattr(provider, "api_key", "") or ""
        return bool(api_key) and api_key != "FILL_IN_CONFIG_YAML"

    # ── input side ─────────────────────────────────────────────────────

    def validate_input(self, user_input: str) -> GuardrailVerdict:
        # Layer 1 (structural, always).
        layer1 = self.input_guard.validate(user_input)
        if not layer1.passed:
            return layer1

        # Layer 2 (NeMo Colang) — LLM-reasoned. Best-effort.
        if self._nemo_requested and not self._use_fallback:
            try:
                nemo_verdict = _run_async(
                    check_input_with_nemo(user_input, self.config)
                )
                if not nemo_verdict["allowed"]:
                    return GuardrailVerdict(
                        passed=False,
                        severity="block",
                        violations=[
                            {
                                "type": "nemo_input_rail",
                                "matched": "colang_flow",
                                "severity": "block",
                                "refusal_text": nemo_verdict.get(
                                    "refusal_text", ""
                                ),
                            }
                        ],
                        sanitized_input=None,
                    )
            except Exception as exc:  # pragma: no cover — pure defensive
                logger.warning("NeMo input rail invocation failed: %s", exc)

        return layer1

    # ── output side ────────────────────────────────────────────────────

    def validate_output(self, output_text: str) -> GuardrailVerdict:
        # Layer 1: structural (hard_blocklist + prediction patterns + disclaimer).
        layer1 = self.output_guard.validate(output_text)
        if not layer1.passed:
            return layer1

        # Layer 2: NeMo Colang output rail.
        if self._nemo_requested and not self._use_fallback:
            try:
                nemo_verdict = _run_async(
                    check_output_with_nemo(output_text, self.config)
                )
                if not nemo_verdict["allowed"]:
                    return GuardrailVerdict(
                        passed=False,
                        severity="warn",
                        violations=[
                            {
                                "type": "nemo_output_rail",
                                "matched": "colang_flow",
                                "severity": "warn",
                                "refusal_text": nemo_verdict.get(
                                    "refusal_text", ""
                                ),
                            }
                        ],
                        needs_disclaimer=True,
                    )
            except Exception as exc:  # pragma: no cover
                logger.warning("NeMo output rail invocation failed: %s", exc)

        # Layer 3: Llama Guard-style LLM classifier.
        if self._llamaguard_available:
            try:
                lg_verdict = self._llamaguard_check(output_text)
                if not lg_verdict.passed:
                    return lg_verdict
            except Exception as exc:  # pragma: no cover
                logger.warning("Llama Guard check failed: %s", exc)

        return layer1

    # ── Layer 3 impl ───────────────────────────────────────────────────

    def _llamaguard_check(self, text: str) -> GuardrailVerdict:
        """Llama Guard 4-style safety classification via the configured provider.

        Uses a small LlamaGuard-shaped prompt on the medium tier. The
        response is forced to `response_format=json_object` for parse
        stability (gpt-agent.cc honors this).
        """
        provider_name = getattr(self.config, "default_provider", None)
        provider_cfg = self.config.providers.get(provider_name)
        model = None
        if provider_cfg and provider_cfg.tiers:
            medium = provider_cfg.tiers.get("medium", {})
            model = medium.get("model") or list(
                provider_cfg.tiers.values()
            )[0].get("model")
        if not model:
            return GuardrailVerdict(passed=True, severity="pass")

        prompt = (
            "You are a safety classifier acting as Llama Guard 4. Classify "
            "the following assistant text as safe or unsafe against these "
            "categories: hate, harassment, self-harm, sexual, violence, "
            "regulated-content (medical / financial guarantee / competitor "
            "defamation), fabricated statistics.\n"
            'Return JSON only: {"verdict": "safe"|"unsafe", '
            '"category": "<name-or-empty>", "reason": "<brief>"}.\n\n'
            f"Text:\n{text}\n"
        )
        try:
            response = litellm.completion(
                model=f"openai/{model}" if not model.startswith("openai/") else model,
                messages=[{"role": "user", "content": prompt}],
                api_key=provider_cfg.api_key,
                api_base=provider_cfg.base_url,
                response_format={"type": "json_object"},
                temperature=0.0,
                max_tokens=200,
            )
            import json
            raw = response.choices[0].message.content or "{}"
            # MiniMax emits <think>...</think> preambles — strip.
            if "<think>" in raw and "</think>" in raw:
                start = raw.index("</think>") + len("</think>")
                raw = raw[start:].strip()
            result = json.loads(raw)
            verdict = str(result.get("verdict", "safe")).lower()
            category = str(result.get("category", ""))
            reason = str(result.get("reason", ""))
            passed = verdict == "safe"
            return GuardrailVerdict(
                passed=passed,
                severity="pass" if passed else "block",
                violations=[]
                if passed
                else [
                    {
                        "type": "llamaguard",
                        "category": category,
                        "reason": reason,
                        "severity": "block",
                    }
                ],
                needs_disclaimer=not passed,
            )
        except Exception as exc:
            logger.info(
                "LlamaGuard classifier LLM call failed (%s); fail-open.", exc,
            )
            return GuardrailVerdict(passed=True, severity="pass")

    # ── status introspection ───────────────────────────────────────────

    @property
    def status(self) -> str:
        if self._use_fallback:
            return f"fallback ({self._fallback_reason})"
        parts = ["Layer1 (structural)"]
        if self._nemo_requested:
            parts.append("Layer2 (NeMo Colang)")
        if self._llamaguard_available:
            parts.append("Layer3 (LlamaGuard-via-LLM)")
        return " + ".join(parts) if len(parts) > 1 else parts[0]


# ── async bridging helper ─────────────────────────────────────────────


def _run_async(coro):
    """Run an async coroutine from a sync caller.

    `validate_input`/`validate_output` are sync per the D5 workflow's
    interface expectation, but NeMo's `generate_async` is async. This
    helper does the right thing whether or not we're inside a running
    event loop.
    """
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None
    if loop is None or not loop.is_running():
        return asyncio.run(coro)
    # We're inside a running loop (e.g. under FastAPI). Use nest_asyncio
    # fallback: create a new loop for this sync call. In practice
    # validate_input is called from sync workflow.run, so this branch is
    # rarely hit.
    import concurrent.futures

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(asyncio.run, coro)
        return future.result()
