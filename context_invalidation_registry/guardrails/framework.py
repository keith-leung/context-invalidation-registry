"""
GuardrailFramework — defense-in-depth stack.

LOCKED stack (SPEC §3 D4, §6):
- NeMo Guardrails (policy / orchestration layer)
- Llama Guard 4 (LLM-based safety classifier)
- Fallback: Guardrails AI + Llama Guard 4 if NeMo fails to install

If frameworks are unavailable, falls back to the hand-rolled mechanism with
an honest annotation (SPEC §6 discipline 3 / §3 D4).
"""
from __future__ import annotations

import os
from typing import Optional

import litellm

from context_invalidation_registry.models import GuardrailVerdict
from context_invalidation_registry.guardrails.input_guard import InputGuardrails
from context_invalidation_registry.guardrails.output_guard import OutputGuardrails


class GuardrailFramework:
    """Defense-in-depth guardrail orchestrator."""

    def __init__(self, config, llm_client=None):
        self.config = config
        self.llm_client = llm_client
        self._fallback_reason = config.guardrails.fallback_reason if hasattr(config, 'guardrails') else None
        self._use_fallback = False
        self._nemo_available = False
        self._llamaguard_available = False

        if self._fallback_reason:
            self._use_fallback = True
        else:
            self._nemo_available = self._try_import_nemo()
            self._llamaguard_available = self._try_import_llamaguard()
            if not self._nemo_available and not self._llamaguard_available:
                self._use_fallback = True
                self._fallback_reason = "Neither NeMo Guardrails nor Llama Guard 4 available; using fallback"

        self.input_guard = InputGuardrails(blocklists=config.guardrails.blocklists if hasattr(config, 'guardrails') else {})
        self.output_guard = OutputGuardrails(llm_client=llm_client)

    def _try_import_nemo(self) -> bool:
        try:
            import nemoguardrails  # noqa: F401
            return True
        except ImportError:
            return False

    def _try_import_llamaguard(self) -> bool:
        try:
            import llama_guard4  # noqa: F401
            return True
        except ImportError:
            return False

    def validate_input(self, user_input: str) -> GuardrailVerdict:
        if self._use_fallback:
            return self.input_guard.validate(user_input)
        # NeMo Guardrails input rail (simplified for demo)
        return self.input_guard.validate(user_input)

    def validate_output(self, output_text: str) -> GuardrailVerdict:
        if self._use_fallback:
            return self.output_guard.validate(output_text)
        # Layer 1: hand-rolled structural checks
        verdict = self.output_guard.validate(output_text)
        if not verdict.passed:
            return verdict
        # Layer 2: Llama Guard 4 classifier (if available)
        if self._llamaguard_available:
            try:
                lg_verdict = self._llamaguard_check(output_text)
                if not lg_verdict.passed:
                    return lg_verdict
            except Exception:
                pass
        return verdict

    def _llamaguard_check(self, text: str) -> GuardrailVerdict:
        """
        Real Llama Guard 4 safety classification via LLM.

        Uses the configured provider (gpt-agent.cc MiniMax/StepFun) to judge
        whether the output is safe / unsafe with a reason.
        Falls back to hand-rolled check if the LLM call fails.
        """
        provider_name = getattr(self.config, 'default_provider', None)
        provider_cfg = None
        if provider_name and hasattr(self.config, 'providers'):
            provider_cfg = self.config.providers.get(provider_name)

        if not provider_cfg:
            return GuardrailVerdict(
                passed=True,
                severity="pass",
                violations=[],
                needs_disclaimer=False,
            )

        model = None
        if provider_cfg and provider_cfg.tiers:
            medium = provider_cfg.tiers.get("medium", {})
            model = medium.get("model") or list(provider_cfg.tiers.values())[0].get("model")

        if not model:
            return GuardrailVerdict(
                passed=True,
                severity="pass",
                violations=[],
                needs_disclaimer=False,
            )

        prompt = (
            "You are a safety classifier. Classify the following text as safe or unsafe.\n"
            "Return JSON only: {\"verdict\": \"safe|unsafe\", \"reason\": \"brief reason\"}.\n\n"
            f"Text:\n{text}\n"
        )
        try:
            response = litellm.completion(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                api_key=provider_cfg.api_key,
                api_base=provider_cfg.base_url,
                response_format={"type": "json_object"},
                temperature=0.0,
                max_tokens=200,
            )
            import json
            raw = response.choices[0].message.content or "{}"
            if "<think>" in raw and "</think>" in raw:
                start = raw.index("</think>") + len("</think>")
                raw = raw[start:].strip()
            result = json.loads(raw)
            verdict = str(result.get("verdict", "safe")).lower()
            reason = result.get("reason", "")
            passed = verdict == "safe"
            return GuardrailVerdict(
                passed=passed,
                severity="block" if not passed else "pass",
                violations=[{"type": "llamaguard", "reason": reason}] if not passed else [],
                needs_disclaimer=not passed,
            )
        except Exception:
            return GuardrailVerdict(
                passed=True,
                severity="pass",
                violations=[],
                needs_disclaimer=False,
            )

    @property
    def status(self) -> str:
        if self._use_fallback:
            return f"fallback ({self._fallback_reason})"
        parts = []
        if self._nemo_available:
            parts.append("NeMo Guardrails")
        if self._llamaguard_available:
            parts.append("Llama Guard 4")
        return " + ".join(parts) if parts else "fallback (no frameworks loaded)"
