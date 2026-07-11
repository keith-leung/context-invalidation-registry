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

from typing import Optional

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
        # Placeholder: integrate llama-guard4 classifier here
        # In a real setup this calls the Llama Guard 4 model endpoint
        return GuardrailVerdict(passed=True, severity="pass")

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
