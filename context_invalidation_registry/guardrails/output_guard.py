"""
OutputGuardrails — validates LLM output BEFORE user sees it.

Two checks:
1. Hard blocklist pattern match (structural — regex/string match OK)
2. Constitutional-AI principle check (semantic — MUST call LLM)
"""
from __future__ import annotations

from typing import List, Optional

from context_invalidation_registry.models import GuardrailVerdict


# Constitutional-AI principles borrowed for policy design (NOT implemented as runtime mechanism)
# per SPEC §6.5 clarification — output is checked against principles via LLM reasoning.
# Principles are operator-configurable; the shipped set below is a demo-domain
# example (marketing-advisory tone). Real deployments load their own principle
# list from config; the mechanism (LLM-reasoned check against a principle list)
# is what this file demonstrates, not any specific principle wording.
_CONSTITUTIONAL_PRINCIPLES = [
    "Output must comply with the operator-configured regulated-content policy.",
    "Output must not make or imply medical efficacy claims about any product.",
    "Output must not guarantee specific sales figures, revenue, or growth rates.",
    "Competitive analysis in output must be based on public data and must not use disparaging language.",
    "Output must not contain fabricated case data or made-up statistics.",
    "Output must include a risk disclaimer.",
    "Output must not discriminate against any group or use inappropriate stereotypes.",
]


class OutputGuardrails:
    """Validates ALL LLM-generated output before it reaches the user."""

    def __init__(self, llm_client=None, principles: Optional[List[str]] = None):
        self.llm_client = llm_client
        self.principles = principles or list(_CONSTITUTIONAL_PRINCIPLES)
        self._hard_blocklist = [
            "guarantee", "will definitely", "100%", "guaranteed viral", "must go viral", "risk-free profit",
            "therapeutic effect", "cure rate", "clinically verified", "fraud", "scam", "garbage product",
        ]
        self._prediction_patterns = [
            "will reach", "projected sales", "predicted growth", "will sell out",
            "guaranteed sales", "ensure revenue",
        ]
        self._default_disclaimer = (
            "\n\n---\n"
            "**Risk Disclaimer:** The above analysis is based on comparative research of existing industry cases and is for reference only. "
            "Actual marketing results are influenced by market conditions, execution quality, competitive landscape, and other factors. "
            "This system does not make any promises or guarantees regarding specific sales outcomes."
        )

    def validate(self, output_text: str) -> GuardrailVerdict:
        violations = []
        for pattern in self._hard_blocklist:
            if pattern in output_text:
                violations.append({
                    "type": "hard_blocklist",
                    "matched": pattern,
                    "severity": "rewrite",
                })
        for pattern in self._prediction_patterns:
            if pattern in output_text:
                violations.append({
                    "type": "prediction_pattern",
                    "matched": pattern,
                    "severity": "rewrite",
                })

        needs_disclaimer = (
            "disclaimer" not in output_text.lower()
            and "risk" not in output_text.lower()
        )

        passed = len(violations) == 0 and not needs_disclaimer
        return GuardrailVerdict(
            passed=passed,
            severity="pass" if passed else "warn",
            violations=violations,
            needs_disclaimer=needs_disclaimer,
        )

    def validate_with_llm(self, output_text: str) -> GuardrailVerdict:
        """LLM-reasoned principle check (semantic, not regex)."""
        if self.llm_client is None:
            return self.validate(output_text)
        import json
        prompt = (
            "You are a safety reviewer. Check the following output against these principles. "
            "Return JSON: {\"violations\": [{\"principle\": str, \"severity\": \"block|warn|pass\"}], \"needs_disclaimer\": bool}.\n"
            "Principles:\n" + "\n".join(f"- {p}" for p in self.principles) + "\n\nOutput:\n" + output_text
        )
        try:
            result = self.llm_client.structured(
                messages=[{"role": "user", "content": prompt}],
                json_schema={},
            )
            violations = result.get("violations", [])
            needs_disclaimer = result.get("needs_disclaimer", False)
            passed = len(violations) == 0 and not needs_disclaimer
            return GuardrailVerdict(
                passed=passed,
                severity="warn" if not passed else "pass",
                violations=violations,
                needs_disclaimer=needs_disclaimer,
            )
        except Exception:
            return self.validate(output_text)

    def append_disclaimer(self, output_text: str) -> str:
        if "disclaimer" not in output_text.lower() and "risk" not in output_text.lower():
            return output_text + self._default_disclaimer
        return output_text
