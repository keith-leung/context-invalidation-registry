"""
InputGuardrails — validates user input BEFORE routing/LLM.

Block/warn/pass severity model. Blocklists loaded from config (not hardcoded).
Regex/string-pattern match is structural (allowed); semantic verdicts go to LLM.
"""
from __future__ import annotations

from typing import List, Optional

from context_invalidation_registry.models import GuardrailVerdict


class InputGuardrails:
    """Validates user input before it enters the pipeline."""

    def __init__(self, blocklists: dict):
        self.blocklists = blocklists or {}

    def validate(self, user_input: str) -> GuardrailVerdict:
        """Structural blocklist match. Severity derives from the category
        name prefix: `block_*` → block, `warn_*` → warn, else → warn.
        The prefix convention lets operators add new categories without
        touching this code — see config.ci.yaml for the shipped shape."""
        violations = []
        blocked = False

        for category, keywords in self.blocklists.items():
            if not keywords:
                continue
            action = "block" if category.startswith("block_") else "warn"
            for kw in keywords:
                if kw in user_input:
                    if action == "block":
                        blocked = True
                    violations.append({
                        "type": category,
                        "matched": kw,
                        "severity": action,
                    })

        passed = not blocked
        return GuardrailVerdict(
            passed=passed,
            severity="block" if blocked else ("warn" if violations else "pass"),
            violations=violations,
            sanitized_input=None if blocked else user_input,
        )
