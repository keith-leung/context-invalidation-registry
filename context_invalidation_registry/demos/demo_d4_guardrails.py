"""
Demo D4 — input/output guardrails.

Tests block/warn/pass input branches; output hard-blocklist + disclaimer append.
Blocklists loaded from config (SPEC §6).
"""
from __future__ import annotations

from typing import Any, Dict

from context_invalidation_registry.config import AppConfig
from context_invalidation_registry.guardrails.framework import GuardrailFramework
from context_invalidation_registry.guardrails.output_guard import OutputGuardrails


def _load_config() -> AppConfig:
    return AppConfig.load()


def demo_d4_guardrails() -> Dict[str, Any]:
    config = _load_config()
    framework = GuardrailFramework(config)
    output_guard = OutputGuardrails()

    # Input: blocklist keyword
    input_block = framework.validate_input("This input contains taiwan independence keyword")
    # Input: clean
    input_pass = framework.validate_input("Please write a spring marketing copy for me")

    # Output: hard blocklist pattern
    bad_output = "This product guarantees to cure all diseases, 100% effective."
    output_verdict = output_guard.validate(bad_output)
    # Output: missing disclaimer
    no_disclaimer = "According to our analysis, sales are projected to reach new highs."
    no_disc_verdict = output_guard.validate(no_disclaimer)

    passed = (
        input_block.passed is False
        and input_pass.passed is True
        and output_verdict.passed is False
        and no_disc_verdict.needs_disclaimer is True
    )

    return {
        "input_block_passed": input_block.passed,
        "input_block_severity": input_block.severity,
        "input_pass_passed": input_pass.passed,
        "output_block_passed": output_verdict.passed,
        "output_block_violations": len(output_verdict.violations),
        "output_needs_disclaimer": no_disc_verdict.needs_disclaimer,
        "guardrail_status": framework.status,
        "passed": passed,
    }
