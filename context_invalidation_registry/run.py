"""
Smoke test runner — runs D1-D5 demos and reports pass/fail.

Usage:
    python -m context_invalidation_registry.run --all
    python -m context_invalidation_registry.run --demo d1
"""
from __future__ import annotations

import argparse
import json
import sys
from typing import Dict, List

from context_invalidation_registry.demos import (
    demo_d1_staleness,
    demo_d2_signal,
    demo_d3_routing,
    demo_d4_guardrails,
    demo_d5_workflow,
)


DEMOS = {
    "d1": ("Critical Events Registry staleness", demo_d1_staleness),
    "d2": ("Invalidation signal emission", demo_d2_signal),
    "d3": ("Staleness-aware routing + reranking", demo_d3_routing),
    "d4": ("I/O guardrails", demo_d4_guardrails),
    "d5": ("LlamaIndex Workflow", demo_d5_workflow),
}


def run_all() -> int:
    results: Dict[str, Dict] = {}
    all_pass = True
    for key, (name, fn) in DEMOS.items():
        try:
            res = fn()
            results[key] = res
            status = "PASS" if res.get("passed") else "FAIL"
            if not res.get("passed"):
                all_pass = False
            print(f"[{status}] {name}")
        except Exception as exc:  # pragma: no cover
            results[key] = {"passed": False, "error": str(exc)}
            all_pass = False
            print(f"[FAIL] {name}: {exc}")

    print("\n--- Summary ---")
    print(json.dumps(results, ensure_ascii=False, indent=2))
    return 0 if all_pass else 1


def main() -> int:
    parser = argparse.ArgumentParser(description="Context Invalidation Registry runner")
    parser.add_argument("--config", default=None, help="Path to config YAML")
    parser.add_argument("--all", action="store_true", help="Run all demos")
    parser.add_argument("--demo", choices=list(DEMOS.keys()), help="Run a single demo")
    args = parser.parse_args()

    if args.config:
        import os
        os.environ["CONTEXT_INVALIDATION_CONFIG"] = args.config

    if args.all:
        return run_all()
    if args.demo:
        name, fn = DEMOS[args.demo]
        res = fn()
        print(json.dumps(res, ensure_ascii=False, indent=2))
        return 0 if res.get("passed") else 1
    parser.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
