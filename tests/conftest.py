"""Shared pytest fixtures for D repo."""
from __future__ import annotations

import os
from pathlib import Path

import pytest


@pytest.fixture
def repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


@pytest.fixture
def ci_config_path(repo_root: Path) -> Path:
    return repo_root / "config.ci.yaml"


@pytest.fixture
def real_config_path(repo_root: Path) -> Path:
    return repo_root / "config.yaml"


@pytest.fixture
def has_real_llm(real_config_path: Path) -> bool:
    """True iff config.yaml exists with a non-placeholder api_key.

    Integration tests skip when False so CI without a real key stays green.
    """
    if not real_config_path.exists():
        return False
    import yaml
    with open(real_config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    providers = (cfg.get("providers") or {})
    for provider in providers.values():
        key = (provider or {}).get("api_key") or ""
        if key and key != "FILL_IN_CONFIG_YAML":
            return True
    return False
