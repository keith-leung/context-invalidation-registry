"""AppConfig loader — mode + provider + blocklist parsing."""
from __future__ import annotations

import pytest

from context_invalidation_registry.config import AppConfig


def test_load_ci_config(ci_config_path):
    cfg = AppConfig.load(str(ci_config_path))
    assert cfg.mode == "mock"
    assert cfg.embedding.provider == "mock"
    assert "block_category_a" in cfg.guardrails.blocklists
    assert "warn_category_a" in cfg.guardrails.blocklists


def test_ci_config_blocklist_uses_prefix_convention(ci_config_path):
    """Confirms the block_*/warn_* prefix convention is preserved.

    input_guard.py derives severity from this prefix — if a future edit
    breaks the naming, this test catches it before demos silently fail.
    """
    cfg = AppConfig.load(str(ci_config_path))
    for category in cfg.guardrails.blocklists:
        assert category.startswith("block_") or category.startswith("warn_"), (
            f"category '{category}' violates block_*/warn_* prefix convention"
        )


def test_config_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        AppConfig.load(str(tmp_path / "does-not-exist.yaml"))
