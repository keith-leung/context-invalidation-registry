"""LLM client wrapper — litellm-based, provider-agnostic."""
from __future__ import annotations

import os
from typing import Any, Dict, Optional

import litellm


class LLMClient:
    """Thin wrapper around litellm for chat completions."""

    def __init__(self, provider: str, model: str, base_url: str, api_key: str):
        self.provider = provider
        self.model = model
        self.base_url = base_url
        self.api_key = api_key
        litellm.api_key = api_key
        if base_url:
            litellm.api_base = base_url

    def chat(self, messages: list[dict], **kwargs) -> str:
        response = litellm.completion(
            model=self.model,
            messages=messages,
            api_key=self.api_key,
            api_base=self.base_url,
            **kwargs,
        )
        text = response.choices[0].message.content or ""
        # Strip MiniMax <think> blocks if present
        if "<think>" in text and "</think>" in text:
            start = text.index("</think>") + len("</think>")
            text = text[start:].strip()
        return text

    def structured(self, messages: list[dict], json_schema: dict, **kwargs) -> dict:
        """Force JSON output using response_format."""
        import json
        response = litellm.completion(
            model=self.model,
            messages=messages,
            api_key=self.api_key,
            api_base=self.base_url,
            response_format={"type": "json_object"},
            **kwargs,
        )
        text = response.choices[0].message.content or "{}"
        if "<think>" in text and "</think>" in text:
            start = text.index("</think>") + len("</think>")
            text = text[start:].strip()
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return {"raw": text}
