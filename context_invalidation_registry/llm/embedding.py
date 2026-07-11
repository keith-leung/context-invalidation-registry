"""Embedding provider — mock, Voyage, OpenAI, Cohere."""
from __future__ import annotations

import hashlib
import math
from typing import List, Optional


def mock_embedding(text: str, dim: int = 1536) -> List[float]:
    """
    Deterministic bag-of-ngrams pseudo-embedding.
    Each character and character-bigram maps to a fixed dimension via md5.
    Chinese stopwords are stripped so content keywords dominate similarity.
    Same text always produces the same vector across processes.
    """
    vec = [0.0] * dim
    cleaned = _strip_stopchars(text.lower().replace(" ", ""))
    chars = list(cleaned)

    for ch in chars:
        idx = _stable_hash(ch) % dim
        vec[idx] += 1.0

    for i in range(len(chars) - 1):
        bigram = chars[i] + chars[i + 1]
        idx = _stable_hash(bigram) % dim
        vec[idx] += 0.5

    norm = math.sqrt(sum(x * x for x in vec))
    if norm > 0:
        vec = [x / norm for x in vec]
    else:
        vec[0] = 1.0
    return vec


def _stable_hash(s: str) -> int:
    return int(hashlib.md5(s.encode("utf-8")).hexdigest(), 16)


_ZH_STOPCHARS = set(
    "我你他她它们的了在是做想要有一不也都这那就人和与或"
    "可以通过面向目前目标如何怎么什么吗呢了吧把被比从到"
    "对于关于或者以及因为所以如果虽然但是而且然后之后以"
    "上下中大小多少很非常更最某每各个些里边向着得地"
    "能够应该需要已经正在进行分析评价一下帮"
)


def _strip_stopchars(text: str) -> str:
    return "".join(ch for ch in text if ch not in _ZH_STOPCHARS)


class EmbeddingProvider:
    """Unified embedding interface."""

    def __init__(self, provider: str, model: Optional[str] = None,
                 api_key: Optional[str] = None, base_url: Optional[str] = None,
                 dimensions: int = 1536):
        self.provider = provider
        self.model = model
        self.api_key = api_key
        self.base_url = base_url
        self.dimensions = dimensions
        self._client = None

    def embed(self, text: str) -> List[float]:
        if self.provider == "mock":
            return mock_embedding(text, dim=self.dimensions)
        if self._client is None:
            self._client = self._build_client()
        return self._client.embed(text)

    def _build_client(self):
        if self.provider == "openai":
            from openai import OpenAI
            return OpenAI(api_key=self.api_key, base_url=self.base_url or "https://api.openai.com/v1")
        if self.provider == "voyage":
            import voyageai
            return voyageai.Client(api_key=self.api_key)
        if self.provider == "cohere":
            import cohere
            return cohere.Client(api_key=self.api_key)
        raise ValueError(f"Unsupported embedding provider: {self.provider}")

    def _embed_openai(self, text: str) -> List[float]:
        resp = self._client.embeddings.create(input=[text], model=self.model or "text-embedding-3-large")
        return resp.data[0].embedding

    def _embed_voyage(self, text: str) -> List[float]:
        resp = self._client.embed([text], model=self.model or "voyage-3-large", input_type="document")
        return resp.embeddings[0]

    def _embed_cohere(self, text: str) -> List[float]:
        resp = self._client.embed(texts=[text], model=self.model or "embed-english-v3.0", input_type="search_document")
        return resp.embeddings[0]
