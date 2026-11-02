"""Provider interfaces. Everything model-related goes through these so models are swappable."""

from typing import Protocol


class Embedder(Protocol):
    def embed(self, texts: list[str]) -> list[list[float]]: ...


class LLMProvider(Protocol):
    def generate(
        self,
        messages: list[dict],
        *,
        json_schema: dict | None = None,
        temperature: float = 0.0,
    ) -> str: ...


class Reranker(Protocol):
    def score(self, query: str, texts: list[str]) -> list[float]:
        """Relevance of each text to the query, in [0, 1]; higher is more relevant."""
        ...
