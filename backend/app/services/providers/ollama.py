"""Ollama implementations of the provider interfaces (local, free)."""

import httpx

# Keep models in GPU memory between questions; Ollama's default (5 min) meant the first
# question after a short break waited for the model to reload.
KEEP_ALIVE = "30m"


class OllamaEmbedder:
    def __init__(self, base_url: str, model: str, timeout: float = 120.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts. Returns one vector per input, in order."""
        if not texts:
            return []
        resp = httpx.post(
            f"{self.base_url}/api/embed",
            json={"model": self.model, "input": texts, "keep_alive": KEEP_ALIVE},
            timeout=self.timeout,
        )
        resp.raise_for_status()
        return resp.json()["embeddings"]


class OllamaLLM:
    def __init__(
        self, base_url: str, model: str, timeout: float = 120.0, max_tokens: int = 512
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout
        self.max_tokens = max_tokens  # answers are 1-2 sentences; stops runaway generation

    def generate(
        self,
        messages: list[dict],
        *,
        json_schema: dict | None = None,
        temperature: float = 0.0,
    ) -> str:
        """Run a non-streaming chat completion. With json_schema, output is constrained to it."""
        payload: dict = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "keep_alive": KEEP_ALIVE,
            "options": {"temperature": temperature, "num_predict": self.max_tokens},
        }
        if json_schema is not None:
            payload["format"] = json_schema
        resp = httpx.post(f"{self.base_url}/api/chat", json=payload, timeout=self.timeout)
        resp.raise_for_status()
        return resp.json()["message"]["content"]

    def load(self) -> None:
        """Load the model into memory without generating (a chat request with no messages)."""
        payload = {"model": self.model, "messages": [], "keep_alive": KEEP_ALIVE}
        resp = httpx.post(f"{self.base_url}/api/chat", json=payload, timeout=self.timeout)
        resp.raise_for_status()
