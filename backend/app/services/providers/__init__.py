import logging
import threading
from functools import lru_cache

from app.config import get_settings
from app.services.providers.base import Embedder, LLMProvider, Reranker
from app.services.providers.cross_encoder import CrossEncoderReranker
from app.services.providers.ollama import OllamaEmbedder, OllamaLLM

__all__ = [
    "Embedder",
    "LLMProvider",
    "Reranker",
    "get_embedder",
    "get_llm",
    "get_reranker",
    "warm_up_in_background",
]

logger = logging.getLogger(__name__)


@lru_cache
def get_embedder() -> Embedder:
    s = get_settings()
    return OllamaEmbedder(s.ollama_url, s.embed_model)


@lru_cache
def get_llm() -> LLMProvider:
    s = get_settings()
    return OllamaLLM(s.ollama_url, s.llm_model, timeout=s.llm_timeout_s)


@lru_cache
def get_reranker() -> Reranker | None:
    """The configured reranker, or None when RERANK_ENABLED=false."""
    s = get_settings()
    if not s.rerank_enabled:
        return None
    return CrossEncoderReranker(s.reranker_model, s.reranker_device)


def _warm_up() -> None:
    # Loading the embedder, reranker and LLM took ~60 s, all paid by the first question.
    steps = [
        ("embedder", lambda: get_embedder().embed(["warm-up"])),
        ("reranker", lambda: (r := get_reranker()) and r.score("warm-up", ["warm-up"])),
        ("LLM", lambda: getattr(get_llm(), "load", lambda: None)()),
    ]
    for name, step in steps:
        try:
            step()
            logger.info("Warmed up the %s", name)
        except Exception as exc:  # e.g. Ollama not running yet: the first question loads it
            # Stop here: without Ollama no question can be answered, so the 2 GB reranker
            # download can wait (this also keeps the CI smoke test, which has no Ollama, light).
            logger.warning("Could not warm up the %s: %s", name, exc)
            return


def warm_up_in_background() -> None:
    """Load the models at startup so the first question doesn't wait for them."""
    threading.Thread(target=_warm_up, name="warm-up", daemon=True).start()
