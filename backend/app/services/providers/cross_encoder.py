"""Cross-encoder reranker (sentence-transformers). Loaded lazily: the model is ~2 GB."""

import logging
import threading

logger = logging.getLogger(__name__)


class CrossEncoderReranker:
    def __init__(self, model_name: str, device: str | None = None) -> None:
        self.model_name = model_name
        self.device = device
        self._model = None
        self._lock = threading.Lock()

    def _load(self, device: str | None = None):
        with self._lock:
            if self._model is None:
                import torch  # heavy imports, only when used
                from sentence_transformers import CrossEncoder

                device = device or self.device or ("cuda" if torch.cuda.is_available() else "cpu")
                # Half precision on the GPU: ~1.1 GB, so it fits next to Ollama's models on an
                # 8 GB card. On CPU it ran 14-28 s per question; on the GPU well under a second.
                kwargs = {"torch_dtype": torch.float16} if device == "cuda" else {}
                self._model = CrossEncoder(
                    self.model_name, device=device, max_length=512, model_kwargs=kwargs
                )
                logger.info("Reranker loaded on %s", device)
        return self._model

    def score(self, query: str, texts: list[str]) -> list[float]:
        if not texts:
            return []
        pairs = [(query, t) for t in texts]
        try:
            scores = self._load().predict(pairs, batch_size=16)
        except RuntimeError as exc:  # e.g. CUDA out of memory: keep answering, on the CPU
            if "cuda" not in str(exc).lower() and "CUDA" not in type(exc).__name__:
                raise
            logger.warning("Reranker failed on the GPU (%s); falling back to the CPU", exc)
            with self._lock:
                self._model = None
            scores = self._load(device="cpu").predict(pairs, batch_size=16)
        # bge-reranker outputs one logit; CrossEncoder applies a sigmoid, giving [0, 1].
        return [float(s) for s in scores]
