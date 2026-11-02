"""Application settings, read from environment variables only."""

from functools import lru_cache

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    database_url: str = "postgresql+psycopg://cardocs:cardocs@localhost:5432/cardocs"

    jwt_secret: str = "dev-only-insecure-secret-change-me-in-env"
    jwt_expire_minutes: int = 480

    s3_endpoint: str = "http://localhost:9000"
    s3_public_endpoint: str = "http://localhost:9000"
    s3_access_key: str = "cardocs"
    s3_secret_key: str = "cardocs-secret"
    s3_bucket: str = "car-docs"

    ollama_url: str = "http://localhost:11434"
    embed_model: str = "bge-m3"  # must output 1024-dim vectors (see db.models.EMBED_DIM)
    llm_model: str = "qwen2.5:7b-instruct"
    llm_timeout_s: float = 120.0  # raise on CPU-only machines, where answers take minutes
    warm_up_models: bool = True  # load the models at startup, not on the first question

    # Chunk sizes are in whitespace-separated words (~1.3 bge-m3 tokens per word),
    # so 300 words is roughly 400 tokens.
    chunk_size_words: int = 300
    chunk_overlap_words: int = 40
    embed_batch_size: int = 32

    max_upload_mb: int = 25
    cors_origins: list[str] = ["http://localhost:3000"]  # the Next.js frontend

    # OCR for photos and scanned pages (pages with fewer text-layer characters than this)
    ocr_enabled: bool = True
    ocr_languages: str = "eng+fil"
    ocr_dpi: int = 300
    ocr_min_text_chars: int = 50

    # Retrieval
    retrieval_top_k: int = 20  # candidates from each of the vector and keyword searches
    final_k: int = 5  # chunks given to the LLM
    rerank_enabled: bool = True
    reranker_model: str = "BAAI/bge-reranker-v2-m3"
    reranker_device: str | None = None  # None = auto (cuda if available)
    rerank_candidates: int = 12  # best fused results sent to the reranker (it's the slow step)

    # Abstain when the best chunk scores below these (tuned on eval/dataset.jsonl)
    abstain_rerank_threshold: float = 0.2
    abstain_vector_threshold: float = 0.5  # cosine similarity, used when reranking is off
    # Lower bar when the top chunk contains a plate/VIN/number named in the question: the
    # reranker underrates terse form text such as "ENGINE NO.: HKD40527".
    abstain_identifier_threshold: float = 0.05


@lru_cache
def get_settings() -> Settings:
    return Settings()
