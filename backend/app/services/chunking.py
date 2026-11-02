"""Split page text into overlapping chunks and build the per-chunk context header."""

import re

# A "word" plus its trailing whitespace, so joining words reproduces the original text exactly.
# That matters later: quote verification checks quotes verbatim against chunk text.
_WORD_RE = re.compile(r"\S+\s*")
_NORM_RE = re.compile(r"[\s\-]+")


def chunk_text(text: str, max_words: int = 300, overlap: int = 40) -> list[str]:
    """Split one page of text into chunks of at most ``max_words`` words.

    Consecutive chunks share ``overlap`` words. Chunks never span pages because
    callers pass a single page at a time.
    """
    if overlap >= max_words:
        raise ValueError("overlap must be smaller than max_words")
    words = _WORD_RE.findall(text.strip())
    if not words:
        return []

    step = max_words - overlap
    chunks: list[str] = []
    for start in range(0, len(words), step):
        chunks.append("".join(words[start : start + max_words]).strip())
        if start + max_words >= len(words):
            break
    return chunks


def context_header(
    *, plate: str | None, make: str | None, model: str | None, doc_type: str, page: int
) -> str:
    """Header prepended to each chunk so retrieval and the LLM see which vehicle/doc it is from."""
    # "none" made the LLM treat an unlinked document as being about no vehicle at all.
    vehicle = " ".join(p for p in (plate, make, model) if p) or "not linked"
    return f"Vehicle: {vehicle} | Doc: {doc_type} | Page {page}"


def normalize_for_match(text: str) -> str:
    """Lowercase and strip spaces/hyphens so 'ABC 1234', 'ABC-1234' and 'abc1234' compare equal."""
    return _NORM_RE.sub("", text).lower()
