"""Hybrid retrieval: vector + keyword search, fused with Reciprocal Rank Fusion, then reranked."""

import re
import time
from dataclasses import dataclass, field
from typing import Literal

from sqlalchemy import select, text
from sqlalchemy.orm import Session

from app.db.models import Chunk, Document, Vehicle
from app.services.chunking import normalize_for_match
from app.services.providers import Embedder, Reranker

RRF_K = 60

# Common question words (English + Filipino). The 'simple' text config keeps every word,
# so without this list "ano", "the", "ng" would match almost every chunk.
STOPWORDS = frozenset(
    """a an the of for to in on is are was were be been what who whom when where which why how
    much many does do did and or with by at from this that these those it its has have had there
    me my i you your can could please show tell give about any all
    ang ng sa na si ni mga ano sino kanino kailan saan magkano ilan ba yung iyong nang at ay ko mo
    niya kung para po naman lang din rin pa may meron mayroon wala dito nito yan iyan ito kay nina
    """.split()
)

# Identifiers that full-text parsing splits badly: plates ("NBC 1234", "NBC-1234"),
# and any token containing a digit (VIN, OR no., policy no., amounts).
_PLATE_RE = re.compile(r"\b[A-Za-z]{2,3}[\s-]?\d{3,5}\b")
_IDENT_RE = re.compile(r"\b(?=[A-Za-z0-9-]*\d)[A-Za-z0-9][A-Za-z0-9-]{3,}\b")
_WORD_RE = re.compile(r"[a-z0-9]+")


@dataclass
class Candidate:
    chunk_id: int
    document_id: int
    filename: str
    doc_type: str
    vehicle_id: int | None
    page: int
    text: str
    vector_sim: float | None = None
    vector_rank: int | None = None
    keyword_score: float | None = None
    keyword_rank: int | None = None
    rrf: float = 0.0
    rerank: float | None = None
    ocr_confidence: float | None = None  # set if the chunk was read from a photo/scan
    identifier_matches: int = 0  # plates/VINs/numbers from the question found in this chunk

    @property
    def body(self) -> str:
        """Chunk text without the context header line."""
        return self.text.split("\n", 1)[1] if "\n" in self.text else self.text


@dataclass
class RetrievalConfig:
    mode: Literal["vector", "hybrid"] = "hybrid"
    top_k: int = 20
    final_k: int = 5
    rerank: bool = True
    rerank_candidates: int = 12  # only the best fused results are reranked


@dataclass
class RetrievalResult:
    candidates: list[Candidate]  # final top-k, best first
    confidence: float  # best rerank score, or best cosine similarity if not reranked
    reranked: bool
    timings: dict[str, float] = field(default_factory=dict)


def keyword_terms(question: str) -> list[str]:
    """Lowercased content words for the OR full-text query."""
    words = _WORD_RE.findall(question.lower())
    return list(dict.fromkeys(w for w in words if len(w) > 1 and w not in STOPWORDS))


def identifier_terms(question: str) -> list[str]:
    """Normalized identifiers (plates, VINs, numbers) to match against chunks.text_norm."""
    found = _PLATE_RE.findall(question) + _IDENT_RE.findall(question)
    norms = (normalize_for_match(f) for f in found)
    return list(dict.fromkeys(n for n in norms if len(n) >= 4))


# Model names that are also everyday words; they only count when the make is named too.
AMBIGUOUS_MODEL_NAMES = frozenset({"city", "star", "life", "jazz", "civic", "cross"})


def resolve_vehicle(db: Session, question: str) -> int | None:
    """Return the id of the single vehicle the question names, or None.

    A vehicle is named by plate or VIN (any spacing/hyphens), or by model name
    ("the Vios", "Mirage G4"). If zero or several vehicles match, returns None
    so the search stays unscoped (e.g. questions comparing two cars).
    """
    q_norm = normalize_for_match(question)
    q_lower = question.lower()
    matches: set[int] = set()
    for v in db.scalars(select(Vehicle)):
        if v.plate and normalize_for_match(v.plate) in q_norm:
            matches.add(v.id)
        elif v.vin and len(v.vin) >= 8 and normalize_for_match(v.vin) in q_norm:
            matches.add(v.id)
        else:
            model = v.model.lower().split()[0]
            make = v.make.lower()
            named = re.search(rf"\b{re.escape(model)}\b", q_lower)
            with_make = re.search(rf"\b{re.escape(make)}\s+{re.escape(model)}\b", q_lower)
            if named and (model not in AMBIGUOUS_MODEL_NAMES or with_make):
                matches.add(v.id)
    return matches.pop() if len(matches) == 1 else None


def vehicle_named(db: Session, question: str) -> str | None:
    """How the question names a vehicle ("Toyota Vios NBC 1234", or a plate written in an
    uploaded document such as an unlinked CR photo), or None if it names none."""
    vehicle_id = resolve_vehicle(db, question)
    if vehicle_id is not None:
        v = db.get(Vehicle, vehicle_id)
        return " ".join(p for p in (v.make, v.model, v.plate) if p)
    for plate in _PLATE_RE.findall(question):
        norm = normalize_for_match(plate)
        found = db.scalar(
            select(Chunk.id).where(Chunk.text_norm.contains(norm, autoescape=True)).limit(1)
        )
        if found is not None:
            return plate.upper()
    return None


def _filters(query, vehicle_id: int | None, doc_type: str | None):
    query = query.where(Document.status == "ready")
    if vehicle_id is not None:
        query = query.where(Document.vehicle_id == vehicle_id)
    if doc_type is not None:
        query = query.where(Document.doc_type == doc_type)
    return query


def _candidate(row) -> Candidate:
    return Candidate(
        chunk_id=row.id,
        document_id=row.document_id,
        filename=row.filename,
        doc_type=row.doc_type,
        vehicle_id=row.vehicle_id,
        page=row.page,
        text=row.text,
        ocr_confidence=row.ocr_confidence,
    )


def vector_search(
    db: Session, query_vec: list[float], k: int, vehicle_id=None, doc_type=None
) -> list[Candidate]:
    distance = Chunk.embedding.cosine_distance(query_vec).label("distance")
    query = (
        select(
            Chunk.id,
            Chunk.document_id,
            Chunk.page,
            Chunk.text,
            Chunk.ocr_confidence,
            Document.filename,
            Document.doc_type,
            Document.vehicle_id,
            distance,
        )
        .join(Document, Document.id == Chunk.document_id)
        .order_by(distance)
        .limit(k)
    )
    results = []
    for rank, row in enumerate(db.execute(_filters(query, vehicle_id, doc_type)), start=1):
        c = _candidate(row)
        c.vector_sim, c.vector_rank = 1.0 - float(row.distance), rank
        results.append(c)
    return results


_KEYWORD_SQL = """
SELECT c.id, c.document_id, c.page, c.text, c.ocr_confidence, d.filename, d.doc_type,
       d.vehicle_id,
       ts_rank_cd(c.tsv, q.query) + ids.matches AS score, ids.matches AS id_matches
FROM chunks c
JOIN documents d ON d.id = c.document_id
CROSS JOIN (SELECT CASE WHEN :tsq = '' THEN NULL ELSE to_tsquery('simple', :tsq) END AS query) q
CROSS JOIN LATERAL (
    SELECT count(*) AS matches FROM unnest(CAST(:ids AS text[])) AS ident
    WHERE c.text_norm LIKE '%' || ident || '%'
) ids
WHERE d.status = 'ready'
  AND (CAST(:vehicle_id AS int) IS NULL OR d.vehicle_id = :vehicle_id)
  AND (CAST(:doc_type AS text) IS NULL OR d.doc_type = :doc_type)
  AND ((q.query IS NOT NULL AND c.tsv @@ q.query) OR ids.matches > 0)
ORDER BY score DESC, c.id
LIMIT :k
"""


def keyword_search(
    db: Session, question: str, k: int, vehicle_id=None, doc_type=None
) -> list[Candidate]:
    """Full-text OR query over content words, plus +1 per identifier found in text_norm.

    Identifier matches dominate the score on purpose: if the question names a plate
    or VIN, chunks containing it should rank first.
    """
    terms, idents = keyword_terms(question), identifier_terms(question)
    if not terms and not idents:
        return []
    rows = db.execute(
        text(_KEYWORD_SQL),
        {
            "tsq": " | ".join(terms),  # terms are [a-z0-9]+ only, so safe for to_tsquery
            "ids": idents,
            "vehicle_id": vehicle_id,
            "doc_type": doc_type,
            "k": k,
        },
    )
    results = []
    for rank, row in enumerate(rows, start=1):
        c = _candidate(row)
        c.keyword_score, c.keyword_rank = float(row.score), rank
        c.identifier_matches = int(row.id_matches)
        results.append(c)
    return results


def reciprocal_rank_fusion(*ranked_lists: list[Candidate], k: int = RRF_K) -> list[Candidate]:
    """Merge ranked lists: score = sum of 1 / (k + rank). Keeps per-list ranks and scores."""
    merged: dict[int, Candidate] = {}
    for ranked in ranked_lists:
        for rank, cand in enumerate(ranked, start=1):
            existing = merged.setdefault(cand.chunk_id, cand)
            if existing is not cand:
                for attr in ("vector_sim", "vector_rank", "keyword_score", "keyword_rank"):
                    if getattr(existing, attr) is None:
                        setattr(existing, attr, getattr(cand, attr))
                existing.identifier_matches = max(
                    existing.identifier_matches, cand.identifier_matches
                )
            existing.rrf += 1.0 / (k + rank)
    return sorted(merged.values(), key=lambda c: c.rrf, reverse=True)


def retrieve(
    db: Session,
    question: str,
    embedder: Embedder,
    reranker: Reranker | None,
    config: RetrievalConfig,
    *,
    vehicle_id: int | None = None,
    doc_type: str | None = None,
    alt_queries: list[str] | None = None,
) -> RetrievalResult:
    """Run the retrieval half of the pipeline and report a confidence for the abstain check.

    `alt_queries` (e.g. an English translation of a Taglish question) add keyword terms,
    and each candidate keeps its best rerank score over the question and its alternatives.
    """
    queries = [question, *(q for q in alt_queries or [] if q and q != question)]
    timings: dict[str, float] = {}
    t0 = time.perf_counter()
    query_vec = embedder.embed([question])[0]
    timings["embed"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    vector_hits = vector_search(db, query_vec, config.top_k, vehicle_id, doc_type)
    if config.mode == "hybrid":
        keyword_hits = keyword_search(db, " ".join(queries), config.top_k, vehicle_id, doc_type)
        pool = reciprocal_rank_fusion(vector_hits, keyword_hits)
    else:
        pool = reciprocal_rank_fusion(vector_hits)
    timings["search"] = time.perf_counter() - t0

    best_vector_sim = max((c.vector_sim for c in pool if c.vector_sim is not None), default=0.0)
    use_rerank = config.rerank and reranker is not None and bool(pool)
    if use_rerank:
        t0 = time.perf_counter()
        # Reranking is the slowest step; RRF already ranks the right chunk near the top.
        pool = pool[: config.rerank_candidates]
        texts = [c.text for c in pool]
        per_query = [reranker.score(q, texts) for q in queries]
        for i, cand in enumerate(pool):
            cand.rerank = max(scores[i] for scores in per_query)
        pool.sort(key=lambda c: c.rerank, reverse=True)
        timings["rerank"] = time.perf_counter() - t0
        confidence = pool[0].rerank
    else:
        confidence = best_vector_sim

    return RetrievalResult(
        candidates=pool[: config.final_k],
        confidence=confidence,
        reranked=use_rerank,
        timings=timings,
    )
