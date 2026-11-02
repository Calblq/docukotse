import pytest

from app.services.retrieval import (
    Candidate,
    RetrievalConfig,
    identifier_terms,
    keyword_search,
    keyword_terms,
    reciprocal_rank_fusion,
    resolve_vehicle,
    retrieve,
    vector_search,
)
from tests.conftest import FakeReranker


def _cand(chunk_id: int) -> Candidate:
    return Candidate(chunk_id, 1, "f.pdf", "other", None, 1, "text")


def test_keyword_terms_drop_question_words():
    assert keyword_terms("Ano ang deductible ng ABC-5678?") == ["deductible", "abc", "5678"]


def test_identifier_terms_normalize_plates_and_ids():
    assert identifier_terms("plate NBC-1234, OR No. 8820391457") == ["nbc1234", "8820391457"]
    assert identifier_terms("who bought it?") == []


def test_rrf_rewards_agreement_between_lists():
    fused = reciprocal_rank_fusion([_cand(1), _cand(2)], [_cand(2), _cand(3)])
    assert [x.chunk_id for x in fused] == [2, 1, 3]
    assert fused[0].rrf == pytest.approx(1 / 62 + 1 / 61)


@pytest.mark.parametrize("plate", ["NBC 1234", "NBC-1234", "nbc1234"])
def test_keyword_search_matches_plate_in_any_format(db, corpus, plate):
    hits = keyword_search(db, f"documents for {plate}", k=10)
    vios_id = corpus["vehicles"]["vios"].id
    assert hits, "expected keyword hits"
    assert {h.vehicle_id for h in hits[:3]} == {vios_id}


def test_keyword_search_finds_exact_receipt_number(db, corpus):
    hits = keyword_search(db, "OR 8820391457", k=5)
    assert (hits[0].filename, hits[0].page) == ("vios_orcr.pdf", 2)


def test_filters_apply_to_both_searches(db, corpus, fake_embedder):
    city_id = corpus["vehicles"]["city"].id
    vec = fake_embedder.embed(["deductible"])[0]
    assert {c.vehicle_id for c in vector_search(db, vec, 20, vehicle_id=city_id)} == {city_id}
    assert {c.doc_type for c in vector_search(db, vec, 20, doc_type="or_cr")} == {"or_cr"}
    assert keyword_search(db, "NBC 1234", 20, doc_type="insurance") == []


def test_documents_not_ready_are_excluded(db, corpus, fake_embedder):
    corpus["docs"]["city_insurance.pdf"].status = "processing"
    db.commit()
    vec = fake_embedder.embed(["x"])[0]
    assert all(c.filename != "city_insurance.pdf" for c in vector_search(db, vec, 20))
    hits = keyword_search(db, "SMI-MC-2024-0057210", 20)
    assert all(c.filename != "city_insurance.pdf" for c in hits)


def test_hybrid_retrieve_with_rerank_puts_relevant_chunk_first(db, corpus, fake_embedder):
    result = retrieve(
        db, "deductible for ABC 5678", fake_embedder, FakeReranker(), RetrievalConfig()
    )
    assert result.reranked
    assert result.candidates[0].filename == "city_insurance.pdf"
    assert result.confidence == result.candidates[0].rerank
    assert len(result.candidates) <= 5


def test_vector_only_mode_skips_keyword_search(db, corpus, fake_embedder):
    result = retrieve(
        db, "8820391457", fake_embedder, None, RetrievalConfig(mode="vector", rerank=False)
    )
    assert not result.reranked
    assert all(c.keyword_rank is None for c in result.candidates)


@pytest.mark.parametrize(
    ("question", "expected"),
    [
        ("Who bought the Vios?", "vios"),
        ("deductible of abc-5678", "city"),
        ("Is the Honda City insured?", "city"),
        ("What is the policy number of the Mirage?", "mirage"),
        ("Kanino nakapangalan ang DAE9012?", "mirage"),
        ("Which cars are registered in Quezon City?", None),  # 'City' alone is ambiguous
        ("Compare the Vios and the Mirage", None),  # two vehicles: don't scope
        ("Who is the insurer?", None),
    ],
)
def test_resolve_vehicle(db, corpus, question, expected):
    got = resolve_vehicle(db, question)
    assert got == (corpus["vehicles"][expected].id if expected else None)
