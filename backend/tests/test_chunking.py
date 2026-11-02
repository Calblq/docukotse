import pytest

from app.schemas import canonical_plate
from app.services.chunking import chunk_text, context_header, normalize_for_match


def words(n: int) -> str:
    return " ".join(f"w{i}" for i in range(n))


def test_short_text_is_one_chunk():
    assert chunk_text("Plate No: ABC 1234\nMake: Toyota") == ["Plate No: ABC 1234\nMake: Toyota"]


def test_empty_text_gives_no_chunks():
    assert chunk_text("   \n\n ") == []


def test_long_text_splits_with_overlap():
    chunks = chunk_text(words(250), max_words=100, overlap=20)
    assert [len(c.split()) for c in chunks] == [100, 100, 90]
    # Last 20 words of a chunk start the next one
    assert chunks[0].split()[-20:] == chunks[1].split()[:20]
    assert chunks[1].split()[-20:] == chunks[2].split()[:20]


def test_chunks_cover_every_word():
    chunks = chunk_text(words(250), max_words=100, overlap=20)
    covered = {w for c in chunks for w in c.split()}
    assert covered == set(words(250).split())


def test_exact_fit_has_no_tiny_trailing_chunk():
    assert len(chunk_text(words(100), max_words=100, overlap=20)) == 1


def test_original_whitespace_is_preserved():
    text = "Seller: Juan Dela Cruz\nBuyer:  Maria Santos"
    assert chunk_text(text) == [text]


def test_overlap_must_be_smaller_than_size():
    with pytest.raises(ValueError):
        chunk_text("a b c", max_words=10, overlap=10)


def test_context_header():
    header = context_header(plate="ABC 1234", make="Toyota", model="Vios", doc_type="or_cr", page=2)
    assert header == "Vehicle: ABC 1234 Toyota Vios | Doc: or_cr | Page 2"
    assert context_header(plate=None, make=None, model=None, doc_type="id", page=1).startswith(
        "Vehicle: not linked |"
    )


@pytest.mark.parametrize("plate", ["ABC 1234", "ABC-1234", "abc1234", "ABC  - 1234"])
def test_plate_variants_normalize_equal(plate):
    assert normalize_for_match(plate) == "abc1234"
    assert canonical_plate(plate) == "ABC 1234"
