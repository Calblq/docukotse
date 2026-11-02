import pytest

from app.services.verification import canonical_number, quote_in_excerpts, verify_answer

DEED = (
    "Vehicle: NBC 1234 Toyota Vios | Doc: deed_of_sale | Page 1\n"
    "Seller: Juan Dela Cruz\nBuyer:  Maria Santos\nPrice: PHP 485,000.00\n"
    "Date of sale: September 18, 2024"
)


def test_grounded_answer_passes():
    result = verify_answer(
        "Maria Santos bought it for PHP 485,000.00 on September 18, 2024.",
        "Price: PHP 485,000.00",
        [DEED],
    )
    assert result.ok, result.problems


def test_quote_matching_ignores_punctuation():
    assert quote_in_excerpts("Chassis No / VIN", ["Chassis No. / VIN: MR0FB8CD"])


def test_quote_must_keep_the_words_in_order():
    assert not quote_in_excerpts("Santos Maria Buyer", [DEED])
    assert not quote_in_excerpts("Buyer: Maria Santos Price", ["Buyer: Maria Santos\nSeller"])


def test_quote_matching_ignores_whitespace_case_and_quote_style():
    assert quote_in_excerpts("buyer: maria   santos", [DEED])
    assert quote_in_excerpts("“Buyer: Maria Santos”", [DEED])


def test_quote_matching_ignores_dot_leader_length():
    form = "Vehicle: X | Doc: financing | Page 1\nMonthly amortization ........... PHP 31,764.67"
    assert quote_in_excerpts("Monthly amortization ..... PHP 31,764.67", [form])
    assert quote_in_excerpts("Monthly amortization........PHP 31,764.67", [form])
    assert not quote_in_excerpts("Monthly amortization ..... PHP 31,000.00", [form])


def test_paraphrased_quote_fails():
    result = verify_answer("Maria Santos.", "The buyer is Maria Santos", [DEED])
    assert not result.ok
    assert "quote" in result.problems[0]


def test_empty_quote_fails():
    assert not verify_answer("Maria Santos.", "", [DEED]).ok


@pytest.mark.parametrize("amount", ["PHP 485,000", "485000", "PHP 485,000.00"])
def test_number_formats_match(amount):
    assert verify_answer(f"It sold for {amount}.", "Price: PHP 485,000.00", [DEED]).ok


def test_hallucinated_number_fails():
    result = verify_answer("It sold for PHP 500,000.", "Price: PHP 485,000.00", [DEED])
    assert not result.ok
    assert "500000" in result.problems[0]


def test_wrong_plate_letters_fail():
    result = verify_answer("The plate is NBX 1234.", "Price: PHP 485,000.00", [DEED])
    assert not result.ok
    assert "NBX 1234" in " ".join(result.problems)


def test_filipino_month_name_matches_english_source():
    result = verify_answer(
        "Nabenta noong Setyembre 18, 2024.", "Date of sale: September 18, 2024", [DEED]
    )
    assert result.ok, result.problems


def test_wrong_month_fails():
    result = verify_answer("Sold on October 18, 2024.", "Date of sale: September 18, 2024", [DEED])
    assert not result.ok
    assert "october" in result.problems[0]


def test_filipino_may_is_not_a_month():
    assert verify_answer("May buyer na: Maria Santos.", "Buyer:  Maria Santos", [DEED]).ok


def test_citation_markers_are_not_numbers():
    assert verify_answer("Maria Santos [1].", "Buyer:  Maria Santos", [DEED]).ok


def test_no_excerpts_fails():
    assert not verify_answer("Maria Santos.", "Buyer: Maria Santos", []).ok


@pytest.mark.parametrize(
    ("raw", "expected"),
    [("485,000.00", "485000"), ("6.50", "6.5"), ("0031875", "0031875"), ("12", "12")],
)
def test_canonical_number(raw, expected):
    assert canonical_number(raw) == expected


@pytest.mark.parametrize(
    ("answer", "refusal"),
    [
        ("The document does not provide information about who bought the Honda City.", True),
        ("The document does not explicitly state who bought it. However, X is the owner.", True),
        ("The registration expiry is not mentioned in the excerpts.", True),
        ("Walang impormasyon tungkol dito.", True),
        ("Hindi nakasaad sa dokumento kung sino ang bumili.", True),
        ("", True),
        ("No, flood damage is not covered by the Vios policy.", False),
        ("Maria L. Santos bought it for PHP 485,000.00.", False),
        ("May sira ang ilaw at electrical.", False),
    ],
)
def test_looks_like_refusal(answer, refusal):
    from app.services.generation import looks_like_refusal

    assert looks_like_refusal(answer) is refusal


def test_identifier_answer_must_be_in_the_quote_as_a_whole_token():
    form = "CHASSIS NO.: 9XTRK4LP2HKD40527\nENGINE NO.: HKD40527"
    # Engine number given as the answer, chassis line quoted: caught even though HKD40527
    # is in the excerpt and is a substring of the chassis number.
    wrong = verify_answer("HKD40527", "CHASSIS NO.: 9XTRK4LP2HKD40527", [form])
    assert not wrong.ok
    assert "HKD40527" in wrong.problems[0]
    assert verify_answer("9XTRK4LP2HKD40527", "CHASSIS NO.: 9XTRK4LP2HKD40527", [form]).ok


def test_identifiers_from_the_question_may_be_repeated():
    form = "PLATE NO.: XYZ7788\nOWNER'S NAME: JUAN DELA CRUZ"
    result = verify_answer(
        "XYZ7788 is registered to Juan Dela Cruz.",
        "OWNER'S NAME: JUAN DELA CRUZ",
        [form],
        question="Who owns XYZ7788?",
    )
    assert result.ok, result.problems


def test_ordinary_words_before_numbers_are_not_plates():
    address = "OWNER'S ADDRESS: 245 EXAMPLE STREET, SAMPLE VILLAGE"
    result = verify_answer(
        "The owner lives at 245 Example Street, Sample Village.", "245 EXAMPLE STREET", [address]
    )
    assert result.ok, result.problems


def test_numeric_dates_support_month_names():
    receipt = "OR. DATE: 04/18/2019\nAMOUNT: PHP 6,418.35"
    assert verify_answer("Paid on April 18, 2019.", "OR. DATE: 04/18/2019", [receipt]).ok
    wrong = verify_answer("Paid on June 18, 2019.", "OR. DATE: 04/18/2019", [receipt])
    assert not wrong.ok


def test_multi_line_answers_quote_each_line():
    receipt = "OR. DATE: 04/18/2019\nAMOUNT: PHP 6,418.35"
    both = verify_answer(
        "Paid on 04/18/2019, PHP 6,418.35.",
        "OR. DATE: 04/18/2019 | AMOUNT: PHP 6,418.35",
        [receipt],
    )
    assert both.ok, both.problems
    invented = verify_answer(
        "Paid on 04/18/2019.", "OR. DATE: 04/18/2019 | PAID IN CASH", [receipt]
    )
    assert not invented.ok


def test_quote_lines_joined_by_line_breaks_are_checked_one_by_one():
    cr = "PLATE NO.: XYZ7788\nENGINE NO.: HKD40527\nMAKE/BRAND: NISSAN\nSERIES: PATROL 4X4"
    assert quote_in_excerpts("MAKE/BRAND: NISSAN\nSERIES: PATROL 4X4", [cr])
    assert quote_in_excerpts("PLATE NO.: XYZ7788\nSERIES: PATROL 4X4", [cr])
    assert not quote_in_excerpts("MAKE/BRAND: NISSAN\nSERIES: NAVARA", [cr])
