"""OCR tests. The ones that read text need the tesseract binary (Docker image and CI have
it); they are skipped elsewhere. Upload/API behavior is tested without it too."""

import io

import pymupdf
import pytest
from PIL import Image
from sqlalchemy import select

from app.db.models import Chunk
from app.services import ingestion
from app.services.files import to_pdf
from app.services.ocr import ocr_available, ocr_page
from app.services.verification import quote_in_excerpts

needs_tesseract = pytest.mark.skipif(not ocr_available(), reason="tesseract not installed")

CR_TEXT = (
    "CERTIFICATE OF REGISTRATION\n"
    "Owner: JUAN P. DELA CRUZ\n"
    "Plate No.: NBC 1234\n"
    "Engine No.: 2NR1321959\n"
    "Registration valid until: March 31, 2025"
)


def photo_of_text(text: str, *, tilt: float = 0.0, sideways: bool = False) -> bytes:
    """Render text on a page, then rasterize it like a phone photo (JPEG, no text layer)."""
    pdf = pymupdf.open()
    page = pdf.new_page(width=595, height=842)
    page.insert_textbox(pymupdf.Rect(60, 60, 540, 400), text, fontsize=13, fontname="helv")
    pix = page.get_pixmap(dpi=200)
    image = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
    image = image.rotate(tilt, expand=True, fillcolor=(90, 75, 60))
    if sideways:
        image = image.rotate(90, expand=True)
    buf = io.BytesIO()
    image.save(buf, format="JPEG", quality=80)
    return buf.getvalue()


def ocr_text(jpeg: bytes) -> str:
    page = pymupdf.open(stream=to_pdf(jpeg), filetype="pdf")[0]
    return ocr_page(page).text


@needs_tesseract
def test_reads_a_straight_photo():
    text = ocr_text(photo_of_text(CR_TEXT))
    assert quote_in_excerpts("Plate No.: NBC 1234", [text])
    assert quote_in_excerpts("Owner: JUAN P. DELA CRUZ", [text])


@needs_tesseract
def test_straightens_a_tilted_photo():
    text = ocr_text(photo_of_text(CR_TEXT, tilt=6))
    assert quote_in_excerpts("Engine No.: 2NR1321959", [text])


@needs_tesseract
def test_turns_a_sideways_photo_upright():
    text = ocr_text(photo_of_text(CR_TEXT, sideways=True))
    assert quote_in_excerpts("Registration valid until: March 31, 2025", [text])


@needs_tesseract
def test_photo_upload_is_ocrd_and_searchable(client, staff_headers, db):
    resp = client.post(
        "/documents",
        files={"file": ("orcr_photo.jpg", photo_of_text(CR_TEXT, tilt=3), "image/jpeg")},
        data={"doc_type": "or_cr"},
        headers=staff_headers,
    )
    assert resp.status_code == 202, resp.text
    doc = client.get(f"/documents/{resp.json()['id']}", headers=staff_headers).json()
    assert doc["status"] == "ready", doc["error"]
    assert doc["filename"] == "orcr_photo.jpg"
    assert 50 < doc["ocr_confidence"] <= 100

    chunk = db.scalar(select(Chunk).where(Chunk.document_id == doc["id"]))
    assert chunk.ocr_confidence == doc["ocr_confidence"]
    assert "nbc1234" in chunk.text_norm  # plate search works on OCR'd text


def test_photo_upload_without_tesseract_fails_with_a_clear_message(
    client, staff_headers, monkeypatch
):
    monkeypatch.setattr(ingestion, "ocr_available", lambda: False)
    resp = client.post(
        "/documents",
        files={"file": ("orcr.png", photo_of_text(CR_TEXT), "image/png")},
        data={"doc_type": "or_cr"},
        headers=staff_headers,
    )
    assert resp.status_code == 202  # photos are accepted...
    doc = client.get(f"/documents/{resp.json()['id']}", headers=staff_headers).json()
    assert doc["status"] == "failed"  # ...but can't be read without OCR, and it says why
    assert "Tesseract" in doc["error"]


def test_text_pdfs_are_not_ocrd(client, staff_headers, monkeypatch):
    def fail(*args, **kwargs):
        raise AssertionError("OCR should not run on a page that has a text layer")

    monkeypatch.setattr(ingestion, "ocr_page", fail)
    pdf = pymupdf.open()
    pdf.new_page().insert_text((72, 72), CR_TEXT, fontsize=11)
    resp = client.post(
        "/documents",
        files={"file": ("orcr.pdf", pdf.tobytes(), "application/pdf")},
        data={"doc_type": "or_cr"},
        headers=staff_headers,
    )
    doc = client.get(f"/documents/{resp.json()['id']}", headers=staff_headers).json()
    assert doc["status"] == "ready"
    assert doc["ocr_confidence"] is None


@pytest.mark.parametrize("tilt", [-8.0, -2.5, 0.0, 4.0, 6.0, 11.0])
def test_estimates_page_tilt_even_on_a_dark_desk(tilt):
    import numpy as np

    from app.services.ocr import estimate_skew

    image = Image.open(io.BytesIO(photo_of_text(CR_TEXT * 3, tilt=tilt))).convert("L")
    # PIL rotates counter-clockwise; undoing it means rotating back by the same amount.
    assert estimate_skew(np.asarray(image)) == pytest.approx(-tilt, abs=0.5)


def lto_style_form() -> bytes:
    """A ruled form like an LTO CR: light-blue background, blue grid, small labels above
    bold values in each box. Rasterized to a JPEG photo (no text layer)."""
    cells = [
        ("PLATE NO.", "NBC 1234"), ("ENGINE NO.", "2NR1321959"), ("MAKE/BRAND", "TOYOTA"),
        ("COLOR", "SILVER METALLIC"), ("BODY TYPE", "SEDAN"), ("YEAR MODEL", "2019"),
        ("OWNER'S NAME", "JUAN P. DELA CRUZ"), ("O.R. NO.", "8820391457"),
        ("AMOUNT", "PHP 1,820.00"),
    ]  # fmt: skip
    pdf = pymupdf.open()
    page = pdf.new_page(width=595, height=842)
    blue, light = (0.15, 0.2, 0.55), (0.86, 0.9, 0.98)
    page.draw_rect(page.rect, color=None, fill=light)
    for i, (label, value) in enumerate(cells):
        row, col = divmod(i, 3)
        rect = pymupdf.Rect(
            40 + col * 175, 120 + row * 70, 40 + (col + 1) * 175, 120 + (row + 1) * 70
        )
        page.draw_rect(rect, color=blue, width=1.5)
        page.insert_text(
            (rect.x0 + 6, rect.y0 + 14), label, fontsize=7, fontname="helv", color=blue
        )
        page.insert_text((rect.x0 + 14, rect.y0 + 45), value, fontsize=12, fontname="hebo")
    pix = page.get_pixmap(dpi=200)
    image = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
    buf = io.BytesIO()
    image.save(buf, format="JPEG", quality=85)
    return buf.getvalue()


def test_finds_the_boxes_of_a_ruled_form():
    import numpy as np

    from app.services.ocr import find_form_cells

    page = pymupdf.open(stream=to_pdf(lto_style_form()), filetype="pdf")[0]
    pix = page.get_pixmap(dpi=300, colorspace=pymupdf.csGRAY)
    gray = np.frombuffer(pix.samples, np.uint8).reshape(pix.height, pix.width)
    assert len(find_form_cells(gray)) == 9
    plain_text = pymupdf.open()  # a page without a grid is not treated as a form
    plain_text.new_page().insert_text((72, 72), CR_TEXT, fontsize=11)
    pix = plain_text[0].get_pixmap(dpi=300, colorspace=pymupdf.csGRAY)
    assert (
        find_form_cells(np.frombuffer(pix.samples, np.uint8).reshape(pix.height, pix.width)) == []
    )


@needs_tesseract
def test_reads_each_form_box_as_label_and_value():
    page = pymupdf.open(stream=to_pdf(lto_style_form()), filetype="pdf")[0]
    lines = ocr_page(page).text.splitlines()

    def line_with(value: str) -> str:
        return next((ln.upper() for ln in lines if value in ln), "")

    # Each value comes out on the same line as its label (small OCR slips in labels,
    # like "C.R. NO" for "O.R. NO", are fine: what matters is the pairing).
    assert "PLATE" in line_with("NBC 1234")
    assert "ENGINE" in line_with("2NR1321959")
    assert "OWNER" in line_with("JUAN P. DELA CRUZ")
    assert "R. NO" in line_with("8820391457")
    assert "AMOUNT" in line_with("1,820.00")


@pytest.mark.parametrize(
    ("text", "confs", "noise"),
    [
        ("M NAL YAYAIN NAKA AK", [30, 35, 28, 31, 25], True),  # security pattern
        ("— SS | ——— =", [60, 60, 60, 60, 60], True),  # mostly symbols
        ("PLATE NO.: XYZ7788", [91, 90, 95], False),
        ("Odometer: 58,903 km", [88, 93, 90], False),
    ],
)
def test_noise_lines_are_dropped(text, confs, noise):
    from app.services.ocr import is_noise

    words = text.split()[: len(confs)]
    assert is_noise(text, confs[: len(words)], [len(w) for w in words]) is noise


@pytest.mark.parametrize(
    ("texts", "expected"),
    [
        (["CHASSIS NO.", "9XTRK4LP2HKD 40527"], "CHASSIS NO.: 9XTRK4LP2HKD40527"),
        (["PLATE NO,", "XYZ 7788"], "PLATE NO,: XYZ 7788"),  # plates keep their space
        (["O.R. NO.", "0003055 48207419"], "O.R. NO.: 000305548207419"),
        (["OWNER'S NAME", "JUAN P. DELA CRUZ"], "OWNER'S NAME: JUAN P. DELA CRUZ"),  # not an ID
        (["COLOR", "PEARL WHITE"], "COLOR: PEARL WHITE"),
        (["REGISTRANT'S SIGNATURE"], "REGISTRANT'S SIGNATURE"),
        (["YEAR MODEL", "{New/Used imported CBU)", "2016"], "YEAR MODEL: 2016"),  # hint dropped
        (["YEAR REBUILT", "(For Rebuilt MVs)"], "YEAR REBUILT: (For Rebuilt MVs)"),  # no value
    ],
)
def test_form_cells_become_label_value_lines(texts, expected):
    from app.services.ocr import format_cell

    assert format_cell(texts) == expected


@needs_tesseract
def test_reads_a_line_on_curled_paper():
    """The bottom line slopes 4° more than the rest, like the bottom of a curled photo."""
    body = photo_of_text(CR_TEXT)
    page_img = Image.open(io.BytesIO(body)).convert("RGB")
    footer = pymupdf.open()
    fp = footer.new_page(width=595, height=60)
    fp.insert_text((60, 35), "CR No. 0420817396", fontsize=13, fontname="cour")
    pix = fp.get_pixmap(dpi=200)
    strip = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
    strip = strip.rotate(4, expand=True, fillcolor="white")
    canvas = Image.new("RGB", (page_img.width, page_img.height + strip.height + 40), "white")
    canvas.paste(page_img, (0, 0))
    canvas.paste(strip, (0, page_img.height))
    buf = io.BytesIO()
    canvas.save(buf, format="JPEG", quality=85)
    assert "0420817396" in ocr_text(buf.getvalue()).replace(" ", "")
