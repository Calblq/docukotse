import io

import pymupdf
import pytest
from PIL import Image

from app.services.files import UnsupportedFile, sniff, to_pdf


def image_bytes(fmt: str, size=(300, 200), **save_kwargs) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", size, "white").save(buf, format=fmt, **save_kwargs)
    return buf.getvalue()


@pytest.mark.parametrize(("fmt", "kind"), [("JPEG", "jpeg"), ("PNG", "png"), ("WEBP", "webp")])
def test_sniff_detects_images_by_content(fmt, kind):
    assert sniff(image_bytes(fmt)) == kind


def test_sniff_pdf_and_unknown():
    assert sniff(b"%PDF-1.7 ...") == "pdf"
    assert sniff(b"hello world") is None


def test_pdf_passes_through_unchanged():
    data = b"%PDF-1.4\n%fake"
    assert to_pdf(data) is data


@pytest.mark.parametrize("fmt", ["JPEG", "PNG", "WEBP"])
def test_photo_becomes_one_page_pdf_with_the_image(fmt):
    pdf = pymupdf.open(stream=to_pdf(image_bytes(fmt, size=(600, 900))), filetype="pdf")
    assert pdf.page_count == 1
    page = pdf[0]
    assert len(page.get_images()) == 1
    assert page.rect.height > page.rect.width  # portrait photo stays portrait


def test_heic_photo_is_supported():
    heic = image_bytes("HEIF")
    assert sniff(heic) == "heic"
    assert pymupdf.open(stream=to_pdf(heic), filetype="pdf").page_count == 1


def test_exif_rotation_is_applied():
    # A landscape sensor image tagged "rotate 90°" (EXIF orientation 6) is a portrait photo.
    exif = Image.Exif()
    exif[0x0112] = 6
    tagged = image_bytes("JPEG", size=(900, 600), exif=exif.tobytes())
    page = pymupdf.open(stream=to_pdf(tagged), filetype="pdf")[0]
    assert page.rect.height > page.rect.width


def test_huge_photo_is_scaled_down():
    page = pymupdf.open(stream=to_pdf(image_bytes("JPEG", size=(8000, 6000))), filetype="pdf")[0]
    width_px = page.rect.width * 300 / 72
    assert width_px <= 3500 + 1


@pytest.mark.parametrize("data", [b"plain text", b"\xff\xd8\xff garbage that is not a jpeg"])
def test_unsupported_or_broken_files_are_rejected(data):
    with pytest.raises(UnsupportedFile):
        to_pdf(data)
