"""Normalize uploads: PDFs pass through, photos become a one-page PDF.

Storing everything as PDF keeps one code path for ingestion, page citations and the
"open page N" viewer, whether staff upload a PDF or snap the OR/CR with a phone.
"""

import io

import pymupdf
from PIL import Image, ImageOps, UnidentifiedImageError
from pillow_heif import register_heif_opener

register_heif_opener()  # lets Pillow open iPhone HEIC photos

SUPPORTED_DESCRIPTION = "a PDF or a photo (JPG, PNG, WEBP or HEIC)"
MAX_IMAGE_SIDE = 3500  # px; larger phone photos are scaled down, OCR doesn't need more
IMAGE_DPI = 300  # page size given to photos, so rendering at 300 DPI keeps native pixels


class UnsupportedFile(ValueError):
    """The upload is not a PDF or a readable image."""


def sniff(data: bytes) -> str | None:
    """Identify the file type from its first bytes (never trust the file name)."""
    if data.startswith(b"%PDF"):
        return "pdf"
    if data.startswith(b"\xff\xd8\xff"):
        return "jpeg"
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return "png"
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "webp"
    if data[4:8] == b"ftyp" and data[8:12] in (b"heic", b"heix", b"mif1", b"msf1", b"heim"):
        return "heic"
    return None


def to_pdf(data: bytes) -> bytes:
    """Return PDF bytes for an upload. Raises UnsupportedFile for anything else."""
    kind = sniff(data)
    if kind == "pdf":
        return data
    if kind is None:
        raise UnsupportedFile(f"Unsupported file. Upload {SUPPORTED_DESCRIPTION}.")
    try:
        image = Image.open(io.BytesIO(data))
        image = ImageOps.exif_transpose(image)  # phones store rotation in EXIF
        image = image.convert("RGB")
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise UnsupportedFile(f"Could not read the image: {exc}") from exc

    image.thumbnail((MAX_IMAGE_SIDE, MAX_IMAGE_SIDE))
    jpeg = io.BytesIO()
    image.save(jpeg, format="JPEG", quality=90)

    pdf = pymupdf.open()
    width_pt = image.width * 72 / IMAGE_DPI
    height_pt = image.height * 72 / IMAGE_DPI
    page = pdf.new_page(width=width_pt, height=height_pt)
    page.insert_image(page.rect, stream=jpeg.getvalue())
    out = pdf.tobytes(garbage=3, deflate=True)
    pdf.close()
    return out
