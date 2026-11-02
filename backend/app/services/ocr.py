"""OCR for pages without a text layer: photos of OR/CRs, scanned deeds, faxes.

Pipeline per page: render in color → fix 90/180/270° rotation (Tesseract OSD) → straighten
small skew → Tesseract passes, merged (later passes add only lines not already present):

1. ruled forms only: each box read on its own, as "LABEL: value" (see read_form_cells);
2. grayscale, automatic layout (psm 3): paragraphs, and text printed in color;
3. "color-free" image with ruled lines removed, single block (psm 6);
4. the same image, sparse text (psm 11): stray values outside boxes;
5. curled paper: tiles tilted differently from the rest of the page, re-read straightened.

LTO forms drove this design: the OR/CR is a grid of boxed values on a blue security
pattern. Read like a letter (pass 2 alone), the labels survive but the values don't; passes
3-4 recover values but not which label they belong to; pass 1 keeps the pairs together
("ENGINE NO.: HKD40527").

The confidence (0-100, mean word confidence) is stored so the UI can flag low-quality scans.
"""

import logging
import re
import shutil
from dataclasses import dataclass
from difflib import SequenceMatcher
from functools import lru_cache

import cv2
import numpy as np
import pymupdf
import pytesseract

logger = logging.getLogger(__name__)

_ROTATE_RE = re.compile(r"Rotate:\s+(\d+)")
_OSD_MIN_CONFIDENCE = 1.5  # Tesseract's orientation confidence; below this, keep as is
_TURNS = {90: cv2.ROTATE_90_CLOCKWISE, 180: cv2.ROTATE_180, 270: cv2.ROTATE_90_COUNTERCLOCKWISE}
_NORM_RE = re.compile(r"[\W_]+")
# Symbols Tesseract outputs for letters on low-contrast print.
_SYMBOL_FIXES = str.maketrans({"©": "C", "®": "R"})
MIN_FORM_CELLS = 6  # fewer boxes than this: treat the page as ordinary text
NOISE_MIN_CONFIDENCE = 45.0  # lines below this mean word confidence are dropped as noise
STRICT_MIN_ALNUM = 6  # extra passes: shorter lines are almost always pattern fragments
KNOWN_COVERAGE = 0.8  # share of a word's characters matching a kept word to count as seen
# Boxes holding an identifier: OCR often splits them ("9XTRK4LP2HKD 40527"), so spaces
# between all-caps/digit tokens are removed. Not plates: "NBC 1234" is written with a space.
_ID_LABEL_RE = re.compile(r"\b(CHASSIS|ENGINE|FILE|VIN|MV|O\.?\s?R|C\.?\s?R)\b", re.IGNORECASE)
_HINT_RE = re.compile(r"^[({\[].*[)}\]]$")  # a whole line in brackets: a printed form hint
_ID_TOKENS_RE = re.compile(r"^[A-Z0-9]+( [A-Z0-9]+)+$")
CURL_ROWS = 14  # tile height = page height / CURL_ROWS (half-width tiles, 50% overlap)
CURL_SKEW_MIN = 2.0  # re-read tiles whose own tilt differs from the page's by at least this


@dataclass
class OcrResult:
    text: str
    confidence: float  # mean word confidence, 0-100


@lru_cache
def ocr_available() -> bool:
    """True if the tesseract binary is installed (it is in the Docker image)."""
    return shutil.which(pytesseract.pytesseract.tesseract_cmd) is not None


def render_page(page: pymupdf.Page, dpi: int) -> np.ndarray:
    """Render a PDF page to an RGB image."""
    pix = page.get_pixmap(dpi=dpi, colorspace=pymupdf.csRGB, alpha=False)
    return np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, 3).copy()


def color_free(rgb: np.ndarray) -> np.ndarray:
    """Grayscale that keeps black/gray ink but fades colored printing.

    Taking each pixel's brightest channel turns light-blue security patterns, blue borders
    and red stamps nearly white, while black typed values stay dark.
    """
    return rgb.max(axis=2)


def detect_rotation(gray: np.ndarray) -> int:
    """Clockwise quarter-turn (0/90/180/270) that makes the page upright."""
    try:
        osd = pytesseract.image_to_osd(gray, config="--psm 0")
    except pytesseract.TesseractError:
        return 0  # too little text to tell; leave it
    match = _ROTATE_RE.search(osd)
    conf = re.search(r"Orientation confidence:\s+([\d.]+)", osd)
    if not match or (conf and float(conf.group(1)) < _OSD_MIN_CONFIDENCE):
        return 0
    return int(match.group(1)) if int(match.group(1)) in _TURNS else 0


def _rotate(image: np.ndarray, angle: float, border: int) -> np.ndarray:
    h, w = image.shape
    matrix = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    return cv2.warpAffine(image, matrix, (w, h), flags=cv2.INTER_LINEAR, borderValue=border)


def _line_sharpness(strokes: np.ndarray) -> float:
    """High when text lines are horizontal: rows alternate between ink and blank."""
    rows = strokes.sum(axis=1, dtype=np.float64)
    return float(np.sum(np.diff(rows) ** 2))


def estimate_skew(gray: np.ndarray, max_angle: float = 15.0) -> float:
    """Tilt of the text lines in degrees (projection-profile search).

    Only text strokes are used (local thresholding ignores big dark areas such as the
    desk around a photographed page). The angle where row sums are sharpest wins.
    """
    scale = 800 / max(gray.shape)
    small = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    strokes = cv2.adaptiveThreshold(
        small, 1, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 15, 15
    )
    if strokes.sum() < 200:  # (almost) no text
        return 0.0

    def best(angles: np.ndarray) -> float:
        return max(angles, key=lambda a: _line_sharpness(_rotate(strokes, a, 0)))

    coarse = best(np.arange(-max_angle, max_angle + 0.01, 1.0))
    return float(best(np.arange(coarse - 1.0, coarse + 1.01, 0.1)))


def straighten(image: np.ndarray, angle: float) -> np.ndarray:
    if abs(angle) < 0.3:
        return image
    h, w = image.shape
    matrix = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    return cv2.warpAffine(
        image, matrix, (w, h), flags=cv2.INTER_CUBIC, borderValue=int(np.median(image))
    )


def remove_ruled_lines(gray: np.ndarray) -> np.ndarray:
    """Whiten long horizontal/vertical lines (form grids, underlines), keeping text."""
    ink = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 25, 15
    )
    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max(gray.shape[1] // 25, 10), 1))
    v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(gray.shape[0] // 25, 10)))
    lines = cv2.bitwise_or(
        cv2.morphologyEx(ink, cv2.MORPH_OPEN, h_kernel),
        cv2.morphologyEx(ink, cv2.MORPH_OPEN, v_kernel),
    )
    out = gray.copy()
    out[cv2.dilate(lines, np.ones((3, 3), np.uint8)) > 0] = 255
    return out


def enhance(gray: np.ndarray) -> np.ndarray:
    """Denoise and boost local contrast (shadows, dim light, faded print)."""
    if gray.shape[1] < 1600:  # Tesseract reads best with ~30 px tall capitals
        gray = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    denoised = cv2.fastNlMeansDenoising(gray, h=10)
    return cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(denoised)


def read_lines(image: np.ndarray, lang: str, psm: int) -> list[tuple[str, list[float], list[int]]]:
    """Run Tesseract; return each text line with its word confidences and word lengths."""
    data = pytesseract.image_to_data(
        image, lang=lang, config=f"--psm {psm}", output_type=pytesseract.Output.DICT
    )
    lines: dict[tuple, tuple[list[str], list[float], list[int]]] = {}
    for i, word in enumerate(data["text"]):
        word = word.strip().translate(_SYMBOL_FIXES)
        conf = float(data["conf"][i])
        if not word or conf < 0:
            continue
        key = (data["block_num"][i], data["par_num"][i], data["line_num"][i])
        words, confs, lens = lines.setdefault(key, ([], [], []))
        words.append(word)
        confs.append(conf)
        lens.append(len(word))
    return [(" ".join(w), c, n) for w, c, n in lines.values()]


def find_form_cells(gray: np.ndarray) -> list[tuple[int, int, int, int]]:
    """Boxes of a ruled form (x, y, w, h), in reading order. Empty if the page isn't a grid.

    Long horizontal/vertical strokes are the grid; the white regions they enclose are the
    cells. Detected on the plain grayscale image, where the (often blue) grid is dark.
    """
    ink = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 25, 15
    )
    height, width = gray.shape
    h_lines = cv2.morphologyEx(
        ink, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (max(width // 25, 10), 1))
    )
    v_lines = cv2.morphologyEx(
        ink, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(height // 40, 10)))
    )
    grid = cv2.dilate(cv2.bitwise_or(h_lines, v_lines), np.ones((5, 5), np.uint8))
    count, _, stats, _ = cv2.connectedComponentsWithStats(cv2.bitwise_not(grid), connectivity=4)
    cells = []
    for x, y, w, h, area in stats[1:]:
        too_small = w < width * 0.06 or h < height * 0.012
        too_big = w > width * 0.95 or h > height * 0.2  # page background, not a box
        if too_small or too_big or area < 0.6 * w * h:  # not a rectangle
            continue
        cells.append((int(x), int(y), int(w), int(h)))
    if len(cells) < MIN_FORM_CELLS:
        return []
    row = height * 0.02  # cells whose tops are this close count as one row
    return sorted(cells, key=lambda c: (round(c[1] / row), c[0]))


def format_cell(texts: list[str]) -> str:
    """One form box as "LABEL: value". Identifier values lose stray OCR spaces
    ("CHASSIS NO.", "9XTRK4LP2HKD 40527" -> "CHASSIS NO.: 9XTRK4LP2HKD40527")."""
    if len(texts) == 1:
        return texts[0]
    # Printed hints under a label, e.g. "(New/Used imported CBU)", aren't part of the value.
    rest = [t for t in texts[1:] if not _HINT_RE.match(t)] or texts[1:]
    label, value = texts[0], " ".join(rest)
    if _ID_LABEL_RE.search(label) and _ID_TOKENS_RE.match(value):
        value = value.replace(" ", "")
    return f"{label}: {value}"


def read_form_cells(gray: np.ndarray, lang: str) -> list[tuple[str, list[float], list[int]]]:
    """Read each box of a ruled form on its own, as "LABEL: value" lines.

    Read as a whole page, a CR's labels and values come out as separate streams ("PLATE NO.
    ENGINE NO. ..." then "XYZ7788 HKD40527 ..."), so nothing ties a value to its label.
    Per box, a local black/white threshold (Otsu) also separates the printed label and value
    from the light security pattern far better than any whole-page setting.
    """
    cells = find_form_cells(gray)
    if not cells:
        return []
    # Grayscale, not color-free: labels are often printed in blue, and must stay dark.
    clean = remove_ruled_lines(gray)
    out = []
    for x, y, w, h in cells:
        crop = cv2.GaussianBlur(clean[y + 4 : y + h - 4, x + 4 : x + w - 4], (3, 3), 0)
        if crop.size == 0:
            continue
        crop = cv2.threshold(crop, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
        crop = cv2.copyMakeBorder(crop, 20, 20, 20, 20, cv2.BORDER_CONSTANT, value=255)
        lines = read_lines(crop, lang, psm=6)
        if not lines:
            continue
        text = format_cell([text for text, _, _ in lines])
        out.append(
            (text, [c for _, cs, _ in lines for c in cs], [n for _, _, ns in lines for n in ns])
        )
    return out


def read_curved_regions(gray: np.ndarray, lang: str) -> list[tuple[str, list[float], list[int]]]:
    """Re-read regions of the page that are tilted differently from the rest.

    Photographed paper curls, so one page-wide straightening can leave parts crooked. The
    page is cut into overlapping half-width tiles (curl differs left to right too); only
    tiles tilted by at least CURL_SKEW_MIN are re-read, so flat pages cost almost nothing.
    """
    height, width = gray.shape
    tile_h, tile_w = max(height // CURL_ROWS, 60), max(width // 2, 60)
    out = []
    for top in range(0, max(height - tile_h // 2, 1), tile_h // 2):
        for left in range(0, max(width - tile_w // 2, 1), tile_w // 2):
            tile = gray[top : top + tile_h, left : left + tile_w]
            angle = estimate_skew(tile, max_angle=8.0)
            if abs(angle) < CURL_SKEW_MIN:
                continue
            tile = cv2.GaussianBlur(straighten(tile, angle), (3, 3), 0)
            tile = cv2.threshold(tile, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
            out.extend(read_lines(tile, lang, psm=6))
    return out


def is_noise(text: str, confs: list[float], lens: list[int], strict: bool = False) -> bool:
    """OCR garbage from backgrounds and security patterns ("M NAL YAYAIN NAKA AK", "— SS |").

    Dropped: lines Tesseract itself is unsure of, and lines that are mostly symbols. With
    `strict` (the extra passes, which mostly re-find text already read), also short
    fragments ("IM", "Wy X") and lines made mostly of 1-2 character tokens.
    """
    if np.average(confs, weights=lens) < NOISE_MIN_CONFIDENCE:
        return True
    alnum = sum(ch.isalnum() for ch in text)
    if alnum < 2 or alnum < 0.5 * len(text.replace(" ", "")):
        return True
    if strict:
        sizes = [sum(ch.isalnum() for ch in t) for t in text.split()]
        sizes = [n for n in sizes if n]
        tiny = sum(1 for n in sizes if n <= 2)
        has_long = any(n >= 5 for n in sizes)  # "CR No. 0420817396" is not a fragment
        return alnum < STRICT_MIN_ALNUM or (tiny > len(sizes) / 2 and not has_long)
    return False


def _tokens(text: str) -> list[str]:
    """Words/numbers of 4+ characters, normalized, for novelty checks."""
    return [t for t in (_NORM_RE.sub("", w.lower()) for w in text.split()) if len(t) >= 4]


def _is_known(token: str, seen: set[str]) -> bool:
    """True if the token, or an OCR variant of it, was already kept.

    Numbers: containment ("40527" inside "9xtrk4lp2hkd40527"), or for long numbers a
    near-identical one of about the same length ("090200000264937" vs "070200000264937");
    looser fuzzy matching on digits finds chance overlaps. Tokens with letters: most
    characters line up with a known token ("4XTRKALPZHKD40527" vs "9XTRK4LP2HKD40527").
    """
    if token in seen:
        return True
    if token.isdigit():
        return any(
            token in s
            or (s.isdigit() and s in token)
            or (
                len(token) >= 8
                and s.isdigit()
                and abs(len(token) - len(s)) <= 1
                and SequenceMatcher(None, token, s, autojunk=False).ratio() >= 0.85
            )
            for s in seen
        )
    for s in seen:
        matcher = SequenceMatcher(None, token, s, autojunk=False)
        if sum(b.size for b in matcher.get_matching_blocks()) >= KNOWN_COVERAGE * len(token):
            return True
    return False


def merge_passes(
    passes: list[list[tuple[str, list[float], list[int]]]], strict_from: int = 2
) -> OcrResult:
    """Union of the passes' lines: the first pass in reading order, then any line from a
    later pass whose content isn't already present. Passes from index `strict_from` on
    get the strict noise filter and must add at least one new word or number."""
    kept: list[str] = []
    seen = ""
    seen_tokens: set[str] = set()
    confs: list[float] = []
    weights: list[int] = []
    for index, lines in enumerate(passes):
        strict = index >= strict_from
        for text, line_confs, lens in lines:
            norm = _NORM_RE.sub("", text.lower())
            if not norm or (len(norm) >= 3 and norm in seen):
                continue
            if is_noise(text, line_confs, lens, strict=strict):
                continue
            tokens = _tokens(text)
            # Extra passes mostly re-read text already kept, often with errors
            # ("4XTRKALPZHKD40527" for "9XTRK4LP2HKD40527"): keep a line only if it adds
            # something substantial that isn't already present in any form.
            if strict:
                new = [t for t in tokens if not _is_known(t, seen_tokens)]
                substantial = any(len(t) >= 6 for t in new) or len(new) >= len(tokens) / 2
                if not new or not substantial:
                    continue
            seen_tokens.update(tokens)
            kept.append(text)
            seen += "|" + norm
            confs.extend(line_confs)
            weights.extend(lens)
    confidence = float(np.average(confs, weights=weights)) if confs else 0.0
    return OcrResult(text="\n".join(kept), confidence=round(confidence, 1))


def ocr_page(page: pymupdf.Page, lang: str = "eng+fil", dpi: int = 300) -> OcrResult:
    """OCR one PDF page (see the module docstring for the passes)."""
    rgb = render_page(page, dpi)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    plain = color_free(rgb)

    turn = detect_rotation(gray)
    if turn:
        gray, plain = cv2.rotate(gray, _TURNS[turn]), cv2.rotate(plain, _TURNS[turn])
    angle = estimate_skew(gray)
    gray, plain = straighten(gray, angle), straighten(plain, angle)

    form = enhance(remove_ruled_lines(plain))
    cells = read_form_cells(gray, lang)  # empty unless the page is a ruled form
    result = merge_passes(
        [
            cells,
            read_lines(enhance(gray), lang, psm=3),
            read_lines(form, lang, psm=6),
            read_lines(form, lang, psm=11),
            read_curved_regions(gray, lang),
        ],
        # On a form, even the page-level reading must add something new: its jumbled header
        # rows ("PLATE NO. ENGINE NO. CHASSIS NO." then "XYZ7788 HKD40527") invited the
        # model to mix up which value belongs to which label.
        strict_from=1 if cells else 2,
    )
    logger.info(
        "OCR page %s: %.0f%% confidence, %d chars",
        page.number + 1,
        result.confidence,
        len(result.text),
    )
    return result
