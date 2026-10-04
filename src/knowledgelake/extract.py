"""Stage 1 - Extract text from PDFs, images, text files and zip archives.

PDFs with a real text layer are read with `pdftotext -layout`. Scanned PDFs
(almost no extractable text per page) and images (JPG/PNG/GIF) fall back to
Tesseract OCR.
"""
from __future__ import annotations

import logging
import shutil
import subprocess
import tempfile
import zipfile
from pathlib import Path

from .models import Document

log = logging.getLogger(__name__)

PDF_EXT = {".pdf"}
IMG_EXT = {".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp", ".tif", ".tiff"}
TXT_EXT = {".txt", ".md", ".csv"}
MIN_CHARS_PER_PAGE = 50  # below this a PDF page is treated as scanned


def _run(cmd: list[str]) -> str:
    return subprocess.run(cmd, capture_output=True, text=True, check=False).stdout


def pdf_page_count(path: Path) -> int:
    for line in _run(["pdfinfo", str(path)]).splitlines():
        if line.startswith("Pages:"):
            return int(line.split()[1])
    return 1


def ocr_image(path: Path) -> str:
    from PIL import Image  # lazy import
    import pytesseract

    img = Image.open(path)
    if getattr(img, "is_animated", False):  # GIF carousels: first frame
        img.seek(0)
    return pytesseract.image_to_string(img.convert("RGB"))


def extract_pdf(path: Path, dpi: int = 200) -> Document:
    pages = pdf_page_count(path)
    text = _run(["pdftotext", "-layout", str(path), "-"])
    if len(text.replace(" ", "").replace("\n", "")) >= MIN_CHARS_PER_PAGE * pages:
        return Document(path.name, text, "text-layer", pages)

    log.info("OCR fallback for scanned PDF %s (%d pages)", path.name, pages)
    from pdf2image import convert_from_path
    import pytesseract

    chunks = [pytesseract.image_to_string(img) for img in convert_from_path(str(path), dpi=dpi)]
    return Document(path.name, "\n\f".join(chunks), "ocr-pdf", pages)


def extract_file(path: Path) -> Document | None:
    ext = path.suffix.lower()
    try:
        if ext in PDF_EXT:
            return extract_pdf(path)
        if ext in IMG_EXT:
            return Document(path.name, ocr_image(path), "ocr-image")
        if ext in TXT_EXT:
            return Document(path.name, path.read_text(errors="ignore"), "plain-text")
    except Exception as exc:  # one bad file must not stop the batch
        log.warning("Skipping %s: %s", path.name, exc)
        return None
    log.info("Unsupported file type skipped: %s", path.name)
    return None


def extract_inputs(inputs: list[Path]) -> list[Document]:
    """Accepts files, folders or .zip archives and returns one Document per file."""
    docs: list[Document] = []
    tmp = Path(tempfile.mkdtemp(prefix="kb_"))
    try:
        files: list[Path] = []
        for p in inputs:
            if p.is_dir():
                files += sorted(x for x in p.rglob("*") if x.is_file())
            elif p.suffix.lower() == ".zip":
                target = tmp / p.stem
                with zipfile.ZipFile(p) as z:
                    z.extractall(target)
                files += sorted(x for x in target.rglob("*") if x.is_file() and "__MACOSX" not in x.parts)
            else:
                files.append(p)
        for f in files:
            doc = extract_file(f)
            if doc and doc.text.strip():
                docs.append(doc)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return docs
