"""PDF -> pages -> chunks. PyMuPDF is imported lazily so tests run without it."""
import hashlib
import os
import re
from typing import List, Tuple

Page = Tuple[int, str]  # (page_number starting at 1, text)


def file_hash(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()[:16]


def read_pdf_bytes(data: bytes) -> List[Page]:
    """Extract text per page. Raises RuntimeError with a clear message if PyMuPDF is missing."""
    try:
        import fitz  # PyMuPDF
    except ImportError as e:  # pragma: no cover
        raise RuntimeError("PyMuPDF is not installed. Run: pip install pymupdf") from e
    pages: List[Page] = []
    with fitz.open(stream=data, filetype="pdf") as doc:
        for i, page in enumerate(doc, start=1):
            pages.append((i, clean(page.get_text("text"))))
    return pages


def read_txt_bytes(data: bytes, page_size: int = 3000) -> List[Page]:
    """Fallback for .txt input: split into pseudo-pages so the rest of the pipeline is identical."""
    text = clean(data.decode("utf-8", errors="ignore"))
    if "[PAGE " in text:  # already page-marked
        parts = re.split(r"\[PAGE (\d+)\]", text)
        pages = []
        for i in range(1, len(parts), 2):
            pages.append((int(parts[i]), parts[i + 1].strip()))
        return pages
    return [(i + 1, text[p:p + page_size]) for i, p in enumerate(range(0, max(len(text), 1), page_size))]


def read_any(path: str) -> List[Page]:
    with open(path, "rb") as f:
        data = f.read()
    return read_pdf_bytes(data) if path.lower().endswith(".pdf") else read_txt_bytes(data)


def clean(text: str) -> str:
    text = text.replace("\xa0", " ").replace("​", "")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def is_scanned(pages: List[Page], min_chars: int = 120) -> bool:
    """True if most pages carry almost no text - i.e. an image-only (scanned) PDF."""
    if not pages:
        return True
    empty = sum(1 for _, t in pages if len(t.strip()) < min_chars)
    return empty > 0.7 * len(pages)


def make_chunks(pages: List[Page], max_chars: int = 45000) -> List[str]:
    """Group pages into prompt-sized chunks, each page tagged with [PAGE n] so citations survive."""
    chunks, cur, size = [], [], 0
    for no, text in pages:
        block = f"[PAGE {no}]\n{text}\n"
        if cur and size + len(block) > max_chars:
            chunks.append("".join(cur))
            cur, size = [], 0
        cur.append(block)
        size += len(block)
    if cur:
        chunks.append("".join(cur))
    return chunks


def page_text(pages: List[Page], no: int) -> str:
    for n, t in pages:
        if n == no:
            return t
    return ""
