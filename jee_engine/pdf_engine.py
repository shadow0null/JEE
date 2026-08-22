"""
pdf_engine.py
=============
Local JEE PDF processing, built on PyMuPDF (fitz).

Everything here runs entirely offline. A PDF is NEVER sent to Gemini
or any other network service - extraction, question detection and
metadata all happen locally with PyMuPDF + regex/RapidFuzz heuristics.

    JEE PYQ 2024.pdf -> PyMuPDF -> extract text -> detect questions
                      -> return structured questions

Safety:
    * Hard byte-size ceiling (safety.MAX_PDF_SIZE_BYTES) checked before
      the file is ever opened.
    * Hard page-count ceiling (safety.MAX_PDF_PAGES).
    * Wall-clock timeout (safety.PDF_PROCESSING_TIMEOUT_SECONDS).
    * No JavaScript execution, no embedded-file execution, no writing
      back to the original file - PyMuPDF is only used in read mode.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from .safety import (
    MAX_PDF_SIZE_BYTES,
    MAX_PDF_PAGES,
    PDF_PROCESSING_TIMEOUT_SECONDS,
    SafetyError,
    time_limit,
)

try:
    import fitz  # PyMuPDF

    PYMUPDF_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised only when PyMuPDF is absent
    fitz = None
    PYMUPDF_AVAILABLE = False


def _fail(msg: str) -> dict:
    return {"success": False, "error": msg}


def _ok(**kwargs) -> dict:
    return {"success": True, **kwargs}


def _require_pymupdf() -> Optional[dict]:
    if not PYMUPDF_AVAILABLE:
        return _fail("PyMuPDF is not installed. Run: pip install -r requirements.txt")
    return None


def _validate_pdf_bytes(pdf_bytes: bytes) -> None:
    if not isinstance(pdf_bytes, (bytes, bytearray)):
        raise SafetyError("PDF content must be raw bytes.")
    if len(pdf_bytes) == 0:
        raise SafetyError("Empty PDF.")
    if len(pdf_bytes) > MAX_PDF_SIZE_BYTES:
        raise SafetyError(f"PDF exceeds the maximum allowed size ({MAX_PDF_SIZE_BYTES} bytes).")
    # PDFs start with '%PDF-'. Reject anything that isn't actually a PDF
    # before handing it to the parser.
    if not pdf_bytes[:5] == b"%PDF-":
        raise SafetyError("File does not appear to be a valid PDF.")


# --------------------------------------------------------------------------- #
# Question / topic detection heuristics
# --------------------------------------------------------------------------- #

# Matches leading question numbers like "1.", "Q1.", "Question 12)", "(3)"
_QUESTION_NUMBER_RE = re.compile(
    r"(?:^|\n)\s*(?:Q(?:uestion)?\.?\s*)?(\d{1,3})[\.\)]\s+",
    re.IGNORECASE,
)

_CHAPTER_HEADING_RE = re.compile(
    r"(?:^|\n)\s*(?:SECTION|CHAPTER|TOPIC)\s*[:\-]?\s*([A-Za-z][A-Za-z \-&]{2,60})",
    re.IGNORECASE,
)


def _split_questions(page_text: str) -> List[Dict[str, Any]]:
    """Split a page of text into individual question blocks using the
    leading question-number heuristic. Returns [] if no numbering is
    detected (the caller falls back to returning the whole page)."""
    matches = list(_QUESTION_NUMBER_RE.finditer(page_text))
    if not matches:
        return []

    questions = []
    for i, match in enumerate(matches):
        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(page_text)
        body = page_text[start:end].strip()
        if body:
            questions.append({
                "question_number": match.group(1),
                "text": body,
            })
    return questions


def _detect_chapter_headings(text: str) -> List[str]:
    return sorted({m.group(1).strip() for m in _CHAPTER_HEADING_RE.finditer(text)})


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #

def extract_text(pdf_bytes: bytes, page_range: Optional[List[int]] = None) -> Dict[str, Any]:
    """Extract raw text, page by page. `page_range` is an optional
    [start, end] (1-indexed, inclusive) to limit extraction."""
    if (err := _require_pymupdf()) is not None:
        return err
    try:
        _validate_pdf_bytes(pdf_bytes)

        with time_limit(PDF_PROCESSING_TIMEOUT_SECONDS):
            doc = fitz.open(stream=bytes(pdf_bytes), filetype="pdf")
            try:
                if doc.is_encrypted:
                    return _fail("Encrypted PDFs are not supported.")

                page_count = doc.page_count
                if page_count > MAX_PDF_PAGES:
                    return _fail(f"PDF has too many pages (max {MAX_PDF_PAGES}).")

                start, end = 1, page_count
                if page_range:
                    start = max(1, int(page_range[0]))
                    end = min(page_count, int(page_range[1]))
                    if start > end:
                        return _fail("Invalid page_range: start is after end.")

                pages = []
                for page_index in range(start - 1, end):
                    page = doc.load_page(page_index)
                    pages.append({"page": page_index + 1, "text": page.get_text("text")})
            finally:
                doc.close()

        return _ok(page_count=len(pages), pages=pages)
    except SafetyError as e:
        return _fail(str(e))
    except Exception as e:  # noqa: BLE001
        return _fail(f"Unable to safely extract text from this PDF. ({e})")


def extract_metadata(pdf_bytes: bytes) -> Dict[str, Any]:
    """Extract basic, non-sensitive PDF metadata (title, author, page
    count, etc). Never exposes filesystem paths or embedded scripts."""
    if (err := _require_pymupdf()) is not None:
        return err
    try:
        _validate_pdf_bytes(pdf_bytes)
        with time_limit(PDF_PROCESSING_TIMEOUT_SECONDS):
            doc = fitz.open(stream=bytes(pdf_bytes), filetype="pdf")
            try:
                meta = dict(doc.metadata or {})
                result = {
                    "title": meta.get("title") or None,
                    "author": meta.get("author") or None,
                    "subject": meta.get("subject") or None,
                    "page_count": doc.page_count,
                    "is_encrypted": bool(doc.is_encrypted),
                }
            finally:
                doc.close()
        return _ok(metadata=result)
    except SafetyError as e:
        return _fail(str(e))
    except Exception as e:  # noqa: BLE001
        return _fail(f"Unable to safely read PDF metadata. ({e})")


def extract_questions(pdf_bytes: bytes, page_range: Optional[List[int]] = None) -> Dict[str, Any]:
    """The main JEE PDF pipeline: extract text, detect question
    numbering, detect chapter/topic headings, and return structured
    page-to-question mapping. Does not call Gemini or any topic
    classifier that requires network access - chapter detection here
    is purely heading-based; fine-grained topic matching is available
    separately via topic_matcher.match_topic()."""
    text_result = extract_text(pdf_bytes, page_range)
    if not text_result.get("success"):
        return text_result

    all_questions = []
    all_headings: List[str] = []

    for page in text_result["pages"]:
        page_num = page["page"]
        page_text = page["text"]

        headings = _detect_chapter_headings(page_text)
        all_headings.extend(headings)

        page_questions = _split_questions(page_text)
        if page_questions:
            for q in page_questions:
                all_questions.append({
                    "page": page_num,
                    "question_number": q["question_number"],
                    "text": q["text"][:4000],  # bound memory even on malformed PDFs
                })
        else:
            # No numbering detected on this page - still return the raw
            # page text so nothing is silently dropped.
            all_questions.append({
                "page": page_num,
                "question_number": None,
                "text": page_text.strip()[:4000],
            })

    return _ok(
        page_count=text_result["page_count"],
        chapter_headings=sorted(set(all_headings)),
        question_count=sum(1 for q in all_questions if q["question_number"] is not None),
        questions=all_questions,
    )

# --------------------------------------------------------------------------- #
# StudyDesk JEE Engine 2.0: Tutor extraction, OCR fallback, chunking + TF-IDF
# --------------------------------------------------------------------------- #
def ocr_available() -> bool:
    try:
        import shutil, pytesseract  # type: ignore
        return shutil.which('tesseract') is not None
    except Exception:
        return False


def _ocr_page(page) -> str:
    if not ocr_available():
        return ''
    try:
        import io
        from PIL import Image
        import pytesseract  # type: ignore
        pix = page.get_pixmap(matrix=fitz.Matrix(1.8, 1.8), alpha=False)
        img = Image.open(io.BytesIO(pix.tobytes('png')))
        return pytesseract.image_to_string(img, config='--psm 6') or ''
    except Exception:
        return ''


def extract_tutor_pages(pdf_bytes: bytes, use_ocr: bool = True) -> Dict[str, Any]:
    """Page-aware extraction used by StudyDesk Tutor.
    OCR is attempted only on pages with almost no embedded text and only when
    the tesseract binary is available on the server.
    """
    if (err := _require_pymupdf()) is not None:
        return err
    try:
        _validate_pdf_bytes(pdf_bytes)
        with time_limit(PDF_PROCESSING_TIMEOUT_SECONDS):
            doc = fitz.open(stream=bytes(pdf_bytes), filetype='pdf')
            try:
                if doc.is_encrypted:
                    return _fail('Encrypted PDFs are not supported.')
                if doc.page_count > MAX_PDF_PAGES:
                    return _fail(f'PDF has too many pages (max {MAX_PDF_PAGES}).')
                pages=[]; total=0; ocr_pages=0
                for i in range(doc.page_count):
                    page=doc.load_page(i)
                    text=(page.get_text('text') or '').strip()
                    source='text'
                    if use_ocr and len(text) < 30:
                        ocr=_ocr_page(page).strip()
                        if len(ocr) > len(text):
                            text=ocr; source='ocr'; ocr_pages += 1
                    text=text[:14000]
                    total += len(text)
                    pages.append({'page':i+1,'text':text,'source':source})
            finally:
                doc.close()
        return _ok(ok=True, page_count=len(pages), chars=total, pages=pages,
                   scanned=(total < 80), ocr_available=ocr_available(), ocr_pages=ocr_pages)
    except SafetyError as e:
        return _fail(str(e))
    except Exception as e:
        return _fail(f'Unable to safely extract this PDF. ({e})')


def make_chunks(pages: List[Dict[str, Any]], target_chars: int = 5200) -> List[Dict[str, Any]]:
    target_chars=max(1200,min(int(target_chars),9000))
    chunks=[]; buf=''; start=None; end=None; n=0
    for p in pages:
        text=str(p.get('text','')).strip(); page=int(p.get('page',1))
        if not text: continue
        block=f'[Page {page}]\n{text}'
        if buf and len(buf)+len(block)>target_chars:
            chunks.append({'chunk_no':n,'page_start':start,'page_end':end,'text':buf}); n+=1; buf=''; start=None
        if start is None: start=page
        end=page; buf += ('\n\n' if buf else '') + block
    if buf: chunks.append({'chunk_no':n,'page_start':start,'page_end':end,'text':buf})
    return chunks


def extract_and_chunk(pdf_bytes: bytes, use_ocr: bool = True) -> Dict[str, Any]:
    data=extract_tutor_pages(pdf_bytes,use_ocr=use_ocr)
    if not data.get('success'): return data
    chunks=make_chunks(data.get('pages',[]))
    return {**data,'chunks':chunks,'chunk_count':len(chunks)}
