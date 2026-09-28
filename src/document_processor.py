"""
document_processor.py
=====================
EvidencePro — Document Ingestion and Multi-Candidate Evidence Extraction
------------------------------------------------------------------------

Accepts an uploaded forensic document (PDF, DOCX, JPG, PNG, TXT) and:

  1. Extracts full text using Docling (preferred) or a lightweight fallback.
  2. Identifies multiple evidence candidate descriptions from the extracted text.
  3. Returns a ProcessedDocument dataclass ready for investigator review.

PIPELINE POSITION
-----------------
Uploaded file
  → process_document()              ← this module
  → investigator reviews candidates (app.py)
  → extract_features() per candidate (extractor.py)
  → EvidenceItem construction + triage pipeline

DOCLING NOTES
-------------
Basic usage:
    from docling.document_converter import DocumentConverter
    converter = DocumentConverter()
    result = converter.convert(file_path)
    text = result.document.export_to_markdown()

If Docling is unavailable or raises, fallback processors are used:
  - PDF: pdfminer.six (if installed) → else raw text read
  - DOCX: python-docx (if installed) → else raw text read
  - Images: pytesseract (if installed) → else "(image — OCR unavailable)"
  - TXT: plain read

EVIDENCE CANDIDATE EXTRACTION
------------------------------
A best-effort heuristic scan of the text for likely evidence items.
Each candidate is a short string (50-120 chars) that can be passed to
extract_features() for field suggestion.

The heuristic looks for lines / sentences that contain forensic keywords
such as "swab", "knife", "phone", "clothing", "blood", "firearm", etc.

SCALABILITY
-----------
The document text is split into logical chunks (~100 lines each).
Each chunk is scanned independently for evidence-keyword sentences.
Candidates are collected in document order and deduplicated.

The practical cap MAX_CANDIDATES (default 50) is a UI/performance
safeguard only — it prevents the investigator from being overwhelmed
by thousands of noise fragments from a very large document.  It is set
high enough that a real large forensic form (typically 10–40 evidence
items) will never be silently truncated.

The full document text is always stored without truncation.

All candidates are suggestions — the investigator reviews, edits, and
removes them before they enter the triage pipeline.
"""

from __future__ import annotations

import os
import re
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# CONSTANTS
# ---------------------------------------------------------------------------

# Maximum candidates returned. Set high enough that real forensic forms
# (typically 5–40 items) are never silently dropped. A limit is kept to
# prevent the UI from becoming unmanageable with noise-heavy documents.
MAX_CANDIDATES = 50

# Lines per chunk for large-document processing.
# Each chunk is scanned independently; results are merged in document order.
_CHUNK_LINES = 120

MAX_TEXT_PREVIEW = 3000   # characters shown in UI preview

SUPPORTED_EXTENSIONS = {".pdf", ".jpg", ".jpeg", ".png", ".docx", ".txt"}

# Patterns that suggest an evidence item in running text
_EVIDENCE_SENTENCE_KEYWORDS = [
    "swab", "blood", "dna", "saliva", "semen", "biological",
    "knife", "blade", "weapon", "firearm", "gun", "bullet", "cartridge",
    "phone", "mobile", "laptop", "computer", "device", "tablet",
    "cctv", "camera", "footage", "video",
    "fingerprint", "latent", "print",
    "clothing", "garment", "shoe", "footwear",
    "fibre", "fiber", "glass", "soil", "trace",
    "drug", "toxicology", "narcotic",
    "document", "signature", "forged",
    "exhibit", "item", "evidence", "recovered", "seized", "collected",
]


# ---------------------------------------------------------------------------
# PUBLIC DATA CLASSES
# ---------------------------------------------------------------------------

@dataclass
class ProcessedDocument:
    """
    Result of processing an uploaded document.

    Attributes
    ----------
    full_text          : Complete extracted text (never truncated).
    processor          : "docling" | "pdfminer" | "docx" | "tesseract" | "plain_text" | "unavailable"
    candidates         : Candidate evidence description strings (capped at MAX_CANDIDATES).
    candidates_total   : Total candidates found before the cap was applied. If equal
                         to len(candidates), no candidates were dropped.
    filename           : Original upload filename.
    error              : Non-empty string if extraction failed; empty string on success.
    """
    full_text:         str
    processor:         str
    candidates:        list[str]
    filename:          str
    error:             str = ""
    candidates_total:  int = 0    # set by process_document()

    @property
    def success(self) -> bool:
        return bool(self.full_text) and not self.error

    @property
    def preview(self) -> str:
        return self.full_text[:MAX_TEXT_PREVIEW] + (
            f"\n\n… *(truncated — {len(self.full_text)} chars total)*"
            if len(self.full_text) > MAX_TEXT_PREVIEW else ""
        )


# ---------------------------------------------------------------------------
# PUBLIC API
# ---------------------------------------------------------------------------

def process_document(file_path: str | Path, filename: str = "") -> ProcessedDocument:
    """
    Extract text and evidence candidates from an uploaded file.

    Parameters
    ----------
    file_path : Path to a temporary file on disk.
    filename  : Original upload filename (used for extension detection and display).

    Returns
    -------
    ProcessedDocument with full text, processor label, and candidates.
    Never raises — all errors are caught and returned in .error.
    """
    file_path = Path(file_path)
    fname = filename or file_path.name
    ext = Path(fname).suffix.lower()

    if ext not in SUPPORTED_EXTENSIONS:
        return ProcessedDocument(
            full_text="",
            processor="unsupported",
            candidates=[],
            filename=fname,
            error=f"Unsupported file type: {ext}. Supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}",
        )

    # Try Docling first
    text, processor = _try_docling(file_path, fname)

    # Fallback chain
    if not text:
        text, processor = _fallback_extract(file_path, ext, fname)

    if not text:
        return ProcessedDocument(
            full_text="",
            processor=processor or "unavailable",
            candidates=[],
            filename=fname,
            error="Could not extract text from this document.",
        )

    candidates, total_found = _extract_candidates(text)
    return ProcessedDocument(
        full_text=text,
        processor=processor,
        candidates=candidates,
        candidates_total=total_found,
        filename=fname,
    )


# ---------------------------------------------------------------------------
# DOCLING
# ---------------------------------------------------------------------------

def _try_docling(file_path: Path, filename: str) -> tuple[str, str]:
    """
    Attempt Docling extraction. Returns (text, "docling") or ("", "").
    """
    try:
        from docling.document_converter import DocumentConverter  # type: ignore
        converter = DocumentConverter()
        result = converter.convert(str(file_path))
        text = result.document.export_to_markdown()
        if text and text.strip():
            return text.strip(), "docling"
        return "", ""
    except ImportError:
        return "", ""
    except Exception:
        return "", ""


# ---------------------------------------------------------------------------
# FALLBACK EXTRACTORS
# ---------------------------------------------------------------------------

def _fallback_extract(file_path: Path, ext: str, filename: str) -> tuple[str, str]:
    if ext == ".txt":
        return _read_txt(file_path)
    if ext == ".pdf":
        text, proc = _read_pdf_pdfminer(file_path)
        if text:
            return text, proc
        return _read_binary_fallback(file_path)
    if ext == ".docx":
        text, proc = _read_docx(file_path)
        if text:
            return text, proc
        return _read_binary_fallback(file_path)
    if ext in (".jpg", ".jpeg", ".png"):
        text, proc = _read_image_tesseract(file_path)
        if text:
            return text, proc
        return "(Image file — OCR unavailable. Please type evidence descriptions manually.)", "plain_text"
    return "", "unavailable"


def _read_txt(file_path: Path) -> tuple[str, str]:
    try:
        for enc in ("utf-8", "latin-1", "cp1252"):
            try:
                text = file_path.read_text(encoding=enc)
                return text.strip(), "plain_text"
            except UnicodeDecodeError:
                continue
    except Exception:
        pass
    return "", "plain_text"


def _read_pdf_pdfminer(file_path: Path) -> tuple[str, str]:
    try:
        from pdfminer.high_level import extract_text as pdfminer_extract  # type: ignore
        text = pdfminer_extract(str(file_path))
        return (text.strip() if text else ""), "pdfminer"
    except ImportError:
        return "", ""
    except Exception:
        return "", ""


def _read_docx(file_path: Path) -> tuple[str, str]:
    try:
        import docx  # type: ignore
        doc = docx.Document(str(file_path))
        paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
        text = "\n".join(paragraphs)
        return text, "docx"
    except ImportError:
        return "", ""
    except Exception:
        return "", ""


def _read_image_tesseract(file_path: Path) -> tuple[str, str]:
    try:
        from PIL import Image  # type: ignore
        import pytesseract  # type: ignore
        img = Image.open(str(file_path))
        text = pytesseract.image_to_string(img)
        return (text.strip() if text else ""), "tesseract"
    except ImportError:
        return "", ""
    except Exception:
        return "", ""


def _read_binary_fallback(file_path: Path) -> tuple[str, str]:
    """Last-resort: try reading as UTF-8 text, ignore decode errors."""
    try:
        text = file_path.read_text(encoding="utf-8", errors="ignore")
        # Filter to printable-ish lines only
        lines = [l for l in text.splitlines() if len(l.strip()) > 4 and l.strip().isprintable()]
        return "\n".join(lines[:500]), "plain_text"
    except Exception:
        return "", "unavailable"


# ---------------------------------------------------------------------------
# CANDIDATE EXTRACTION
# ---------------------------------------------------------------------------

def _extract_candidates(text: str) -> tuple[list[str], int]:
    """
    Scan extracted text for sentence fragments that likely describe evidence items.

    Returns
    -------
    (candidates, total_found) where:
      candidates   : up to MAX_CANDIDATES strings, in document order
      total_found  : total unique candidates before the cap was applied

    Strategy
    --------
    1. Split the full text into logical line-chunks (_CHUNK_LINES lines each).
       This keeps memory usage flat for very large documents and means a large
       section of low-scoring noise cannot push genuine early evidence off a
       global top-N list.
    2. Within each chunk, split further on sentence/bullet boundaries.
    3. Score each fragment by forensic keyword hits (≥1 = candidate).
    4. Collect candidates in DOCUMENT ORDER (not sorted by score).
       This preserves the natural exhibit sequence that forensic forms use.
    5. Deduplicate across chunks using normalised 50-char prefix matching.
    6. Cap at MAX_CANDIDATES only as a UI safeguard; real forensic forms
       (typically 5–40 items) will never hit this cap.
    """
    lines = text.splitlines()
    all_candidates: list[str] = []   # document-order, deduplicated
    seen: set[str] = set()           # deduplication keys

    # Process in chunks of _CHUNK_LINES lines
    for chunk_start in range(0, len(lines), _CHUNK_LINES):
        chunk_lines = lines[chunk_start : chunk_start + _CHUNK_LINES]
        chunk_text = "\n".join(chunk_lines)

        # Split chunk into sentence/bullet fragments
        raw_fragments = re.split(
            r"(?<=[.!?])\s+|[\n\r]+|(?<=\d\.)\s+|[-•–]\s+",
            chunk_text,
        )

        for frag in raw_fragments:
            frag = frag.strip()
            # Filter: 20–350 chars, not a pure-number / symbol-only line
            if len(frag) < 20 or len(frag) > 350:
                continue
            if re.match(r"^[\d\s\W]{1,10}$", frag):
                continue
            low = frag.lower()
            score = sum(1 for kw in _EVIDENCE_SENTENCE_KEYWORDS if kw in low)
            if score < 1:
                continue

            # Deduplicate: normalise the first 50 chars as the key.
            # 50 chars is long enough to distinguish different exhibits that
            # start similarly (e.g. "Blood swab #1" vs "Blood swab #2")
            # while still catching chunking duplicates that are genuinely identical.
            dedup_key = re.sub(r"\s+", " ", frag[:50].lower()).strip()
            if dedup_key in seen:
                continue
            seen.add(dedup_key)
            all_candidates.append(frag[:200])   # trim to readable length

    total_found = len(all_candidates)
    # Apply the UI cap and return both the capped list and the pre-cap total
    return all_candidates[:MAX_CANDIDATES], total_found
