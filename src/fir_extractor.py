"""
fir_extractor.py
================
EvidencePro — FIR Document Text Extraction via Docling (Python package)
-----------------------------------------------------------------------

PURPOSE
-------
Accepts an uploaded FIR (First Information Report) document (PDF, DOCX, or
image) and extracts structured text from it using the ``docling`` Python
package (DocumentConverter).

The extracted markdown is then parsed with regex / keyword rules to suggest

values for both:

  Tab 1 — Case Context form:
    • FIR / Case Number  → fir_number
    • Offence Type       → offence_type  (mapped to nearest OFFENCE_TYPES entry)
    • Case Narrative     → narrative     (cleaned markdown text, editable)

  Tab 2 — Evidence Items:
    • Per-row evidence items parsed from the Forensic Evidence Inventory table
      (label, evidence_type, collection_age_hours, evidence_condition)
    • Per-row triage scores parsed from the Evidence Classification table
      (probative_value, perishability, exclusionary_power, testing_lead_time,
       contamination_risk, specialist_required, specialist_type)
    • The two tables are joined on row-position (S.No / sequence).

PIPELINE POSITION
-----------------
Uploaded FIR file (bytes)
  → extract_fir_document()          ← this module  (DocumentConverter)
  → _parse_inventory_table()        ← parse "Forensic Evidence Inventory" table
  → _parse_triage_table()           ← parse "Evidence Classification" table
  → _merge_evidence_rows()          ← join both tables by row position
  → FIRExtractResult                ← returned to app.py
      .fir_number / .offence_type / .narrative  → Tab 1 pre-fill
      .evidence_items                           → Tab 2 pre-fill list
  → investigator review / edit      (all suggestions are editable, never auto-added)
  → CaseContext + EvidenceItem      (models/evidence_item.py)
  → rest of the triage pipeline

DEPENDENCIES
------------
``docling`` — install with:
    pip install docling

No API key, no cloud service, no Docker container required.
Docling runs fully locally.

IMPORTANT
---------
All field values produced by this module are SUGGESTIONS only. The
investigator must review and confirm every value before anything enters the
pipeline. No evidence item is automatically added — the investigator
explicitly approves each one in Tab 2.
"""

from __future__ import annotations

import os
import re
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from models.evidence_item import EVIDENCE_TYPES, OFFENCE_TYPES

# ---------------------------------------------------------------------------
# CONSTANTS
# ---------------------------------------------------------------------------

# Regex patterns to detect a FIR / case number in extracted text.
# Tried in order; first match wins.
#
# Design notes
# ------------
# • Docling OCR sometimes drops characters (e.g. "FIR No." → "FIR o."),
#   so we match both "No" and "o" after FIR.
# • Indian police docs use formats like:  S.D.E/FIR No., C.R. No., Crime No.
# • The value itself can be  0217/2026  or  FIR-2024-009  or  CR/101/2023.
# • A leading "FIR" or bare number followed by "dated" is stripped of the
#   "dated …" trailer so only the case number is returned.
_FIR_NUMBER_PATTERNS = [
    # Bare FIR-YYYY-NNN  (most specific, try first)
    r"\b(FIR[-/]\d{4}[-/]\d+)\b",

    # "S.D.E/FIR No.:" or "FIR No.:" — value may itself start with "FIR o. 0217/2026"
    # Colon, dash, or plain whitespace all accepted as separator.
    r"(?:S\.D\.E/)?FIR\s*[No\.o]*\.?\s*[:\-\s]\s*(?:FIR\s*[Nno\.o]*\.?\s*)?(\d[\d/\-]+)",

    # "FIR No / Number / # :" with alphanumeric value (colon/dash/space separator)
    r"FIR\s*(?:No\.?|Number|Num\.?|#)\s*[:\-\s]\s*([A-Z0-9][\w\-/]{2,})",

    # Indian: "C.R. No.", "CR No.", "Crime No.", "C.No."
    r"(?:C\.R\.|CR|Crime|C)\s*[Nn]o\.?\s*[:\-]?\s*(\d[\d/\-]+)",

    # "Case No / Number :"
    r"Case\s*(?:No\.?|Number|Num\.?|#)\s*[:\-]\s*([A-Z0-9][\w\-/]{2,})",

    # "Registration No / Report No :"
    r"(?:Registration|Report)\s*(?:No\.?|Number)\s*[:\-]\s*([A-Z0-9][\w\-/]{3,})",
]

# Keyword → OFFENCE_TYPES value.
# Scored independently; highest-count match wins.
_OFFENCE_KEYWORD_MAP: list[tuple[list[str], str]] = [
    (
        ["murder", "killed", "homicide", "death", "deceased", "culpable homicide", "stab"],
        "homicide",
    ),
    (
        ["rape", "sexual assault", "molestation", "sexual offence", "indecent assault"],
        "sexual_assault",
    ),
    (["robbery", "dacoity", "snatching", "armed robbery"], "robbery"),
    (
        ["burglary", "break-in", "breaking and entering", "house breaking", "housebreaking"],
        "burglary",
    ),
    (["drug", "narcotic", "narcotics", "trafficking", "contraband", "substance"], "drug_supply"),
    (["fraud", "cheating", "forgery", "embezzlement", "financial crime", "scam"], "fraud"),
    (["arson", "fire", "burnt", "set ablaze", "intentional fire"], "arson"),
    (
        ["vehicle", "car theft", "motor vehicle", "theft of vehicle", "auto theft", "motorcycle theft"],
        "vehicle_crime",
    ),
]

# Maps condition strings (from table cells) to evidence_condition int 1-3
_CONDITION_MAP = {
    "poor":      1,
    "bad":       1,
    "damaged":   1,
    "fair":      2,
    "average":   2,
    "moderate":  2,
    "good":      3,
    "excellent": 3,
    "intact":    3,
}

# Narrative is capped so the text area stays usable
_NARRATIVE_MAX_CHARS = 2000

# Heading phrases that identify the inventory and triage tables.
# Covers both structured reports (our .md sample) and real forensic scene
# inspection PDFs (Section 5: "Biological / Explosive / Physical / Ballistics
# Clue Materials").
_INVENTORY_HEADINGS = [
    # Structured report headings
    "forensic evidence inventory", "evidence inventory",
    "evidence items", "items collected",
    # Real scene inspection form headings (Section 5)
    "biological", "explosive", "clue material", "clue materials",
    "sample", "samples",
]
_TRIAGE_HEADINGS = [
    "evidence classification", "triage score",
    "classification and triage", "pdes",
]

# ---------------------------------------------------------------------------
# EVIDENCE-TYPE INFERENCE MAPS
# Mirrors extractor.py heuristics — kept here to avoid circular imports.
# ---------------------------------------------------------------------------

# keyword list → canonical evidence_type  (order matters: touch DNA before DNA)
_EVIDENCE_KEYWORDS: list[tuple[list[str], str]] = [
    (["touch dna", "contact dna"],                              "biological_touch_dna"),
    (["dna", "blood", "saliva", "semen",
      "swab", "biological", "tissue", "cell",
      "nail", "hair strand", "clipping"],                       "biological_dna"),
    (["latent fingerprint", "latent print",
      "fingerprint powder", "latent fp", "fingerprint"],        "fingerprint_latent"),
    (["patent fingerprint", "visible fingerprint",
      "bloody fingerprint"],                                    "fingerprint_patent"),
    (["cctv", "camera", "surveillance",
      "video footage", "dashcam", "footage"],                   "digital_cctv"),
    (["phone", "mobile", "laptop", "tablet",
      "computer", "device", "digital",
      "hard drive", "usb", "memory"],                           "digital_device"),
    (["bullet", "cartridge", "casing",
      "ballistic", "firearm", "gun",
      "projectile"],                                            "firearm_ballistic"),
    (["gsr", "gunshot residue", "firearm residue"],             "firearm_residue"),
    (["glass", "broken glass",
      "glass fragment", "glass shard", "bangle"],               "trace_glass"),
    (["soil", "dirt", "mud", "earth",
      "shoe impression", "footwear impression",
      "impression", "footprint"],                               "impression_footwear"),
    (["tyre", "tire", "tyre mark", "wheel track"],              "impression_tyre"),
    (["fibre", "fiber", "thread", "textile",
      "clothing trace", "bedsheet", "garment"],                 "trace_fibre"),
    (["document", "cheque", "check",
      "signature", "handwriting",
      "questioned document", "forged"],                         "document_questioned"),
    (["rolling pin", "weapon", "stick",
      "rod", "blunt"],                                          "firearm_ballistic"),
]

# evidence_type → default probative_value (1-3)
_PROBATIVE_MAP: dict[str, int] = {
    "biological_dna":       3,
    "biological_touch_dna": 2,
    "fingerprint_latent":   2,
    "fingerprint_patent":   2,
    "digital_cctv":         3,
    "digital_device":       3,
    "firearm_ballistic":    3,
    "firearm_residue":      2,
    "toxicology_blood":     3,
    "toxicology_hair":      2,
    "document_questioned":  2,
    "trace_fibre":          1,
    "trace_glass":          2,
    "trace_soil":           1,
    "impression_footwear":  2,
    "impression_tyre":      1,
}

# evidence_type → default perishability (1-3)
_PERISHABILITY_MAP: dict[str, int] = {
    "biological_dna":       3,
    "biological_touch_dna": 3,
    "toxicology_blood":     3,
    "firearm_residue":      3,
    "toxicology_hair":      2,
    "trace_fibre":          2,
    "fingerprint_latent":   2,
    "trace_glass":          1,
    "trace_soil":           1,
    "fingerprint_patent":   1,
    "digital_device":       1,
    "digital_cctv":         1,
    "firearm_ballistic":    1,
    "document_questioned":  1,
    "impression_footwear":  1,
    "impression_tyre":      1,
}

# evidence_type → default exclusionary_power (1-3)
_EXCLUSIONARY_MAP: dict[str, int] = {
    "biological_dna":       3,
    "biological_touch_dna": 3,
    "fingerprint_latent":   2,
    "fingerprint_patent":   2,
    "digital_cctv":         2,
    "digital_device":       2,
    "firearm_ballistic":    3,
    "firearm_residue":      2,
    "toxicology_blood":     3,
    "toxicology_hair":      2,
    "document_questioned":  2,
    "trace_fibre":          1,
    "trace_glass":          2,
    "trace_soil":           1,
    "impression_footwear":  2,
    "impression_tyre":      1,
}

# evidence_type → default testing_lead_time days
_LEAD_TIME_MAP: dict[str, int] = {
    "biological_dna":       5,
    "biological_touch_dna": 7,
    "toxicology_blood":     3,
    "toxicology_hair":      21,
    "firearm_ballistic":    10,
    "firearm_residue":      4,
    "fingerprint_latent":   5,
    "fingerprint_patent":   3,
    "digital_device":       14,
    "digital_cctv":         2,
    "document_questioned":  14,
    "trace_fibre":          7,
    "trace_glass":          5,
    "trace_soil":           10,
    "impression_footwear":  7,
    "impression_tyre":      7,
}

# evidence_type → specialist_required (0/1)
_SPECIALIST_REQUIRED_MAP: dict[str, int] = {
    "biological_dna":       1,
    "biological_touch_dna": 1,
    "toxicology_blood":     1,
    "toxicology_hair":      1,
    "firearm_ballistic":    1,
    "firearm_residue":      1,
    "document_questioned":  1,
    "trace_fibre":          1,
    "trace_glass":          1,
    "trace_soil":           1,
    "digital_device":       1,
    "digital_cctv":         0,
    "fingerprint_latent":   0,
    "fingerprint_patent":   0,
    "impression_footwear":  1,
    "impression_tyre":      1,
}

# evidence_type → specialist_type label
_SPECIALIST_TYPE_MAP: dict[str, str] = {
    "biological_dna":       "DNA analyst",
    "biological_touch_dna": "DNA analyst",
    "fingerprint_latent":   "Fingerprint specialist",
    "fingerprint_patent":   "Fingerprint specialist",
    "digital_cctv":         "Digital forensics examiner",
    "digital_device":       "Digital forensics examiner",
    "firearm_ballistic":    "Ballistics/firearms examiner",
    "firearm_residue":      "Ballistics/firearms examiner",
    "toxicology_blood":     "Toxicologist",
    "toxicology_hair":      "Toxicologist",
    "document_questioned":  "Document examiner",
    "trace_fibre":          "Trace evidence examiner",
    "trace_glass":          "Trace evidence examiner",
    "trace_soil":           "Trace evidence examiner",
    "impression_footwear":  "Impression evidence specialist",
    "impression_tyre":      "Impression evidence specialist",
}

# contamination_risk defaults by evidence_type (1-3)
_CONTAMINATION_MAP: dict[str, int] = {
    "biological_dna":       3,
    "biological_touch_dna": 3,
    "fingerprint_latent":   2,
    "fingerprint_patent":   1,
    "digital_cctv":         1,
    "digital_device":       1,
    "firearm_ballistic":    1,
    "firearm_residue":      3,
    "toxicology_blood":     3,
    "toxicology_hair":      2,
    "document_questioned":  1,
    "trace_fibre":          2,
    "trace_glass":          2,
    "trace_soil":           2,
    "impression_footwear":  2,
    "impression_tyre":      1,
}


# ---------------------------------------------------------------------------
# DATACLASSES
# ---------------------------------------------------------------------------

@dataclass
class EvidenceRowSuggestion:
    """
    A single evidence item suggestion parsed from the FIR document.

    All fields mirror EvidenceItem but are typed as Any / optional so the
    investigator can choose to override or blank any of them.
    """
    # Identity
    label: str = ""

    # ML features
    evidence_type: str = ""
    probative_value: int = 2
    perishability: int = 1
    exclusionary_power: int = 2
    contamination_risk: int = 1
    specialist_required: int = 0
    testing_lead_time: int = 7

    # Operational
    collection_age_hours: int = 0
    evidence_condition: int = 2
    specialist_type: str = ""

    # Provenance flag — always True since these come from doc extraction
    from_document: bool = True


@dataclass
class FIRExtractResult:
    """
    Result returned by ``extract_fir_document()``.

    Attributes
    ----------
    success : bool
        True when text was extracted successfully.
    raw_text : str
        Full markdown text produced by Docling (export_to_markdown).
    fir_number : str
        Detected FIR / case number, or "" if not found.
    offence_type : str
        Best-matched OFFENCE_TYPES value, or "" if undetermined.
    narrative : str
        Cleaned document text suitable for the Case Narrative field
        (capped at _NARRATIVE_MAX_CHARS characters).
    evidence_items : list[EvidenceRowSuggestion]
        Evidence items parsed from the document tables.
        Empty list when no structured evidence table was found.
    error : str
        Human-readable error message when success=False.
    source : str
        "docling" when successful, "none" otherwise.
    """
    success: bool = False
    raw_text: str = ""
    fir_number: str = ""
    offence_type: str = ""
    narrative: str = ""
    evidence_items: list[EvidenceRowSuggestion] = field(default_factory=list)
    error: str = ""
    source: str = "none"


# ---------------------------------------------------------------------------
# PUBLIC API
# ---------------------------------------------------------------------------

def is_docling_available() -> bool:
    """
    Return True if the ``docling`` package is installed.

    Uses ``importlib.util.find_spec`` so it never actually imports the package
    or triggers model loading — safe to call on every Streamlit rerender.
    """
    import importlib.util
    return importlib.util.find_spec("docling") is not None


# Extensions treated as image files that genuinely need OCR
_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tiff", ".tif", ".bmp", ".webp"}


def _build_converter(suffix: str):
    """
    Build a DocumentConverter configured for the given file extension.

    For PDF and document formats (DOCX, DOC, MD …) OCR is disabled —
    these formats have an embedded text layer so no model download is needed.

    For image files (PNG, JPG, TIFF …) OCR is left at its default setting,
    but we still catch any model-download failure and surface a clear message.
    """
    from docling.datamodel.base_models import InputFormat
    from docling.document_converter import DocumentConverter, PdfFormatOption
    from docling.datamodel.pipeline_options import PdfPipelineOptions

    is_image = suffix.lower() in _IMAGE_EXTENSIONS

    if not is_image:
        # Disable OCR entirely — text-layer extraction only, no model downloads
        pdf_opts = PdfPipelineOptions(do_ocr=False)
        return DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(pipeline_options=pdf_opts),
            }
        )

    # Image path: use default (OCR enabled) but don't suppress any errors here —
    # let the caller catch and surface the model-download message.
    return DocumentConverter()


def extract_fir_document(file_bytes: bytes, filename: str) -> FIRExtractResult:
    """
    Convert a FIR document to markdown using Docling's DocumentConverter,
    then parse case-context fields and structured evidence items from the text.

    OCR is disabled for PDF/DOCX/MD files (they have an embedded text layer),
    which avoids any model downloads. Image files (PNG/JPG/TIFF) use OCR but
    require a network connection for the first run to download the OCR model.

    Parameters
    ----------
    file_bytes : bytes
        Raw bytes of the uploaded file (PDF / DOCX / PNG / JPG / TIFF).
    filename : str
        Original filename including extension, e.g. ``"FIR-2024-001.pdf"``.

    Returns
    -------
    FIRExtractResult
        Always returns a result — never raises.  Check ``.success``.
    """
    try:
        from docling.document_converter import DocumentConverter  # noqa: F401
    except ImportError:
        return FIRExtractResult(
            success=False,
            error=(
                "The 'docling' package is not installed.\n"
                "Install it with:  pip install docling"
            ),
        )

    suffix = Path(filename).suffix or ".pdf"

    # Write the uploaded bytes to a temp file so DocumentConverter can read it
    try:
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(file_bytes)
            tmp_path = tmp.name
    except OSError as exc:
        return FIRExtractResult(
            success=False,
            error=f"Could not write temporary file for conversion: {exc}",
        )

    try:
        converter = _build_converter(suffix)
        result = converter.convert(tmp_path)
        raw_text = result.document.export_to_markdown()
    except Exception as exc:
        err_str = str(exc)
        # Give a specific, actionable message for model-download failures
        if "download" in err_str.lower() or "modelscope" in err_str.lower() or "huggingface" in err_str.lower():
            return FIRExtractResult(
                success=False,
                error=(
                    "Docling could not download an OCR model (network unreachable).\n"
                    "For image files (PNG/JPG/TIFF) an internet connection is required "
                    "on first use to download the OCR model.\n"
                    "For PDF or DOCX files, re-upload the file — OCR is not needed and "
                    "this error should not occur.\n\n"
                    f"Detail: {exc}"
                ),
            )
        return FIRExtractResult(
            success=False,
            error=f"Docling conversion failed: {exc}",
        )
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass

    if not raw_text.strip():
        return FIRExtractResult(
            success=False,
            error=(
                "Docling extracted no text from this document. "
                "Check the file format or try a higher-quality scan."
            ),
        )

    evidence_items = _parse_evidence_items(raw_text)

    return FIRExtractResult(
        success=True,
        raw_text=raw_text,
        fir_number=_detect_fir_number(raw_text),
        offence_type=_detect_offence_type(raw_text),
        narrative=_clean_narrative(raw_text),
        evidence_items=evidence_items,
        error="",
        source="docling",
    )


def extract_fir_from_markdown(markdown_text: str) -> FIRExtractResult:
    """
    Parse an already-extracted markdown string (e.g. from a .md file read
    directly) without going through Docling conversion.

    Useful for testing with the sample ``forensic_examination_report.md``.
    """
    if not markdown_text.strip():
        return FIRExtractResult(
            success=False,
            error="Empty markdown text provided.",
        )

    evidence_items = _parse_evidence_items(markdown_text)

    return FIRExtractResult(
        success=True,
        raw_text=markdown_text,
        fir_number=_detect_fir_number(markdown_text),
        offence_type=_detect_offence_type(markdown_text),
        narrative=_clean_narrative(markdown_text),
        evidence_items=evidence_items,
        error="",
        source="docling",
    )


# ---------------------------------------------------------------------------
# EVIDENCE TABLE PARSING
# ---------------------------------------------------------------------------

def _parse_evidence_items(text: str) -> list[EvidenceRowSuggestion]:
    """
    Parse evidence items from a FIR / forensic scene inspection document.

    Strategy (tried in order):
    ──────────────────────────
    1. Structured report format — looks for a heading matching
       _INVENTORY_HEADINGS followed by a table with explicit Evidence Type,
       P/D/E columns (our .md sample report).
       If a matching triage table also exists the two are merged by row.

    2. Scene inspection form format (real PDFs) — looks for a samples table
       (Section 5: "Biological / Explosive / Physical / Ballistics Clue
       Materials") with columns "Name of the Sample" and "Collected from".
       Evidence type and triage scores are inferred from the description text
       using keyword matching.

    Returns an empty list when no usable table is found.
    """
    inventory_rows = _parse_inventory_table(text)
    triage_rows    = _parse_triage_table(text)

    if inventory_rows:
        # Structured format: merge with triage rows if available
        return _merge_evidence_rows(inventory_rows, triage_rows)

    # Fallback: parse as scene inspection samples table
    return _parse_samples_table(text)


def _markdown_tables(text: str) -> list[list[list[str]]]:
    """
    Extract all markdown pipe-tables from the text.

    Returns a list of tables; each table is a list of rows; each row is a
    list of cell strings (stripped, pipe-delimited).
    Skips the separator row (---|---).
    """
    tables: list[list[list[str]]] = []
    current: list[list[str]] = []

    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("|") and stripped.endswith("|"):
            cells = [c.strip() for c in stripped.strip("|").split("|")]
            # Skip pure separator rows like |---|---|
            if all(re.fullmatch(r"[-: ]+", c) for c in cells if c):
                continue
            current.append(cells)
        else:
            if current:
                tables.append(current)
                current = []

    if current:
        tables.append(current)

    return tables


def _table_near_heading(text: str, headings: list[str]) -> list[list[str]] | None:
    """
    Return the first markdown table that follows one of the given heading phrases
    (case-insensitive match anywhere in the heading line).
    Returns None if not found.
    """
    lines = text.splitlines()
    heading_line_numbers: list[int] = []
    for i, line in enumerate(lines):
        lower = line.lower()
        if any(h in lower for h in headings):
            heading_line_numbers.append(i)

    if not heading_line_numbers:
        return None

    # For each heading position, find the first table that starts after it
    all_tables = _markdown_tables(text)

    # Build a lookup: which line does each table start on?
    # We find them by scanning linearly
    table_start_lines: list[int] = []
    in_table = False
    for i, line in enumerate(lines):
        stripped = line.strip()
        is_table_line = stripped.startswith("|") and stripped.endswith("|")
        if is_table_line and not in_table:
            table_start_lines.append(i)
            in_table = True
        elif not is_table_line:
            in_table = False

    for heading_pos in heading_line_numbers:
        for ti, start_line in enumerate(table_start_lines):
            if start_line > heading_pos and ti < len(all_tables):
                return all_tables[ti]

    return None


def _normalise_header(h: str) -> str:
    """Lowercase, strip markdown bold/italic markers and extra spaces."""
    h = re.sub(r"[*_`]", "", h)
    return re.sub(r"\s+", " ", h).strip().lower()


def _parse_inventory_table(text: str) -> list[dict]:
    """
    Parse a structured "Forensic Evidence Inventory" table.

    Only matches tables that have an explicit evidence_type / collection_age /
    condition column — i.e. our structured .md report format.
    Returns [] when the matched table looks like a raw samples list instead
    (no recognisable typed columns), so the caller falls back to
    _parse_samples_table().
    """
    table = _table_near_heading(text, [
        "forensic evidence inventory", "evidence inventory",
        "evidence items", "items collected",
    ])
    if not table or len(table) < 2:
        return []

    headers = [_normalise_header(h) for h in table[0]]

    # Require at least one typed column to confirm this is the structured format
    typed_keywords = ["evidence type", "collection age", "condition", "probative"]
    if not any(any(kw in h for kw in typed_keywords) for h in headers):
        return []

    def _col(row: list[str], *keywords: str) -> str:
        for kw in keywords:
            for i, h in enumerate(headers):
                if kw in h and i < len(row):
                    return row[i].strip()
        return ""

    rows = []
    for row in table[1:]:
        if not any(c.strip() for c in row):
            continue
        description = _col(row, "description", "label", "item", "detail")
        raw_et      = _col(row, "evidence type", "type")
        raw_age     = _col(row, "collection age", "age")
        raw_cond    = _col(row, "condition")

        et   = _match_evidence_type(raw_et)
        age  = _parse_int(raw_age, default=0)
        cond = _CONDITION_MAP.get(raw_cond.lower(), 2)

        rows.append({
            "label":                description,
            "evidence_type":        et,
            "collection_age_hours": age,
            "evidence_condition":   cond,
        })

    return rows


def _parse_samples_table(text: str) -> list[EvidenceRowSuggestion]:
    """
    Parse a raw samples/clue-materials table from a scene inspection form.

    Handles the Section 5 format found in real forensic PDFs:

        | S. No. | Name of the Sample          | Marked as | Collected from        |
        |--------|-----------------------------|-----------|-----------------------|
        | (i)    | Blood stain scraped floor   | A         | Living room floor     |

    There are no P/D/E columns — all triage fields are inferred from the
    sample description text using keyword matching.

    Blank rows (where Name of the Sample is empty) are silently skipped.
    """
    table = _table_near_heading(text, _INVENTORY_HEADINGS)
    if not table or len(table) < 2:
        return []

    headers = [_normalise_header(h) for h in table[0]]

    def _col(row: list[str], *keywords: str) -> str:
        for kw in keywords:
            for i, h in enumerate(headers):
                if kw in h and i < len(row):
                    return row[i].strip()
        return ""

    items: list[EvidenceRowSuggestion] = []
    for row in table[1:]:
        name = _col(row, "name of the sample", "name of sample", "sample", "description", "label", "item")
        location = _col(row, "collected from", "source", "place", "location", "from")

        # Skip placeholder empty rows  e.g. "(ix) | | |"
        if not name or not name.strip():
            continue

        label = name
        if location:
            label = f"{name} — {location}"

        items.append(_infer_from_description(label))

    return items


def _parse_triage_table(text: str) -> list[dict]:
    """
    Parse the "Evidence Classification and Triage Scores" table.

    Expected columns (flexible matching):
      probative value / P
      degradation risk / D / perishability
      exclusionary power / E
      testing lead time / lead time / days
      contamination risk
      specialist required
      specialist type
    """
    table = _table_near_heading(text, _TRIAGE_HEADINGS)
    if not table or len(table) < 2:
        return []

    headers = [_normalise_header(h) for h in table[0]]

    def _col(row: list[str], *keywords: str) -> str:
        for kw in keywords:
            for i, h in enumerate(headers):
                if kw in h and i < len(row):
                    return row[i].strip()
        return ""

    rows = []
    for row in table[1:]:
        if not any(c.strip() for c in row):
            continue
        rows.append({
            "probative_value":    _clamp(_parse_int(_col(row, "probative", " p)"), default=2), 1, 3),
            "perishability":      _clamp(_parse_int(_col(row, "degradation", "perishab", " d)"), default=1), 1, 3),
            "exclusionary_power": _clamp(_parse_int(_col(row, "exclusionary", " e)"), default=2), 1, 3),
            "testing_lead_time":  _clamp(_parse_int(_col(row, "lead time", "lead", "testing"), default=7), 1, 60),
            "contamination_risk": _clamp(_parse_int(_col(row, "contamination"), default=1), 1, 3),
            "specialist_required":_clamp(_parse_int(_col(row, "specialist req"), default=0), 0, 1),
            "specialist_type":    _col(row, "specialist type", "specialist typ"),
        })

    return rows


def _merge_evidence_rows(
    inventory: list[dict],
    triage: list[dict],
) -> list[EvidenceRowSuggestion]:
    """
    Merge structured inventory rows and triage rows by position.

    When the inventory row already has an evidence_type the triage fields are
    taken verbatim from the triage table (if present) or inferred from the
    evidence_type via the heuristic maps.
    """
    n = max(len(inventory), len(triage)) if (inventory or triage) else 0
    if n == 0:
        return []

    items: list[EvidenceRowSuggestion] = []
    for i in range(n):
        inv = inventory[i] if i < len(inventory) else {}
        tri = triage[i]    if i < len(triage)    else {}

        et    = inv.get("evidence_type", "")
        label = inv.get("label", f"Evidence item {i+1}")

        # Prefer explicit triage values; fall back to heuristic defaults for the type
        item = EvidenceRowSuggestion(
            label                = label,
            evidence_type        = et,
            collection_age_hours = inv.get("collection_age_hours", 0),
            evidence_condition   = inv.get("evidence_condition", 2),
            probative_value      = tri.get("probative_value",    _PROBATIVE_MAP.get(et, 2)),
            perishability        = tri.get("perishability",      _PERISHABILITY_MAP.get(et, 1)),
            exclusionary_power   = tri.get("exclusionary_power", _EXCLUSIONARY_MAP.get(et, 2)),
            testing_lead_time    = tri.get("testing_lead_time",  _LEAD_TIME_MAP.get(et, 7)),
            contamination_risk   = tri.get("contamination_risk", _CONTAMINATION_MAP.get(et, 1)),
            specialist_required  = tri.get("specialist_required",_SPECIALIST_REQUIRED_MAP.get(et, 0)),
            specialist_type      = tri.get("specialist_type",    _SPECIALIST_TYPE_MAP.get(et, "")),
        )
        items.append(item)

    return items


def _infer_from_description(description: str) -> EvidenceRowSuggestion:
    """
    Build an EvidenceRowSuggestion by inferring all triage fields from the
    free-text sample description using keyword matching.

    Used for scene inspection form rows that have no explicit P/D/E columns.
    """
    lower = description.lower()

    # Infer evidence_type — first match wins (order in _EVIDENCE_KEYWORDS matters)
    et = ""
    for keywords, candidate in _EVIDENCE_KEYWORDS:
        if any(kw in lower for kw in keywords):
            et = candidate
            break

    return EvidenceRowSuggestion(
        label                = description,
        evidence_type        = et,
        probative_value      = _PROBATIVE_MAP.get(et, 2),
        perishability        = _PERISHABILITY_MAP.get(et, 1),
        exclusionary_power   = _EXCLUSIONARY_MAP.get(et, 2),
        testing_lead_time    = _LEAD_TIME_MAP.get(et, 7),
        contamination_risk   = _CONTAMINATION_MAP.get(et, 1),
        specialist_required  = _SPECIALIST_REQUIRED_MAP.get(et, 0),
        specialist_type      = _SPECIALIST_TYPE_MAP.get(et, ""),
        collection_age_hours = 0,
        evidence_condition   = 2,
    )


# ---------------------------------------------------------------------------
# PRIVATE HELPERS
# ---------------------------------------------------------------------------

def _match_evidence_type(raw: str) -> str:
    """
    Map a raw evidence_type cell to the canonical EVIDENCE_TYPES value.

    First tries an exact match, then substring matching.
    Returns "" if nothing matches.
    """
    clean = raw.strip().lower().replace(" ", "_")
    # Exact match
    if clean in EVIDENCE_TYPES:
        return clean
    # Substring: canonical value contained in raw, or raw contained in canonical
    for et in EVIDENCE_TYPES:
        if et in clean or clean in et:
            return et
    return ""


def _parse_int(val: str, default: int = 0) -> int:
    """Extract the first integer from a string, returning default on failure."""
    m = re.search(r"\d+", val)
    return int(m.group()) if m else default


def _clamp(val: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, val))


def _detect_fir_number(text: str) -> str:
    """
    Return the first FIR / case number found in the text, or ''.

    Post-processes the matched value to strip trailing noise such as
    " dated 14/09/2026" that Docling OCR sometimes includes in the same token.
    """
    for pattern in _FIR_NUMBER_PATTERNS:
        m = re.search(pattern, text, re.IGNORECASE)
        if m:
            value = m.group(1).strip()
            # Strip " dated …" or " dt …" suffix that OCR sometimes attaches
            value = re.sub(r"\s+(?:dated?|dt\.?)\s.*$", "", value, flags=re.IGNORECASE).strip()
            # Strip trailing punctuation
            value = value.rstrip(".,;:")
            if value:
                return value
    return ""


def _detect_offence_type(text: str) -> str:
    """
    Return the best-matching OFFENCE_TYPES value from the document text.

    Scores each candidate by counting how many of its keywords appear in
    the lowercased text; returns the highest-scoring match.
    Returns "" when nothing scores above zero.
    """
    lower = text.lower()
    best_offence = ""
    best_score = 0
    for keywords, offence in _OFFENCE_KEYWORD_MAP:
        score = sum(1 for kw in keywords if kw in lower)
        if score > best_score:
            best_score = score
            best_offence = offence
    return best_offence


def _clean_narrative(text: str) -> str:
    """
    Normalise raw markdown for use in the Case Narrative text area.

    - Strips leading/trailing whitespace
    - Collapses 3+ consecutive blank lines to 2
    - Collapses repeated inline spaces / tabs to one space
    - Truncates to _NARRATIVE_MAX_CHARS
    """
    cleaned = text.strip()
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    cleaned = re.sub(r"[ \t]{2,}", " ", cleaned)
    return cleaned[:_NARRATIVE_MAX_CHARS]
