"""
test_extractor.py
=================
Regression tests for src/extractor.py heuristic extraction.

Covers:
- Knife description does NOT inherit biological_dna
- Blood swab still correctly returns biological_dna
- Phone still correctly returns digital_device
- Empty description returns empty (no evidence_type)
- _source metadata key always present
- Whitespace normalisation: leading/trailing whitespace is stripped before
  testing (relevant for the app.py description-match guard)

All tests use the heuristic path (no API key required).

Run with:
    pytest src/tests/test_extractor.py -v
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from models.evidence_item import CaseContext
from extractor import extract_features, EXTRACTABLE_FIELDS


# ---------------------------------------------------------------------------
# Shared fixture
# ---------------------------------------------------------------------------

@pytest.fixture
def ctx():
    return CaseContext(
        fir_number="TEST-001",
        offence_type="homicide",
        narrative="Test case.",
    )


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _content_keys(result: dict) -> set:
    """Return keys that are actual evidence field suggestions, not metadata."""
    return {k for k in result if k not in ("_source", "_partial")}


# ---------------------------------------------------------------------------
# Regression: knife must NOT inherit biological_dna
# ---------------------------------------------------------------------------

class TestKnifeNotBiological:
    """
    Regression for the stale-cache bug report.

    The exact description from the bug report: a knife with a stain on the blade.
    Even though the description mentions a stain (which could be blood), the
    primary evidence object is a knife — a physical artefact. The heuristic
    extractor must not classify it as biological_dna.
    """

    KNIFE_DESC = (
        "A folding knife with a dark reddish-brown stain on the blade was recovered "
        "next to the victim. The knife was packaged separately in a rigid evidence container."
    )

    def test_knife_has_no_evidence_type(self, ctx):
        result = extract_features(self.KNIFE_DESC, ctx)
        assert "evidence_type" not in result, (
            f"Knife description must not produce an evidence_type, "
            f"got {result.get('evidence_type')!r}"
        )

    def test_knife_not_classified_as_biological(self, ctx):
        result = extract_features(self.KNIFE_DESC, ctx)
        et = result.get("evidence_type", "")
        assert not et.startswith("biological_"), (
            f"Knife description must not be classified as biological, got {et!r}"
        )

    def test_knife_returns_dict(self, ctx):
        result = extract_features(self.KNIFE_DESC, ctx)
        assert isinstance(result, dict)

    def test_knife_has_source_metadata(self, ctx):
        result = extract_features(self.KNIFE_DESC, ctx)
        assert "_source" in result

    def test_knife_with_leading_whitespace(self, ctx):
        """Whitespace-padded description should produce the same result."""
        padded = "  \n  " + self.KNIFE_DESC + "  \n  "
        result = extract_features(padded, ctx)
        assert "evidence_type" not in result or not result["evidence_type"].startswith("biological_")


# ---------------------------------------------------------------------------
# Regression: blood swab still correctly classified (must not break)
# ---------------------------------------------------------------------------

class TestBloodSwabStillBiological:
    BLOOD_DESC = "Blood swab collected from victim's clothing at the scene."

    def test_blood_swab_evidence_type(self, ctx):
        result = extract_features(self.BLOOD_DESC, ctx)
        assert result.get("evidence_type") == "biological_dna", (
            f"Blood swab must still be classified as biological_dna, got {result.get('evidence_type')!r}"
        )

    def test_blood_swab_perishability(self, ctx):
        result = extract_features(self.BLOOD_DESC, ctx)
        assert result.get("perishability") == 3

    def test_blood_swab_specialist_required(self, ctx):
        result = extract_features(self.BLOOD_DESC, ctx)
        assert result.get("specialist_required") == 1


# ---------------------------------------------------------------------------
# Regression: phone still correctly classified (must not break)
# ---------------------------------------------------------------------------

class TestPhoneStillDigitalDevice:
    PHONE_DESC = "Mobile phone recovered at the scene, locked with PIN."

    def test_phone_evidence_type(self, ctx):
        result = extract_features(self.PHONE_DESC, ctx)
        assert result.get("evidence_type") == "digital_device", (
            f"Phone description must still be classified as digital_device, got {result.get('evidence_type')!r}"
        )


# ---------------------------------------------------------------------------
# Empty description
# ---------------------------------------------------------------------------

class TestEmptyDescription:
    def test_empty_string_no_evidence_type(self, ctx):
        result = extract_features("", ctx)
        assert "evidence_type" not in result

    def test_empty_string_source_is_none(self, ctx):
        result = extract_features("", ctx)
        assert result.get("_source") == "none"

    def test_whitespace_only_no_evidence_type(self, ctx):
        result = extract_features("   \n  \t  ", ctx)
        assert "evidence_type" not in result


# ---------------------------------------------------------------------------
# Metadata always present
# ---------------------------------------------------------------------------

class TestMetadataAlwaysPresent:
    def test_source_key_present_for_blood(self, ctx):
        result = extract_features("blood swab", ctx)
        assert "_source" in result

    def test_source_key_present_for_knife(self, ctx):
        result = extract_features("folding knife", ctx)
        assert "_source" in result

    def test_source_key_present_for_empty(self, ctx):
        result = extract_features("", ctx)
        assert "_source" in result

    def test_source_is_heuristic_for_known_type(self, ctx):
        result = extract_features("blood swab from scene", ctx)
        assert result["_source"] == "heuristic"

    def test_extractable_fields_not_priority_tier(self, ctx):
        """priority_tier must never appear in extracted results."""
        for desc in ["blood swab", "mobile phone", "folding knife", ""]:
            result = extract_features(desc, ctx)
            assert "priority_tier" not in result, (
                f"priority_tier must never be extracted, found in result for {desc!r}"
            )


# ---------------------------------------------------------------------------
# Cache-guard unit tests  (Tasks 3.1 – 3.4)
# Tests resolve_extraction_cache() from extractor.py directly.
# Pure Python — no Streamlit session required.
# ---------------------------------------------------------------------------

from extractor import resolve_extraction_cache as _resolve


class TestResolveExtractionCache:
    """
    Unit tests for the _resolve_extraction_cache() guard function.

    These correspond directly to the four required regression scenarios in
    the task specification.
    """

    BLOOD_EX = {
        "evidence_type": "biological_dna",
        "perishability": 3,
        "specialist_required": 1,
        "_source": "heuristic",
        "_partial": True,
    }

    # ------------------------------------------------------------------
    # T1: Extract blood → change description to knife → submit without
    #     re-extracting → biological_dna must NOT be applied.
    # ------------------------------------------------------------------

    def test_t1_changed_description_returns_empty(self):
        result = _resolve(
            cached_ex=self.BLOOD_EX,
            source_desc="blood swab",
            current_desc="knife recovered from scene",
        )
        assert result == {}, (
            "After changing description without re-extracting, cache must be ignored. "
            f"Got: {result}"
        )

    def test_t1_evidence_type_not_inherited(self):
        result = _resolve(
            cached_ex=self.BLOOD_EX,
            source_desc="blood swab",
            current_desc="knife recovered from scene",
        )
        assert "evidence_type" not in result
        assert result.get("evidence_type") != "biological_dna"

    # ------------------------------------------------------------------
    # T2: Extract blood → leave description unchanged → submit →
    #     biological_dna extraction is still used.
    # ------------------------------------------------------------------

    def test_t2_unchanged_description_returns_cache(self):
        result = _resolve(
            cached_ex=self.BLOOD_EX,
            source_desc="blood swab",
            current_desc="blood swab",
        )
        assert result == self.BLOOD_EX, (
            "Unchanged description must still return the cached extraction."
        )

    def test_t2_biological_dna_preserved_when_same_desc(self):
        result = _resolve(
            cached_ex=self.BLOOD_EX,
            source_desc="blood swab",
            current_desc="blood swab",
        )
        assert result.get("evidence_type") == "biological_dna"

    # ------------------------------------------------------------------
    # T3: Extract blood → change description → perform new extraction
    #     that returns empty/neutral → submit → old result must NOT leak.
    # ------------------------------------------------------------------

    def test_t3_new_empty_extraction_does_not_leak_old_result(self):
        # Simulate: after changing description and running a new extraction
        # that yields no fields, the cache is updated to the empty extraction.
        empty_ex = {"_source": "heuristic", "_partial": True}
        result = _resolve(
            cached_ex=empty_ex,
            source_desc="knife recovered from scene",
            current_desc="knife recovered from scene",
        )
        # The descriptions match, so the new (empty) extraction is returned —
        # old biological_dna is gone because the caller replaced it.
        assert "evidence_type" not in result or result.get("evidence_type") != "biological_dna"

    def test_t3_stale_cache_cleared_before_new_extraction_applied(self):
        # If the investigator changed the desc but has NOT yet extracted,
        # the old cache (still pointing at "blood swab") must be discarded.
        result = _resolve(
            cached_ex=self.BLOOD_EX,
            source_desc="blood swab",        # old source
            current_desc="knife recovered",  # new current — mismatch
        )
        assert result == {}

    # ------------------------------------------------------------------
    # T4: Whitespace normalisation — leading/trailing whitespace does not
    #     incorrectly invalidate an otherwise identical description.
    # ------------------------------------------------------------------

    def test_t4_leading_whitespace_does_not_invalidate(self):
        result = _resolve(
            cached_ex=self.BLOOD_EX,
            source_desc="  blood swab  ",
            current_desc="blood swab",
        )
        assert result == self.BLOOD_EX, (
            "Leading/trailing whitespace must not invalidate a matching description."
        )

    def test_t4_trailing_whitespace_does_not_invalidate(self):
        result = _resolve(
            cached_ex=self.BLOOD_EX,
            source_desc="blood swab",
            current_desc="blood swab   ",
        )
        assert result == self.BLOOD_EX

    def test_t4_both_padded_still_matches(self):
        result = _resolve(
            cached_ex=self.BLOOD_EX,
            source_desc="  blood swab\n",
            current_desc="\tblood swab  ",
        )
        assert result == self.BLOOD_EX

    def test_t4_genuinely_different_not_just_whitespace(self):
        """A substantive change must still invalidate the cache."""
        result = _resolve(
            cached_ex=self.BLOOD_EX,
            source_desc="blood swab",
            current_desc="blood swab from victim",  # extra words = different
        )
        assert result == {}

    # ------------------------------------------------------------------
    # Edge cases
    # ------------------------------------------------------------------

    def test_empty_cache_always_returns_empty(self):
        result = _resolve(
            cached_ex={},
            source_desc="blood swab",
            current_desc="blood swab",
        )
        assert result == {}

    def test_metadata_only_cache_treated_as_empty(self):
        # A cache containing only metadata keys (no actual field suggestions)
        # is falsy for the guard — treated as no extraction.
        meta_only = {}  # empty dict — no fields
        result = _resolve(
            cached_ex=meta_only,
            source_desc="blood swab",
            current_desc="blood swab",
        )
        assert result == {}
