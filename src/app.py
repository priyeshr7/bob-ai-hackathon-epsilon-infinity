"""
app.py
======
EvidencePro — Streamlit Investigator Platform
----------------------------------------------
Launch with:
    streamlit run src/app.py

IMPORTANT NOTICES
-----------------
• All AI/ML outputs in this application are RECOMMENDATIONS only.
  The investigator is the final decision-maker for all forensic triage
  and evidence submission decisions.

• This prototype is trained on SYNTHETIC / DEMONSTRATION DATA. Accuracy
  figures do not represent real-world forensic performance.

• Any policy weights are a CONFIGURABLE PROTOTYPE MECHANISM and are NOT
  official forensic standards.

Architecture
------------
This file is the presentation and orchestration layer only.
All ML, policy, scheduling, and report logic lives in the imported modules:

  models/triage_models.py  → ML training, prediction, explanation
  models/evidence_item.py  → EvidenceItem, CaseContext dataclasses
  models/policy.py         → TriagePolicy, governance, apply_policy
  scheduler.py             → build_schedule, ScheduledItem
  report.py                → generate_report, ReportData
  document_processor.py   → process_document, ProcessedDocument
  feedback.py             → compute_override_stats, request_candidate_retraining
  pdf_report.py           → generate_pdf_report (ReportLab A4)

No ML logic, policy formulas, or batch assignment rules are defined here.
"""

import sys
import os
import tempfile
from datetime import datetime, timezone

import streamlit as st

# Ensure src/ is on the path when launched from project root
sys.path.insert(0, os.path.dirname(__file__))

from extractor import (
    extract_features, extraction_mode_label, is_watsonx_available,
    resolve_extraction_cache,
)
from document_processor import process_document, SUPPORTED_EXTENSIONS
from feedback import (
    compute_override_stats, request_candidate_retraining,
    CANDIDATE_HISTORY, get_pending_candidates,
    RETRAINING_THRESHOLD,
)

from models.evidence_item import (
    EvidenceItem, CaseContext, EVIDENCE_TYPES, OFFENCE_TYPES, PRIORITY_LABELS
)
from models.triage_models import (
    get_trained_model, predict_priority, validate_item, DATA_DISCLAIMER,
    TrainingResult,
)
from models.policy import (
    TriagePolicy, PolicyRole,
    POLICY_HISTORY,
    propose_policy, approve_policy, reject_policy, get_active_policy,
    apply_policy, policy_summary,
)
from scheduler import build_schedule, schedule_summary, BATCH_IMMEDIATE, BATCH_SECONDARY, BATCH_ARCHIVE
from report import ReportData, generate_report

# ---------------------------------------------------------------------------
# PAGE CONFIG (must be first Streamlit call)
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="EvidencePro",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# SESSION STATE INITIALISATION
# ---------------------------------------------------------------------------

def _init_state():
    defaults = {
        "case_context":         None,        # CaseContext
        "evidence_items":       [],          # list[EvidenceItem]
        "training_result":      None,        # TrainingResult
        "triage_results":       [],          # list[TriageResult]
        "overrides":            {},          # dict[item_id -> (tier, reason)]
        "schedule":             [],          # list[ScheduledItem]
        "role":                 "investigator",
        "model_name":           "decision_tree",
        "item_counter":         0,
        "triage_run":           False,
        "schedule_built":       False,
        "_extraction":          {},
        "_extraction_source_desc": "",
        # Document upload state
        "doc_processed":          None,        # ProcessedDocument | None
        "doc_candidates":         [],          # list of candidate strings from document
        "doc_candidates_edited":  [],          # investigator-edited candidate strings
        "doc_candidates_imported":False,       # True once candidates have been bulk-imported as EvidenceItems
        "doc_item_ids":           set(),       # item_ids created from document import (for re-import cleanup)
        "doc_input_mode":         "document",  # "document" | "manual"
        # Retraining state
        "retrain_requested":    False,
        "retrain_requester":    "",
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val

_init_state()

# ---------------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------------

MODEL_DISPLAY = {
    "decision_tree":      "Decision Tree",
    "random_forest":      "Random Forest",
    "gradient_boosting":  "Gradient Boosting",
}

ROLE_DISPLAY = {
    "investigator": "Investigator",
    "policy_admin": "Policy Admin",
    "approver":     "Approver",
}

PRIORITY_COLOURS = {
    "Critical": "🔴",
    "High":     "🟠",
    "Standard": "🟡",
    "Low":      "🟢",
}

ORDINAL_LABELS = {1: "Low (1)", 2: "Medium (2)", 3: "High (3)"}
BINARY_LABELS  = {0: "No", 1: "Yes"}

def _tier_badge(tier: str) -> str:
    return f"{PRIORITY_COLOURS.get(tier, '')} **{tier}**"

def _active_policy() -> TriagePolicy | None:
    return get_active_policy()

def _role_enum() -> PolicyRole:
    r = st.session_state.role
    return {
        "investigator": PolicyRole.INVESTIGATOR,
        "policy_admin": PolicyRole.POLICY_ADMIN,
        "approver":     PolicyRole.APPROVER,
    }[r]

def _next_item_id() -> str:
    st.session_state.item_counter += 1
    return f"E{st.session_state.item_counter:03d}"

def _retrain():
    with st.spinner(f"Training {MODEL_DISPLAY[st.session_state.model_name]}…"):
        st.session_state.training_result = get_trained_model(st.session_state.model_name)
    st.session_state.triage_run     = False
    st.session_state.triage_results = []
    st.session_state.schedule       = []
    st.session_state.schedule_built = False

# ---------------------------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------------------------

with st.sidebar:
    st.title("🔬 EvidencePro")
    st.caption("AI-Assisted Forensic Evidence Triage · Prototype")

    st.warning(
        "⚠️ **AI Recommendations Only**\n\n"
        "All outputs are AI-assisted recommendations. "
        "The investigator is the **final decision-maker**.",
        icon=None,
    )

    st.divider()

    st.subheader("Role")
    new_role = st.selectbox(
        "Session role (prototype simulation — no real authentication)",
        options=list(ROLE_DISPLAY.keys()),
        format_func=lambda k: ROLE_DISPLAY[k],
        key="role",
    )

    st.divider()

    st.subheader("ML Model")
    prev_model = st.session_state.model_name
    new_model = st.selectbox(
        "Model",
        options=list(MODEL_DISPLAY.keys()),
        format_func=lambda k: MODEL_DISPLAY[k],
        key="model_name",
    )

    if st.button("Train / Retrain Model", use_container_width=True):
        _retrain()
        st.success(f"Model trained: {MODEL_DISPLAY[st.session_state.model_name]}")

    if st.session_state.training_result is not None:
        tr = st.session_state.training_result
        st.caption(
            f"Active: **{MODEL_DISPLAY[tr.model_name]}** · "
            f"CV accuracy: {tr.cv_accuracy * 100:.1f}%"
        )
        st.caption(f"⚠️ {DATA_DISCLAIMER}")
    else:
        st.info("No model trained yet. Click **Train / Retrain Model** above.")

    st.divider()

    st.subheader("Active Policy")
    active_pol = _active_policy()
    if active_pol:
        st.success(
            f"**{active_pol.label}** · v{active_pol.version}\n\n"
            f"P={active_pol.weight_P} · D={active_pol.weight_D} · "
            f"E={active_pol.weight_E} · S={active_pol.weight_S}"
        )
        st.caption(
            "⚠️ Policy weights are a CONFIGURABLE PROTOTYPE MECHANISM, "
            "not official forensic standards."
        )
    else:
        st.info("No active policy — unweighted ML in use.")

# ---------------------------------------------------------------------------
# MAIN CONTENT — TABS
# ---------------------------------------------------------------------------

tab_doc, tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
    "📄 Document",
    "1 · Case Context",
    "2 · Evidence",
    "3 · Run Triage",
    "4 · Explanations",
    "5 · Review",
    "6 · Schedule",
    "7 · Report",
    "8 · Policy Admin",
])

# ===========================================================================
# TAB 0 — DOCUMENT UPLOAD (new primary entry point)
# ===========================================================================
with tab_doc:
    st.header("Upload Case Document")
    st.caption(
        "Upload a forensic case document. EvidencePro will extract the text and "
        "identify candidate evidence items for your review. "
        "Supported formats: **PDF, DOCX, JPG, PNG, TXT**."
    )

    st.info(
        "**Optional but recommended:** Upload a forensic form here to auto-populate evidence candidates. "
        "You can also skip this step and enter evidence manually in **Tab 2 · Evidence**."
    )

    uploaded_file = st.file_uploader(
        "Choose a document",
        type=["pdf", "docx", "jpg", "jpeg", "png", "txt"],
        help="The document will be processed automatically once uploaded.",
    )

    if uploaded_file is not None:
        # Auto-process on upload (only when a new file is detected)
        if (
            st.session_state.doc_processed is None
            or st.session_state.doc_processed.filename != uploaded_file.name
        ):
            with st.spinner(f"Processing **{uploaded_file.name}**…"):
                suffix = os.path.splitext(uploaded_file.name)[1]
                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                    tmp.write(uploaded_file.getvalue())
                    tmp_path = tmp.name
                try:
                    doc = process_document(tmp_path, uploaded_file.name)
                finally:
                    try:
                        os.unlink(tmp_path)
                    except Exception:
                        pass

            st.session_state.doc_processed            = doc
            st.session_state.doc_candidates           = list(doc.candidates)
            st.session_state.doc_candidates_edited    = list(doc.candidates)
            st.session_state.doc_candidates_imported  = False   # reset on new document

        doc = st.session_state.doc_processed

        if doc.success:
            if doc.processor == "docling":
                st.success(f"✓ Processed with **Docling** · {len(doc.full_text):,} characters extracted")
            else:
                st.info(f"✓ Processed with **{doc.processor}** fallback · {len(doc.full_text):,} characters extracted")

            with st.expander("📄 Extracted document text (review)"):
                st.text_area(
                    "Full extracted text",
                    value=doc.preview,
                    height=300,
                    disabled=True,
                    label_visibility="collapsed",
                )

            st.divider()

            _n_shown = len(doc.candidates)
            _n_total = getattr(doc, "candidates_total", _n_shown)
            st.subheader(f"🔍 Detected Evidence Candidates ({_n_shown})")

            if not doc.candidates:
                st.warning(
                    "No clear evidence items were automatically detected in this document. "
                    "You can add candidates manually below, or enter evidence in Tab 2 directly."
                )
            elif _n_total > _n_shown:
                st.warning(
                    f"⚠️ **{_n_total} candidates were detected; showing the first {_n_shown}** "
                    f"(practical display limit). "
                    "If your document has more exhibits, add them manually using the form below."
                )

            # ---- Editable candidate list (pre-import review) ----
            edited_candidates = []
            candidates_to_remove = []
            current_candidates = st.session_state.doc_candidates_edited or []

            if current_candidates:
                st.caption(
                    f"{len(current_candidates)} candidate(s) — review and edit the text, remove any "
                    "that are not evidence items, then click **Import All as Evidence Items**."
                )
                for i, cand in enumerate(current_candidates):
                    col_text, col_del = st.columns([9, 1])
                    with col_text:
                        edited = st.text_input(
                            f"Candidate {i+1}",
                            value=cand,
                            key=f"cand_{i}",
                            label_visibility="collapsed",
                        )
                        edited_candidates.append(edited)
                    with col_del:
                        if st.button("✕", key=f"del_cand_{i}", help="Remove this candidate"):
                            candidates_to_remove.append(i)

                if candidates_to_remove:
                    st.session_state.doc_candidates_edited = [
                        c for j, c in enumerate(st.session_state.doc_candidates_edited)
                        if j not in candidates_to_remove
                    ]
                    st.rerun()
                else:
                    st.session_state.doc_candidates_edited = edited_candidates

            # Add a new candidate manually
            with st.form("add_candidate_form"):
                new_cand = st.text_input(
                    "Add a candidate not detected above",
                    placeholder="e.g. Knife with reddish-brown stain recovered at scene",
                )
                if st.form_submit_button("➕ Add Candidate"):
                    if new_cand.strip():
                        st.session_state.doc_candidates_edited.append(new_cand.strip())
                        st.session_state.doc_candidates_imported = False
                        st.rerun()

            st.divider()

            # ---- IMPORT BUTTON (the key UX change) ----
            ready_candidates = [c for c in st.session_state.doc_candidates_edited if c.strip()]

            if st.session_state.doc_candidates_imported:
                st.success(
                    f"✅ **{len(ready_candidates)} evidence item(s) already imported** into the Evidence tab. "
                    "Go to **Tab 2 · Evidence** to review and edit them, then click **Analyze Evidence**."
                )
                if st.button("↩ Re-import (clears existing document-sourced items)", type="secondary"):
                    # Remove items created by the previous import, identified by stored IDs
                    prev_doc_ids = st.session_state.doc_item_ids
                    st.session_state.evidence_items = [
                        it for it in st.session_state.evidence_items
                        if it.item_id not in prev_doc_ids
                    ]
                    st.session_state.doc_item_ids            = set()
                    st.session_state.doc_candidates_imported = False
                    st.session_state.triage_run     = False
                    st.session_state.triage_results = []
                    st.session_state.schedule       = []
                    st.session_state.schedule_built = False
                    st.rerun()
            else:
                if not st.session_state.case_context:
                    st.warning(
                        "⚠️ Complete **Tab 1 · Case Context** first (FIR number + offence type required "
                        "for AI field extraction), then return here to import."
                    )
                else:
                    if ready_candidates:
                        if st.button(
                            f"✅ Import All {len(ready_candidates)} Candidate(s) as Evidence Items →",
                            type="primary",
                            use_container_width=True,
                            help="Runs AI extraction on each candidate and creates editable evidence items in Tab 2.",
                        ):
                            ctx_for_import = st.session_state.case_context
                            # Remove any previously doc-imported items before re-importing
                            prev_doc_ids = st.session_state.doc_item_ids
                            st.session_state.evidence_items = [
                                it for it in st.session_state.evidence_items
                                if it.item_id not in prev_doc_ids
                            ]
                            new_doc_ids = set()
                            imported_count = 0
                            with st.spinner(f"Extracting fields for {len(ready_candidates)} candidate(s)…"):
                                for cand_text in ready_candidates:
                                    if not cand_text.strip():
                                        continue
                                    # Run AI/heuristic extraction for field suggestions
                                    ex = extract_features(
                                        description=cand_text,
                                        context=ctx_for_import,
                                    )
                                    new_id = _next_item_id()
                                    new_item = EvidenceItem(
                                        item_id=new_id,
                                        label=cand_text.strip()[:200],
                                        evidence_type=ex.get("evidence_type", EVIDENCE_TYPES[0]),
                                        offence_type=ctx_for_import.offence_type,
                                        probative_value=int(ex.get("probative_value", 2)),
                                        perishability=int(ex.get("perishability", 1)),
                                        exclusionary_power=int(ex.get("exclusionary_power", 2)),
                                        contamination_risk=int(ex.get("contamination_risk", 1)),
                                        specialist_required=int(ex.get("specialist_required", 0)),
                                        testing_lead_time=int(ex.get("testing_lead_time", 7)),
                                        collection_age_hours=0,
                                        evidence_condition=int(ex.get("evidence_condition", 2)),
                                        specialist_type=ex.get("specialist_type", ""),
                                        ai_extracted=True,
                                    )
                                    st.session_state.evidence_items.append(new_item)
                                    new_doc_ids.add(new_id)
                                    imported_count += 1

                            st.session_state.doc_item_ids         = new_doc_ids
                            st.session_state.doc_candidates_imported = True
                            st.session_state.triage_run     = False
                            st.session_state.triage_results = []
                            st.session_state.schedule       = []
                            st.session_state.schedule_built = False
                            st.success(
                                f"✅ **{imported_count} evidence item(s) imported** with AI-suggested fields. "
                                "Go to **Tab 2 · Evidence** to review, edit, and confirm each item, "
                                "then click **Analyze Evidence** in Tab 3."
                            )
                            st.rerun()
                    else:
                        st.info("Add at least one candidate above before importing.")

        else:
            st.error(f"❌ Document processing failed: {doc.error}")
            st.info("You can still enter evidence manually in **Tab 2 · Evidence**.")

    else:
        st.info(
            "No document uploaded yet. "
            "Upload a PDF, DOCX, image, or text file above, "
            "or go directly to **Tab 2 · Evidence** to enter items manually."
        )
        if st.session_state.doc_processed:
            st.caption("Previously processed document cleared.")
            if st.button("Clear document state"):
                st.session_state.doc_processed           = None
                st.session_state.doc_candidates          = []
                st.session_state.doc_candidates_edited   = []
                st.session_state.doc_candidates_imported = False
                st.rerun()


# ===========================================================================
# TAB 1 — CASE CONTEXT
# ===========================================================================
with tab1:
    st.header("Step 1 — Case / FIR Context")
    st.caption(
        "Enter the case details. The offence type is used by the ML model "
        "as a feature for every evidence item in this session."
    )

    with st.form("case_context_form"):
        fir_number   = st.text_input("FIR / Case Number", placeholder="e.g. FIR-2024-001")
        offence_type = st.selectbox("Offence Type", options=OFFENCE_TYPES)
        narrative    = st.text_area(
            "Case Narrative (optional)",
            placeholder="Brief description of the incident…",
            height=120,
        )
        submitted = st.form_submit_button("Save Case Context", use_container_width=True)

    if submitted:
        if not fir_number.strip():
            st.error("FIR / Case Number is required.")
        else:
            st.session_state.case_context = CaseContext(
                fir_number=fir_number.strip(),
                offence_type=offence_type,
                narrative=narrative.strip(),
            )
            st.success(f"Case context saved: **{fir_number}** · {offence_type}")

    if st.session_state.case_context:
        ctx = st.session_state.case_context
        st.info(
            f"**Current case:** {ctx.fir_number} · {ctx.offence_type}"
            + (f"\n\n_{ctx.narrative}_" if ctx.narrative else "")
        )


# ===========================================================================
# TAB 2 — EVIDENCE ITEMS
# ===========================================================================
with tab2:
    st.header("Step 2 — Evidence Items")

    if not st.session_state.case_context:
        st.warning("⬅️ Please complete **Step 1 — Case Context** first.")
    else:
        ctx = st.session_state.case_context

        # ---- DOCUMENT-IMPORTED ITEMS NOTICE ----
        if st.session_state.doc_candidates_imported and st.session_state.doc_item_ids:
            n_imported = len(st.session_state.doc_item_ids)
            st.info(
                f"📄 **{n_imported} evidence item(s) imported from the uploaded document** "
                "with AI-suggested fields (marked 🤖). "
                "Review each item below — edit any field, remove items you don't need, "
                "then click **Analyze Evidence** in Tab 3. "
                "Use the form below to add additional items manually."
            )
        elif st.session_state.doc_processed and st.session_state.doc_processed.success and not st.session_state.doc_candidates_imported:
            st.warning(
                "📄 A document has been processed but candidates have not been imported yet. "
                "Go to the **📄 Document** tab and click **Import All as Evidence Items**."
            )

        # ---- ADD EVIDENCE FORM ----
        with st.expander("➕ Add a new evidence item manually", expanded=not st.session_state.evidence_items):

            # ------------------------------------------------------------------
            # AI EXTRACTION PANEL (outside the form)
            # ------------------------------------------------------------------
            st.markdown("#### AI-Assisted Feature Extraction *(optional)*")
            st.caption(
                f"Extraction mode: **{extraction_mode_label()}** — "
                "extracted values are suggestions only. Review every field before adding the item."
            )

            desc_input = st.text_area(
                "Evidence description (free text)",
                value=st.session_state._extraction_source_desc,
                placeholder=(
                    "e.g. Blood swab collected from victim's clothing at the scene, "
                    "stored in sealed forensic bag."
                ),
                height=90,
                key="_desc_textarea",
            )

            ex = resolve_extraction_cache(
                cached_ex=st.session_state._extraction,
                source_desc=st.session_state._extraction_source_desc,
                current_desc=desc_input,
            )

            if st.button(
                "🔍 Extract with AI",
                help="Analyse the description and suggest field values. You can correct any value before adding.",
                use_container_width=False,
            ):
                if not desc_input.strip():
                    st.warning("Enter a description before extracting.")
                else:
                    with st.spinner("Extracting features…"):
                        extracted = extract_features(description=desc_input, context=ctx)
                    st.session_state._extraction = extracted
                    st.session_state._extraction_source_desc = desc_input
                    ex = extracted
                    source = extracted.get("_source", "none")
                    if source == "watsonx.ai":
                        st.success("✅ Extracted via IBM watsonx.ai. Review all values below.")
                    elif source == "heuristic":
                        st.info("ℹ️ Heuristic extraction (no API key). Some fields pre-filled from keywords. Review carefully.")
                    else:
                        st.warning("No features could be extracted. Please fill in the form manually.")

            def _ai_badge(field: str) -> str:
                if field in ex and "_source" in ex and ex.get("_source") != "none":
                    return " 🤖"
                return ""

            st.divider()

            # ------------------------------------------------------------------
            # EVIDENCE FORM
            # ------------------------------------------------------------------
            with st.form("add_evidence_form", clear_on_submit=True):
                st.markdown("**Item details**")
                label = st.text_input(
                    "Short label / description",
                    value=desc_input if ex else "",
                    placeholder="e.g. Blood swab from victim's clothing",
                )

                col1, col2 = st.columns(2)
                with col1:
                    _et_default = ex.get("evidence_type", EVIDENCE_TYPES[0])
                    _et_index   = EVIDENCE_TYPES.index(_et_default) if _et_default in EVIDENCE_TYPES else 0
                    evidence_type = st.selectbox(
                        f"Evidence Type{_ai_badge('evidence_type')}",
                        EVIDENCE_TYPES,
                        index=_et_index,
                    )
                with col2:
                    st.text_input("Offence Type (from case context)", value=ctx.offence_type, disabled=True)

                st.markdown("**PDES dimensions**")
                st.caption(
                    "P = Probative Value · D = Degradation/Perishability Risk · "
                    "E = Exclusionary Power · S = Processing Speed (via lead time)"
                )

                col_p, col_d, col_e = st.columns(3)
                with col_p:
                    _pv_default = int(ex.get("probative_value", 2)) - 1
                    probative_value = st.selectbox(
                        f"Probative Value (P){_ai_badge('probative_value')}",
                        options=[1, 2, 3],
                        format_func=lambda v: ORDINAL_LABELS[v],
                        index=max(0, min(2, _pv_default)),
                    )
                with col_d:
                    _per_default = int(ex.get("perishability", 1)) - 1
                    perishability = st.selectbox(
                        f"Degradation Risk (D){_ai_badge('perishability')}",
                        options=[1, 2, 3],
                        format_func=lambda v: {1: "Stable (1)", 2: "Degrades over days (2)", 3: "Degrades in hours (3)"}[v],
                        index=max(0, min(2, _per_default)),
                    )
                with col_e:
                    _ep_default = int(ex.get("exclusionary_power", 2)) - 1
                    exclusionary_power = st.selectbox(
                        f"Exclusionary Power (E){_ai_badge('exclusionary_power')}",
                        options=[1, 2, 3],
                        format_func=lambda v: ORDINAL_LABELS[v],
                        index=max(0, min(2, _ep_default)),
                    )

                st.markdown("**Secondary features**")
                col_a, col_b, col_c = st.columns(3)
                with col_a:
                    _cr_default = int(ex.get("contamination_risk", 1)) - 1
                    contamination_risk = st.selectbox(
                        f"Contamination Risk{_ai_badge('contamination_risk')}",
                        options=[1, 2, 3],
                        format_func=lambda v: ORDINAL_LABELS[v],
                        index=max(0, min(2, _cr_default)),
                    )
                with col_b:
                    _sr_default = int(ex.get("specialist_required", 0))
                    specialist_required = st.selectbox(
                        f"Specialist Required{_ai_badge('specialist_required')}",
                        options=[0, 1],
                        format_func=lambda v: BINARY_LABELS[v],
                        index=max(0, min(1, _sr_default)),
                    )
                with col_c:
                    _lt_default = int(ex.get("testing_lead_time", 7))
                    testing_lead_time = st.number_input(
                        f"Testing Lead Time (days){_ai_badge('testing_lead_time')}",
                        min_value=1, max_value=60,
                        value=max(1, min(60, _lt_default)),
                    )

                st.markdown("**Operational details**")
                col_op1, col_op2, col_op3 = st.columns(3)
                with col_op1:
                    collection_age_hours = st.number_input(
                        "Collection Age (hours)",
                        min_value=0, max_value=720, value=0,
                    )
                with col_op2:
                    _ec_default = int(ex.get("evidence_condition", 2)) - 1
                    evidence_condition = st.selectbox(
                        f"Evidence Condition{_ai_badge('evidence_condition')}",
                        options=[1, 2, 3],
                        format_func=lambda v: {1: "Poor (1)", 2: "Fair (2)", 3: "Good (3)"}[v],
                        index=max(0, min(2, _ec_default)),
                    )
                with col_op3:
                    specialist_type = st.text_input(
                        f"Specialist Type (optional){_ai_badge('specialist_type')}",
                        value=ex.get("specialist_type", ""),
                        placeholder="e.g. DNA analyst",
                    )

                if ex and any(k not in ("_source", "_partial") for k in ex):
                    st.info(
                        f"🤖 Fields marked **🤖** were suggested by AI extraction "
                        f"({ex.get('_source', 'unknown')} mode). "
                        "Review and correct before adding. "
                        "**All AI suggestions require investigator confirmation.**"
                    )

                add_submitted = st.form_submit_button("✅ Add Evidence Item", use_container_width=True)

            if add_submitted:
                if not label.strip():
                    st.error("Label / description is required.")
                else:
                    _ai_extracted = bool(ex and any(k not in ("_source", "_partial") for k in ex))
                    new_item = EvidenceItem(
                        item_id=_next_item_id(),
                        label=label.strip(),
                        evidence_type=evidence_type,
                        offence_type=ctx.offence_type,
                        probative_value=probative_value,
                        perishability=perishability,
                        exclusionary_power=exclusionary_power,
                        contamination_risk=contamination_risk,
                        specialist_required=specialist_required,
                        testing_lead_time=int(testing_lead_time),
                        collection_age_hours=int(collection_age_hours),
                        evidence_condition=evidence_condition,
                        specialist_type=specialist_type.strip(),
                        ai_extracted=_ai_extracted,
                    )
                    errors = validate_item(new_item)
                    if errors:
                        for e in errors:
                            st.error(e)
                    else:
                        st.session_state.evidence_items.append(new_item)
                        st.session_state._extraction = {}
                        st.session_state._extraction_source_desc = ""
                        st.session_state.triage_run     = False
                        st.session_state.triage_results = []
                        st.session_state.schedule       = []
                        st.session_state.schedule_built = False
                        ai_note = " *(AI-assisted)*" if _ai_extracted else ""
                        st.success(f"Item added: **{new_item.item_id}** — {new_item.label}{ai_note}")

        # ---- EVIDENCE LIST ----
        items = st.session_state.evidence_items
        doc_item_ids = st.session_state.doc_item_ids  # set of IDs imported from document
        if not items:
            st.info("No evidence items yet. Upload a document and import candidates, or use the form above.")
        else:
            st.subheader(f"Evidence items ({len(items)}) — review before analysis")
            st.caption(
                "🤖 = AI-suggested fields (from document extraction). "
                "Review and correct any field before clicking **Analyze Evidence**."
            )
            remove_id = None
            for item in items:
                is_doc_item = item.item_id in doc_item_ids
                label_prefix = "🤖 " if is_doc_item else ""
                # Doc-imported items default to expanded so investigators notice them
                with st.expander(
                    f"{label_prefix}**{item.item_id}** · {item.label}",
                    expanded=is_doc_item,
                ):
                    if is_doc_item:
                        st.caption(
                            "🤖 Fields below were suggested by AI extraction from the uploaded document. "
                            "**Review and correct each field before running analysis.**"
                        )
                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("Evidence type", item.evidence_type.replace("_", " "))
                    c2.metric("Probative (P)", item.probative_value)
                    c3.metric("Degradation (D)", item.perishability)
                    c4.metric("Exclusionary (E)", item.exclusionary_power)
                    c5, c6, c7, c8 = st.columns(4)
                    c5.metric("Lead time (days)", item.testing_lead_time)
                    c6.metric("Contamination risk", item.contamination_risk)
                    c7.metric("Specialist req.", "Yes" if item.specialist_required else "No")
                    c8.metric("Collection age (h)", item.collection_age_hours if item.collection_age_hours else "Unknown")
                    if item.specialist_type:
                        st.caption(f"Specialist type: {item.specialist_type}")
                    if st.button(f"🗑 Remove {item.item_id}", key=f"remove_{item.item_id}"):
                        remove_id = item.item_id
            if remove_id:
                st.session_state.evidence_items = [
                    i for i in st.session_state.evidence_items if i.item_id != remove_id
                ]
                # If a doc item is removed, untrack it
                st.session_state.doc_item_ids = {
                    iid for iid in st.session_state.doc_item_ids if iid != remove_id
                }
                st.session_state.triage_run     = False
                st.session_state.triage_results = []
                st.session_state.schedule       = []
                st.session_state.schedule_built = False
                st.rerun()

        # ---- TRAIN MODEL PROMPT ----
        if not st.session_state.training_result and items:
            st.warning(
                "⬅️ No model trained yet. Go to the **sidebar** and click "
                "**Train / Retrain Model** before running triage."
            )


# ===========================================================================
# TAB 3 — RUN TRIAGE
# ===========================================================================
with tab3:
    st.header("Step 3 — Run Triage")
    st.caption(
        "The ML model classifies each evidence item and assigns a priority tier. "
        "All results are **AI-assisted recommendations** — the investigator reviews "
        "and makes the final decision in Step 5."
    )

    items = st.session_state.evidence_items
    tr    = st.session_state.training_result

    col_run1, col_run2 = st.columns([2, 1])
    with col_run1:
        if not st.session_state.case_context:
            st.warning("Complete Step 1 (Case Context) first.")
        elif not items:
            st.warning("Add at least one evidence item in Step 2.")
        elif tr is None:
            st.warning("Train a model using the sidebar first.")
        else:
            if st.button(
                f"▶ Analyze Evidence ({len(items)} item{'s' if len(items) != 1 else ''})",
                use_container_width=True,
                type="primary",
            ):
                active_pol = _active_policy()
                with st.spinner("Running triage…"):
                    results = [predict_priority(item, tr) for item in items]
                st.session_state.triage_results  = results
                st.session_state.triage_run      = True
                st.session_state.schedule_built  = False
                st.session_state.schedule        = []
                st.success(f"Triage complete. {len(results)} item(s) classified.")

    with col_run2:
        if tr:
            st.info(f"Model: **{MODEL_DISPLAY[tr.model_name]}**\nCV accuracy: {tr.cv_accuracy*100:.1f}%")

    if st.session_state.triage_run and st.session_state.triage_results:
        active_pol = _active_policy()
        results    = st.session_state.triage_results

        st.divider()
        st.subheader("AI Triage Results")
        st.caption(
            "🔴 Critical · 🟠 High · 🟡 Standard · 🟢 Low\n\n"
            "⚠️ = Urgent (perishable or recently collected degrading evidence)"
        )

        import pandas as pd
        rows = []
        for item, result in zip(items, results):
            pol_result = apply_policy(result, item, active_pol) if active_pol else None
            urgency = "⚠️ YES" if result.urgency_flag else "No"
            raw_tier = f"{PRIORITY_COLOURS.get(result.priority_tier, '')} {result.priority_tier}"
            pol_tier = (
                f"{PRIORITY_COLOURS.get(pol_result.priority_tier, '')} {pol_result.priority_tier}"
                if pol_result and pol_result.priority_tier != result.priority_tier
                else "—"
            )
            rows.append({
                "ID": item.item_id,
                "Label": item.label,
                "Raw ML Tier": raw_tier,
                "Policy Tier": pol_tier,
                "Urgency": urgency,
                "Score": f"{result.priority_score:.3f}",
                "Specialist": "Yes" if item.specialist_required else "No",
            })

        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        if active_pol:
            st.info(
                f"Active policy: **{active_pol.label}** v{active_pol.version} · "
                "Policy Tier column shows adjusted tier where it differs from raw ML tier.\n\n"
                "⚠️ Policy weights are a CONFIGURABLE PROTOTYPE MECHANISM, not official forensic standards."
            )
        else:
            st.caption("No active policy — raw ML tiers shown.")

        st.info("→ Proceed to **Tab 4 · Explanations** or **Tab 5 · Review** to review and override results.")


# ===========================================================================
# TAB 4 — EXPLANATIONS
# ===========================================================================
with tab4:
    st.header("Step 4 — Explanations")
    st.caption(
        "Understand why each item received its recommendation. "
        "**Decision Tree** shows the exact rule path followed for this item. "
        "**Random Forest / Gradient Boosting** show the item's feature values "
        "and the model's global feature importances (not per-item causation)."
    )

    if not st.session_state.triage_run or not st.session_state.triage_results:
        st.info("Run triage in Step 3 first.")
    else:
        import pandas as pd
        items   = st.session_state.evidence_items
        results = st.session_state.triage_results
        active_pol = _active_policy()
        tr = st.session_state.training_result

        # Model accuracy context
        if tr:
            with st.expander("📊 Model Accuracy & Validation Context", expanded=False):
                st.markdown(
                    f"**Model:** {MODEL_DISPLAY[tr.model_name]}  \n"
                    f"**Cross-validated accuracy:** {tr.cv_accuracy * 100:.1f}%  \n"
                )
                st.warning(
                    f"⚠️ {DATA_DISCLAIMER}\n\n"
                    "This accuracy was calculated dynamically using stratified k-fold "
                    "cross-validation on the **synthetic demonstration dataset** used for "
                    "training. It does NOT represent validated real-world forensic performance. "
                    "The model has not been tested against real forensic casework."
                )

        for item, result in zip(items, results):
            pol_result = apply_policy(result, item, active_pol) if active_pol else None
            urgent_str = " · ⚠️ **URGENT**" if result.urgency_flag else ""

            with st.expander(
                f"**{item.item_id}** · {item.label} — {PRIORITY_COLOURS.get(result.priority_tier,'')} {result.priority_tier}{urgent_str}",
                expanded=False,
            ):
                col_raw, col_pol = st.columns(2) if pol_result else (st.container(), None)

                with col_raw:
                    st.markdown("##### Raw ML Recommendation")
                    st.markdown(f"**Priority tier:** {_tier_badge(result.priority_tier)}")
                    st.markdown(f"**Confidence score:** {result.priority_score:.3f}")
                    if result.urgency_flag:
                        st.error("⚠️ URGENT — this evidence requires immediate attention.")
                    st.markdown("**PDES summary:**")
                    st.markdown(
                        f"- P (Probative Value): **{result.item_feature_values.get('probative_value', '?')}**\n"
                        f"- D (Degradation Risk): **{result.item_feature_values.get('perishability', '?')}**\n"
                        f"- E (Exclusionary Power): **{result.item_feature_values.get('exclusionary_power', '?')}**\n"
                        f"- S (Lead time): **{result.item_feature_values.get('testing_lead_time', '?')} days**"
                    )
                    if result.decision_path:
                        st.markdown("**Decision path (actual rules for this item):**")
                        st.code(result.decision_path, language=None)
                    elif result.model_importances:
                        st.markdown("**This item's feature values:**")
                        fv_data = [
                            {"Feature": k, "Value": str(v)}
                            for k, v in result.item_feature_values.items()
                        ]
                        st.dataframe(pd.DataFrame(fv_data), use_container_width=True, hide_index=True)
                        st.markdown("**Model-level feature importance (GLOBAL — not per-item causation):**")
                        st.caption(
                            "These importances are a property of the trained model across all training data. "
                            "They are NOT a causal explanation of this specific item's classification."
                        )
                        fi_data = [
                            {"Rank": e["rank"], "Feature": e["feature"], "Importance": f"{e['importance']:.4f}"}
                            for e in result.model_importances
                        ]
                        st.dataframe(pd.DataFrame(fi_data), use_container_width=True, hide_index=True)

                if pol_result and col_pol:
                    with col_pol:
                        st.markdown("##### Policy-Adjusted Recommendation")
                        st.caption(
                            f"Active policy: **{active_pol.label}** v{active_pol.version}\n\n"
                            "⚠️ CONFIGURABLE PROTOTYPE MECHANISM — not an official forensic standard."
                        )
                        changed = pol_result.priority_tier != result.priority_tier
                        if changed:
                            st.warning(
                                f"Policy changes tier: **{result.priority_tier}** → "
                                f"**{pol_result.priority_tier}**"
                            )
                        else:
                            st.info(f"Policy-adjusted tier: **{pol_result.priority_tier}** (unchanged)")
                        st.markdown(f"**Policy-adjusted score:** {pol_result.priority_score:.4f}")

                st.caption(
                    "⚠️ This is an AI-assisted recommendation. "
                    "The investigator is the final decision-maker."
                )


# ===========================================================================
# TAB 5 — HUMAN REVIEW
# ===========================================================================
with tab5:
    st.header("Step 5 — Human Review")
    st.caption(
        "Review each AI recommendation. **Accept** it or **override** it with a reason. "
        "Your decision — not the AI's — determines the final tier used for scheduling."
    )

    if not st.session_state.triage_run or not st.session_state.triage_results:
        st.info("Run triage in Step 3 first.")
    else:
        items      = st.session_state.evidence_items
        results    = st.session_state.triage_results
        active_pol = _active_policy()
        overrides  = st.session_state.overrides

        for item, result in zip(items, results):
            pol_result   = apply_policy(result, item, active_pol) if active_pol else None
            display_tier = pol_result.priority_tier if pol_result else result.priority_tier
            is_overridden = item.item_id in overrides
            current_override = overrides.get(item.item_id, (None, None))

            with st.expander(
                f"**{item.item_id}** · {item.label} "
                f"— AI: {PRIORITY_COLOURS.get(result.priority_tier,'')} {result.priority_tier}"
                + (" · ✏️ Overridden" if is_overridden else " · ✅ Accepted"),
                expanded=not is_overridden,
            ):
                col_info, col_action = st.columns([2, 3])
                with col_info:
                    st.markdown(f"**Raw ML:** {_tier_badge(result.priority_tier)} (score: {result.priority_score:.3f})")
                    if pol_result:
                        st.markdown(f"**Policy-adjusted:** {_tier_badge(pol_result.priority_tier)}")
                    if result.urgency_flag:
                        st.warning("⚠️ Urgency flag is set.")
                    if is_overridden:
                        ov_tier, ov_reason = current_override
                        st.success(f"✏️ **Your decision:** {_tier_badge(ov_tier)}\n\n**Reason:** {ov_reason}")

                with col_action:
                    if is_overridden:
                        if st.button(
                            f"Revert to AI recommendation ({display_tier})",
                            key=f"revert_{item.item_id}",
                        ):
                            del st.session_state.overrides[item.item_id]
                            st.session_state.schedule_built = False
                            st.rerun()
                    else:
                        st.markdown("**Override this recommendation?**")
                        with st.form(key=f"override_form_{item.item_id}"):
                            new_tier = st.selectbox(
                                "Change tier to",
                                options=PRIORITY_LABELS,
                                index=PRIORITY_LABELS.index(display_tier) if display_tier in PRIORITY_LABELS else 0,
                                key=f"tier_sel_{item.item_id}",
                            )
                            reason = st.text_input(
                                "Override reason (required)",
                                placeholder="e.g. Physical connection to suspect confirmed on-scene",
                                key=f"reason_{item.item_id}",
                            )
                            if st.form_submit_button("Save override"):
                                if not reason.strip():
                                    st.error("An override reason is required.")
                                elif new_tier == display_tier:
                                    st.warning("Selected tier is the same as the recommendation — no override saved.")
                                else:
                                    st.session_state.overrides[item.item_id] = (new_tier, reason.strip())
                                    st.session_state.schedule_built = False
                                    st.rerun()

        st.divider()
        st.subheader("Review Summary")
        ov_count  = len(overrides)
        acc_count = len(items) - ov_count
        col_s1, col_s2, col_s3 = st.columns(3)
        col_s1.metric("Items reviewed", len(items))
        col_s2.metric("Accepted", acc_count)
        col_s3.metric("Overridden", ov_count)

        if ov_count > 0:
            st.caption(f"**{ov_count} of {len(items)} evidence item(s) were overridden by the investigator.**")

        # ---- RETRAINING FEEDBACK ----
        st.divider()
        stats = compute_override_stats(total_items=len(items), override_count=ov_count)

        if stats["recommend_retrain"]:
            st.warning(
                f"🔄 **Candidate retraining recommended**\n\n"
                f"Override rate: **{stats['override_rate'] * 100:.0f}%** "
                f"({ov_count}/{len(items)} items) — above the {RETRAINING_THRESHOLD * 100:.0f}% threshold.\n\n"
                "A high override rate suggests the current model may not match investigator "
                "judgement for this case type. You may request a candidate model evaluation."
            )

            if not st.session_state.retrain_requested:
                requester = st.text_input(
                    "Your name / ID (for retraining request record)",
                    key="retrain_requester_input",
                    placeholder="e.g. DC Smith",
                )
                if st.button("📤 Request Candidate Retraining", type="secondary"):
                    if not requester.strip():
                        st.error("Enter your name/ID before requesting retraining.")
                    else:
                        candidate = request_candidate_retraining(
                            model_name=st.session_state.model_name,
                            requested_by=requester.strip(),
                            override_rate=stats["override_rate"],
                        )
                        st.session_state.retrain_requested = True
                        st.session_state.retrain_requester = requester.strip()
                        st.success(
                            f"Candidate retraining request recorded: **{candidate.candidate_id}** (status: pending). "
                            "An authorised reviewer must evaluate and approve the candidate before "
                            "it can replace the active model."
                        )
                        st.rerun()
            else:
                st.success(
                    f"✅ Retraining request submitted by **{st.session_state.retrain_requester}**. "
                    "Candidate status: **pending**. The active model remains unchanged until "
                    "an authorised reviewer approves the candidate."
                )

        else:
            if ov_count > 0:
                st.caption(
                    f"Override rate {stats['override_rate'] * 100:.0f}% is below the "
                    f"{RETRAINING_THRESHOLD * 100:.0f}% retraining threshold. "
                    "No retraining recommendation at this time."
                )


# ===========================================================================
# TAB 6 — FSL SCHEDULE
# ===========================================================================
with tab6:
    st.header("Step 6 — FSL Examination Schedule")
    st.caption(
        "The schedule assigns each evidence item to an examination batch using "
        "**deterministic, rule-based logic** — not another ML model. "
        "Batch assignment is based on final tier, urgency, and specialist requirement."
    )

    if not st.session_state.triage_run or not st.session_state.triage_results:
        st.info("Run triage in Step 3 first.")
    else:
        items   = st.session_state.evidence_items
        results = st.session_state.triage_results

        if st.button("🗓 Generate FSL Schedule", use_container_width=True, type="primary"):
            active_pol = _active_policy()
            overrides  = st.session_state.overrides
            with st.spinner("Building schedule…"):
                schedule = build_schedule(items, results, overrides, active_pol)
            st.session_state.schedule       = schedule
            st.session_state.schedule_built = True
            st.success(f"Schedule built — {len(schedule)} item(s) assigned to batches.")

        if st.session_state.schedule_built and st.session_state.schedule:
            schedule = st.session_state.schedule
            groups   = schedule_summary(schedule)

            batch_info = {
                BATCH_IMMEDIATE: ("🔴", "Critical priority or urgent evidence — immediate laboratory examination"),
                BATCH_SECONDARY: ("🟠", "High priority (non-urgent) or Standard with specialist requirement"),
                BATCH_ARCHIVE:   ("🟢", "Low priority or Standard with no specialist requirement — routine processing"),
            }

            for batch_label in [BATCH_IMMEDIATE, BATCH_SECONDARY, BATCH_ARCHIVE]:
                icon, desc = batch_info[batch_label]
                batch_items = groups[batch_label]
                st.subheader(f"{icon} {batch_label}")
                st.caption(desc)

                if not batch_items:
                    st.info("No items in this batch.")
                else:
                    for rank, si in enumerate(batch_items, 1):
                        urgency_tag = " · ⚠️ URGENT" if si.triage_result.urgency_flag else ""
                        ov_tag      = " · ✏️ Overridden" if si.investigator_decision == "overridden" else ""
                        with st.container():
                            col_n, col_body = st.columns([1, 9])
                            col_n.markdown(f"**#{rank}**")
                            with col_body:
                                st.markdown(
                                    f"**{si.item.item_id}** · {si.item.label}{urgency_tag}{ov_tag}\n\n"
                                    f"Final tier: {_tier_badge(si.final_tier)} · "
                                    f"Lead time: {si.item.testing_lead_time}d · "
                                    f"Specialist: {'Yes' if si.item.specialist_required else 'No'}"
                                )
                                st.caption(f"Reason: {si.batch_reason}")
                                if si.override_reason:
                                    st.caption(f"Override reason: {si.override_reason}")
                        st.divider()

            st.caption(
                "Items within each batch are sorted by degradation risk (highest first) "
                "then by testing lead time (shortest first)."
            )


# ===========================================================================
# TAB 7 — REPORT
# ===========================================================================
with tab7:
    st.header("Step 7 — Report")
    st.caption(
        "Generate a complete report covering the case, evidence classification, "
        "explanations, FSL schedule, and investigator decisions. "
        "Download as **Markdown** or **PDF**."
    )

    can_report = (
        st.session_state.case_context is not None
        and st.session_state.triage_run
        and len(st.session_state.triage_results) > 0
        and st.session_state.schedule_built
        and len(st.session_state.schedule) > 0
    )

    if not can_report:
        missing = []
        if not st.session_state.case_context:
            missing.append("Case context (Step 1)")
        if not st.session_state.triage_run:
            missing.append("Triage results (Step 3)")
        if not st.session_state.schedule_built:
            missing.append("FSL schedule (Step 6)")
        st.warning("To generate a report, complete: " + ", ".join(missing))
    else:
        col_gen1, col_gen2 = st.columns(2)

        with col_gen1:
            if st.button("📄 Generate Markdown Report", use_container_width=True):
                now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
                active_pol = _active_policy()
                tr = st.session_state.training_result

                report_data = ReportData(
                    context=st.session_state.case_context,
                    items=st.session_state.evidence_items,
                    results=st.session_state.triage_results,
                    schedule=st.session_state.schedule,
                    policy=active_pol,
                    generated_at=now,
                    model_used=MODEL_DISPLAY.get(tr.model_name, tr.model_name) if tr else "Unknown",
                    cv_accuracy=tr.cv_accuracy if tr else 0.0,
                )
                report_md = generate_report(report_data)
                st.session_state["_report_md"]   = report_md
                st.session_state["_report_data"] = report_data
                st.success("Markdown report generated.")

        with col_gen2:
            if st.button("📥 Generate PDF Report", use_container_width=True, type="primary"):
                now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
                active_pol = _active_policy()
                tr = st.session_state.training_result

                report_data = ReportData(
                    context=st.session_state.case_context,
                    items=st.session_state.evidence_items,
                    results=st.session_state.triage_results,
                    schedule=st.session_state.schedule,
                    policy=active_pol,
                    generated_at=now,
                    model_used=MODEL_DISPLAY.get(tr.model_name, tr.model_name) if tr else "Unknown",
                    cv_accuracy=tr.cv_accuracy if tr else 0.0,
                )
                st.session_state["_report_data"] = report_data

                with st.spinner("Generating PDF…"):
                    from pdf_report import generate_pdf_report
                    pdf_bytes = generate_pdf_report(
                        report_data,
                        overrides=st.session_state.overrides,
                    )

                if pdf_bytes:
                    st.session_state["_report_pdf"] = pdf_bytes
                    st.success("PDF report generated.")
                else:
                    st.warning(
                        "ReportLab is not installed — PDF generation unavailable. "
                        "Use the Markdown report instead, or install reportlab."
                    )
                    # Ensure markdown is generated as fallback
                    if "_report_md" not in st.session_state:
                        from report import generate_report
                        report_md = generate_report(report_data)
                        st.session_state["_report_md"] = report_md

        # Download buttons
        fir = st.session_state.case_context.fir_number.replace("/", "-")

        if "_report_pdf" in st.session_state and st.session_state["_report_pdf"]:
            st.download_button(
                label="⬇️ Download PDF Report",
                data=st.session_state["_report_pdf"],
                file_name=f"EvidencePro_{fir}.pdf",
                mime="application/pdf",
                use_container_width=True,
                type="primary",
            )

        if "_report_md" in st.session_state:
            st.download_button(
                label="⬇️ Download Markdown Report",
                data=st.session_state["_report_md"].encode("utf-8"),
                file_name=f"EvidencePro_{fir}.md",
                mime="text/markdown",
                use_container_width=True,
            )

        # Document source info
        if st.session_state.doc_processed and st.session_state.doc_processed.success:
            doc = st.session_state.doc_processed
            st.info(
                f"📄 Source document: **{doc.filename}** · "
                f"Processed with **{doc.processor}** · "
                f"{len(doc.full_text):,} characters extracted."
            )

        # Markdown preview
        if "_report_md" in st.session_state:
            st.divider()
            with st.expander("📄 Report Preview (Markdown)", expanded=False):
                st.markdown(st.session_state["_report_md"])


# ===========================================================================
# TAB 8 — POLICY ADMIN
# ===========================================================================
with tab8:
    st.header("Step 8 — Policy Administration")
    st.caption(
        "Policy weights are a **CONFIGURABLE PROTOTYPE MECHANISM** only. "
        "They are NOT official forensic standards, legally validated thresholds, "
        "or scientifically peer-reviewed forensic triage guidelines."
    )

    import pandas as pd
    role = _role_enum()

    # ---- CANDIDATE RETRAINING HISTORY ----
    if CANDIDATE_HISTORY:
        st.subheader("Candidate Retraining Requests")
        cand_rows = [
            {
                "ID":          c.candidate_id,
                "Model":       c.model_name,
                "Requested by": c.requested_by,
                "Override rate": f"{c.override_rate * 100:.0f}%",
                "Status":      c.status,
                "Approved by": c.approved_by or "—",
                "Rejected by": c.rejected_by or "—",
            }
            for c in CANDIDATE_HISTORY
        ]
        st.dataframe(pd.DataFrame(cand_rows), use_container_width=True, hide_index=True)
        st.caption(
            "Candidates are evaluated independently. A rejected candidate NEVER becomes active. "
            "The active model is NEVER silently replaced."
        )
        st.divider()

    # ---- POLICY HISTORY ----
    st.subheader("Policy History")
    if not POLICY_HISTORY:
        st.info("No policies have been proposed yet.")
    else:
        hist_rows = [
            {
                "Version": p.version,
                "Label": p.label,
                "P": p.weight_P, "D": p.weight_D,
                "E": p.weight_E, "S": p.weight_S,
                "Status": p.status,
                "Proposed by": p.proposed_by,
                "Approved by": p.approved_by or "—",
            }
            for p in POLICY_HISTORY
        ]
        st.dataframe(pd.DataFrame(hist_rows), use_container_width=True, hide_index=True)

    active_pol = _active_policy()
    if active_pol:
        st.success(
            f"**Active policy:** {active_pol.label} · v{active_pol.version} · "
            f"P={active_pol.weight_P} · D={active_pol.weight_D} · "
            f"E={active_pol.weight_E} · S={active_pol.weight_S}"
        )
    else:
        st.info("No active policy — system is running in unweighted ML mode.")

    st.divider()

    # ---- PROPOSE POLICY (POLICY_ADMIN only) ----
    if role == PolicyRole.POLICY_ADMIN:
        st.subheader("Propose a New Policy")
        st.warning(
            "⚠️ Any policy you propose must be reviewed and approved by an Approver "
            "before it becomes active. Your proposal does NOT immediately change the system."
        )
        with st.form("propose_policy_form"):
            p_version     = st.text_input("Version", placeholder="e.g. 1.0.0")
            p_label       = st.text_input("Label", placeholder="e.g. Homicide Focus Policy")
            p_description = st.text_area("Description", placeholder="What does this policy emphasise?")
            p_proposed_by = st.text_input("Your name / ID")
            st.markdown("**PDES Dimension Weights** (1.0 = default, >1 = more emphasis)")
            c1, c2, c3, c4 = st.columns(4)
            w_P = c1.number_input("Weight P", min_value=0.0, max_value=10.0, value=1.0, step=0.5)
            w_D = c2.number_input("Weight D", min_value=0.0, max_value=10.0, value=1.0, step=0.5)
            w_E = c3.number_input("Weight E", min_value=0.0, max_value=10.0, value=1.0, step=0.5)
            w_S = c4.number_input("Weight S", min_value=0.0, max_value=10.0, value=1.0, step=0.5)

            if st.form_submit_button("Submit Proposal"):
                if not p_version.strip() or not p_label.strip() or not p_proposed_by.strip():
                    st.error("Version, label, and proposer name are required.")
                else:
                    new_pol = TriagePolicy(
                        version=p_version.strip(),
                        label=p_label.strip(),
                        description=p_description.strip(),
                        weight_P=w_P, weight_D=w_D, weight_E=w_E, weight_S=w_S,
                        proposed_by=p_proposed_by.strip(),
                    )
                    try:
                        propose_policy(new_pol, role)
                        st.success(
                            f"Policy '{new_pol.label}' v{new_pol.version} submitted for approval. "
                            "An Approver must review it before it becomes active."
                        )
                    except ValueError as e:
                        st.error(str(e))

    # ---- APPROVE / REJECT (APPROVER only) ----
    elif role == PolicyRole.APPROVER:
        st.subheader("Review Pending Proposals")
        pending = [p for p in POLICY_HISTORY if p.status == "proposed"]
        if not pending:
            st.info("No pending proposals.")
        else:
            approver_name = st.text_input("Your name / ID (Approver)")
            for pol in pending:
                with st.expander(f"**{pol.label}** v{pol.version} · proposed by {pol.proposed_by}"):
                    st.markdown(
                        f"- Description: {pol.description or '—'}\n"
                        f"- P={pol.weight_P} · D={pol.weight_D} · E={pol.weight_E} · S={pol.weight_S}"
                    )
                    col_a, col_r = st.columns(2)
                    with col_a:
                        if st.button(f"Approve v{pol.version}", key=f"approve_{pol.version}"):
                            if not approver_name.strip():
                                st.error("Enter your name before approving.")
                            else:
                                try:
                                    approve_policy(pol, approver_name.strip(), role)
                                    st.success(
                                        f"Policy '{pol.label}' v{pol.version} approved. "
                                        "It is now the active policy."
                                    )
                                    st.rerun()
                                except ValueError as e:
                                    st.error(str(e))
                    with col_r:
                        if st.button(f"Reject v{pol.version}", key=f"reject_{pol.version}"):
                            if not approver_name.strip():
                                st.error("Enter your name before rejecting.")
                            else:
                                try:
                                    reject_policy(pol, approver_name.strip(), role)
                                    st.warning(f"Policy v{pol.version} rejected.")
                                    st.rerun()
                                except ValueError as e:
                                    st.error(str(e))

    # ---- INVESTIGATOR VIEW ----
    elif role == PolicyRole.INVESTIGATOR:
        st.info(
            "As an **Investigator**, you can view the active policy but cannot "
            "propose or approve policy changes. Use the **Policy Admin** or "
            "**Approver** role (in the sidebar) to manage policies."
        )
        if active_pol:
            st.markdown(
                f"**Active policy:** {active_pol.label} · v{active_pol.version}\n\n"
                f"- P (Probative Value weight): {active_pol.weight_P}\n"
                f"- D (Degradation Risk weight): {active_pol.weight_D}\n"
                f"- E (Exclusionary Power weight): {active_pol.weight_E}\n"
                f"- S (Processing Speed weight): {active_pol.weight_S}"
            )
        st.caption(
            "Policy weights are a CONFIGURABLE PROTOTYPE MECHANISM. "
            "They are NOT official forensic standards."
        )

    st.divider()
    st.caption(
        "⚠️ This policy governance panel is a PROTOTYPE SIMULATION. "
        "There is no real authentication. Role selection is for demonstration purposes only."
    )
