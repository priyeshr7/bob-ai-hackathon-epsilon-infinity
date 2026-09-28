"""
pdf_report.py
=============
EvidencePro — A4 PDF Report Generator
--------------------------------------

Produces a professional A4 PDF report using ReportLab.
Falls back to returning None if ReportLab is not installed
(the caller should offer the Markdown download instead).

REPORT SECTIONS
---------------
1.  Cover / Header + Disclaimer
2.  Case Summary
3.  Evidence Inventory
4.  AI Triage Results
5.  Model Explanation & Accuracy Context
6.  Human Overrides
7.  Policy Adjustments
8.  FSL Examination Schedule
9.  Limitations / Disclaimer

All text is written in natural-language sentences (not raw data dumps).
"""

from __future__ import annotations

import io
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from report import ReportData

from models.triage_models import DATA_DISCLAIMER
from scheduler import BATCH_IMMEDIATE, BATCH_SECONDARY, BATCH_ARCHIVE, schedule_summary


# ---------------------------------------------------------------------------
# PALETTE
# ---------------------------------------------------------------------------
_DARK     = (0.118, 0.137, 0.157)    # #1e232a  headers
_MID      = (0.337, 0.376, 0.420)    # #566070  body
_LIGHT    = (0.965, 0.969, 0.980)    # #f6f7fa  alt row bg
_ACCENT   = (0.173, 0.384, 0.729)    # #2c62ba  section rule
_RED      = (0.722, 0.145, 0.145)    # #b82525  critical
_ORANGE   = (0.871, 0.502, 0.102)    # #de801a  high
_GREEN    = (0.165, 0.533, 0.227)    # #2a883a  low


# ---------------------------------------------------------------------------
# PUBLIC API
# ---------------------------------------------------------------------------

def generate_pdf_report(data: "ReportData", overrides: dict | None = None) -> Optional[bytes]:
    """
    Generate an A4 PDF report.

    Parameters
    ----------
    data      : ReportData from report.py
    overrides : dict mapping item_id → (tier, reason) from session state

    Returns
    -------
    PDF bytes, or None if ReportLab is not installed.
    """
    try:
        from reportlab.lib import colors
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import cm
        from reportlab.platypus import (
            SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
            HRFlowable, PageBreak,
        )
        from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_JUSTIFY
    except ImportError:
        return None

    overrides = overrides or {}
    buf = io.BytesIO()

    doc = SimpleDocTemplate(
        buf,
        pagesize=A4,
        leftMargin=2.2 * cm,
        rightMargin=2.2 * cm,
        topMargin=2.5 * cm,
        bottomMargin=2.5 * cm,
        title=f"EvidencePro — {data.context.fir_number}",
        author="EvidencePro (prototype)",
    )

    # ---- Styles ----
    styles = getSampleStyleSheet()

    def _style(name, **kw):
        base = styles["Normal"]
        return ParagraphStyle(name, parent=base, **kw)

    H1   = _style("H1",  fontSize=18, textColor=colors.Color(*_DARK),  spaceAfter=6,  spaceBefore=12, fontName="Helvetica-Bold")
    H2   = _style("H2",  fontSize=13, textColor=colors.Color(*_DARK),  spaceAfter=4,  spaceBefore=10, fontName="Helvetica-Bold")
    H3   = _style("H3",  fontSize=11, textColor=colors.Color(*_ACCENT), spaceAfter=3, spaceBefore=8,  fontName="Helvetica-Bold")
    BODY = _style("BODY", fontSize=9.5, textColor=colors.Color(*_MID),  spaceAfter=4,  leading=14, alignment=TA_JUSTIFY)
    CAPTION = _style("CAP", fontSize=8, textColor=colors.Color(*_MID),  spaceAfter=3, fontStyle="italic", leading=11)
    DISCLAIMER = _style("DISC", fontSize=8, textColor=colors.Color(0.5, 0.2, 0.2), spaceAfter=3, leading=11, borderPad=4)
    MONO = _style("MONO", fontSize=8, fontName="Courier", textColor=colors.Color(*_DARK), spaceAfter=3, leading=11)
    BOLD_BODY = _style("BB", fontSize=9.5, fontName="Helvetica-Bold", textColor=colors.Color(*_DARK), spaceAfter=3)

    def _hr():
        return HRFlowable(width="100%", thickness=0.5,
                          color=colors.Color(*_ACCENT), spaceAfter=4, spaceBefore=4)

    def _tier_text(tier: str) -> str:
        icons = {"Critical": "■ CRITICAL", "High": "▲ HIGH", "Standard": "● STANDARD", "Low": "○ LOW"}
        return icons.get(tier, tier)

    # ---- Assemble flowables ----
    story = []

    # COVER
    story.append(Spacer(1, 0.5 * cm))
    story.append(Paragraph("EvidencePro", H1))
    story.append(Paragraph("Forensic Evidence Triage Report", H2))
    story.append(_hr())
    story.append(Paragraph(
        f"<b>Case / FIR:</b> {data.context.fir_number} &nbsp;&nbsp; "
        f"<b>Offence:</b> {data.context.offence_type} &nbsp;&nbsp; "
        f"<b>Generated:</b> {data.generated_at}",
        BODY,
    ))
    story.append(Spacer(1, 0.3 * cm))
    story.append(Paragraph(
        "<b>IMPORTANT DISCLAIMER:</b> EvidencePro is an AI-assisted decision-support tool. "
        "All recommendations are produced by a machine-learning model trained on synthetic "
        "demonstration data. The investigator is the final decision-maker. No recommendation "
        "constitutes a final forensic, legal, or evidentiary decision. " + DATA_DISCLAIMER,
        DISCLAIMER,
    ))

    # SECTION 1 — CASE SUMMARY
    story.append(Spacer(1, 0.4 * cm))
    story.append(Paragraph("1. Case Summary", H2))
    story.append(_hr())
    narrative = data.context.narrative.strip() if data.context.narrative else "Not provided."
    story.append(Paragraph(
        f"This report covers case <b>{data.context.fir_number}</b>, "
        f"classified as a <b>{data.context.offence_type}</b> matter. "
        f"A total of <b>{len(data.items)}</b> evidence item(s) were triaged using the "
        f"<b>{data.model_used}</b> model.",
        BODY,
    ))
    story.append(Paragraph(f"<b>Case narrative:</b> {narrative}", BODY))

    # SECTION 2 — EVIDENCE INVENTORY
    story.append(Spacer(1, 0.4 * cm))
    story.append(Paragraph("2. Evidence Inventory", H2))
    story.append(_hr())
    story.append(Paragraph(
        f"The following {len(data.items)} item(s) were submitted for triage.", BODY
    ))

    cond_lbl = {1: "Poor", 2: "Fair", 3: "Good"}
    inv_data = [["#", "ID", "Label", "Type", "Age (h)", "Condition"]]
    for i, item in enumerate(data.items, 1):
        age = str(item.collection_age_hours) if item.collection_age_hours else "Unknown"
        inv_data.append([
            str(i), item.item_id,
            item.label[:55] + ("…" if len(item.label) > 55 else ""),
            item.evidence_type.replace("_", " "),
            age, cond_lbl.get(item.evidence_condition, "?"),
        ])
    story.append(_make_table(inv_data))

    # SECTION 3 — AI TRIAGE RESULTS
    story.append(Spacer(1, 0.4 * cm))
    story.append(Paragraph("3. AI Triage Results", H2))
    story.append(_hr())
    story.append(Paragraph(
        "Each item was classified by the AI model. The following table shows the raw ML "
        "recommendation before any policy adjustment or investigator override.", BODY
    ))

    triage_data = [["ID", "Label", "P", "D", "E", "Lead", "Raw Tier", "Urgent", "Specialist"]]
    groups = schedule_summary(data.schedule)
    sched_lookup = {si.item.item_id: si for si in data.schedule}
    for item, result in zip(data.items, data.results):
        urg = "YES" if result.urgency_flag else "No"
        spec = "Yes" if item.specialist_required else "No"
        triage_data.append([
            item.item_id,
            item.label[:40] + ("…" if len(item.label) > 40 else ""),
            str(item.probative_value), str(item.perishability),
            str(item.exclusionary_power), f"{item.testing_lead_time}d",
            _tier_text(result.priority_tier), urg, spec,
        ])
    story.append(_make_table(triage_data, small=True))
    story.append(Paragraph(
        "P = Probative Value · D = Degradation Risk · E = Exclusionary Power · "
        "Lead = Testing Lead Time. Urgent = evidence flagged as perishable and recently collected.",
        CAPTION,
    ))

    # SECTION 4 — MODEL EXPLANATION & ACCURACY
    story.append(Spacer(1, 0.4 * cm))
    story.append(Paragraph("4. Model Explanation & Accuracy Context", H2))
    story.append(_hr())
    cv_pct = f"{data.cv_accuracy * 100:.1f}%"
    story.append(Paragraph(
        f"The active model is <b>{data.model_used}</b>, which achieved a "
        f"cross-validated accuracy of <b>{cv_pct}</b> on the synthetic demonstration dataset. "
        "This figure was calculated dynamically during this session using stratified "
        "k-fold cross-validation on the same dataset used for training.",
        BODY,
    ))
    story.append(Paragraph(
        "IMPORTANT: This accuracy figure was computed on synthetic/demonstration data only. "
        "It does not represent validated real-world forensic performance. "
        "The model has not been evaluated against real forensic casework and must not "
        "be used for actual forensic decisions without appropriate validation.",
        DISCLAIMER,
    ))

    # Per-item explanations
    for item, result in zip(data.items, data.results):
        story.append(Paragraph(f"{item.item_id}: {item.label[:70]}", H3))
        story.append(Paragraph(
            f"The model assigned a priority tier of <b>{result.priority_tier}</b> "
            f"with a confidence score of <b>{result.priority_score:.3f}</b>.",
            BODY,
        ))
        if result.explanation:
            story.append(Paragraph(result.explanation, MONO))
        if result.decision_path:
            story.append(Paragraph("<b>Decision path:</b>", BOLD_BODY))
            story.append(Paragraph(result.decision_path.replace("\n", "<br/>"), MONO))

    # SECTION 5 — HUMAN OVERRIDES
    story.append(Spacer(1, 0.4 * cm))
    story.append(Paragraph("5. Human Overrides", H2))
    story.append(_hr())

    ov_count = len(overrides)
    total    = len(data.items)
    ov_rate  = ov_count / total if total > 0 else 0.0
    story.append(Paragraph(
        f"{ov_count} of {total} evidence item(s) were overridden by the investigator "
        f"({ov_rate * 100:.0f}% override rate). "
        "Every investigator change is recorded below.",
        BODY,
    ))

    if ov_count > 0:
        ov_data = [["ID", "Label", "AI Tier", "Investigator Tier", "Reason"]]
        for item, result in zip(data.items, data.results):
            if item.item_id in overrides:
                ov_tier, ov_reason = overrides[item.item_id]
                ov_data.append([
                    item.item_id,
                    item.label[:35] + ("…" if len(item.label) > 35 else ""),
                    result.priority_tier, ov_tier,
                    ov_reason[:60] + ("…" if len(ov_reason) > 60 else ""),
                ])
        story.append(_make_table(ov_data))
    else:
        story.append(Paragraph("No investigator overrides were recorded in this session.", BODY))

    # SECTION 6 — POLICY ADJUSTMENTS
    story.append(Spacer(1, 0.4 * cm))
    story.append(Paragraph("6. Policy Adjustments", H2))
    story.append(_hr())
    if data.policy:
        p = data.policy
        story.append(Paragraph(
            f"Active policy: <b>{p.label}</b> v{p.version} "
            f"(proposed by {p.proposed_by}, approved by {p.approved_by or '—'}). "
            f"Weights: P={p.weight_P}, D={p.weight_D}, E={p.weight_E}, S={p.weight_S}.",
            BODY,
        ))
        # Count items where policy changed the tier
        adj_count = 0
        for si in data.schedule:
            if si.policy_result and si.policy_result.priority_tier != si.triage_result.priority_tier:
                adj_count += 1
        story.append(Paragraph(
            f"The policy adjusted the priority tier for {adj_count} item(s). "
            "⚠️ Policy weights are a CONFIGURABLE PROTOTYPE MECHANISM — not official forensic standards.",
            BODY,
        ))
    else:
        story.append(Paragraph(
            "No active policy was applied. All triage results reflect raw ML model output.", BODY
        ))

    # SECTION 7 — FSL SCHEDULE
    story.append(Spacer(1, 0.4 * cm))
    story.append(Paragraph("7. FSL Examination Schedule", H2))
    story.append(_hr())
    story.append(Paragraph(
        "The examination schedule was produced by a deterministic, rule-based scheduling "
        "algorithm — not an ML model. Each item is assigned to one of three examination "
        "batches based on final priority tier, urgency flag, and specialist requirement. "
        "Items within each batch are sorted by degradation risk (highest first) then lead time (shortest first).",
        BODY,
    ))

    batch_info = {
        BATCH_IMMEDIATE: ("Immediate Examination", "Critical priority or urgent/perishable evidence."),
        BATCH_SECONDARY: ("Secondary Scheduling", "High priority (non-urgent) or Standard requiring specialist analysis."),
        BATCH_ARCHIVE:   ("Routine Archive", "Low priority or Standard with no specialist requirement."),
    }

    for batch_label in [BATCH_IMMEDIATE, BATCH_SECONDARY, BATCH_ARCHIVE]:
        title, desc = batch_info[batch_label]
        batch_items = groups[batch_label]
        story.append(Paragraph(f"Batch: {title}", H3))
        story.append(Paragraph(desc, CAPTION))
        if not batch_items:
            story.append(Paragraph("No items assigned to this batch.", BODY))
            continue
        sched_data = [["#", "ID", "Label", "Final Tier", "Urgent", "Specialist", "Lead"]]
        for rank, si in enumerate(batch_items, 1):
            urg  = "YES" if si.triage_result.urgency_flag else "No"
            spec = "Yes" if si.item.specialist_required else "No"
            sched_data.append([
                str(rank), si.item.item_id,
                si.item.label[:38] + ("…" if len(si.item.label) > 38 else ""),
                _tier_text(si.final_tier), urg, spec,
                f"{si.item.testing_lead_time}d",
            ])
        story.append(_make_table(sched_data, small=True))

    # SECTION 8 — LIMITATIONS & DISCLAIMER
    story.append(PageBreak())
    story.append(Paragraph("8. Limitations & Disclaimer", H2))
    story.append(_hr())
    story.append(Paragraph(
        "EvidencePro is a prototype decision-support system. The following limitations "
        "apply to all outputs in this report:", BODY
    ))
    for point in [
        "All AI/ML outputs are recommendations. The investigator is the final decision-maker "
        "for all forensic triage and evidence submission decisions.",
        "The model was trained on a synthetic/demonstration dataset. Accuracy figures do not "
        "represent real-world forensic validation.",
        "Policy weights are a configurable prototype mechanism and are NOT official forensic "
        "standards, legally validated thresholds, or scientifically peer-reviewed guidelines.",
        "The FSL schedule is produced by rule-based logic and must be reviewed by a "
        "qualified forensic practitioner before implementation.",
        "This system must not be used for actual forensic decisions without appropriate "
        "validation and authorisation by a qualified forensic authority.",
    ]:
        story.append(Paragraph(f"• {point}", BODY))

    story.append(Spacer(1, 0.8 * cm))
    story.append(_hr())
    story.append(Paragraph("Report generated by EvidencePro (prototype) — AI-Assisted Forensic Evidence Triage", CAPTION))

    # Build
    doc.build(story)
    return buf.getvalue()


# ---------------------------------------------------------------------------
# TABLE HELPER
# ---------------------------------------------------------------------------

def _make_table(data: list[list], small: bool = False) -> "Table":
    from reportlab.platypus import Table, TableStyle
    from reportlab.lib import colors

    font_size = 7.5 if small else 8.5
    header_bg = colors.Color(*_DARK)
    alt_bg    = colors.Color(*_LIGHT)
    body_color = colors.Color(*_MID)

    col_count = len(data[0]) if data else 1
    style = TableStyle([
        # Header row
        ("BACKGROUND",  (0, 0), (-1, 0),  header_bg),
        ("TEXTCOLOR",   (0, 0), (-1, 0),  colors.white),
        ("FONTNAME",    (0, 0), (-1, 0),  "Helvetica-Bold"),
        ("FONTSIZE",    (0, 0), (-1, 0),  font_size),
        ("BOTTOMPADDING", (0, 0), (-1, 0), 5),
        ("TOPPADDING",  (0, 0), (-1, 0),  5),
        # Body rows
        ("FONTNAME",    (0, 1), (-1, -1), "Helvetica"),
        ("FONTSIZE",    (0, 1), (-1, -1), font_size),
        ("TEXTCOLOR",   (0, 1), (-1, -1), body_color),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, alt_bg]),
        ("GRID",        (0, 0), (-1, -1), 0.3, colors.Color(0.8, 0.8, 0.8)),
        ("TOPPADDING",  (0, 1), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 1), (-1, -1), 3),
        ("VALIGN",      (0, 0), (-1, -1), "MIDDLE"),
    ])

    table = Table(data, repeatRows=1)
    table.setStyle(style)
    return table
