"""
feedback.py
===========
EvidencePro — Override Feedback and Candidate Retraining Workflow
-----------------------------------------------------------------

PURPOSE
-------
Tracks investigator overrides accumulated during a session and provides
a recommendation when enough overrides have occurred to suggest that the
current model may benefit from a candidate retraining cycle.

LIFECYCLE
---------
Override recorded
  → override_rate exceeds RETRAINING_THRESHOLD
  → Investigator sees "Candidate retraining recommended"
  → Investigator clicks "Request Candidate Retraining"
  → CandidateModel created with status "pending"
  → (In a full system) an admin evaluates the candidate
  → Candidate transitions: pending → evaluated → approved → active
     OR: pending → rejected

GOVERNANCE RULES
----------------
- A rejected candidate NEVER becomes active.
- The active model is NEVER silently replaced.
- Retraining requests create a CANDIDATE, not an active model.
- The threshold is configurable (default: 0.25 = 25% of items overridden).

This module has no Streamlit dependency and can be unit-tested directly.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Literal, Optional

# ---------------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------------

#: Fraction of evidence items that must be overridden to trigger the
#: retraining recommendation banner.
RETRAINING_THRESHOLD = 0.25

#: Valid candidate model lifecycle states.
CandidateStatus = Literal["pending", "evaluated", "approved", "active", "rejected"]


# ---------------------------------------------------------------------------
# DATACLASSES
# ---------------------------------------------------------------------------

@dataclass
class OverrideFeedback:
    """
    A single investigator override recorded for feedback purposes.

    Attributes
    ----------
    item_id    : EvidenceItem identifier.
    ai_tier    : The AI-recommended priority tier.
    final_tier : The investigator's final decision.
    reason     : Override reason text (required by app.py).
    timestamp  : UTC timestamp of the override.
    """
    item_id:    str
    ai_tier:    str
    final_tier: str
    reason:     str
    timestamp:  str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


@dataclass
class CandidateModel:
    """
    A candidate replacement model, created in response to a retraining request.

    A candidate NEVER becomes active automatically.
    It must be approved by an authorised reviewer.

    Attributes
    ----------
    candidate_id  : Unique identifier (e.g. "CANDIDATE-001").
    model_name    : Base model type requested ("decision_tree", etc.).
    requested_by  : Name/ID of the investigator who requested retraining.
    requested_at  : UTC ISO-8601 timestamp.
    override_rate : Override fraction that triggered the request (0.0-1.0).
    status        : Current lifecycle state.
    evaluated_by  : Set when status moves to "evaluated".
    approved_by   : Set when status moves to "approved" or "active".
    rejected_by   : Set when status moves to "rejected".
    rejection_note: Reason for rejection (if rejected).
    notes         : Free-text notes.
    """
    candidate_id:   str
    model_name:     str
    requested_by:   str
    requested_at:   str
    override_rate:  float
    status:         CandidateStatus = "pending"
    evaluated_by:   Optional[str]   = None
    approved_by:    Optional[str]   = None
    rejected_by:    Optional[str]   = None
    rejection_note: Optional[str]   = None
    notes:          Optional[str]   = None

    def is_final(self) -> bool:
        """Return True if the candidate has reached a terminal state."""
        return self.status in ("active", "rejected")


# ---------------------------------------------------------------------------
# SESSION-LEVEL STORE
# ---------------------------------------------------------------------------

#: In-session list of all candidate models (prototype: in-memory only).
CANDIDATE_HISTORY: list[CandidateModel] = []

#: Counter for generating candidate IDs.
_candidate_counter: int = 0


# ---------------------------------------------------------------------------
# PUBLIC API
# ---------------------------------------------------------------------------

def compute_override_stats(
    total_items: int,
    override_count: int,
) -> dict:
    """
    Compute override statistics and retraining recommendation.

    Parameters
    ----------
    total_items    : Number of evidence items in the session.
    override_count : Number of items overridden by the investigator.

    Returns
    -------
    dict with keys:
      total_items      : int
      override_count   : int
      override_rate    : float (0.0 to 1.0)
      recommend_retrain: bool (True if override_rate >= RETRAINING_THRESHOLD)
      threshold        : float (the configured threshold)
    """
    rate = override_count / total_items if total_items > 0 else 0.0
    return {
        "total_items":       total_items,
        "override_count":    override_count,
        "override_rate":     rate,
        "recommend_retrain": rate >= RETRAINING_THRESHOLD,
        "threshold":         RETRAINING_THRESHOLD,
    }


def request_candidate_retraining(
    model_name:    str,
    requested_by:  str,
    override_rate: float,
    notes:         str = "",
) -> CandidateModel:
    """
    Create a new CandidateModel and add it to CANDIDATE_HISTORY.

    The candidate has status "pending". It does NOT affect the active model.

    Parameters
    ----------
    model_name    : Name of the base model type to retrain.
    requested_by  : Name/ID of the requesting investigator.
    override_rate : Override fraction that triggered the request.
    notes         : Optional free-text note.

    Returns
    -------
    The newly created CandidateModel (status="pending").
    """
    global _candidate_counter
    _candidate_counter += 1
    candidate = CandidateModel(
        candidate_id=f"CANDIDATE-{_candidate_counter:03d}",
        model_name=model_name,
        requested_by=requested_by,
        requested_at=datetime.now(timezone.utc).isoformat(),
        override_rate=override_rate,
        status="pending",
        notes=notes or None,
    )
    CANDIDATE_HISTORY.append(candidate)
    return candidate


def evaluate_candidate(candidate: CandidateModel, evaluator: str) -> None:
    """
    Mark a pending candidate as evaluated.

    Only a candidate in "pending" status can be evaluated.
    """
    if candidate.status != "pending":
        raise ValueError(
            f"Candidate {candidate.candidate_id} is in status '{candidate.status}' "
            "and cannot be evaluated."
        )
    candidate.status       = "evaluated"
    candidate.evaluated_by = evaluator


def approve_candidate(candidate: CandidateModel, approver: str) -> None:
    """
    Approve a candidate, setting it to "active" status.

    Only a candidate in "evaluated" status can be approved.
    A rejected candidate cannot be approved.
    """
    if candidate.status == "rejected":
        raise ValueError(
            f"Candidate {candidate.candidate_id} has been rejected and cannot be approved."
        )
    if candidate.status not in ("pending", "evaluated"):
        raise ValueError(
            f"Candidate {candidate.candidate_id} cannot be approved from status '{candidate.status}'."
        )
    candidate.status      = "active"
    candidate.approved_by = approver


def reject_candidate(
    candidate: CandidateModel,
    rejector:  str,
    note:      str = "",
) -> None:
    """
    Reject a candidate permanently.

    A rejected candidate NEVER becomes active.
    A candidate already in "active" status cannot be rejected.
    """
    if candidate.status == "active":
        raise ValueError(
            f"Candidate {candidate.candidate_id} is already active and cannot be rejected."
        )
    if candidate.status == "rejected":
        return  # idempotent
    candidate.status         = "rejected"
    candidate.rejected_by    = rejector
    candidate.rejection_note = note or None


def get_pending_candidates() -> list[CandidateModel]:
    """Return candidates in "pending" or "evaluated" status."""
    return [c for c in CANDIDATE_HISTORY if c.status in ("pending", "evaluated")]


def get_active_candidate() -> Optional[CandidateModel]:
    """Return the most recently activated candidate, or None."""
    active = [c for c in CANDIDATE_HISTORY if c.status == "active"]
    return active[-1] if active else None
