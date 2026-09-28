"""
synthetic_dataset.py
====================
Synthetic / curated demonstration dataset for EvidencePro.

WARNING — SYNTHETIC DATA
------------------------
This dataset is ENTIRELY SYNTHETIC. It was constructed to demonstrate the
EvidencePro pipeline and to provide enough variation for multi-model comparison
with cross-validation. It is NOT derived from real casework, real forensic
laboratory data, or any official forensic triage standard.

Cross-validated accuracy figures produced from this dataset are DEMONSTRATIVE
ONLY. They show that the pipeline is functional and that the three models can
be compared. They do NOT represent real-world forensic triage performance.

Do not use this dataset or any accuracy figures derived from it to make real
forensic or legal decisions.

Dataset structure
-----------------
Columns:
  evidence_type        — one of EVIDENCE_TYPES (16 categories)
  offence_type         — one of OFFENCE_TYPES  (8 categories)
  probative_value      — 1 (low) / 2 (medium) / 3 (high)      [P]
  perishability        — 1 (stable) / 2 (days) / 3 (hours)    [D]
  exclusionary_power   — 1 (low) / 2 (medium) / 3 (high)      [E]
  contamination_risk   — 1 (low) / 2 (medium) / 3 (high)
  specialist_required  — 0 / 1
  testing_lead_time    — days (proxy for S, inverted)
  priority_label       — Critical / High / Standard / Low  (target)

Priority labelling rationale
-----------------------------
  Critical  — serious offence (homicide / sexual_assault) AND high probative
               value OR perishable biological/toxicological evidence; or any
               evidence with perishability=3 + probative_value=3
  High      — important evidence with moderate urgency: serious offence but
               more stable evidence, or moderate probative value in a serious
               offence, or high probative value in a moderate-severity offence
  Standard  — routine processing: moderate offences (burglary, arson, vehicle
               crime, drug supply, fraud) with no extreme perishability
  Low       — supplementary: low probative value, stable, routine offence

Blocks
------
  Block 1 (rows   0-47)  — Original 48 rows from evidence_triage.py baseline
  Block 2 (rows  48-95)  — Arson, drug supply, fraud coverage across all evidence types
  Block 3 (rows  96-143) — Varied perishability/exclusionary_power combinations
  Block 4 (rows 144-191) — Edge and boundary cases; ensures each label is well represented
"""

import pandas as pd

COLUMNS = [
    "evidence_type", "offence_type",
    "probative_value", "perishability", "exclusionary_power",
    "contamination_risk", "specialist_required", "testing_lead_time",
    "priority_label",
]


def build_dataset() -> pd.DataFrame:
    """
    Return the full synthetic dataset as a DataFrame.

    Each row represents one piece of evidence in one crime scenario.
    The 'priority_label' column is the synthetic ground-truth triage class.

    Returns
    -------
    pd.DataFrame with columns defined in COLUMNS.
    """
    rows = _block1() + _block2() + _block3() + _block4()
    return pd.DataFrame(rows, columns=COLUMNS)


# ---------------------------------------------------------------------------
# BLOCK 1 — Original 48 rows (preserved verbatim from evidence_triage.py)
#           exclusionary_power added inline; values assigned by rationale.
# ---------------------------------------------------------------------------
# exclusionary_power rationale for Block 1:
#   biological DNA / touch DNA — 3 (can directly exclude suspects)
#   toxicology blood/hair — 3 if high PV, 2 otherwise
#   firearm ballistic/residue — 2-3
#   fingerprint latent/patent — 2
#   digital device/CCTV — 2-3 depending on PV
#   trace / impression — 1-2
#   document_questioned — 1-2

def _block1():
    return [
        # fmt: off
        # --- CRITICAL: Homicide + biological DNA ---
        #  ev_type                  offence          PV  D   E   CR  SR  TL  label
        ("biological_dna",       "homicide",        3, 3, 3, 3, 1,  5, "Critical"),
        ("biological_touch_dna", "homicide",        3, 3, 3, 3, 1,  7, "Critical"),
        ("toxicology_blood",     "homicide",        3, 3, 3, 2, 1,  3, "Critical"),
        ("firearm_residue",      "homicide",        3, 3, 2, 3, 1,  4, "Critical"),
        ("firearm_ballistic",    "homicide",        3, 2, 3, 1, 1, 10, "Critical"),
        ("biological_dna",       "sexual_assault",  3, 3, 3, 3, 1,  5, "Critical"),
        ("toxicology_blood",     "sexual_assault",  3, 3, 3, 2, 1,  3, "Critical"),
        ("biological_touch_dna", "sexual_assault",  2, 3, 3, 3, 1,  7, "Critical"),
        ("digital_device",       "homicide",        3, 1, 3, 1, 1, 14, "Critical"),
        ("digital_cctv",         "homicide",        3, 2, 2, 1, 0,  2, "Critical"),

        # --- HIGH ---
        ("fingerprint_latent",   "homicide",        2, 2, 2, 2, 0,  5, "High"),
        ("fingerprint_latent",   "sexual_assault",  2, 2, 2, 2, 0,  5, "High"),
        ("biological_dna",       "robbery",         3, 2, 3, 2, 1,  5, "High"),
        ("digital_device",       "sexual_assault",  3, 1, 3, 1, 1, 14, "High"),
        ("digital_cctv",         "robbery",         3, 1, 2, 1, 0,  2, "High"),
        ("toxicology_blood",     "drug_supply",     3, 3, 3, 2, 1,  3, "High"),
        ("firearm_ballistic",    "robbery",         3, 2, 3, 1, 1, 10, "High"),
        ("firearm_residue",      "robbery",         2, 2, 2, 2, 1,  4, "High"),
        ("digital_device",       "fraud",           3, 1, 2, 1, 1, 14, "High"),
        ("trace_fibre",          "homicide",        2, 2, 2, 2, 1,  7, "High"),

        # --- STANDARD ---
        ("fingerprint_latent",   "burglary",        2, 2, 2, 2, 0,  5, "Standard"),
        ("fingerprint_patent",   "burglary",        2, 1, 1, 1, 0,  3, "Standard"),
        ("biological_dna",       "burglary",        2, 1, 2, 1, 1,  5, "Standard"),
        ("trace_glass",          "burglary",        2, 2, 1, 1, 0,  5, "Standard"),
        ("trace_fibre",          "burglary",        1, 2, 1, 2, 1,  7, "Standard"),
        ("impression_footwear",  "burglary",        2, 1, 1, 2, 0,  7, "Standard"),
        ("digital_cctv",         "burglary",        2, 1, 1, 1, 0,  2, "Standard"),
        ("digital_device",       "drug_supply",     2, 1, 2, 1, 1, 14, "Standard"),
        ("toxicology_hair",      "drug_supply",     2, 1, 2, 1, 1, 21, "Standard"),
        ("impression_tyre",      "arson",           2, 1, 1, 2, 0,  7, "Standard"),
        ("trace_soil",           "arson",           1, 1, 1, 1, 1, 10, "Standard"),
        ("digital_cctv",         "vehicle_crime",   2, 1, 1, 1, 0,  2, "Standard"),
        ("impression_footwear",  "robbery",         1, 1, 1, 2, 0,  7, "Standard"),
        ("document_questioned",  "fraud",           2, 1, 2, 1, 1, 14, "Standard"),

        # --- LOW ---
        ("trace_soil",           "burglary",        1, 1, 1, 1, 1, 10, "Low"),
        ("trace_fibre",          "vehicle_crime",   1, 1, 1, 2, 1,  7, "Low"),
        ("impression_tyre",      "vehicle_crime",   1, 1, 1, 2, 0,  7, "Low"),
        ("fingerprint_patent",   "vehicle_crime",   1, 1, 1, 1, 0,  3, "Low"),
        ("document_questioned",  "vehicle_crime",   1, 1, 1, 1, 1, 14, "Low"),
        ("trace_glass",          "vehicle_crime",   1, 1, 1, 1, 0,  5, "Low"),
        ("toxicology_hair",      "fraud",           1, 1, 1, 1, 1, 21, "Low"),
        ("impression_footwear",  "vehicle_crime",   1, 1, 1, 2, 0,  7, "Low"),
        ("trace_fibre",          "fraud",           1, 1, 1, 1, 1,  7, "Low"),
        ("trace_soil",           "vehicle_crime",   1, 1, 1, 1, 1, 10, "Low"),
        ("fingerprint_latent",   "vehicle_crime",   1, 1, 1, 1, 0,  5, "Low"),
        ("digital_cctv",         "fraud",           1, 1, 1, 1, 0,  2, "Low"),
        ("impression_tyre",      "burglary",        1, 1, 1, 2, 0,  7, "Low"),
        ("document_questioned",  "burglary",        1, 1, 1, 1, 1, 14, "Low"),
        # fmt: on
    ]


# ---------------------------------------------------------------------------
# BLOCK 2 — Broader offence-type coverage
#           Fills gaps in arson, drug_supply, fraud with varied evidence types.
# ---------------------------------------------------------------------------

def _block2():
    return [
        # fmt: off
        # Arson
        ("biological_dna",       "arson",           3, 3, 3, 3, 1,  5, "Critical"),
        ("toxicology_blood",     "arson",            2, 3, 2, 2, 1,  3, "High"),
        ("trace_glass",          "arson",            2, 2, 1, 1, 0,  5, "Standard"),
        ("fingerprint_latent",   "arson",            2, 2, 2, 2, 0,  5, "Standard"),
        ("digital_cctv",         "arson",            3, 1, 2, 1, 0,  2, "High"),
        ("digital_device",       "arson",            2, 1, 2, 1, 1, 14, "Standard"),
        ("trace_fibre",          "arson",            1, 2, 1, 2, 1,  7, "Standard"),
        ("trace_soil",           "arson",            1, 1, 1, 1, 1, 10, "Standard"),
        ("impression_footwear",  "arson",            1, 1, 1, 2, 0,  7, "Low"),
        ("document_questioned",  "arson",            1, 1, 1, 1, 1, 14, "Low"),
        ("toxicology_hair",      "arson",            1, 1, 1, 1, 1, 21, "Low"),
        ("impression_tyre",      "arson",            2, 1, 1, 2, 0,  7, "Standard"),

        # Drug supply
        ("biological_dna",       "drug_supply",     2, 2, 2, 2, 1,  5, "Standard"),
        ("biological_touch_dna", "drug_supply",     2, 3, 2, 3, 1,  7, "High"),
        ("fingerprint_latent",   "drug_supply",     2, 2, 1, 2, 0,  5, "Standard"),
        ("fingerprint_patent",   "drug_supply",     1, 1, 1, 1, 0,  3, "Low"),
        ("digital_cctv",         "drug_supply",     3, 1, 2, 1, 0,  2, "High"),
        ("digital_device",       "drug_supply",     3, 1, 2, 1, 1, 14, "High"),
        ("document_questioned",  "drug_supply",     2, 1, 2, 1, 1, 14, "Standard"),
        ("trace_fibre",          "drug_supply",     1, 1, 1, 1, 1,  7, "Low"),
        ("trace_glass",          "drug_supply",     1, 1, 1, 1, 0,  5, "Low"),
        ("impression_footwear",  "drug_supply",     1, 1, 1, 2, 0,  7, "Low"),
        ("firearm_ballistic",    "drug_supply",     3, 2, 3, 1, 1, 10, "High"),
        ("firearm_residue",      "drug_supply",     2, 2, 2, 2, 1,  4, "Standard"),

        # Fraud
        ("biological_dna",       "fraud",           1, 1, 1, 1, 1,  5, "Low"),
        ("fingerprint_latent",   "fraud",           2, 1, 2, 1, 0,  5, "Standard"),
        ("fingerprint_patent",   "fraud",           1, 1, 1, 1, 0,  3, "Low"),
        ("digital_device",       "fraud",           3, 1, 3, 1, 1, 14, "High"),
        ("digital_cctv",         "fraud",           2, 1, 2, 1, 0,  2, "Standard"),
        ("trace_fibre",          "fraud",           1, 1, 1, 1, 1,  7, "Low"),
        ("document_questioned",  "fraud",           3, 1, 3, 1, 1, 14, "High"),
        ("toxicology_hair",      "fraud",           2, 1, 1, 1, 1, 21, "Standard"),
        ("impression_footwear",  "fraud",           1, 1, 1, 1, 0,  7, "Low"),
        ("impression_tyre",      "fraud",           1, 1, 1, 1, 0,  7, "Low"),
        ("trace_soil",           "fraud",           1, 1, 1, 1, 1, 10, "Low"),
        ("trace_glass",          "fraud",           1, 1, 1, 1, 0,  5, "Low"),
        # fmt: on
    ]


# ---------------------------------------------------------------------------
# BLOCK 3 — Varied perishability / exclusionary_power combinations
#           Ensures the model sees high-E / low-D and vice versa for each
#           serious offence, providing better class separation.
# ---------------------------------------------------------------------------

def _block3():
    return [
        # fmt: off
        # Homicide — varied E and D
        ("trace_fibre",          "homicide",        3, 3, 2, 2, 1,  7, "Critical"),
        ("trace_glass",          "homicide",        2, 2, 2, 1, 0,  5, "High"),
        ("trace_soil",           "homicide",        1, 1, 1, 1, 1, 10, "Standard"),
        ("impression_footwear",  "homicide",        2, 1, 2, 2, 0,  7, "Standard"),
        ("impression_tyre",      "homicide",        2, 1, 1, 2, 0,  7, "Standard"),
        ("toxicology_hair",      "homicide",        2, 2, 2, 1, 1, 21, "High"),
        ("document_questioned",  "homicide",        1, 1, 1, 1, 1, 14, "Standard"),
        ("fingerprint_patent",   "homicide",        2, 1, 1, 1, 0,  3, "High"),

        # Sexual assault — varied E and D
        ("trace_fibre",          "sexual_assault",  3, 3, 2, 2, 1,  7, "Critical"),
        ("trace_glass",          "sexual_assault",  2, 2, 1, 1, 0,  5, "High"),
        ("fingerprint_patent",   "sexual_assault",  2, 2, 2, 1, 0,  3, "High"),
        ("toxicology_hair",      "sexual_assault",  3, 2, 3, 1, 1, 21, "High"),
        ("impression_footwear",  "sexual_assault",  2, 1, 2, 2, 0,  7, "High"),
        ("digital_cctv",         "sexual_assault",  3, 1, 2, 1, 0,  2, "High"),
        ("trace_soil",           "sexual_assault",  1, 1, 1, 1, 1, 10, "Standard"),
        ("document_questioned",  "sexual_assault",  1, 1, 1, 1, 1, 14, "Standard"),

        # Robbery — varied E and D
        ("biological_touch_dna", "robbery",         3, 3, 3, 3, 1,  7, "Critical"),
        ("toxicology_blood",     "robbery",         2, 3, 2, 2, 1,  3, "High"),
        ("fingerprint_latent",   "robbery",         3, 2, 2, 2, 0,  5, "High"),
        ("fingerprint_patent",   "robbery",         2, 1, 1, 1, 0,  3, "Standard"),
        ("trace_glass",          "robbery",         2, 2, 1, 1, 0,  5, "Standard"),
        ("impression_footwear",  "robbery",         2, 1, 2, 2, 0,  7, "Standard"),
        ("trace_fibre",          "robbery",         1, 1, 1, 1, 1,  7, "Low"),
        ("trace_soil",           "robbery",         1, 1, 1, 1, 1, 10, "Low"),
        ("document_questioned",  "robbery",         1, 1, 1, 1, 1, 14, "Low"),
        ("toxicology_hair",      "robbery",         1, 1, 1, 1, 1, 21, "Low"),

        # Burglary — varied E and D
        ("biological_dna",       "burglary",        3, 2, 3, 2, 1,  5, "High"),
        ("biological_touch_dna", "burglary",        2, 3, 2, 3, 1,  7, "High"),
        ("toxicology_blood",     "burglary",        1, 3, 1, 2, 1,  3, "Standard"),
        ("firearm_ballistic",    "burglary",        2, 1, 2, 1, 1, 10, "Standard"),
        ("toxicology_hair",      "burglary",        1, 1, 1, 1, 1, 21, "Low"),
        ("document_questioned",  "burglary",        2, 1, 1, 1, 1, 14, "Standard"),
        # fmt: on
    ]


# ---------------------------------------------------------------------------
# BLOCK 4 — Edge and boundary cases
#           High perishability + low offence severity (forces Standard/High
#           depending on PV), and high PV + low perishability combos.
# ---------------------------------------------------------------------------

def _block4():
    return [
        # fmt: off
        # High perishability but lower-severity offences (not auto-Critical)
        ("toxicology_blood",     "burglary",        3, 3, 2, 2, 1,  3, "High"),
        ("biological_dna",       "vehicle_crime",   2, 3, 2, 2, 1,  5, "Standard"),
        ("biological_touch_dna", "fraud",           1, 3, 1, 3, 1,  7, "Standard"),
        ("trace_fibre",          "arson",           2, 3, 1, 2, 1,  7, "Standard"),
        ("toxicology_blood",     "arson",           3, 3, 2, 2, 1,  3, "High"),
        ("biological_dna",       "arson",           2, 3, 2, 3, 1,  5, "High"),

        # High PV + stable (no perishability urgency) — High or Standard
        ("biological_dna",       "drug_supply",     3, 1, 3, 2, 1,  5, "High"),
        ("firearm_ballistic",    "fraud",           3, 1, 2, 1, 1, 10, "High"),
        ("digital_device",       "vehicle_crime",   3, 1, 2, 1, 1, 14, "Standard"),
        ("digital_device",       "burglary",        3, 1, 2, 1, 1, 14, "Standard"),
        ("firearm_residue",      "fraud",           2, 1, 2, 1, 1,  4, "Standard"),
        ("firearm_residue",      "vehicle_crime",   1, 1, 1, 1, 1,  4, "Low"),

        # Very low probative value across serious offences
        ("trace_soil",           "homicide",        1, 3, 1, 1, 1, 10, "High"),
        ("impression_tyre",      "sexual_assault",  1, 2, 1, 2, 0,  7, "Standard"),
        ("trace_glass",          "sexual_assault",  1, 3, 1, 1, 0,  5, "Standard"),
        ("fingerprint_patent",   "robbery",         1, 2, 1, 1, 0,  3, "Standard"),

        # Critical boundary: perishability=3 + PV=3 in serious offences
        ("firearm_residue",      "sexual_assault",  3, 3, 2, 3, 1,  4, "Critical"),
        ("toxicology_blood",     "robbery",         3, 3, 2, 2, 1,  3, "Critical"),
        ("biological_touch_dna", "homicide",        3, 3, 3, 3, 1,  7, "Critical"),
        ("trace_fibre",          "sexual_assault",  3, 3, 3, 2, 1,  7, "Critical"),

        # Additional Critical rows for better class balance
        ("firearm_residue",      "homicide",        3, 3, 3, 3, 1,  4, "Critical"),
        ("toxicology_blood",     "sexual_assault",  3, 3, 3, 2, 1,  3, "Critical"),
        ("biological_dna",       "robbery",         3, 3, 3, 3, 1,  5, "Critical"),
        ("biological_touch_dna", "sexual_assault",  3, 3, 3, 3, 1,  7, "Critical"),

        # Low across all dimensions — clearly Low
        ("trace_soil",           "fraud",           1, 1, 1, 1, 1, 10, "Low"),
        ("impression_tyre",      "drug_supply",     1, 1, 1, 2, 0,  7, "Low"),
        ("fingerprint_patent",   "arson",           1, 1, 1, 1, 0,  3, "Low"),
        ("trace_fibre",          "robbery",         1, 1, 1, 1, 1,  7, "Low"),
        ("document_questioned",  "drug_supply",     1, 1, 1, 1, 1, 14, "Low"),
        ("impression_footwear",  "fraud",           1, 1, 1, 1, 0,  7, "Low"),
        ("trace_glass",          "arson",           1, 1, 1, 1, 0,  5, "Low"),
        ("toxicology_hair",      "vehicle_crime",   1, 1, 1, 1, 1, 21, "Low"),
        # fmt: on
    ]
