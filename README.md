# 🚀 [Your Project Title Here]

> ⚠️ **Replace everything in `[ ]` brackets with your actual content before submission.**

---

## 👥 Team

| Field | Value |
|---|---|
| **Team Name** |[Epsilon Infinity ]|
| **Track** | [ Forensic Science] |
| **Team Lead** | [Priyesh Raj] — [Priyeshr067@gmail.com] |
| **Members** | [Gopal Gawande], [Praveen Gundyagol]|

---

## 🎯 Problem Statement

> In 2–3 sentences: What problem does your project solve? Who experiences this problem?

[Crime scene investigators are routinely overwhelmed by volume — the 2019 Hyderabad veterinarian murder case alone produced 3,000+ photos and 200+ physical items — with no systematic way to decide what gets examined first. Without prioritization, perishable, high-value evidence (a DNA-bearing cigarette butt was nearly missed in that case) can be buried under low-value items, while backlogged Forensic Science Laboratories (FSLs) add weeks of delay on top.

]

---

## 💡 Solution

> In 2–3 sentences: What did you build? How does it solve the problem above?

[We built **EvidencePro**, an AI-powered evidence triage assistant with two engines, both running on IBM watsonx.ai. For photos, watsonx's Granite Vision model scans crime scene images and flags probable evidence categories and time-sensitive items directly. For written item lists, watsonx's Granite text model extracts only observable facts from each description (not a priority judgment), which a trained decision tree then classifies deterministically — returning the exact split-by-split reasoning behind every ranking, so investigators get a fully explainable, ranked FSL submission schedule instead of a black-box score.
]

---

## ✨ Key Features

- **Feature 1:** [**Vision-based photo triage:** IBM watsonx.ai's Granite Vision model scans crime scene photographs directly and classifies each into a forensic evidence category, targeting the exact "3,000-photo, missed cigarette butt" failure mode from the Hyderabad case.
]
- **Feature 2:** [**Explainable decision-tree prioritization:** A trained `DecisionTreeClassifier` assigns each item's priority tier and returns the literal sequence of yes/no splits it followed — every "why is this Critical?" has a concrete, auditable answer instead of an opaque LLM-generated score.
]
- **Feature 3:** [**Gen AI narrowed to fact-extraction, not judgment:** watsonx's Granite text model only pulls structured, observable features from free text (is it biological? exposed to weather? a locked device?) — keeping the AI's role low-risk while the tree does all actual prioritizing.
]
- **Feature 4:** [**Perishability-aware urgency flagging:** Both engines flag time-sensitive evidence (DNA degradation, volatile accelerants, devices at risk of battery loss or remote wipe) so fragile evidence surfaces regardless of raw score.
]
- **Feature 5:** [**Ranked, tiered FSL schedule:** Outputs a Critical/High/Standard/Low examination schedule with per-category FSL section and turnaround estimates, built iteratively by running every item through the tree and sorting the results.
]

---

## 🛠️ Tech Stack

| Category | Technologies |
|---|---|
| **Languages** | [Python, JavaScript, HTML, CSS] |
| **Frameworks** | [Flask, Flask-CORS] |
| **IBM Technologies** | [] |
| **Databases** | [None — stateless request/response pipeline ] |
| **Other** | [scikit-learn (trained `DecisionTreeClassifier`), joblib (model serialization)] |

---

## 📁 Repository Structure

```
├── src/                  # All source code
├── docs/                 # Written documentation
│   ├── problem-statement.md
│   ├── solution-overview.md
│   ├── architecture.md
│   └── setup-guide.md
├── demo/                 # Demo artifacts
│   ├── screenshots/      # App screenshots
│   └── demo-video-link.txt  # Link to demo video
├── presentation/         # Slide deck
└── submission.yaml       # Structured submission metadata
```

---

## ⚡ How to Run

> **Copy these exact steps from your [`docs/setup-guide.md`](docs/setup-guide.md)**

```bash
# 1. Clone the repo
git clone https://github.com/[your-repo].git
cd [your-repo]

# 2. Install dependencies
[your install command here]

# 3. Configure environment
cp .env.example .env
# Edit .env with your values

# 4. Run the project
[your run command here]
```

---

## 🖥️ Demo

| Artifact | Link |
|---|---|
| 📹 Demo Video | [See demo/demo-video-link.txt](demo/demo-video-link.txt) |
| 🌐 Live Demo | [See demo/live-demo-url.txt](demo/live-demo-url.txt) |
| 🖼️ Screenshots | [See demo/screenshots/](demo/screenshots/) |
| 📊 Presentation | [See presentation/slides.pdf](presentation/) |

---

## ⚠️ Known Limitations

> Be honest — judges appreciate transparency over overclaiming.

- [Limitation 1: e.g., "Authentication is mocked — not production-ready"]
- [Limitation 2: e.g., "Only tested on Chrome"]
- [Limitation 3: e.g., "Feature X is scaffolded but not fully implemented"]

---

## 🏅 What We're Most Proud Of

[Tell the judges what part of your submission is strongest and worth paying close attention to.]

---
