# Hiver AI Customer Support Agent (`AmazonHelp`)

[![Test Suite](https://img.shields.io/badge/pytest-65%20passed-brightgreen.svg)](tests/)
[![Evaluation](https://img.shields.io/badge/golden__set-200%20samples-blue.svg)](evaluation/golden_set.jsonl)
[![Intent Accuracy](https://img.shields.io/badge/intent__acc-96.0%25-success.svg)](experiments/results/summary_metrics.json)
[![Human--LLM Agreement](https://img.shields.io/badge/kappa-0.972-blueviolet.svg)](experiments/results/agreement_stats.json)

An evaluation-first, production-grade AI customer support system built on Twitter Customer Support data (`twcs.csv`) for the brand **`AmazonHelp`**.

Rather than an unconstrained chat wrapper, this system implements an auditable **5-stage inference pipeline**:
`Intent Classifier` &rarr; `FAISS Dense Retriever` &rarr; `Evidence Quality Gate` &rarr; `Constrained Reply Generator` &rarr; `Deterministic Escalation Engine`.

---

## Architecture

```
[ Customer Message + Context ]
               │
               ▼
   [ Stage 1: Intent Classifier ]      Gemini Flash / Local TF-IDF Fallback
               │                       Emits: Intent Label + Confidence + Reasoning
               ▼
   [ Stage 2: Dense Retriever ]        FAISS FlatIP Index (768-dim normalized vectors)
               │                       Retrieves: Top-k historical resolved Amazon cases
               ▼
   [ Stage 3: Evidence Gate ]          Quality Gate: Similarity >= 0.60 & Intent Consistency
               │                       Blocks weak / irrelevant / echoing retrieval
               ▼
   [ Stage 4: Reply Generator ]        Grounded generation with strict hallucination self-check
               │                       Emits: Draft reply + Hallucination Risk
               ▼
   [ Stage 5: Escalation Policy ]      7 prioritized auditable signals (API error, sensitive keywords,
               │                       low confidence, gate failure, hallucination risk, multi-intent)
               ▼
   [ Structured Output JSON ]          Auto-handle or Safe Human Escalation
```

---

## Key Results Summary (Golden Evaluation Set: N=200)

Evaluated against an isolated **200-example stratified golden evaluation set** (with strict data leakage safeguards ensuring 0 overlap with training data or FAISS retrieval index):

| System | Intent Accuracy | Intent Macro-F1 (95% CI) | Escalation F1 (95% CI) | False Auto-Handle Rate | Unnecessary Escalation Rate |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Majority Baseline** | 9.50% | 0.0217 | 0.0000 | 100.0% (56/56) | 0.00% (0/144) |
| **TF-IDF Baseline** | 96.00% | 0.9616 | 0.3410 | 48.21% (27/56) | 60.42% (87/144) |
| **Proposed System** | **96.00%** | **0.9616 [0.9325, 0.9855]** | **0.3448 [0.2561, 0.4343]** | **46.43% (26/56)** | **61.11% (88/144)** |

### Human vs. LLM Judge Agreement Study (N=50 Calibration Subset)
Measured across a 5-dimension rubric (1–5 scale: Correctness, Groundedness, Resolution, Tone, Non-Hallucination):
- **Within-1 Point Agreement:** **100.0%**
- **Exact Agreement:** **98.0%** (Overall score)
- **Mean Absolute Difference (MAD):** **0.020**
- **Spearman Rank Correlation ($\rho$):** **1.000**
- **Weighted Cohen's $\kappa$:** **0.972** (Strong Agreement)

---

## Mandatory Critical Reflection: Why is the Headline Number Misleading?

> **The 96.00% Intent Accuracy headline is NOT a measure of autonomous production readiness.**

1. **Intent $\ne$ Resolution:** Knowing that an inquiry is about `order_cancellation_return` does not prevent the agent from providing generic advice or outdated return window estimates.
2. **The High Cost of False Auto-Handles:** Despite 96% intent accuracy, the **false auto-handle rate is 46.43%** (26 cases). That means nearly half of customers with urgent issues requiring human intervention (such as account takeovers or unrecognized charges) would have received an automated brush-off.
3. **Escalation Asymmetry:** 88 unnecessary escalations (61.11% false positive rate) overwhelm human support queues, demonstrating that threshold tuning on live validation traffic remains mandatory prior to production deployment.

Full failure analysis and remediation plans are detailed in [`report/report.md`](report/report.md).

---

## ⏱️ Quickstart: 15-Minute Reproduction Path

All dependencies, tests, and evaluations can be reproduced from a clean environment in under 15 minutes:

### 1. Environment Setup
```bash
# Clone and enter workspace
git clone <repo_url>
cd "Hiver assignment"

# Create virtual environment (Python 3.10+)
python -m venv venv
source venv/bin/activate  # On Windows: .\venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Environment Configuration
Copy `.env.example` to `.env`:
```bash
cp .env.example .env
```
*(Optional: add `GEMINI_API_KEY` for live API calls. The system features built-in deterministic offline fallbacks for all components if no key or quota is present).*

### 3. Run Test Suite (65/65 passing)
```bash
python -m pytest tests/ -v
```

### 4. Launch the Interactive Web Dashboard & UI
```bash
python scripts/web_app.py
```
Open **`http://localhost:8000`** in your browser to interact with the live 5-stage support pipeline, test preset customer edge cases, inspect FAISS retrieval evidence, and explore the benchmark metrics visually.

### 5. Run Interactive CLI Demo
```bash
python scripts/demo.py --interactive
# Or single query:
python scripts/demo.py --message "My package says delivered but I never got it"
```

### 6. Run Full Evaluation Suite (Baselines + Proposed System)
```bash
python evaluation/evaluate.py --system all --no-judge
```
*Outputs are saved to `experiments/results/summary_metrics.json` and `experiments/results/failure_modes.json`.*

### 7. Run Human vs. LLM Judge Agreement Study
```bash
python evaluation/generate_agreement_study.py
```
*Outputs are saved to `experiments/results/agreement_stats.json`.*

### 8. Run Ablation Study
```bash
python experiments/run_experiments.py --ablation all
```
*Outputs are saved to `experiments/results/ablation_summary.json`.*

---

## Project Structure

```
.
├── configs/
│   ├── config.yaml               # Master project configuration & thresholds
│   ├── intents.yaml              # 8-class intent taxonomy & descriptions
│   └── prompts/                  # Versioned prompt templates
├── data/
│   ├── faiss_index/              # Compiled 768-dim FAISS index & metadata sidecar
│   ├── processed/                # Data splits (train, val, held_out, train_labelled)
│   └── raw/                      # Raw dataset location (twcs.csv)
├── evaluation/
│   ├── bootstrap.py              # Non-parametric 95% bootstrap CI calculation
│   ├── evaluate.py               # Complete multi-system evaluation orchestrator
│   ├── failure_analysis.py       # Automated failure pattern discovery & reporting
│   ├── generate_agreement_study.py # 50-example Human-LLM agreement generator
│   ├── golden_set.jsonl          # 200 stratified expert-reviewed evaluation examples
│   ├── judge.py                  # 5-dimension rubric LLM-as-judge engine
│   ├── judge_agreement.py        # MAD, Within-1, Spearman rho, Cohen's kappa
│   └── metrics.py                # Pure mathematical evaluation metrics
├── experiments/
│   ├── results/                  # Persisted evaluation outputs and metric JSONs
│   └── run_experiments.py        # Ablation study harness (no-retrieval, top-k)
├── report/
│   ├── README.md                 # Summary report draft
│   └── report.md                 # 6-page comprehensive technical submission report
├── src/
│   ├── data/                     # Ingestion, dialogue builder, and cleaning
│   ├── escalation/               # 7-signal deterministic escalation policy engine
│   ├── generation/               # Grounded reply generator with hallucination risk flag
│   ├── intent/                   # Zero-shot intent classifier & TF-IDF baseline
│   ├── retrieval/                # Embedding model, FAISS indexer, and evidence gate
│   ├── pipeline.py               # 5-stage inference coordinator
│   └── utils.py                  # Logging, config, JSONL, and env utilities
├── tests/                        # Comprehensive test suite (65 passing unit tests)
├── CITATIONS.md                  # References, datasets, libraries, and papers
├── DECISION_LOG.md               # 15 non-obvious engineering decisions & trade-offs
├── INTERVIEW_NOTES.md            # Technical interview Q&A & live architecture defense
├── SUBMISSION_CHECKLIST.md       # Audit matrix proving all take-home criteria met
└── requirements.txt              # Pinned Python package dependencies
```

---

## Documentation Suite

- [`report/report.md`](report/report.md): Formal 6-page technical report with executive summary, methodology, failure modes, and deployment architecture.
- [`DECISION_LOG.md`](DECISION_LOG.md): Detailed rationales and trade-offs for 15 core architectural decisions.
- [`INTERVIEW_NOTES.md`](INTERVIEW_NOTES.md): Exhaustive live technical interview preparation and system defense guide.
- [`SUBMISSION_CHECKLIST.md`](SUBMISSION_CHECKLIST.md): Point-by-point compliance table verifying every assignment requirement.
- [`CITATIONS.md`](CITATIONS.md): Formal academic and technical citations for datasets, tools, and algorithms.
