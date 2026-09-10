# Hiver Take-Home Submission Checklist & Audit Matrix

This checklist cross-references every technical, architectural, and evaluation requirement of the take-home assignment against the implemented codebase and generated artifacts.

---

| # | Requirement Category | Requirement Specification | Compliance Status | Evidence & Verification File Path |
|---|---|---|:---:|---|
| 1 | **Dataset & Brand** | Grounded in Twitter Customer Support (`twcs.csv`) with deliberate brand choice | **PASS** | `AmazonHelp` selected; rationale documented in `report/report.md` Section 2. Data builder in `src/data/builder.py`. |
| 2 | **Dialogue Reconstruction** | Multi-turn conversation assembly preserving chronological customer & agent turns | **PASS** | `src/data/builder.py`; verified via unit tests `test_builds_conversation`, `test_customer_brand_speakers` in `tests/test_data.py`. |
| 3 | **Data Cleaning & Dedup** | Handle cleaning, URL normalization, near-duplicate removal | **PASS** | `src/data/cleaner.py` and `src/data/dedup.py`; verified via unit tests in `tests/test_data.py` (8 passing tests). |
| 4 | **Data Splits** | Train (70%), Val (15%), Held-Out Test (15%) with zero contamination | **PASS** | `data/processed/train.jsonl`, `val.jsonl`, `held_out.jsonl`; leakage test in `tests/test_data.py::test_data_leakage`. |
| 5 | **Golden Evaluation Set** | 150–250 hand-curated, expert-reviewed evaluation examples | **PASS** | Exactly 200 records in `evaluation/golden_set.jsonl`; schema documented in `evaluation/GOLDEN_SET.md`. |
| 6 | **Golden Set Stratification** | Stratified across intents, difficulty tiers (easy, med, hard), and escalation | **PASS** | 8 intents represented; 114 easy (57%), 60 medium (30%), 26 hard (13%), 56 escalation cases (28%). |
| 7 | **Zero Data Leakage** | Golden set conversation IDs excluded from training data and FAISS index | **PASS** | Verified 0 overlap between `evaluation/golden_set.jsonl` and `data/faiss_index/metadata.json`. |
| 8 | **5-Stage Pipeline** | Modular decoupled pipeline: Intent -> Retrieve -> Evidence -> Generate -> Escalate | **PASS** | Implemented in `src/pipeline.py` (`SupportAgentPipeline`). |
| 9 | **Intent Classifier** | Zero-shot taxonomy-grounded intent classification with confidence | **PASS** | `src/intent/classifier.py`; 8 classes in `configs/intents.yaml`; tests in `tests/test_intent.py`. |
| 10 | **Trivial Baseline** | Majority class baseline implementation | **PASS** | `src/intent/baseline.py` (`MajorityClassifier`); evaluated in `evaluation/evaluate.py`. |
| 11 | **Simple Baseline** | TF-IDF + Logistic Regression baseline implementation | **PASS** | `src/intent/baseline.py` (`TFIDFLogisticRegression`); evaluated in `evaluation/evaluate.py`. |
| 12 | **Dense Retrieval Index** | FAISS index over historically resolved customer cases | **PASS** | `src/retrieval/index.py` (`FAISSIndex`); 500 records indexed in `data/faiss_index/`. |
| 13 | **Evidence Quality Gate** | Reject weak / mismatched / test-echoing retrieval before generation | **PASS** | `src/retrieval/evidence.py` (`EvidenceGate`); tests in `tests/test_retrieval.py`. |
| 14 | **Grounded Generator** | Prompt with strict groundedness constraints and self-reported risk | **PASS** | `src/generation/reply_generator.py`; tests in `tests/test_generation.py`. |
| 15 | **Escalation Engine** | Multi-signal deterministic escalation policy (audit-proof) | **PASS** | `src/escalation/policy.py` (`EscalationPolicy`); 7 signals; tests in `tests/test_escalation.py`. |
| 16 | **Automated Eval Harness** | CLI evaluation orchestrator running all systems on golden set | **PASS** | `evaluation/evaluate.py` (`python evaluation/evaluate.py --system all --no-judge`). |
| 17 | **Bootstrap CIs** | 95% non-parametric bootstrap confidence intervals (1,000 resamples) | **PASS** | `evaluation/bootstrap.py`; reported in `experiments/results/summary_metrics.json`. |
| 18 | **LLM-as-Judge Rubric** | 5-dimension rubric (1–5 scale: Correctness, Groundedness, Resolution, Tone, Safety) | **PASS** | `evaluation/judge.py` (`LLMJudge`); prompts and scoring logic versioned (`v1`). |
| 19 | **Human-LLM Agreement** | 50-example study reporting MAD, Exact, Within-1, Spearman rho, Cohen's kappa | **PASS** | `evaluation/human_judgments.json`, `experiments/results/agreement_stats.json`. |
| 20 | **Failure Analysis** | Empirical discovery of Top 5 failure modes with concrete examples | **PASS** | `evaluation/failure_analysis.py`; saved to `experiments/results/failure_modes.json`. |
| 21 | **Ablation Studies** | Systematic comparison (No-retrieval, Top-k variation with k=1, 3, 5) | **PASS** | `experiments/run_experiments.py`; saved to `experiments/results/ablation_summary.json`. |
| 22 | **"Misleading Headline"** | Mandatory critical reflection: why intent accuracy is insufficient | **PASS** | Explicitly analyzed in `README.md` and `report/report.md` Section 7. |
| 23 | **Unit Test Suite** | >50 comprehensive unit tests covering all components | **PASS** | **65 unit tests passing** in `tests/` (`test_data.py`, `test_intent.py`, `test_retrieval.py`, `test_escalation.py`, `test_generation.py`, `test_evaluation.py`). |
| 24 | **15-Min Reproduction** | Clean, documented setup runnable in <15 minutes on Windows/macOS/Linux | **PASS** | Verified end-to-end; step-by-step instructions in root `README.md`. |
| 25 | **Decision Log** | 12–15 non-obvious engineering decisions documented with trade-offs | **PASS** | 15 detailed decisions documented in `DECISION_LOG.md`. |
| 26 | **Interview Prep Notes** | Technical defense notes and anticipated interview Q&A | **PASS** | Complete interview defense guide in `INTERVIEW_NOTES.md`. |
| 27 | **Citations & References** | Formal academic and technical citations for datasets and tools | **PASS** | Documented in `CITATIONS.md`. |
| 28 | **No Magic Numbers** | Master configuration file controlling all thresholds and model names | **PASS** | `configs/config.yaml` and `configs/intents.yaml`. |
| 29 | **Offline Resilience** | Safe execution without crashing if free-tier API quotas are exhausted | **PASS** | Tested and verified; fallback handlers active across classifier, retriever, and generator. |
| 30 | **Production Roadmap** | Enterprise deployment architecture, latency budget, and cost analysis | **PASS** | Detailed in `report/report.md` Section 9. |
