# Engineering Decision Log

This document records the 15 major architecture and engineering design decisions made while developing the **Hiver AI Customer Support Agent**. Each decision details the technical context, alternatives evaluated, rationale, and real-world trade-offs.

---

### Decision 1: Single Brand Focus on `AmazonHelp`
- **Context:** The Twitter Customer Support dataset (`twcs.csv`) contains customer interactions across dozens of distinct enterprises (Apple, Delta, Uber, Spotify, etc.).
- **Alternatives Considered:**
  1. Train a multi-brand generalist support bot.
  2. Sample equal proportions across all 108 brands.
  3. Narrow down to a single brand with high conversational density.
- **Decision:** Focus entirely on `AmazonHelp`.
- **Rationale:** Customer support resolutions are deeply brand- and policy-specific. An agent cannot recommend visiting "Your Orders" or "Prime Membership Settings" if it is simultaneously answering questions for an airline. `AmazonHelp` had the highest turn volume (>160,000 turns) and realistic resolutions (replacement dispatches, locker redirections, refund verifications).
- **Trade-offs:** The system is specialized for Amazon retail support and cannot be applied zero-shot to an airline without replacing the retrieval corpus and intent taxonomy.

---

### Decision 2: 5-Stage Modular Pipeline vs. End-to-End LLM Prompting
- **Context:** A modern LLM can be prompted to "read customer message and history, decide intent, retrieve context, draft a reply, and say if you need human help" in a single prompt.
- **Alternatives Considered:**
  1. Single large prompt (LLM handles classification, retrieval ranking, generation, and escalation simultaneously).
  2. Two-stage pipeline (Classifier -> Generator).
  3. 5-stage decoupled pipeline (Intent -> Retriever -> Evidence Gate -> Generator -> Escalation Policy).
- **Decision:** Implemented a 5-stage decoupled pipeline with explicit data contracts.
- **Rationale:** Single-prompt agents are black boxes: they fail silently, hallucinate retrieval grounding, and make arbitrary escalation decisions that cannot be tuned or debugged. Decoupling allows unit testing each stage independently, setting strict mathematical thresholds on retrieval quality, and enforcing deterministic escalation rules.
- **Trade-offs:** Adds latency overhead of sequential stage execution and requires maintaining schemas between modules.

---

### Decision 3: Deterministic Rule-Based Escalation Engine vs. LLM-Prompted Escalation
- **Context:** How to decide whether an issue should be handed off to a human agent.
- **Alternatives Considered:**
  1. Prompting the LLM: `Should this inquiry be escalated? Answer yes or no.`
  2. Training a binary classifier on historical escalation outcomes.
  3. A deterministic, ordered multi-signal policy engine evaluating measurable signals.
- **Decision:** Implemented `EscalationPolicy` evaluating 7 prioritized rule-based signals.
- **Rationale:** In enterprise customer support, escalation decisions must be audit-proof, deterministic, and compliance-driven. Prompting an LLM for escalation introduces variance, prompt injection vulnerabilities, and non-deterministic false auto-handles. Prioritized signals (e.g., sensitive legal/fraud keywords override high intent confidence) ensure safety-first behavior.
- **Trade-offs:** Requires manual maintenance of keyword lists and confidence thresholds.

---

### Decision 4: Flat Cosine Index (`IndexFlatIP`) vs. Approximate Vector Index (`HNSW` / `IVF`)
- **Context:** Storing and searching dense embeddings over historical resolved cases.
- **Alternatives Considered:**
  1. Approximate Nearest Neighbors (ANN) via FAISS HNSW or IVF-PQ.
  2. Exact brute-force inner product search (`IndexFlatIP`) with normalized vectors.
- **Decision:** Used `IndexFlatIP`.
- **Rationale:** The curated retrieval corpus comprises 500–5,000 highly resolved cases. For corpora $<100,000$ vectors, brute-force exact search runs in $<5\text{ ms}$ on CPU with zero recall degradation, zero indexing hyperparameters (no $M$ or $efSearch$ tuning), and 100% deterministic reproducibility across machines.
- **Trade-offs:** Linear scaling complexity $O(N \cdot D)$; will require migration to HNSW if the corpus expands past 100,000 vectors.

---

### Decision 5: Dedicated Evidence Quality Gate (`EvidenceGate`)
- **Context:** RAG pipelines frequently suffer from "retrieval pollution," where unrelated nearest neighbors are passed to the generator.
- **Alternatives Considered:**
  1. Pass top-k nearest neighbors unconditionally to the LLM generator.
  2. Filter solely on vector cosine similarity score.
  3. Dual-check gate: minimum cosine similarity ($\ge 0.60$) AND intent consistency verification ($\ge 1$ retrieved case must match classified intent) plus near-duplicate filter ($< 0.98$).
- **Decision:** Implemented a multi-rule `EvidenceGate`.
- **Rationale:** Vector similarity in Twitter text can be high due to shared stop words or greetings while describing completely different issues. Checking intent consistency prevents passing billing cases to delivery queries. The near-duplicate filter ($>0.98$) eliminates test-query echoing.
- **Trade-offs:** Rejects marginally relevant context, causing the escalation policy to trigger more frequently.

---

### Decision 6: Zero-Leakage Golden Set Isolation
- **Context:** Ensuring evaluation integrity and preventing retrieval contamination.
- **Alternatives Considered:**
  1. Random split after indexing the full corpus.
  2. Exclude golden set items only from the classifier training data.
  3. Strict isolation: golden set drawn from held-out split; all golden conversation IDs permanently excised from FAISS index and metadata sidecar.
- **Decision:** Enforced strict isolation with programmatic unit test verification (`test_data_leakage`).
- **Rationale:** If an evaluation query can retrieve its own historical conversation thread from FAISS, retrieval similarity will hit $1.00$ and the generator will trivially copy the ground-truth resolution, completely invalidating the benchmark.
- **Trade-offs:** Reduces the size of the retrievable index by the 200 held-out cases.

---

### Decision 7: Stratified 200-Example Golden Set with Difficulty Tiers
- **Context:** Evaluating system performance realistically across customer distributions.
- **Alternatives Considered:**
  1. Random sample of 100 raw tweets.
  2. Hand-crafting 20 idealized synthetic questions.
  3. 200-example stratified evaluation set with 8 intent classes and 3 difficulty tiers (Easy: 57%, Medium: 30%, Hard: 13%) plus 28% escalation ground truth.
- **Decision:** Created the 200-example stratified golden set (`evaluation/golden_set.jsonl`).
- **Rationale:** Production customer support is not uniform: customers use slang, typos, incomplete information, and multi-intent grievances. Stratification guarantees every intent has statistical support while hard cases test the boundaries of escalation.
- **Trade-offs:** Required significant effort in careful annotation and verification of gold reply criteria.

---

### Decision 8: Free-Tier API Rate Limit Resilience and Deterministic Fallback
- **Context:** External LLM APIs (Gemini Flash/Pro) enforce strict free-tier rate limits (15 RPM, 1,000 requests/day).
- **Alternatives Considered:**
  1. Require a paid API key and fail with unhandled exceptions on 429 errors.
  2. Implement aggressive exponential sleep retries (taking hours to complete evaluation).
  3. Seamless in-memory fallback: if rate limits hit, permanently switch to local TF-IDF classifier and deterministic grounded templates.
- **Decision:** Built resilient in-memory offline fallbacks across all pipeline components.
- **Rationale:** Guarantees that any reviewer can clone the repository and run the full 200-example evaluation suite or test harness in seconds without needing API credentials or incurring unexpected charges.
- **Trade-offs:** When offline fallback is active, classification and generation rely on the local n-gram models rather than frontier LLM reasoning.

---

### Decision 9: Non-Parametric Bootstrap Confidence Intervals (N=1000)
- **Context:** Reporting single point-estimate metrics on a 200-example evaluation set.
- **Alternatives Considered:**
  1. Report single point estimates (e.g., "Accuracy = 96.0%").
  2. Parametric normal approximation ($p \pm 1.96 \sqrt{p(1-p)/n}$).
  3. 1,000-sample non-parametric bootstrap percentile intervals.
- **Decision:** Computed 95% bootstrap confidence intervals for all primary metrics.
- **Rationale:** On a 200-sample test set, point estimates can be misleading due to small sample noise. Non-parametric bootstrapping makes no Gaussian distribution assumptions and provides honest upper and lower bounds (e.g., Escalation F1: `[0.2561, 0.4343]`).
- **Trade-offs:** Adds ~1 second of computational overhead during evaluation reporting.

---

### Decision 10: LLM-as-Judge with Fixed 5-Dimension Rubric
- **Context:** Evaluating customer support reply quality without manual review of every output.
- **Alternatives Considered:**
  1. BLEU / ROUGE against historical agent tweets.
  2. Single holistic 1–10 prompt rating.
  3. 5-dimension rubric (1–5 scale: Correctness, Groundedness, Resolution, Tone, Non-Hallucination) with independent overall score.
- **Decision:** Implemented a versioned 5-dimension rubric with structured JSON output.
- **Rationale:** BLEU/ROUGE penalize valid alternate phrasings and reward superficial n-gram overlap with old agent tweets. Multi-dimensional rubrics allow diagnosing specific flaws (e.g., high tone score but poor resolution or hallucinated claims).
- **Trade-offs:** LLM judges can be slightly more lenient on polite but non-actionable template responses.

---

### Decision 11: Human vs. LLM Judge Calibration Study (N=50)
- **Context:** A skeptical reviewer will ask: *"How do you know your LLM judge isn't just hallucinating high scores for your own generator?"*
- **Alternatives Considered:**
  1. Assume the LLM judge is accurate based on published literature.
  2. Conduct a formal agreement study comparing human expert ratings against LLM judge ratings using MAD, Exact, Within-1, Spearman $\rho$, and weighted Cohen's $\kappa$.
- **Decision:** Conducted and persisted the 50-example calibration study (`experiments/results/agreement_stats.json`).
- **Rationale:** Establishes empirical justification for using the LLM judge. Demonstrates 100% within-1-point agreement and weighted Cohen's $\kappa = 0.972$.
- **Trade-offs:** Required hand-scoring 50 diverse customer service interactions across all 5 rubric dimensions.

---

### Decision 12: Zero-Division and Imbalanced Confusion Matrix Protections
- **Context:** Calculating precision, recall, and F1 on rare classes or edge cases where predicted counts are zero.
- **Alternatives Considered:**
  1. Let standard scikit-learn warnings fire.
  2. Explicitly pass `zero_division=0` and custom mathematical guardrails in `evaluation/metrics.py`.
- **Decision:** Guarded all metric computations with explicit zero-division handlers.
- **Rationale:** Prevents runtime crashes or NaN metrics when evaluating baseline models (e.g., Majority baseline predicting 0 escalations).
- **Trade-offs:** Treats undefined metrics as 0.0, which must be clearly interpreted in documentation.

---

### Decision 13: Grounding Self-Assessment by Generator
- **Context:** Measuring hallucination before sending replies to end users.
- **Alternatives Considered:**
  1. Trust prompt instructions to avoid hallucinations.
  2. Separate post-hoc LLM verification call.
  3. Co-emitted structured JSON flags (`grounding_used: bool`, `hallucination_risk: "low" | "medium" | "high"`).
- **Decision:** Generator co-emits grounding flags as part of its structured output contract.
- **Rationale:** Co-emission adds 0 additional API latency or token cost while providing an immediate signal to the downstream escalation engine. If the generator itself flags `high` hallucination risk, the policy instantly overrides auto-handling and escalates to a human.
- **Trade-offs:** Self-reported risk can occasionally suffer from model overconfidence.

---

### Decision 14: Top-K Retrieval Parameter Set to K=3
- **Context:** Choosing the number of historical cases to retrieve and inject into the generation prompt.
- **Alternatives Considered:**
  1. $k=1$ (Minimal context).
  2. $k=3$ (Multi-case synthesis).
  3. $k=5$ (Broadest context).
- **Decision:** Set $k=3$ based on empirical ablation study.
- **Rationale:** The ablation study showed $k=1$ had a high false auto-handle rate (41.07%) because single-case retrieval is fragile. $k=5$ introduced cross-intent noise and higher token consumption. $k=3$ achieved the highest escalation F1 (0.4400) and lowest false auto-handle rate (0.0179).
- **Trade-offs:** Uses ~250 tokens per prompt for context injection.

---

### Decision 15: Cross-Platform Windows & UTF-8 Encoding Compatibility
- **Context:** The development and execution environment runs on Windows with Python 3.13 where standard output defaults to Windows `cp1252` encoding.
- **Alternatives Considered:**
  1. Assume a Linux terminal and use arbitrary unicode characters ($\mu$, em-dashes `—`, box-drawing `─`).
  2. Enforce strict ASCII-compatible CLI tables and explicit `utf-8` file I/O throughout the codebase.
- **Decision:** Replaced all special unicode characters in console logs and CLI tables with clean ASCII equivalents (`Human Mean`, `--`, `-`), and enforced explicit `encoding="utf-8"` on all file handlers.
- **Rationale:** Prevents fatal `UnicodeEncodeError` crashes when reproducing evaluations or running tests on Windows, ensuring any evaluator can clone and run seamlessly.
- **Trade-offs:** Slightly simpler visual formatting in terminal outputs.
