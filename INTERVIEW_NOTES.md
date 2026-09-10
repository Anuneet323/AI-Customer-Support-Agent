# Interview Preparation & Live System Defense Notes

This guide is designed for defending the **Hiver AI Customer Support Agent** during live technical interviews, architecture reviews, and code walkthroughs.

---

## 1. The 60-Second System Elevator Pitch

> *"Most customer support chatbots fail in production because they treat support as an unconstrained text generation problem. When an LLM hallucinates a refund promise or misses an account takeover, it causes severe real-world harm.*  
> 
> *In this project, I built an evaluation-first, production-oriented customer support AI for AmazonHelp on Twitter. It uses a decoupled 5-stage architecture: Zero-shot Intent Classifier, FAISS Dense Retriever, an Evidence Quality Gate, a Constrained Grounded Reply Generator, and an Auditable 7-Signal Escalation Engine.*  
> 
> *I evaluated the system on an isolated 200-sample stratified golden set. While the system achieves 96% intent accuracy, I openly highlight that its escalation F1 is 0.345 and false auto-handle rate is 46.4%. This honest metric dissection, backed by a 50-example Human-LLM judge calibration study (kappa=0.972) and non-parametric bootstrap intervals, is what distinguishes a production-minded engineer from someone who just writes prompts."*

---

## 2. Deep-Dive Q&A: Defending Architecture & Engineering Choices

### Q1: "Why didn't you just fine-tune an open-source model (e.g., Llama-3 or Mistral)?"
**Answer:**
1. **Separation of Concerns:** Fine-tuning combines classification, retrieval, and policy logic into weights that cannot be inspected or tuned independently. If business policy changes tomorrow (e.g., lower the refund threshold), a fine-tuned model must be completely re-trained and re-evaluated.
2. **Cold-Start Agility:** In real support environments, intent taxonomies shift constantly. Zero-shot classification with YAML-defined taxonomy allows updating business rules in minutes without retraining pipelines.
3. **Data Quality Risk:** Training directly on historical raw Twitter agent replies bakes in historical human errors, bad links, and outdated policies. Grounded retrieval with explicit evidence gating ensures the model only surfaces active, verified resolutions.

---

### Q2: "Why use FAISS FlatIP instead of HNSW or an enterprise vector database like Milvus or Pinecone?"
**Answer:**
1. **Corpus Scale Reality:** Our curated historical resolution database has 500–5,000 cases. At this scale, FlatIP brute-force search executes in under 5 milliseconds on a single CPU core.
2. **Deterministic Reproducibility:** Approximate Nearest Neighbor (ANN) algorithms like HNSW introduce recall approximation errors and require tuning hyperparameters ($M$, $efSearch$). FlatIP guarantees 100% exact cosine search that produces identical results on any machine.
3. **No External Infrastructure Dependency:** By embedding FAISS with a JSON metadata sidecar, the entire repository runs self-contained without needing Docker containers or cloud vector database API keys.
4. **Production Path:** In production with $>500,000$ cases, I would migrate to Milvus or Qdrant with HNSW indexing and distributed sharding.

---

### Q3: "How did you prove that there is zero data leakage between your evaluation set and retrieval index?"
**Answer:**
1. **Conversation ID Segregation:** Every customer interaction has a unique conversation ID. During data preparation, the 200 golden set conversations were drawn exclusively from `held_out.jsonl`.
2. **Pre-Index Sanitization:** Before compiling the FAISS index, the indexing script filtered out any record whose ID existed in `evaluation/golden_set.jsonl`.
3. **Automated Unit Test Proof:** In `tests/test_data.py`, test `test_data_leakage` programmatically computes the set intersection between `golden_set.jsonl` IDs and `data/faiss_index/metadata.json` case IDs, asserting `len(intersection) == 0`.
4. **Near-Duplicate Gating:** Even if an identical problem was submitted by a different user, the `EvidenceGate` enforces a `max_similarity_for_dedup: 0.98` filter that rejects near-duplicate queries.

---

### Q4: "Your intent accuracy is 96%, but your escalation F1 is only 0.345. Why?"
**Answer:**
1. **The 'Misleading Headline' Reality:** Intent classification is a standard NLP problem where keywords like 'package', 'card', or 'login' provide strong signals. Escalation, however, requires understanding subtle pragmatics, unspoken severity, and missing context.
2. **Extreme Class Asymmetry:** Escalation errors are asymmetric. The escalation engine prioritizes safety, resulting in 88 unnecessary escalations (false positives). Because precision is low ($0.2542$), the resulting F1 is mathematically pulled down to $0.3448$.
3. **Short Query Sparsity:** Many customer tweets are extremely short (e.g., *"where package???"*). These fail the dense retrieval similarity threshold ($<0.60$) because they lack lexical overlap with detailed resolved cases. The evidence gate correctly flags them as low evidence and triggers escalation, increasing false positives.

---

### Q5: "How does the Evidence Gate prevent hallucination?"
**Answer:**
The `EvidenceGate` acts as a firewall between retrieval and generation:
1. **Similarity Hurdle:** Rejects any retrieval result with cosine similarity $< 0.60$.
2. **Intent Consistency Check:** Verifies that at least one of the top retrieved cases matches the classified intent. If a customer has a delivery issue, but the retriever returns a billing dispute, the gate fails with `INTENT_MISMATCH`.
3. **Downstream Action:** When the gate fails, the reply generator is instructed to withhold specific resolution claims, and the escalation policy immediately flags `LOW_EVIDENCE` and routes the ticket to a human.

---

### Q6: "Why did you conduct a Human vs. LLM Judge agreement study?"
**Answer:**
1. **Addressing Self-Grading Skepticism:** Using an LLM to evaluate an LLM creates an inherent risk of circular evaluation bias.
2. **Statistical Validation:** By scoring 50 examples by both a human expert and the LLM judge across 5 explicit dimensions, we proved that the LLM judge has **100% within-1-point agreement**, a **Mean Absolute Difference (MAD) of 0.020**, and a **weighted Cohen's $\kappa$ of 0.972**.
3. **Empirical Calibration:** The study revealed that the LLM judge was +0.20 more generous on tone, while the human evaluator was more skeptical of generic resolutions. This calibration informed our prompt tuning.

---

## 3. Live Code Walkthrough: Which Files to Show the Interviewer

When asked to navigate the codebase, follow this sequential path:

1. **`configs/intents.yaml` & `configs/config.yaml`:** Show that the system has no magic numbers; thresholds, models, and taxonomy are cleanly separated.
2. **`src/pipeline.py` (`SupportAgentPipeline.run`):** Walk through the 5 clean stages. Highlight error trapping and latency logging.
3. **`src/retrieval/evidence.py` (`EvidenceGate.assess`):** Show the exact logic where retrieved context is audited for similarity, intent consistency, and near-duplicate rejection.
4. **`src/escalation/policy.py` (`EscalationPolicy.decide`):** Show the 7 prioritized signals. Emphasize why rule-based ordering is more defensible than asking an LLM "should I escalate?".
5. **`tests/`:** Run `pytest tests/ -v` live. Show 65 passing unit tests covering edge cases, zero-division, and data leakage.
6. **`experiments/results/summary_metrics.json`:** Point out the bootstrap confidence intervals and per-intent precision/recall breakdowns.

---

## 4. Production Scaling & Architecture Vision

If asked how to scale this from a take-home prototype to 100,000 requests per day:

1. **Inference Latency Optimization:**
   - Pre-filter: Regex-based sensitive keyword check runs in $<1\text{ ms}$ before invoking any model.
   - Classification: Run a quantized FastEmbed / ONNX model locally for classification ($<25\text{ ms}$), routing only ambiguous queries ($<0.70$ confidence) to Gemini Flash.
   - Retrieval: Host FAISS or Milvus on a GPU node with $L_2$ normalized inner product indexing ($<10\text{ ms}$).
   - Total P95 latency target: $< 600\text{ ms}$.
2. **Cost Optimization:**
   - 100k daily queries with Gemini Flash: ~20M tokens/day $\approx \$1.50 - \$3.00/\text{day}$.
   - Self-hosted vLLM with Llama-3-8B-Instruct on a single NVIDIA A10G: $\approx \$1.00/\text{hour}$ fixed compute cost.
3. **Continuous Evaluation & Monitoring:**
   - Implement shadow scoring on 2% of live traffic with human agent escalation feedback loops.
   - Monitor embedding drift using Population Stability Index (PSI) to detect when customer vocabulary changes (e.g., during Prime Day or new product releases).
