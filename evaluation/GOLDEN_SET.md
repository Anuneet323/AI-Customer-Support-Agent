# Golden Evaluation Set Documentation

## Overview

The Golden Evaluation Set (`evaluation/golden_set.jsonl`) consists of **200 carefully constructed, hand-reviewed customer support examples** tailored to the **AmazonHelp** brand. It provides the immutable ground truth used to evaluate the entire pipeline: intent classification, historical resolution retrieval, reply generation, and escalation policy decisions.

---

## 1. Sampling & Stratification Methodology

To reflect realistic Twitter support traffic, the 200 examples were drawn from the **held-out split** and deliberately stratified across four primary dimensions:

### A. Intent Distribution (8 Intents)
| Intent | Count | % of Set | Operational Focus |
|---|---|---|---|
| `billing_payment_issue` | 28 | 14.0% | Duplicate charges, unauthorized transactions, Prime fees |
| `delivery_issue` | 27 | 13.5% | Tracking delays, missing packages, misdeliveries |
| `product_quality_issue` | 27 | 13.5% | Damaged screens, defective parts, wrong items |
| `account_access_issue` | 26 | 13.0% | Password resets, 2FA issues, account compromise |
| `order_cancellation_return` | 26 | 13.0% | Accidental orders, return labels, dropoff methods |
| `general_inquiry` | 25 | 12.5% | VAT invoices, locker rules, gift card redemption |
| `subscription_membership_issue` | 22 | 11.0% | Prime renewals, trial cancellation, household sharing |
| `seller_third_party_issue` | 19 | 9.5% | Marketplace merchant disputes, A-to-z claims |
| **Total** | **200** | **100%** | Balanced across all operational buckets |

### B. Difficulty Stratification
- **Easy (114 examples, 57%)**: Clear, single-intent inquiries with standard terminology (e.g., *"How do I return an item? I received it yesterday but it's the wrong size."*).
- **Medium (60 examples, 30%)**: Short/terse messages (≤4 words), non-standard abbreviations, or realistic Twitter typographical errors (e.g., *"wher is my order #394012?? tracking not updateing"*).
- **Hard (26 examples, 13%)**: High-risk sensitive triggers (lawsuit, police report, stolen property, unauthorized credit card charges), account compromise/takeovers, or multi-intent complex inquiries.

### C. Escalation Label Balance
- **Auto-Handle (`gold_should_escalate: false`)**: 144 examples (72.0%)
- **Escalate to Human (`gold_should_escalate: true`)**: 56 examples (28.0%)
  - Critical safety cases (threat of legal action, fraud, hacked accounts)
  - Unresolvable automated requests (driver hit property, stolen packages)
  - Ambiguous or multi-intent scenarios

---

## 2. Schema Specification

Each record in `evaluation/golden_set.jsonl` adheres strictly to the following JSON structure:

```json
{
  "id": "gold_001",
  "conversation_id": "c78df21b-82a",
  "customer_message": "My package says delivered but I never received it. Order #482910.",
  "context": "",
  "gold_intent": "delivery_issue",
  "gold_intent_confidence": 0.95,
  "gold_reply_criteria": "Reply must: (1) clearly address the delivery issue, (2) match historical delivery_issue resolution patterns, (3) maintain professional and concise support tone (2-4 sentences), (4) avoid fabricating unverified timelines, guarantees, or unauthorized refunds.",
  "gold_should_escalate": false,
  "gold_escalation_reason": "Standard single-intent customer service inquiry.",
  "difficulty": "easy",
  "labelling_method": "stratified_expert_reviewed",
  "conversation_length": 2
}
```

---

## 3. Ambiguity & Ground Truth Policies

1. **Multi-Intent Policy**: When a customer mentions multiple issues (e.g., *"package was delivered broken and I want to cancel my Prime subscription"*), the primary operational problem is labeled as the main intent, but the example is flagged as `difficulty: hard` with `gold_should_escalate: true` to test the escalation engine's multi-intent detector.
2. **Context-Independence**: For first-turn evaluation, messages are judged solely on the customer's opening tweet, testing the agent's capability under true cold-start Twitter conditions.
3. **Escalation Priority**: Safety takes absolute precedence over automated handling. Financial disputes, security alerts, and legal threats are strictly marked for human escalation.

---

## 4. Leakage Prevention Safeguards

To prevent contaminated retrieval and overfitted evaluations:
1. **Strict Split Separation**: All 200 golden examples originate strictly from `held_out.jsonl`.
2. **Exclusion from FAISS Index**: `scripts/build_index.py` explicitly loads all 200 golden `conversation_id`s and strips them prior to embedding and indexing. A programmatic assertion verifies zero overlap (`overlap == 0`).
3. **Prompt Insulation**: Few-shot prompts and system instructions do not contain any verbatim examples from `golden_set.jsonl`.
