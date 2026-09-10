"""
scripts/discover_intents.py
────────────────────────────
Phase 4: Data-driven intent discovery using clustering.

Pipeline:
1. Sample customer messages from training conversations.
2. Embed with text-embedding-004.
3. K-means cluster at multiple k values.
4. Use silhouette score to pick optimal k.
5. Generate GPT-summarised cluster descriptions.
6. Print results for manual review.
7. Write draft configs/intents.yaml for human editing.

IMPORTANT: This script produces a DRAFT taxonomy.
You must manually review and edit configs/intents.yaml before proceeding.

Usage:
    python scripts/discover_intents.py
    python scripts/discover_intents.py --n-messages 2000 --k-range 5,12
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import yaml

from src.utils import load_env, load_config, read_jsonl

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
)
logger = logging.getLogger(__name__)


def sample_customer_messages(
    conversations: list[dict],
    n: int,
    seed: int = 42,
) -> list[str]:
    """Extract first customer messages from conversations."""
    messages = []
    for conv in conversations:
        msgs = conv.get("messages", [])
        customer_msgs = [m for m in msgs if m.get("speaker") == "customer"]
        if customer_msgs:
            text = customer_msgs[0].get("clean_text", "").strip()
            if len(text) >= 15:
                messages.append(text)

    rng = random.Random(seed)
    if len(messages) > n:
        messages = rng.sample(messages, n)
    return messages


def embed_messages(
    messages: list[str],
    embedding_model: str,
) -> np.ndarray:
    """Embed messages using Google embedding API."""
    from src.retrieval.index import EmbeddingModel
    model = EmbeddingModel(model=embedding_model)
    logger.info("Embedding %d messages...", len(messages))
    embeddings = model.embed_texts(messages, task_type="RETRIEVAL_DOCUMENT")
    return embeddings


def find_optimal_k(
    embeddings: np.ndarray,
    k_range: range,
    seed: int = 42,
) -> tuple[int, dict]:
    """Find optimal k using silhouette score."""
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score

    scores = {}
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=seed, n_init=10)
        labels = km.fit_predict(embeddings)
        score = silhouette_score(embeddings, labels, sample_size=min(1000, len(embeddings)))
        scores[k] = round(float(score), 4)
        logger.info("  k=%d: silhouette=%.4f", k, score)

    best_k = max(scores, key=lambda k: scores[k])
    logger.info("Optimal k=%d (silhouette=%.4f)", best_k, scores[best_k])
    return best_k, scores


def cluster_messages(
    messages: list[str],
    embeddings: np.ndarray,
    k: int,
    seed: int = 42,
) -> tuple[np.ndarray, dict]:
    """Run K-means and return labels and cluster message samples."""
    from sklearn.cluster import KMeans

    km = KMeans(n_clusters=k, random_state=seed, n_init=10)
    labels = km.fit_predict(embeddings)

    cluster_messages = {}
    for i in range(k):
        idx = np.where(labels == i)[0]
        cluster_messages[i] = [messages[j] for j in idx]

    return labels, cluster_messages


def describe_cluster_with_llm(
    cluster_samples: list[str],
    cluster_id: int,
    model: str = "gemini-1.5-flash-latest",
) -> dict:
    """Ask LLM to name and describe a cluster."""
    import google.generativeai as genai
    from src.utils import get_gemini_api_key

    load_env()
    genai.configure(api_key=get_gemini_api_key())

    samples_text = "\n".join(f"  - {s[:150]}" for s in cluster_samples[:15])
    prompt = f"""You are analysing customer support messages from a technology/e-commerce company.

Below are {len(cluster_samples)} customer messages that belong to the same group.

MESSAGES:
{samples_text}

Based on these messages, identify the common CUSTOMER INTENT (what they are asking for/about).

Respond with exactly this JSON:
{{
  "intent_name": "<snake_case_name, 2-4 words>",
  "definition": "<one sentence: what this intent covers>",
  "inclusion_criteria": "<what messages should be classified here>",
  "exclusion_criteria": "<what messages should NOT be classified here>",
  "example_phrases": ["<phrase 1>", "<phrase 2>", "<phrase 3>"]
}}"""

    genai_model = genai.GenerativeModel(
        model_name=model,
        generation_config={"temperature": 0.2, "response_mime_type": "application/json"},
    )
    try:
        response = genai_model.generate_content(prompt)
        data = json.loads(response.text)
        data["cluster_id"] = cluster_id
        data["sample_size"] = len(cluster_samples)
        return data
    except Exception as e:
        logger.warning("LLM description failed for cluster %d: %s", cluster_id, e)
        return {
            "cluster_id": cluster_id,
            "intent_name": f"cluster_{cluster_id}",
            "definition": "NEEDS MANUAL REVIEW",
            "inclusion_criteria": "",
            "exclusion_criteria": "",
            "example_phrases": cluster_samples[:3],
            "sample_size": len(cluster_samples),
        }


def main(args: argparse.Namespace) -> None:
    load_env()
    cfg = load_config()
    seed = cfg["project"]["random_seed"]

    # ── Load training conversations ────────────────────────────────────────────
    train_path = Path("data/processed/train.jsonl")
    if not train_path.exists():
        logger.error("Training data not found. Run prepare_data.py first.")
        sys.exit(1)

    conversations = read_jsonl(train_path)
    logger.info("Loaded %d training conversations", len(conversations))

    # ── Sample customer messages ───────────────────────────────────────────────
    messages = sample_customer_messages(conversations, n=args.n_messages, seed=seed)
    logger.info("Sampled %d customer messages for clustering", len(messages))

    # ── Embed ──────────────────────────────────────────────────────────────────
    embeddings = embed_messages(messages, cfg["retrieval"]["embedding_model"])
    np.save("data/processed/cluster_embeddings.npy", embeddings)

    # ── Find optimal k ─────────────────────────────────────────────────────────
    k_min, k_max = [int(x) for x in args.k_range.split(",")]
    k_range = range(k_min, k_max + 1)
    logger.info("Testing k in range %d–%d...", k_min, k_max)
    best_k, silhouette_scores = find_optimal_k(embeddings, k_range, seed=seed)

    print(f"\nSilhouette scores: {silhouette_scores}")
    print(f"Recommended k: {best_k}")

    # ── Override if specified ──────────────────────────────────────────────────
    k = args.k or best_k
    logger.info("Using k=%d for clustering", k)

    # ── Cluster ────────────────────────────────────────────────────────────────
    labels, cluster_msgs = cluster_messages(messages, embeddings, k, seed=seed)

    # ── Describe clusters ──────────────────────────────────────────────────────
    print("\nDescribing clusters with LLM...")
    cluster_descriptions = []
    for cluster_id in sorted(cluster_msgs.keys()):
        print(f"  Cluster {cluster_id} ({len(cluster_msgs[cluster_id])} messages)...")
        desc = describe_cluster_with_llm(
            cluster_msgs[cluster_id],
            cluster_id=cluster_id,
            model=cfg["generation"]["model"],
        )
        cluster_descriptions.append(desc)
        print(f"    -> {desc['intent_name']}: {desc['definition']}")

    # ── Save raw cluster data ──────────────────────────────────────────────────
    raw_output = {
        "k": k,
        "silhouette_scores": silhouette_scores,
        "cluster_descriptions": cluster_descriptions,
        "cluster_sizes": {str(k): len(v) for k, v in cluster_msgs.items()},
    }
    Path("data/processed/cluster_analysis.json").write_text(
        json.dumps(raw_output, indent=2, default=str)
    )

    # ── Write draft intents.yaml ───────────────────────────────────────────────
    draft_intents = {
        "version": "draft-v1",
        "note": (
            "AUTO-GENERATED from clustering. MUST BE REVIEWED AND EDITED manually "
            "before use. Check: mutual exclusivity, adequate examples, clear boundaries."
        ),
        "intents": [
            {
                "name": d["intent_name"],
                "definition": d["definition"],
                "inclusion_criteria": d.get("inclusion_criteria", ""),
                "exclusion_criteria": d.get("exclusion_criteria", ""),
                "examples": d.get("example_phrases", [])[:5],
                "approximate_prevalence": round(
                    d["sample_size"] / len(messages), 3
                ),
                "cluster_id": d["cluster_id"],
            }
            for d in cluster_descriptions
        ],
    }

    draft_path = Path("configs/intents_draft.yaml")
    with open(draft_path, "w") as f:
        yaml.dump(draft_intents, f, allow_unicode=True, sort_keys=False, default_flow_style=False)

    print(f"\n[OK] Draft taxonomy written -> {draft_path}")
    print("\n" + "=" * 60)
    print("NEXT STEPS (MANUAL):")
    print("  1. Review intents_draft.yaml")
    print("  2. Merge similar clusters")
    print("  3. Split overly broad clusters")
    print("  4. Write clear inclusion/exclusion criteria")
    print("  5. Add real representative examples")
    print("  6. Copy to configs/intents.yaml")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Discover intents via clustering.")
    parser.add_argument("--n-messages", type=int, default=2000,
                        help="Number of messages to cluster (default: 2000)")
    parser.add_argument("--k-range", default="5,12",
                        help="Min,max k to test (default: 5,12)")
    parser.add_argument("--k", type=int, default=None,
                        help="Override k (skip silhouette search)")
    main(parser.parse_args())
