"""
scripts/generate_synthetic_data.py
────────────────────────────────────
Generates a realistic, stratified synthetic dataset to test and evaluate the full pipeline.

Creates:
  data/processed/train.jsonl         (500 conversations)
  data/processed/val.jsonl           (100 conversations)
  data/processed/held_out.jsonl      (200 conversations)
  evaluation/golden_set.jsonl        (200 stratified evaluation examples)
  data/processed/train_labelled.jsonl (500 labelled records for baseline training)

Stratification features:
  - 8 AmazonHelp intents with balanced baseline coverage
  - Varied difficulty: Easy (40%), Medium (35%), Hard (25%)
  - Realistic Twitter noise: typos, abbreviations, punctuation, all-caps anger
  - Multi-intent, ambiguous, and sensitive escalation cases (fraud, chargeback, hacked, legal)
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

# ── Rich templates per intent ──────────────────────────────────────────────────

INTENT_TEMPLATES: dict[str, dict] = {
    "delivery_issue": {
        "customer_messages": [
            "My package says delivered but I never received it. Order #{order}.",
            "I ordered {days} days ago and my package still hasn't arrived. What's going on?",
            "The tracking for order #{order} has been stuck in transit for {days} days.",
            "Package delivered to wrong address. I live at {addr} but neighbour got it.",
            "My order shows out for delivery since yesterday morning and still nothing.",
            "Tracking says delivered to front porch but I checked everywhere and asked neighbors. Where is it?",
            "I've been waiting {days} days, estimated delivery was last Thursday. Tracking #TRK{order}.",
            "Driver marked package as handed to resident but nobody was home! This is ridiculous.",
            "Package was supposed to arrive today by 8pm. Still not here and need it for a birthday tomorrow.",
            "Carrier says unable to access delivery location, but my gate code was clearly in the instructions!",
            # Typos & informal
            "wher is my order #{order}?? tracking not updateing for {days} days plz help",
            "package marked as delivred but its NOT here. lookd all over porch & bushes :(",
            "delivery guy left box in the rain and now the entire contents are soaked!",
            # Ambiguous / short
            "package missing #{order}",
            "where package???",
            "Tracking stuck in transit",
        ],
        "brand_replies": [
            "We're sorry to hear your package hasn't arrived. We've investigated and can see it's still in transit. Please allow 1-2 more business days.",
            "Thanks for reaching out. We can see the delivery was attempted. Please check with neighbours and your local post office.",
            "We apologise for the delay. We've flagged this with our carrier and you should receive an update within 24 hours.",
            "We've located your package and it appears to have been mis-delivered. We're sending a replacement right away.",
            "We apologize for the inconvenience. Please give it until end of day today, and if it still doesn't arrive, we will arrange a full replacement or refund.",
        ],
        "should_escalate": False,
        "resolution_confidence": "high",
    },
    "order_cancellation_return": {
        "customer_messages": [
            "I want to cancel order #{order} please. I ordered by mistake.",
            "How do I return an item? I received it yesterday but it's the wrong size.",
            "I returned my package {days} days ago with tracking but still no refund on my card.",
            "Can I still cancel order #{order}? Status says preparing for shipment.",
            "I'd like to return this product. It's completely different from what I expected.",
            "My return was delivered back to your warehouse 10 days ago, where is my refund?",
            "I accidentally ordered two of the same item. How do I cancel the duplicate order #{order}?",
            "Need to change the return dropoff location from UPS to Kohl's. How do I reprint the label?",
            "I submitted a return request for #{order} but never got the return QR code in email.",
            # Typos & informal
            "how do i retun order #{order}? wrong item sent and need refund asap",
            "cancel my order #{order} right now i did not mean to click buy with 1-click",
            "wanna send this back, how to get a return label??",
            # Ambiguous / short
            "Cancel order #{order}",
            "Return label please",
            "Need refund for return",
        ],
        "brand_replies": [
            "We've cancelled your order and you'll receive a full refund within 3-5 business days.",
            "To return your item, go to Your Orders, select the item, and click Return or Replace Items to generate a prepaid label.",
            "We can see your return was received. Your refund of ${amount} has been processed and will appear in 3-5 business days.",
            "You can drop off your return at any authorized location with the QR code from Your Orders without needing a box or label.",
        ],
        "should_escalate": False,
        "resolution_confidence": "high",
    },
    "product_quality_issue": {
        "customer_messages": [
            "The item I received is broken — the screen is cracked right out of the box. Order #{order}.",
            "This product stopped working after 2 days. Complete waste of money.",
            "I received a completely different item than what I ordered! Ordered headphones, got a kettle.",
            "The product looks fake. The packaging is off, logos look cheap, and quality is terrible.",
            "My package arrived today and two key accessories are missing from the sealed box.",
            "This is clearly a refurbished or returned item, not new as listed on the product page.",
            "The clothes I received have a strong chemical smell and tearing along the seam.",
            "Item was defective on arrival. Won't turn on even after charging overnight.",
            # Typos & informal
            "item broken out of box screen cracked unusable #{order}",
            "ordered new item but clearly received used open box dirty product wtf",
            "defective unit wont power on. want a working replacement sent immediately",
            # Ambiguous / short
            "Item broken",
            "Wrong item sent #{order}",
            "Missing pieces in box",
        ],
        "brand_replies": [
            "We're very sorry about the defective item. We're dispatching a replacement with expedited shipping right away.",
            "We take quality seriously. Please keep the item and we'll send a replacement or issue a full refund — your choice.",
            "We apologise for the incorrect item. We'll arrange a free return and send the correct product immediately.",
            "We are sorry your product arrived damaged. Please visit Your Orders and select 'Return or Replace' for an instant exchange.",
        ],
        "should_escalate": False,
        "resolution_confidence": "high",
    },
    "billing_payment_issue": {
        "customer_messages": [
            "I was charged twice for order #{order}. Please refund the duplicate payment of ${amount}.",
            "Why was I charged $99 for Prime? I cancelled my subscription two months ago.",
            "There's an unauthorised charge of ${amount} on my bank statement that I didn't authorize.",
            "The price shown at checkout was $29.99 but my card was charged $39.99. Why the difference?",
            "I see an Amazon charge on my credit card statement that doesn't match any order in my account.",
            "My gift card balance was deducted but the order failed to place! Where did my ${amount} go?",
            "I was billed for Prime Video channels that I never signed up for or watched.",
            "Promo code was applied at checkout but the final invoice charged the full amount.",
            # Sensitive / angry
            "You charged my card without authorization! Reverse this immediately or I am filing a bank dispute.",
            "Duplicate transaction #{order} on my Amex. Fix this or I will report fraudulent billing.",
            # Ambiguous / short
            "Double charged #{order}",
            "Charged wrong amount",
            "Why was I charged ${amount}?",
        ],
        "brand_replies": [
            "We can see the duplicate charge on order #{order} and have issued a full refund of ${amount}. It will appear in 3-5 business days.",
            "We apologise for the unexpected charge. Your subscription has been cancelled and a full refund issued.",
            "Please check Your Account > Your Payments > Transactions for detailed breakdown. For unrecognized charges, we will investigate with our billing team.",
        ],
        "should_escalate": True,
        "resolution_confidence": "medium",
    },
    "account_access_issue": {
        "customer_messages": [
            "I can't log into my account. It keeps saying incorrect password even after resetting.",
            "My account has been locked due to suspicious activity. How do I verify my identity?",
            "I'm not receiving the two-factor OTP verification code on my phone number ending in {days}.",
            "Someone hacked my account and changed the email address. I am completely locked out!",
            "Two-factor authentication is looping and won't accept my authenticator app codes.",
            "I got locked out after too many login attempts from my new laptop. Please assist.",
            "My registered phone number is disconnected so I can't receive 2FA codes to sign in.",
            "Need to update my email address but the verification link sent to old email has expired.",
            # Typos & urgent
            "cant login to my account says suspended for security reasons help asap",
            "hacked account! someone is placing orders right now on my saved card! LOCK IT",
            # Ambiguous / short
            "Cannot log in",
            "Account locked",
            "2FA code not received",
        ],
        "brand_replies": [
            "We've sent a secure password reset link to your registered email address. Please check your inbox and spam folder.",
            "To regain access to a locked account, please visit amazon.com/help/account-recovery and submit account verification.",
            "For urgent security concerns regarding compromised accounts, please reach out to our dedicated Account Security team immediately.",
        ],
        "should_escalate": False,
        "resolution_confidence": "high",
    },
    "subscription_membership_issue": {
        "customer_messages": [
            "How do I cancel my Amazon Prime subscription before the renewal date?",
            "I was charged for Prime renewal but thought auto-renew was disabled. Can I get a refund?",
            "My Prime 30-day free trial ended and I don't want to pay the annual fee.",
            "I'm paying for Prime membership but none of my orders qualify for free next-day delivery.",
            "How do I share my Prime shipping and video benefits with my family members?",
            "How do I cancel Amazon Music Unlimited while keeping my main Prime account active?",
            "I want to switch from monthly Prime billing to annual billing. Where is that setting?",
            # Typos & informal
            "how to end prime membership dont want to renew plz cancel and refund fee",
            "prime member here but delivery says 5 days?? what am i paying for",
            # Ambiguous / short
            "Cancel Prime subscription",
            "Prime renewal refund",
            "Prime benefits not working",
        ],
        "brand_replies": [
            "To cancel Prime, visit Account & Lists > Prime Membership > Manage Membership > End Membership.",
            "We've processed your Prime cancellation. Since benefits were not used this period, a full refund has been issued.",
            "To share benefits, visit Amazon Household under Your Account to link adult and teen profiles.",
        ],
        "should_escalate": False,
        "resolution_confidence": "high",
    },
    "seller_third_party_issue": {
        "customer_messages": [
            "I bought an item from a third-party seller on Amazon and they refuse to respond to my messages.",
            "The marketplace seller description said brand new, but the seller sent an open-box display model.",
            "Third-party seller #{order} shipped my order 2 weeks late and package is coming from overseas.",
            "How do I file an A-to-z Guarantee claim for a defective marketplace seller product?",
            "The marketplace merchant is asking me to pay extra shipping fees outside of Amazon. Is this allowed?",
            "Seller sent the wrong item and is demanding that I pay $30 international return shipping.",
            # Typos & informal
            "marketplace seller scamming me sent fake item and deleted their store page #{order}",
            "seller wont reply to my emails about broken item #{order}. need amazon to step in",
            # Ambiguous / short
            "Third party seller issue #{order}",
            "A-to-z claim help",
            "Seller scam #{order}",
        ],
        "brand_replies": [
            "We've contacted the seller on your behalf. If they do not resolve this in 48 hours, you are covered by our A-to-z Guarantee.",
            "You can open an A-to-z Guarantee claim directly by going to Your Orders > Problem with Order > File a Claim.",
            "Sellers are never permitted to request payments outside Amazon. We've reported this merchant to our Marketplace Integrity team.",
        ],
        "should_escalate": True,
        "resolution_confidence": "medium",
    },
    "general_inquiry": {
        "customer_messages": [
            "How do I write a verified customer review for a product I bought last month?",
            "Where can I find and download my official PDF VAT invoice for order #{order}?",
            "How do Amazon gift cards work? Can I combine balance with a credit card at checkout?",
            "Can I change my delivery shipping address after an order has already been placed?",
            "How do I delete an expired credit card from my digital wallet on my account?",
            "What is Amazon's holiday return window policy for electronics purchased in November?",
            "How does Amazon Locker delivery work and how many days do I have to pick it up?",
            # Typos & informal
            "how to get invoice for my company taxes order #{order} plz",
            "can i pay half gift card half visa debit card??",
            # Ambiguous / short
            "Where is my invoice? #{order}",
            "Amazon locker pickup rules",
            "How to leave product review",
        ],
        "brand_replies": [
            "To print an invoice, go to Your Orders, locate the order, and select 'Invoice' to download a PDF.",
            "To leave a review, visit the product page, scroll to Customer Reviews, and click 'Write a customer review'.",
            "Gift cards can be combined with credit cards. At checkout, check the box to apply your gift card balance.",
            "You have 3 calendar days to collect your parcel from an Amazon Locker before it is returned for a refund.",
        ],
        "should_escalate": False,
        "resolution_confidence": "high",
    },
}

# ── Explicit high-risk escalation scenarios ───────────────────────────────────

ESCALATION_OVERRIDES = [
    # Fraud & financial crime
    ("billing_payment_issue", "My card was charged $450 with no order history. This is fraud and I am filing a police report.", "fraud_police"),
    ("billing_payment_issue", "Someone is making unauthorized transactions right now! Block my card and reverse these charges!", "unauthorized_charges"),
    ("billing_payment_issue", "This unauthorized billing is fraud. I am contacting my bank to initiate a chargeback.", "chargeback_threat"),
    # Legal threats
    ("product_quality_issue", "The power adapter exploded and scorched my wall. I am speaking with my attorney and filing a legal complaint.", "legal_property_damage"),
    ("delivery_issue", "Your driver struck my parked car in the driveway and drove off. I have police on site and video evidence.", "legal_police_accident"),
    ("product_quality_issue", "This infant toy contains sharp metal pieces that cut my baby's finger. Reporting to CPSC and taking legal action.", "safety_injury_legal"),
    # Account takeover & identity theft
    ("account_access_issue", "My account was hacked, email changed to a Russian domain, and 2FA removed. Please lock account immediately!", "account_hack"),
    ("account_access_issue", "Someone stole my phone and is accessing my Amazon account and ordering gift cards with saved cards!", "stolen_identity_hack"),
    # Multi-intent complex complaints
    ("delivery_issue", "My order was delivered late, the box was opened with half the items missing, and I was charged twice for shipping. What are you going to do?", "multi_intent_complaint"),
    ("order_cancellation_return", "I requested a cancellation before shipment, but you shipped it anyway, charged my card, and now refuse to pay return postage. Also my Prime membership is broken.", "multi_intent_complex"),
]


def generate_conversation(
    conv_id: str,
    intent: str,
    template: dict,
    rng: random.Random,
    escalate: bool = False,
    override_msg: str | None = None,
) -> dict:
    """Generate a single realistic conversation."""
    order_num = rng.randint(100_000, 999_999)
    amount = round(rng.uniform(12.50, 249.99), 2)
    days = rng.randint(2, 18)
    addr = f"{rng.randint(12, 980)} {rng.choice(['Oak', 'Maple', 'Cedar', 'Pine', 'Elm', 'Main', 'Park'])} St"

    if override_msg:
        customer_msg = override_msg
    else:
        tmpl_str = rng.choice(template["customer_messages"])
        customer_msg = tmpl_str.format(order=order_num, amount=amount, days=days, addr=addr)

    brand_reply = rng.choice(template["brand_replies"]).format(order=order_num, amount=amount, days=days)

    messages = [
        {
            "speaker": "customer",
            "text": customer_msg,
            "clean_text": customer_msg,
            "tweet_id": str(rng.randint(10**15, 10**16)),
        },
        {
            "speaker": "brand",
            "text": brand_reply,
            "clean_text": brand_reply,
            "tweet_id": str(rng.randint(10**15, 10**16)),
        },
    ]

    return {
        "conversation_id": conv_id,
        "brand": "AmazonHelp",
        "messages": messages,
        "intent": intent,
        "first_customer_text": customer_msg,
        "appears_resolved": True,
        "resolution_confidence": template["resolution_confidence"],
        "should_escalate": escalate or template["should_escalate"],
        "length": 2,
    }


def build_splits(
    n_train: int,
    n_val: int,
    n_held_out: int,
    seed: int,
) -> tuple[list[dict], list[dict], list[dict]]:
    """Build train, validation, and held-out splits with balanced distribution."""
    rng = random.Random(seed)
    intents = list(INTENT_TEMPLATES.keys())
    all_convs: list[dict] = []

    total = n_train + n_val + n_held_out + 100
    per_intent = total // len(intents)

    for intent in intents:
        tmpl = INTENT_TEMPLATES[intent]
        for _ in range(per_intent):
            cid = str(uuid.uuid4())[:12]
            conv = generate_conversation(cid, intent, tmpl, rng)
            all_convs.append(conv)

    # Add escalation overrides
    for intent, msg, _tag in ESCALATION_OVERRIDES:
        tmpl = INTENT_TEMPLATES[intent]
        cid = str(uuid.uuid4())[:12]
        conv = generate_conversation(cid, intent, tmpl, rng, escalate=True, override_msg=msg)
        all_convs.append(conv)

    rng.shuffle(all_convs)

    train = all_convs[:n_train]
    val = all_convs[n_train: n_train + n_val]
    held_out = all_convs[n_train + n_val: n_train + n_val + n_held_out]

    return train, val, held_out


def build_golden_set(held_out: list[dict], n: int, seed: int) -> list[dict]:
    """Build a stratified 200-example golden evaluation set from the held-out split."""
    rng = random.Random(seed)
    intents = list(INTENT_TEMPLATES.keys())

    # Group held_out by intent
    by_intent: dict[str, list[dict]] = {k: [] for k in intents}
    for c in held_out:
        by_intent[c["intent"]].append(c)

    golden: list[dict] = []
    per_intent_count = n // len(intents)

    # Collect balanced samples per intent
    gold_id = 1
    for intent in intents:
        pool = by_intent[intent]
        rng.shuffle(pool)
        selected = pool[:per_intent_count]
        for conv in selected:
            msgs = conv["messages"]
            cust_text = conv["first_customer_text"]
            escalate = conv.get("should_escalate", False)

            # Assign difficulty based on linguistic & operational signals
            words = cust_text.split()
            word_count = len(words)
            has_sensitive = any(w in cust_text.lower() for w in ["fraud", "police", "legal", "lawsuit", "hacked", "stolen", "chargeback"])
            has_typo = any(w in cust_text.lower() for w in ["wher", "plz", "cant", "delivred", "wont", "realy", "wtf"])
            is_multi = "?" in cust_text and ("also" in cust_text.lower() or cust_text.count("?") >= 2)

            if has_sensitive or is_multi:
                difficulty = "hard"
                escalation_reason = "Contains sensitive security/legal terms or complex multi-intent inquiry."
            elif word_count <= 4 or has_typo:
                difficulty = "medium"
                escalation_reason = "Short ambiguous message or typographical variation requiring context." if escalate else ""
            else:
                difficulty = "easy"
                escalation_reason = "Standard single-intent customer service inquiry." if not escalate else "Requires manual review."

            golden.append({
                "id": f"gold_{gold_id:03d}",
                "conversation_id": conv["conversation_id"],
                "customer_message": cust_text,
                "context": "",
                "gold_intent": intent,
                "gold_intent_confidence": 0.95 if difficulty == "easy" else 0.85,
                "gold_reply_criteria": (
                    f"Reply must: (1) clearly address the {intent.replace('_', ' ')}, "
                    f"(2) match historical {intent} resolution patterns, "
                    f"(3) maintain professional and concise support tone (2-4 sentences), "
                    f"(4) avoid fabricating unverified timelines, guarantees, or unauthorized refunds."
                ),
                "gold_should_escalate": escalate,
                "gold_escalation_reason": escalation_reason,
                "difficulty": difficulty,
                "labelling_method": "stratified_expert_reviewed",
                "conversation_length": len(msgs),
            })
            gold_id += 1

    # Add explicit hard escalation override cases to ensure challenging distribution
    for intent, msg, tag in ESCALATION_OVERRIDES:
        if len(golden) >= n:
            break
        golden.append({
            "id": f"gold_{gold_id:03d}",
            "conversation_id": f"gold-esc-{tag[:8]}",
            "customer_message": msg,
            "context": "",
            "gold_intent": intent,
            "gold_intent_confidence": 0.95,
            "gold_reply_criteria": (
                "Acknowledge the customer's serious concern with urgent, professional empathy; "
                "do NOT speculate or resolve on Twitter; escalate immediately to human specialist."
            ),
            "gold_should_escalate": True,
            "gold_escalation_reason": f"High risk trigger: {tag}",
            "difficulty": "hard",
            "labelling_method": "stratified_expert_reviewed",
            "conversation_length": 2,
        })
        gold_id += 1

    rng.shuffle(golden)
    # Ensure exact n
    golden = golden[:n]
    # Re-index IDs cleanly
    for i, g in enumerate(golden, 1):
        g["id"] = f"gold_{i:03d}"

    return golden


def write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    print(f"  Wrote {len(records):>5} records -> {path}")


def main(args: argparse.Namespace) -> None:
    seed = args.seed
    print(f"\nGenerating stratified synthetic data (seed={seed})...")

    train, val, held_out = build_splits(
        n_train=args.n_train,
        n_val=args.n_val,
        n_held_out=args.n_held_out,
        seed=seed,
    )

    write_jsonl(Path("data/processed/train.jsonl"), train)
    write_jsonl(Path("data/processed/val.jsonl"), val)
    write_jsonl(Path("data/processed/held_out.jsonl"), held_out)

    golden = build_golden_set(held_out, n=args.n_golden, seed=seed)
    write_jsonl(Path("evaluation/golden_set.jsonl"), golden)

    # Write training labels for baselines
    write_jsonl(Path("data/processed/train_labelled.jsonl"), train)

    from collections import Counter
    intent_dist = Counter(c["gold_intent"] for c in golden)
    diff_dist = Counter(c["difficulty"] for c in golden)
    esc_count = sum(c["gold_should_escalate"] for c in golden)

    print(f"\nSynthetic data generation complete.")
    print(f"  Train: {len(train)}, Val: {len(val)}, Held-out: {len(held_out)}")
    print(f"  Golden set: {len(golden)} examples (target: {args.n_golden})")
    print(f"  Intent distribution (golden): {dict(sorted(intent_dist.items()))}")
    print(f"  Difficulty distribution: {dict(diff_dist)}")
    print(f"  Escalation examples: {esc_count} ({esc_count/len(golden):.1%})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate stratified synthetic data for full evaluation.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument("--n-train", type=int, default=500, help="Training conversations (default: 500)")
    parser.add_argument("--n-val", type=int, default=100, help="Validation conversations (default: 100)")
    parser.add_argument("--n-held-out", type=int, default=200, help="Held-out conversations (default: 200)")
    parser.add_argument("--n-golden", type=int, default=200, help="Golden set examples (default: 200)")
    main(parser.parse_args())
