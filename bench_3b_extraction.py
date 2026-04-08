#!/usr/bin/env python3
"""
Test 3B extraction quality against Phase 0 gold sets.

Loads Llama-3.2-3B-Instruct, runs extraction prompt on gold-set chunks,
grades recall/precision/poison against human-annotated gold sets.

This measures QUALITY, not throughput. ANE throughput is already proven
at 24.6 tok/s. The question is: can 3B extract conceptual memories
that 1B and CPU FactExtractor miss?

Copyright 2026 Nick Lo. MIT License.
"""

import json
import os
import sys
import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "unsloth/Llama-3.2-3B-Instruct"
GOLD_DIR = "/Users/midas/Desktop/cowork/vault/subconscious/gold_sets"
CHUNK_DIR = "/Users/midas/Desktop/cowork/vault/subconscious/chunks"

EXTRACTION_PROMPT = """You are a memory extraction system. Read the conversation and extract discrete facts.

RULES:
- One memory per fact. Never combine multiple facts.
- Use the speakers words as ground truth. Store numbers exactly.
- Only extract what is stated or directly demonstrated.
- Extract decisions, preferences, relationships, and state changes.
- Ignore greetings, filler, speculation reaching no conclusion.

Output a bullet list of facts. Each fact is one complete sentence.

CONVERSATION:
---
{chunk}
---

Facts:
-"""


def load_model():
    print(f"Loading {MODEL}...")
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, torch_dtype=torch.float16, device_map="mps")
    model.eval()
    print(f"  Loaded in {time.time()-t0:.1f}s")
    return model, tokenizer


def extract(model, tokenizer, chunk_text, max_new_tokens=500):
    prompt = EXTRACTION_PROMPT.format(chunk=chunk_text[:2000])

    messages = [{"role": "user", "content": prompt}]
    input_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(input_text, return_tensors="pt").to(model.device)

    t0 = time.perf_counter()
    with torch.no_grad():
        output = model.generate(
            **inputs, max_new_tokens=max_new_tokens,
            do_sample=False, temperature=1.0)
    elapsed = time.perf_counter() - t0
    n_gen = output.shape[1] - inputs["input_ids"].shape[1]
    tps = n_gen / elapsed if elapsed > 0 else 0

    response = tokenizer.decode(output[0][inputs["input_ids"].shape[1]:],
                                skip_special_tokens=True)

    # Parse bullet points (handles -, *, •, numbered lists)
    facts = []
    for line in response.split("\n"):
        line = line.strip()
        # Strip common bullet prefixes
        for prefix in ["- ", "* ", "• ", "· "]:
            if line.startswith(prefix):
                line = line[len(prefix):]
                break
        else:
            # Try numbered: "1. ", "1) ", etc.
            import re
            m = re.match(r'^\d+[\.\)]\s+', line)
            if m:
                line = line[m.end():]
            elif line.startswith("-") or line.startswith("*") or line.startswith("•"):
                line = line[1:].strip()
            else:
                # Skip non-bullet lines (preamble, etc.)
                if not line or line.startswith("Here") or line.startswith("Facts"):
                    continue
                # Include bare lines if they look like facts
                if len(line) < 20:
                    continue

        content = line.strip().rstrip(".")
        if len(content) >= 15:
            facts.append(content)

    return facts, elapsed, n_gen, tps


def grade(extracted, gold):
    """Grade extraction against gold set.
    Returns (recall, precision, matched_gold, missed_gold, poison_count)
    """
    gold_found = 0
    gold_missed = []

    for g in gold:
        g_content = g["content"].lower()
        g_words = set(w for w in g_content.split() if len(w) > 3)

        found = False
        for f in extracted:
            f_lower = f.lower()
            f_words = set(w for w in f_lower.split() if len(w) > 3)
            overlap = len(g_words & f_words)
            if overlap >= 3:
                found = True
                break

        if found:
            gold_found += 1
        else:
            gold_missed.append(g["content"][:90])

    recall = gold_found / len(gold) * 100 if gold else 0
    precision = 100  # no automated poison detection for now

    return recall, precision, gold_found, gold_missed, 0


def main():
    model, tokenizer = load_model()

    print("\n" + "=" * 70)
    print("3B EXTRACTION QUALITY TEST — GOLD SET GRADING")
    print("=" * 70)

    total_recall = 0
    total_chunks = 0

    for chunk_id in ["A1", "B1", "C1"]:
        gold_path = os.path.join(GOLD_DIR, f"gold_{chunk_id}.json")
        chunk_path = os.path.join(CHUNK_DIR, f"chunk_{chunk_id}.txt")

        if not os.path.exists(gold_path):
            print(f"\n  {chunk_id}: gold set not found, skipping")
            continue

        with open(gold_path) as f:
            gold = json.load(f)
        with open(chunk_path) as f:
            chunk_text = f.read()

        print(f"\n--- Chunk {chunk_id} ({len(gold)} gold items) ---")
        print(f"  Extracting...")

        facts, elapsed, n_gen, tps = extract(model, tokenizer, chunk_text)
        print(f"  Generated {n_gen} tokens in {elapsed:.1f}s ({tps:.1f} tok/s)")
        print(f"  Extracted {len(facts)} facts")

        # Grade
        recall, precision, matched, missed, poison = grade(facts, gold)
        total_recall += recall
        total_chunks += 1

        print(f"  RECALL: {recall:.0f}% ({matched}/{len(gold)})")
        if missed:
            print(f"  Missed ({len(missed)}):")
            for m in missed[:5]:
                print(f"    - {m}")
            if len(missed) > 5:
                print(f"    ... +{len(missed)-5} more")

        print(f"  Extracted facts:")
        for i, f in enumerate(facts[:8]):
            print(f"    [{i}] {f[:100]}")
        if len(facts) > 8:
            print(f"    ... +{len(facts)-8} more")

    if total_chunks > 0:
        avg_recall = total_recall / total_chunks
        print(f"\n{'=' * 70}")
        print(f"RESULT: Average recall = {avg_recall:.0f}%")
        print(f"KILL TEST: {'PASS' if avg_recall >= 75 else 'FAIL'} (need >=75%)")
        print(f"{'=' * 70}")

        if avg_recall >= 75:
            print("\n3B CLEARS THE BAR. Use as Subconscious brain.")
        elif avg_recall >= 65:
            print("\n3B marginal — same tier as 1B + FactExtractor.")
            print("Prompt iteration may push it over. Or go to 8B.")
        else:
            print("\n3B below 1B baseline. Something wrong with the prompt.")


if __name__ == "__main__":
    main()
