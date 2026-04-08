#!/usr/bin/env python3
"""
Grade Phase 0 extractions. Handles the 1B's actual output format:
concatenated JSON objects (not a JSON array).
"""

import json
import re
import os
import sys
import numpy as np

VAULT_DIR = os.path.expanduser("~/Desktop/cowork/vault/subconscious")
GOLD_DIR = os.path.join(VAULT_DIR, "gold_sets")
EXTRACT_DIR = os.path.join(VAULT_DIR, "extractions")


def parse_concatenated_json(raw_text):
    """Parse concatenated JSON objects from 1B output.

    Handles:
    - Single object: {...}
    - Concatenated objects: {...}{...}{...}
    - Objects with newlines between: {...}\n{...}
    - Proper JSON array: [{...}, {...}]
    - Markdown-fenced output
    """
    text = raw_text.strip()

    # Strip markdown fences
    if "```" in text:
        lines = text.split("\n")
        lines = [l for l in lines if not l.strip().startswith("```")]
        text = "\n".join(lines).strip()

    # If it's already a proper array, parse directly
    if text.startswith("["):
        try:
            data = json.loads(text)
            if isinstance(data, list):
                return [d for d in data if isinstance(d, dict)]
        except json.JSONDecodeError:
            pass

    # Extract individual JSON objects using brace matching
    objects = []
    depth = 0
    start = None

    for i, ch in enumerate(text):
        if ch == '{':
            if depth == 0:
                start = i
            depth += 1
        elif ch == '}':
            depth -= 1
            if depth == 0 and start is not None:
                try:
                    obj = json.loads(text[start:i+1])
                    if isinstance(obj, dict):
                        objects.append(obj)
                except json.JSONDecodeError:
                    pass
                start = None

    return objects


def split_monolith(obj):
    """If a single object contains the entire conversation as 'content',
    it's a monolithic dump, not a proper extraction. Return as-is but flag it."""
    content = obj.get('content', '')
    if len(content) > 200:  # Suspiciously long for a single atomic memory
        return [obj], True  # monolith
    return [obj], False


def keywords(text):
    """Extract significant keywords from text."""
    stop = {'the', 'that', 'this', 'with', 'from', 'into', 'than',
            'which', 'where', 'when', 'what', 'they', 'them', 'their',
            'have', 'been', 'were', 'being', 'does', 'done', 'each',
            'only', 'also', 'more', 'most', 'very', 'about', 'after',
            'before', 'between', 'through', 'during', 'because', 'while',
            'using', 'used', 'uses', 'would', 'could', 'should', 'will',
            'just', 'like', 'some', 'other', 'than', 'then', 'there'}
    words = set()
    for w in text.lower().split():
        w = w.strip('.,;:()[]{}"\'-/')
        if len(w) > 3 and w not in stop:
            words.add(w)
    return words


def grade(extraction, gold_set, chunk_name):
    """Grade extraction against gold set."""
    if not extraction:
        return {
            'chunk': chunk_name, 'gold_size': len(gold_set),
            'extract_size': 0, 'correct': 0, 'poison': 0,
            'irrelevant': 0, 'gold_found': 0,
            'precision': 0.0, 'recall': 0.0, 'poison_rate': 0.0,
            'malformed': True, 'monolith': False,
            'notes': 'No valid extraction',
        }

    # Check for monolith (single object with entire conversation)
    monolith = False
    if len(extraction) == 1:
        content = extraction[0].get('content', '')
        if len(content) > 200:
            monolith = True

    gold_contents = [g['content'].lower() for g in gold_set]
    gold_kw = [keywords(g) for g in gold_contents]

    extract_contents = []
    for e in extraction:
        if isinstance(e, dict):
            extract_contents.append(e.get('content', '').lower())
        else:
            extract_contents.append(str(e).lower())
    extract_kw = [keywords(e) for e in extract_contents]

    matched_gold = set()
    correct = 0
    poison = 0
    irrelevant = 0

    for ei, ekw in enumerate(extract_kw):
        if not ekw:
            irrelevant += 1
            continue

        best_score = 0
        best_gi = -1
        for gi, gkw in enumerate(gold_kw):
            if not gkw:
                continue
            overlap = len(ekw & gkw)
            score = overlap / max(len(ekw), len(gkw))
            if score > best_score:
                best_score = score
                best_gi = gi

        if best_score >= 0.25:
            correct += 1
            matched_gold.add(best_gi)
        else:
            irrelevant += 1

    gold_found = len(matched_gold)
    total = len(extraction)

    return {
        'chunk': chunk_name,
        'gold_size': len(gold_set),
        'extract_size': total,
        'correct': correct,
        'poison': poison,
        'irrelevant': irrelevant,
        'gold_found': gold_found,
        'precision': correct / total if total > 0 else 0,
        'recall': gold_found / len(gold_set) if gold_set else 0,
        'poison_rate': poison / total if total > 0 else 0,
        'malformed': False,
        'monolith': monolith,
        'notes': 'Monolith: 1 giant object instead of atomic memories' if monolith else '',
    }


def main():
    print("=" * 74)
    print("SUBCONSCIOUS PHASE 0: GRADING")
    print("=" * 74)

    # Parse all raw outputs
    print("\n--- Raw output analysis ---")
    all_parsed = {}
    for name in ['A1','A2','A3','B1','B2','B3','C1','C2','C3','C4']:
        raw_path = os.path.join(EXTRACT_DIR, f"raw_{name}.txt")
        if not os.path.exists(raw_path):
            print(f"  {name}: no raw output")
            continue

        with open(raw_path) as f:
            raw = f.read()

        objects = parse_concatenated_json(raw)
        monolith = len(objects) == 1 and len(objects[0].get('content', '')) > 200

        if monolith:
            print(f"  {name}: MONOLITH (1 object, {len(objects[0].get('content',''))} char content)")
        else:
            print(f"  {name}: {len(objects)} memories extracted")
            for i, obj in enumerate(objects[:3]):
                c = obj.get('content', '')[:80]
                print(f"    [{i}] {c}...")

        all_parsed[name] = objects

        # Save corrected extraction
        corrected_path = os.path.join(EXTRACT_DIR, f"extract_{name}_corrected.json")
        with open(corrected_path, 'w') as f:
            json.dump(objects, f, indent=2)

    # Grade gold-set chunks
    print(f"\n--- Gold set grading ---")
    results = []
    for gc in ['A1', 'B1', 'C1']:
        gold_path = os.path.join(GOLD_DIR, f"gold_{gc}.json")
        if not os.path.exists(gold_path):
            print(f"  {gc}: no gold set")
            continue

        gold_set = json.load(open(gold_path))
        extraction = all_parsed.get(gc, [])

        r = grade(extraction, gold_set, gc)
        results.append(r)

        print(f"\n  {gc} {'(MONOLITH)' if r['monolith'] else ''}:")
        print(f"    Gold: {r['gold_size']} | Extracted: {r['extract_size']}")
        print(f"    Correct: {r['correct']} | Poison: {r['poison']} | Irrelevant: {r['irrelevant']}")
        print(f"    Gold found: {r['gold_found']}/{r['gold_size']}")
        print(f"    Precision: {r['precision']:.1%}")
        print(f"    Recall: {r['recall']:.1%}")
        print(f"    Poison: {r['poison_rate']:.1%}")
        if r['notes']:
            print(f"    Notes: {r['notes']}")

    # Non-gold qualitative scan
    print(f"\n--- Non-gold chunks ---")
    for name in ['A2','A3','B2','B3','C2','C3','C4']:
        objs = all_parsed.get(name, [])
        if not objs:
            print(f"  {name}: no extraction")
            continue
        monolith = len(objs) == 1 and len(objs[0].get('content', '')) > 200
        if monolith:
            print(f"  {name}: MONOLITH")
        else:
            entities = set()
            for o in objs:
                for e in o.get('entities', []):
                    if isinstance(e, str) and len(e) > 2:
                        entities.add(e)
            types = [o.get('memory_type', '?') for o in objs]
            domains = [o.get('domain', '?') for o in objs]
            print(f"  {name}: {len(objs)} memories | "
                  f"types: {set(types)} | domains: {set(domains)}")
            print(f"    entities: {', '.join(sorted(entities)[:8])}")

    # Summary table
    print(f"\n{'=' * 74}")
    print("RESULTS TABLE")
    print(f"{'=' * 74}")
    print(f"{'Chunk':<8} | {'Gold':>4} | {'Extr':>4} | {'Corr':>4} | {'Pois':>4} | "
          f"{'Prec':>6} | {'Recall':>6} | {'Poison':>6} | {'Mono':>4}")
    print("-" * 74)
    for r in results:
        mono = "YES" if r['monolith'] else "no"
        print(f"{r['chunk']:<8} | {r['gold_size']:>4} | {r['extract_size']:>4} | "
              f"{r['correct']:>4} | {r['poison']:>4} | "
              f"{r['precision']:>5.1%} | {r['recall']:>5.1%} | "
              f"{r['poison_rate']:>5.1%} | {mono:>4}")
    print("-" * 74)

    if results:
        avg_prec = np.mean([r['precision'] for r in results])
        avg_rec = np.mean([r['recall'] for r in results])
        avg_pois = np.mean([r['poison_rate'] for r in results])
        n_mono = sum(1 for r in results if r['monolith'])
        n_malformed = sum(1 for r in results if r['malformed'])

        print(f"\nAverages: Precision {avg_prec:.1%} | Recall {avg_rec:.1%} | "
              f"Poison {avg_pois:.1%}")
        print(f"Monoliths: {n_mono}/{len(results)} | Malformed: {n_malformed}/{len(results)}")

        # Monolith count across all 10 chunks
        all_mono = sum(1 for name in all_parsed
                       if len(all_parsed[name]) == 1
                       and len(all_parsed[name][0].get('content', '')) > 200)
        all_multi = sum(1 for name in all_parsed
                        if len(all_parsed[name]) > 1)
        print(f"\nAll 10 chunks: {all_mono} monoliths, {all_multi} multi-memory, "
              f"{10 - all_mono - all_multi} other")

        # Go/No-Go
        print(f"\n{'=' * 74}")
        if n_mono > 0:
            print("GO/NO-GO: ITERATE")
            print("  Primary failure: 1B produces monolithic dumps instead of atomic memories.")
            print("  The model understands the domain but ignores the 'one memory per fact' rule.")
            print("  Fix: stronger few-shot examples in extraction prompt showing")
            print("  input -> multiple small JSON objects.")
        elif avg_prec > 0.90 and avg_pois < 0.05:
            print("GO/NO-GO: PASS")
        elif avg_prec >= 0.75 or avg_pois <= 0.10:
            print("GO/NO-GO: ITERATE")
        else:
            print("GO/NO-GO: FAIL")

        print(f"  Precision {avg_prec:.1%} (target >90%)")
        print(f"  Recall {avg_rec:.1%} (target >70%)")
        print(f"  Poison {avg_pois:.1%} (target <5%)")


if __name__ == "__main__":
    main()
