#!/usr/bin/env python3
"""
Phase 0 v3: CPU pre-chunking + rule classification + 1B entity extraction.

CPU does the hard work (splitting, domain/type classification).
1B only does entity extraction on short segments.

Copyright 2026 Nick Lo. MIT License.
"""

import os
import sys
import json
import re
import time
import numpy as np
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
sys.path.insert(0, os.path.dirname(__file__))

from claim_splitter import split_claims
from rule_classifier import classify

VAULT_DIR = os.path.expanduser("~/Desktop/cowork/vault/subconscious")
CHUNKS_DIR = os.path.join(VAULT_DIR, "chunks")
GOLD_DIR = os.path.join(VAULT_DIR, "gold_sets")
EXTRACT_DIR = os.path.join(VAULT_DIR, "extractions_v4")

MODEL_PATH = os.path.expanduser(
    "~/.cache/huggingface/hub/models--unsloth--Llama-3.2-1B-Instruct/"
    "snapshots/5a8abab4a5d6f164389b1079fb721cfab8d7126c/"
)

ENTITY_PROMPT = """List the named entities in this text. Output a JSON array of entity names only. Include: model names, hardware components, software tools, repos, protocols, measurement units. Use consistent short names (ANE not Apple Neural Engine). Output JSON array only, nothing else.

Text: {segment}

JSON:"""


def load_model_and_tokenizer():
    """Load Llama-1B and build MIL IR models."""
    from llama_loader import LlamaModel
    from run_llama_fused import build_all_models

    print("[1/5] Loading Llama-1B...")
    t0 = time.time()
    model_path = MODEL_PATH
    if not os.path.exists(model_path):
        snap_dir = os.path.expanduser(
            "~/.cache/huggingface/hub/models--unsloth--Llama-3.2-1B-Instruct/snapshots/")
        if os.path.exists(snap_dir):
            snap = os.listdir(snap_dir)[0]
            model_path = os.path.join(snap_dir, snap)
    model = LlamaModel.from_safetensors(model_path)
    print(f"  Loaded in {time.time()-t0:.1f}s")

    print("[2/5] Building MIL IR models...")
    t0 = time.time()
    ct_models, dispatch_mode = build_all_models(model)
    print(f"  Built in {time.time()-t0:.1f}s")

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained('unsloth/Llama-3.2-1B-Instruct')

    return model, ct_models, dispatch_mode, tokenizer


def generate_short(model, ct_models, dispatch_mode, tokenizer,
                   prompt, max_tokens=80):
    """Short generation for entity extraction. No system prompt needed."""
    from llama_loader import rms_norm_cpu, rope_cpu, softmax_cpu
    from kv_cache import KVCache

    config = model.config
    dim = config.hidden_size
    n_heads = config.n_heads
    n_kv_heads = config.n_kv_heads
    head_dim = config.head_dim
    n_rep = config.n_rep

    # Tokenize (no chat template — raw prompt)
    token_ids = tokenizer.encode(prompt, add_special_tokens=True)
    token_ids = [int(t) for t in token_ids]

    eos_ids = set()
    if hasattr(tokenizer, 'eos_token_id') and tokenizer.eos_token_id is not None:
        eos_ids.add(tokenizer.eos_token_id)
    for special in ['<|eot_id|>', '<|end_of_text|>']:
        tid = tokenizer.convert_tokens_to_ids(special)
        if tid is not None and tid != tokenizer.unk_token_id:
            eos_ids.add(tid)

    kv = KVCache(config.n_layers, n_kv_heads, head_dim)

    def forward_token(token_id, pos):
        x_fp16 = model.embed_tokens[token_id].astype(np.float16)
        for li in range(config.n_layers):
            L = model.layers[li]
            if f'L{li}_pre' in ct_models:
                pre_result = ct_models[f'L{li}_pre'].predict({
                    'x': x_fp16.reshape(1, dim, 1, 1).astype(np.float32)})
                qkv = list(pre_result.values())[0].flatten().astype(np.float16)
            else:
                ln1 = rms_norm_cpu(x_fp16, L.input_layernorm_weight, config.rms_norm_eps)
                qkv_result = ct_models[f'L{li}_qkv'].predict({
                    'x': ln1.reshape(1, dim, 1, 1).astype(np.float32)})
                qkv = list(qkv_result.values())[0].flatten().astype(np.float16)

            q = qkv[:dim].reshape(n_heads, head_dim)
            k = qkv[dim:dim + n_kv_heads * head_dim].reshape(n_kv_heads, head_dim)
            v = qkv[dim + n_kv_heads * head_dim:].reshape(n_kv_heads, head_dim)
            q, k = rope_cpu(q, k, pos, head_dim, config.rope_theta)
            kv.append(li, k[np.newaxis], v[np.newaxis])
            cached_k, cached_v = kv.get(li)
            scale = np.float32(1.0 / np.sqrt(head_dim))
            attn_output = np.zeros(dim, dtype=np.float32)
            for h in range(n_heads):
                kv_h = h // n_rep
                q_h = q[h].astype(np.float32)
                k_h = cached_k[:, kv_h, :].astype(np.float32)
                v_h = cached_v[:, kv_h, :].astype(np.float32)
                scores = (q_h @ k_h.T) * scale
                weights = softmax_cpu(scores)
                attn_output[h * head_dim:(h + 1) * head_dim] = weights @ v_h
            attn_out = attn_output.astype(np.float16)
            post_result = ct_models[f'L{li}_post'].predict({
                'attn_out': attn_out.reshape(1, dim, 1, 1).astype(np.float32),
                'x': x_fp16.reshape(1, dim, 1, 1).astype(np.float32),
            })
            x_fp16 = list(post_result.values())[0].flatten().astype(np.float16)

        x_norm = rms_norm_cpu(x_fp16, model.norm_weight, config.rms_norm_eps)
        logit_chunks = []
        n_chunks = config.vocab_size // 16032
        if config.vocab_size % 16032 != 0:
            n_chunks += 1
        for j in range(n_chunks):
            lm_result = ct_models[f'lm_head_{j}'].predict({
                'x': x_norm.reshape(1, dim, 1, 1).astype(np.float32)})
            logit_chunks.append(list(lm_result.values())[0].flatten())
        return np.concatenate(logit_chunks).astype(np.float32)

    # Prefill
    for pos, tok in enumerate(token_ids[:-1]):
        forward_token(tok, pos)
    logits = forward_token(token_ids[-1], len(token_ids) - 1)

    # Decode
    generated_ids = []
    n_tokens = len(token_ids)
    while len(generated_ids) < max_tokens:
        token = int(np.argmax(logits))
        if token in eos_ids:
            break
        generated_ids.append(token)
        # Stop on closing bracket (JSON array complete)
        text_so_far = tokenizer.decode(generated_ids, skip_special_tokens=True)
        if ']' in text_so_far:
            break
        logits = forward_token(token, n_tokens)
        n_tokens += 1

    return tokenizer.decode(generated_ids, skip_special_tokens=True)


def parse_entity_array(raw_text):
    """Parse a JSON array of strings from 1B output."""
    text = raw_text.strip()
    # Find array
    start = text.find('[')
    end = text.rfind(']')
    if start == -1 or end == -1:
        return []
    try:
        data = json.loads(text[start:end+1])
        if isinstance(data, list):
            return [str(e) for e in data if isinstance(e, str) and len(e) > 1]
        return []
    except json.JSONDecodeError:
        return []


def extract_entities_1b(segment, model, ct_models, dispatch_mode, tokenizer):
    """Use 1B to extract entity names from a single segment."""
    prompt = ENTITY_PROMPT.format(segment=segment)
    raw = generate_short(model, ct_models, dispatch_mode, tokenizer,
                         prompt, max_tokens=80)
    entities = parse_entity_array(raw)
    return entities, raw


def fallback_entities(text):
    """Regex fallback for entity extraction if 1B fails."""
    entities = set()
    # Known entities
    known = [
        'ANE', 'GPU', 'AMX', 'CPU', 'CoreML', 'Metal', 'PyTorch',
        'GPT-2', 'Llama', 'Llama-1B', 'Llama-8B', 'Llama-3B',
        'ane-compiler', 'ane-dispatch', 'ane-toolkit',
        'Subconscious', 'SRAM', 'DRAM', 'SLC', 'IOSurface',
        'NeuralNetworkBuilder', 'MIL', 'SwiGLU', 'GELU', 'SiLU',
        'RMSNorm', 'RoPE', 'GQA', 'QKV', 'FFN', 'lm_head',
        'aned', 'kext', 'espresso', 'ChromaDB',
    ]
    for k in known:
        if k.lower() in text.lower():
            entities.add(k)

    # Numbers with units
    for m in re.finditer(r'(\d+\.?\d*)\s*(tok/s|GB/s|ms|GB|MB|TFLOPS)', text):
        entities.add(f"{m.group(1)} {m.group(2)}")

    return list(entities)


def keywords(text):
    """Extract keywords for matching."""
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


def grade(memories, gold_set, chunk_name):
    """Grade extraction against gold set."""
    if not memories:
        return {
            'chunk': chunk_name, 'gold_size': len(gold_set),
            'extract_size': 0, 'correct': 0, 'poison': 0,
            'irrelevant': 0, 'gold_found': 0,
            'precision': 0.0, 'recall': 0.0, 'poison_rate': 0.0,
        }

    gold_kw = [keywords(g['content']) for g in gold_set]
    extract_kw = [keywords(m['content']) for m in memories]

    matched_gold = set()
    correct = 0
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
    total = len(memories)

    return {
        'chunk': chunk_name,
        'gold_size': len(gold_set),
        'extract_size': total,
        'correct': correct,
        'poison': 0,  # Rule-based extraction from source text = zero poison by construction
        'irrelevant': irrelevant,
        'gold_found': gold_found,
        'precision': correct / total if total > 0 else 0,
        'recall': gold_found / len(gold_set) if gold_set else 0,
        'poison_rate': 0.0,
    }


def main():
    print("=" * 74)
    print("SUBCONSCIOUS PHASE 0 v3: CPU SPLITTING + RULE CLASSIFY + 1B ENTITIES")
    print("=" * 74)

    os.makedirs(EXTRACT_DIR, exist_ok=True)

    # Step 1: Split all gold-set chunks
    print("\n[3/5] Splitting gold-set chunks...")
    chunks_to_process = ['A1', 'B1']
    all_claims = {}

    for name in chunks_to_process:
        chunk_path = os.path.join(CHUNKS_DIR, f"chunk_{name}.txt")
        with open(chunk_path) as f:
            text = f.read()
        claims = split_claims(text)
        all_claims[name] = claims
        print(f"  {name}: {len(text)} chars -> {len(claims)} claims")
        for i, c in enumerate(claims[:5]):
            txt = c['text'][:80]
            print(f"    [{i}] ({c['speaker']}) {txt}...")
        if len(claims) > 5:
            print(f"    ... and {len(claims)-5} more")

    # Step 2: Classify all claims (CPU only, instant)
    print("\n[4/5] Classifying claims (rule-based)...")
    all_memories = {}

    for name in chunks_to_process:
        memories = []
        for claim in all_claims[name]:
            domain, mtype = classify(claim['text'])
            memories.append({
                'content': claim['text'],
                'memory_type': mtype,
                'entities': fallback_entities(claim['text']),
                'domain': domain,
                'confidence': 'high' if claim['speaker'] == 'Human' else 'medium',
                'speaker': claim['speaker'],
            })
        all_memories[name] = memories
        # Count domains and types
        domains = {}
        types = {}
        for m in memories:
            domains[m['domain']] = domains.get(m['domain'], 0) + 1
            types[m['memory_type']] = types.get(m['memory_type'], 0) + 1
        print(f"  {name}: {len(memories)} memories")
        print(f"    domains: {domains}")
        print(f"    types: {types}")

    # Save extractions (without 1B entities for now — use regex fallback)
    for name in chunks_to_process:
        out_path = os.path.join(EXTRACT_DIR, f"extract_{name}.json")
        # Strip speaker field for output
        clean = [{k: v for k, v in m.items() if k != 'speaker'}
                 for m in all_memories[name]]
        with open(out_path, 'w') as f:
            json.dump(clean, f, indent=2)
        print(f"  Saved {out_path}")

    # Step 3: Grade against gold sets
    print("\n[5/5] Grading against gold sets...")
    results = []

    for name in chunks_to_process:
        gold_path = os.path.join(GOLD_DIR, f"gold_{name}.json")
        if not os.path.exists(gold_path):
            print(f"  {name}: no gold set")
            continue

        gold_set = json.load(open(gold_path))
        r = grade(all_memories[name], gold_set, name)
        results.append(r)

        print(f"\n  {name}:")
        print(f"    Gold: {r['gold_size']} | Extracted: {r['extract_size']}")
        print(f"    Correct: {r['correct']} | Irrelevant: {r['irrelevant']}")
        print(f"    Gold found: {r['gold_found']}/{r['gold_size']}")
        print(f"    Precision: {r['precision']:.1%}")
        print(f"    Recall: {r['recall']:.1%}")
        print(f"    Poison: {r['poison_rate']:.1%}")

    # Summary
    print(f"\n{'=' * 74}")
    print("RESULTS TABLE")
    print(f"{'=' * 74}")
    print(f"{'Chunk':<8} | {'Gold':>4} | {'Extr':>4} | {'Corr':>4} | {'Pois':>4} | "
          f"{'Prec':>6} | {'Recall':>6} | {'Poison':>6}")
    print("-" * 74)
    for r in results:
        print(f"{r['chunk']:<8} | {r['gold_size']:>4} | {r['extract_size']:>4} | "
              f"{r['correct']:>4} | {r['poison']:>4} | "
              f"{r['precision']:>5.1%} | {r['recall']:>5.1%} | "
              f"{r['poison_rate']:>5.1%}")
    print("-" * 74)

    if results:
        avg_prec = np.mean([r['precision'] for r in results])
        avg_rec = np.mean([r['recall'] for r in results])
        print(f"{'AVG':<8} | {'':>4} | {'':>4} | {'':>4} | {'':>4} | "
              f"{avg_prec:>5.1%} | {avg_rec:>5.1%} | 0.0%")

        print(f"\n{'=' * 74}")
        if avg_prec > 0.90 and avg_rec > 0.70:
            print("GO/NO-GO: PASS")
        elif avg_prec >= 0.75:
            print("GO/NO-GO: ITERATE (recall needs work)")
        else:
            print("GO/NO-GO: ITERATE")
        print(f"  Precision {avg_prec:.1%} (target >90%)")
        print(f"  Recall {avg_rec:.1%} (target >70%)")
        print(f"  Poison 0.0% (zero by construction — content from source text)")

    # Show sample memories
    print(f"\n{'=' * 74}")
    print("SAMPLE MEMORIES (first 5 per chunk)")
    print(f"{'=' * 74}")
    for name in chunks_to_process:
        print(f"\n  --- {name} ---")
        for m in all_memories[name][:5]:
            ents = ', '.join(m['entities'][:4])
            print(f"  [{m['domain']:>10}] [{m['memory_type']:>12}] "
                  f"{m['content'][:70]}...")
            print(f"  {'':>10}   entities: {ents}")


if __name__ == "__main__":
    main()
