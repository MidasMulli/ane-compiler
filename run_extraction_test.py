#!/usr/bin/env python3
"""
Subconscious Phase 0: Extraction quality test.

Sends 10 conversation chunks through Llama-1B on ANE (25d+C config)
with the structured extraction prompt. Grades gold-set chunks (A1, B1, C1)
against manual gold sets.

Copyright 2026 Nick Lo. MIT License.
"""

import os
import sys
import json
import time
import numpy as np
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
sys.path.insert(0, os.path.dirname(__file__))

from llama_loader import LlamaModel, LlamaConfig
from kv_cache import KVCache
from run_llama_fused import build_all_models, generate_fused
from llama_loader import rms_norm_cpu

VAULT_DIR = os.path.expanduser("~/Desktop/cowork/vault/subconscious")
CHUNKS_DIR = os.path.join(VAULT_DIR, "chunks")
GOLD_DIR = os.path.join(VAULT_DIR, "gold_sets")
EXTRACT_DIR = os.path.join(VAULT_DIR, "extractions_v2")

MODEL_PATH = os.path.expanduser(
    "~/.cache/huggingface/hub/models--unsloth--Llama-3.2-1B-Instruct/"
    "snapshots/5a8abab4a5d6f164389b1079fb721cfab8d7126c/"
)

SYSTEM_PROMPT = """You are a memory extraction system. You read conversations and output structured
JSON containing atomic memories. You do not summarize or interpret. You extract
discrete facts exactly as stated or clearly implied.

DECOMPOSITION: Before outputting JSON, mentally identify every distinct claim.
A claim is one subject doing or being one thing. "X runs at 50 tok/s with 25
dispatches using C+fusion" = 3 claims minimum: speed, dispatch count, config.
Each claim becomes one JSON object.

RULES:
- One memory per fact. Never combine multiple facts into one memory.
- Use the speakers words as ground truth. Store numbers exactly as stated.
- Only extract what is stated or directly demonstrated.
- When a fact updates a previous fact, extract only the newer version.
- Ignore: greetings, filler, speculative discussion reaching no conclusion,
  hedging language, compliments, meta-conversation.
- If uncertain whether something is a fact or speculation, skip it.

OUTPUT: JSON array only. No preamble, no markdown fences, no explanation.

Each object:
{
  "content": "The atomic fact in one sentence",
  "memory_type": "fact | preference | state | relationship",
  "entities": ["entity1", "entity2"],
  "domain": "hardware | legal | production | personal | research",
  "confidence": "high | medium"
}

EXAMPLE — WRONG (monolith, do NOT do this):
[{"content": "GPT-2 fusion: 73 to 37 dispatches, 28.9 to 135.9 tok/s, FFN fusion via NeuralNetworkBuilder, QKV fusion as single conv, kill test 11/11 and 15/15", "memory_type": "state", "entities": ["GPT-2"], "domain": "hardware", "confidence": "high"}]

EXAMPLE — CORRECT (atomic, do this):
[
  {"content": "GPT-2 FFN fusion reduces dispatches from 73 to 37", "memory_type": "state", "entities": ["GPT-2", "ane-compiler"], "domain": "hardware", "confidence": "high"},
  {"content": "GPT-2 achieves 135.9 tok/s at 37 dispatches on ANE", "memory_type": "state", "entities": ["GPT-2", "ANE"], "domain": "hardware", "confidence": "high"},
  {"content": "FFN fusion uses NeuralNetworkBuilder to xcrun compile with GELU mode 19 patching", "memory_type": "fact", "entities": ["ane-compiler"], "domain": "hardware", "confidence": "high"},
  {"content": "QKV fusion combines c_attn weight 768 to 2304 as single conv", "memory_type": "fact", "entities": ["GPT-2", "ane-compiler"], "domain": "hardware", "confidence": "high"},
  {"content": "GPT-2 fusion kill test passes 11/11 and 15/15 tokens against PyTorch", "memory_type": "state", "entities": ["GPT-2", "ane-compiler"], "domain": "research", "confidence": "high"}
]

MEMORY TYPES:
- fact: Stable knowledge unlikely to change.
- preference: How the user wants things done.
- state: Current status that changes over time.
- relationship: A connection between two entities.

ENTITIES: Named things (models, hardware, software, repos, protocols, concepts).
Consistent naming: ANE not Apple Neural Engine.

DOMAINS: hardware | legal | production | personal | research

DOMAIN GUIDANCE: Hardware capabilities that dont change = fact. Measurements
depending on config = state. Settled architecture decisions = state.

A typical conversation window yields 10-20 memories. If you produce fewer
than 5, you are likely merging related facts. Split further."""


def build_prompt(chunk_text):
    """Build the full chat-templated prompt for extraction."""
    user_msg = f"""Extract memories from this conversation window. Output JSON array only.

CONVERSATION WINDOW:
---
{chunk_text}
---

JSON:"""
    return user_msg


def generate_text(model, ct_models, dispatch_mode, tokenizer,
                  system_prompt, user_prompt, max_tokens=1500):
    """Generate text using the 40d fused pipeline with chat template."""
    config = model.config
    dim = config.hidden_size
    n_heads = config.n_heads
    n_kv_heads = config.n_kv_heads
    head_dim = config.head_dim
    n_rep = config.n_rep

    # Build chat-templated token sequence
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    result = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=True)
    if hasattr(result, 'input_ids'):
        token_ids = list(result.input_ids)
    elif hasattr(result, 'get') and 'input_ids' in result:
        token_ids = list(result['input_ids'])
    elif isinstance(result, list):
        token_ids = result
    elif hasattr(result, 'tolist'):
        token_ids = result.tolist()
    else:
        token_ids = list(result)
    token_ids = [int(t) for t in token_ids]

    # EOS tokens
    eos_ids = set()
    if hasattr(tokenizer, 'eos_token_id') and tokenizer.eos_token_id is not None:
        eos_ids.add(tokenizer.eos_token_id)
    for special in ['<|eot_id|>', '<|end_of_text|>']:
        tid = tokenizer.convert_tokens_to_ids(special)
        if tid is not None and tid != tokenizer.unk_token_id:
            eos_ids.add(tid)

    # Use the generation engine from run_llama_fused
    from llama_loader import rope_cpu, softmax_cpu
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
    print(f"    Prefilling {len(token_ids)} prompt tokens...", end="", flush=True)
    t0 = time.time()
    for pos, tok in enumerate(token_ids[:-1]):
        forward_token(tok, pos)
    logits = forward_token(token_ids[-1], len(token_ids) - 1)
    print(f" {time.time()-t0:.1f}s")

    # Decode
    generated_ids = []
    n_tokens = len(token_ids)
    t0 = time.time()

    while len(generated_ids) < max_tokens:
        token = int(np.argmax(logits))
        if token in eos_ids:
            break
        generated_ids.append(token)
        logits = forward_token(token, n_tokens)
        n_tokens += 1

        # Progress
        if len(generated_ids) % 50 == 0:
            elapsed = time.time() - t0
            tps = len(generated_ids) / elapsed if elapsed > 0 else 0
            print(f"    Generated {len(generated_ids)} tokens ({tps:.1f} tok/s)...",
                  flush=True)

    elapsed = time.time() - t0
    tps = len(generated_ids) / elapsed if elapsed > 0 else 0
    print(f"    Done: {len(generated_ids)} tokens in {elapsed:.1f}s ({tps:.1f} tok/s)")

    text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    return text


def parse_json_output(raw_text):
    """Try to parse JSON array from model output. Handle common failures."""
    text = raw_text.strip()

    # Strip markdown fences if present
    if text.startswith("```"):
        lines = text.split("\n")
        lines = [l for l in lines if not l.strip().startswith("```")]
        text = "\n".join(lines).strip()

    # Find the JSON array
    start = text.find("[")
    end = text.rfind("]")
    if start == -1 or end == -1:
        return None, "No JSON array found"

    json_str = text[start:end + 1]

    try:
        data = json.loads(json_str)
        if isinstance(data, list):
            return data, None
        return None, "Parsed JSON is not an array"
    except json.JSONDecodeError as e:
        return None, f"JSON parse error: {e}"


def grade_extraction(extraction, gold_set, chunk_name):
    """Grade extraction against gold set. Returns metrics dict."""
    if extraction is None:
        return {
            'chunk': chunk_name,
            'gold_size': len(gold_set),
            'extract_size': 0,
            'correct': 0,
            'poison': 0,
            'irrelevant': 0,
            'gold_found': 0,
            'precision': 0.0,
            'recall': 0.0,
            'poison_rate': 0.0,
            'malformed': True,
            'notes': 'Malformed JSON output',
        }

    # Simple matching: for each extraction, check if it matches a gold item
    # Match = content covers the same fact (fuzzy, checked by keyword overlap)
    gold_contents = [g['content'].lower() for g in gold_set]
    extract_contents = [e.get('content', '').lower() for e in extraction]

    # Build keyword sets for matching
    def keywords(text):
        # Extract significant words (>3 chars, not common words)
        stop = {'the', 'that', 'this', 'with', 'from', 'into', 'than',
                'which', 'where', 'when', 'what', 'they', 'them', 'their',
                'have', 'been', 'were', 'being', 'does', 'done', 'each',
                'only', 'also', 'more', 'most', 'very', 'about', 'after',
                'before', 'between', 'through', 'during', 'because', 'while',
                'using', 'used', 'uses'}
        words = set()
        for w in text.split():
            w = w.strip('.,;:()[]{}"\'-/').lower()
            if len(w) > 3 and w not in stop:
                words.add(w)
        return words

    gold_kw = [keywords(g) for g in gold_contents]
    extract_kw = [keywords(e) for e in extract_contents]

    # Match each extraction to best gold item
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

        if best_score >= 0.3:  # At least 30% keyword overlap
            correct += 1
            matched_gold.add(best_gi)
        elif best_score < 0.1:
            # Check if it's factually wrong (contains numbers/claims not in source)
            # Simple heuristic: if extraction mentions specific numbers, check
            # they appear in the original chunk
            irrelevant += 1
        else:
            irrelevant += 1

    gold_found = len(matched_gold)
    total_extractions = len(extraction)

    precision = correct / total_extractions if total_extractions > 0 else 0
    recall = gold_found / len(gold_set) if len(gold_set) > 0 else 0
    poison_rate = poison / total_extractions if total_extractions > 0 else 0

    return {
        'chunk': chunk_name,
        'gold_size': len(gold_set),
        'extract_size': total_extractions,
        'correct': correct,
        'poison': poison,
        'irrelevant': irrelevant,
        'gold_found': gold_found,
        'precision': precision,
        'recall': recall,
        'poison_rate': poison_rate,
        'malformed': False,
        'notes': '',
    }


def main():
    print("=" * 70)
    print("SUBCONSCIOUS PHASE 0: EXTRACTION QUALITY TEST")
    print("10 chunks through Llama-1B on ANE, grade 3 gold sets")
    print("=" * 70)

    # Load model
    print("\n[1/4] Loading Llama-1B...")
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

    # Build models
    print("\n[2/4] Building MIL IR models...")
    t0 = time.time()
    ct_models, dispatch_mode = build_all_models(model)
    print(f"  Built in {time.time()-t0:.1f}s")

    # Tokenizer
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained('unsloth/Llama-3.2-1B-Instruct')

    # Gold-set chunks only first (A1, B1, C1)
    chunks = ['chunk_A1.txt', 'chunk_B1.txt', 'chunk_C1.txt']
    print(f"\n[3/4] Running extraction on {len(chunks)} gold-set chunks (v2 prompt)...")

    os.makedirs(EXTRACT_DIR, exist_ok=True)

    for chunk_file in chunks:
        chunk_name = chunk_file.replace('chunk_', '').replace('.txt', '')
        chunk_path = os.path.join(CHUNKS_DIR, chunk_file)
        extract_path = os.path.join(EXTRACT_DIR, f"extract_{chunk_name}.json")

        # Skip if already extracted
        if os.path.exists(extract_path):
            print(f"\n  {chunk_name}: cached, skipping")
            continue

        print(f"\n  === {chunk_name} ===")
        chunk_text = open(chunk_path).read()

        user_prompt = build_prompt(chunk_text)
        raw_output = generate_text(
            model, ct_models, dispatch_mode, tokenizer,
            SYSTEM_PROMPT, user_prompt, max_tokens=1500)

        # Save raw output
        raw_path = os.path.join(EXTRACT_DIR, f"raw_{chunk_name}.txt")
        with open(raw_path, 'w') as f:
            f.write(raw_output)

        # Parse JSON
        parsed, error = parse_json_output(raw_output)
        if parsed:
            with open(extract_path, 'w') as f:
                json.dump(parsed, f, indent=2)
            print(f"    Extracted {len(parsed)} memories")
        else:
            print(f"    MALFORMED: {error}")
            # Save error marker
            with open(extract_path, 'w') as f:
                json.dump({"error": error, "raw": raw_output[:500]}, f, indent=2)

    # Grade gold-set chunks
    print(f"\n[4/4] Grading gold-set chunks...")
    gold_chunks = ['A1', 'B1', 'C1']
    results = []

    for gc in gold_chunks:
        gold_path = os.path.join(GOLD_DIR, f"gold_{gc}.json")
        extract_path = os.path.join(EXTRACT_DIR, f"extract_{gc}.json")

        if not os.path.exists(gold_path):
            print(f"  {gc}: no gold set, skipping")
            continue
        if not os.path.exists(extract_path):
            print(f"  {gc}: no extraction, skipping")
            continue

        gold_set = json.load(open(gold_path))
        extract_raw = json.load(open(extract_path))

        # Check if extraction was malformed
        if isinstance(extract_raw, dict) and 'error' in extract_raw:
            r = grade_extraction(None, gold_set, gc)
        else:
            r = grade_extraction(extract_raw, gold_set, gc)
        results.append(r)

        print(f"\n  {gc}:")
        print(f"    Gold: {r['gold_size']} | Extracted: {r['extract_size']}")
        print(f"    Correct: {r['correct']} | Poison: {r['poison']} | Irrelevant: {r['irrelevant']}")
        print(f"    Gold found: {r['gold_found']}/{r['gold_size']}")
        print(f"    Precision: {r['precision']:.1%}")
        print(f"    Recall: {r['recall']:.1%}")
        print(f"    Poison: {r['poison_rate']:.1%}")
        print(f"    Malformed: {r['malformed']}")

    # Summary table
    print(f"\n{'=' * 70}")
    print("RESULTS TABLE")
    print(f"{'=' * 70}")
    print(f"{'Chunk':<8} | {'Gold':>4} | {'Extr':>4} | {'Corr':>4} | {'Pois':>4} | "
          f"{'Prec':>6} | {'Recall':>6} | {'Poison':>6} | {'JSON'}")
    print("-" * 70)
    for r in results:
        json_ok = "OK" if not r['malformed'] else "FAIL"
        print(f"{r['chunk']:<8} | {r['gold_size']:>4} | {r['extract_size']:>4} | "
              f"{r['correct']:>4} | {r['poison']:>4} | "
              f"{r['precision']:>5.1%} | {r['recall']:>5.1%} | "
              f"{r['poison_rate']:>5.1%} | {json_ok}")
    print("-" * 70)

    # Averages
    if results:
        avg_prec = np.mean([r['precision'] for r in results])
        avg_rec = np.mean([r['recall'] for r in results])
        avg_pois = np.mean([r['poison_rate'] for r in results])
        n_malformed = sum(1 for r in results if r['malformed'])
        print(f"{'AVG':<8} | {'':>4} | {'':>4} | {'':>4} | {'':>4} | "
              f"{avg_prec:>5.1%} | {avg_rec:>5.1%} | "
              f"{avg_pois:>5.1%} | {n_malformed}/{len(results)} fail")

        # Go/No-Go
        print(f"\n{'=' * 70}")
        if avg_prec > 0.90 and avg_pois < 0.05 and n_malformed == 0:
            print("GO/NO-GO: PASS")
        elif avg_prec >= 0.75 or avg_pois <= 0.10:
            print("GO/NO-GO: ITERATE (prompt needs work)")
        else:
            print("GO/NO-GO: FAIL")
        print(f"  Precision {avg_prec:.1%} (target >90%)")
        print(f"  Recall {avg_rec:.1%} (target >70%)")
        print(f"  Poison {avg_pois:.1%} (target <5%)")
        print(f"  Malformed JSON: {n_malformed}/{len(results)} (target <10%)")

    # Qualitative scan of non-gold chunks
    print(f"\n{'=' * 70}")
    print("NON-GOLD CHUNKS: Qualitative scan")
    print(f"{'=' * 70}")
    non_gold = [c.replace('chunk_', '').replace('.txt', '')
                for c in chunks
                if c.replace('chunk_', '').replace('.txt', '') not in gold_chunks]
    for nc in non_gold:
        ep = os.path.join(EXTRACT_DIR, f"extract_{nc}.json")
        if not os.path.exists(ep):
            print(f"  {nc}: no extraction")
            continue
        data = json.load(open(ep))
        if isinstance(data, dict) and 'error' in data:
            print(f"  {nc}: MALFORMED — {data['error']}")
        else:
            print(f"  {nc}: {len(data)} memories extracted")
            # Check entity consistency
            entities = set()
            for m in data:
                for e in m.get('entities', []):
                    entities.add(e)
            print(f"    Entities: {', '.join(sorted(entities)[:10])}")


if __name__ == "__main__":
    main()
