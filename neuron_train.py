#!/usr/bin/env python3
"""
PROJECT NEURON Phase 4: Fine-tune FFN-only Pythia-160M on Subconscious memory data.
Multi-task: entity extraction (multi-label), domain classification, embedding projection.

FFN-only architecture processes tokens independently (no cross-token attention),
so entity extraction uses pooled multi-label classification over a known entity
vocabulary rather than per-token BIO tagging.
"""

import json
import os
import time
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
from collections import Counter
import sys
import chromadb

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
SAVE_DIR = "/Users/midas/Desktop/cowork/ane-compiler/neuron_model"
CHROMA_PATH = "/Users/midas/Desktop/cowork/orion-ane/memory/chromadb_live"
MODEL_NAME = "EleutherAI/pythia-160m"
N_LAYERS = 8
MAX_SEQ_LEN = 128
BATCH_SIZE = 16
ENTITY_MIN_COUNT = 3  # Minimum occurrences to include in entity vocabulary

# Domain mapping: 5 canonical classes, extras bucketed into general
DOMAIN_MAP = {"decision": 0, "task": 1, "preference": 2, "quantitative": 3, "general": 4}
DOMAIN_NAMES = ["decision", "task", "preference", "quantitative", "general"]


# ---------------------------------------------------------------------------
# 1. Data extraction
# ---------------------------------------------------------------------------

def load_chromadb_data():
    """Extract all memories from ChromaDB."""
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    col = client.get_collection("conversation_memory")

    count = col.count()
    print(f"ChromaDB has {count} entries")

    all_docs = []
    all_metas = []
    all_embeds = []

    batch_size = 500
    all_ids = col.get(include=[])["ids"]

    for i in range(0, len(all_ids), batch_size):
        batch_ids = all_ids[i:i+batch_size]
        batch = col.get(ids=batch_ids, include=["documents", "metadatas", "embeddings"])
        all_docs.extend(batch["documents"])
        all_metas.extend(batch["metadatas"])
        all_embeds.extend(batch["embeddings"])

    print(f"Loaded {len(all_docs)} documents")
    return all_docs, all_metas, all_embeds


def build_entity_vocab(metas):
    """Build entity vocabulary from metadata. Returns entity->index mapping."""
    counter = Counter()
    for m in metas:
        try:
            entities = json.loads(m.get("entities", "[]"))
            for e in entities:
                if e and isinstance(e, str) and len(e) > 1:
                    counter[e] += 1
        except (json.JSONDecodeError, TypeError):
            pass

    # Filter to entities with enough occurrences
    vocab = {}
    for ent, count in counter.most_common():
        if count >= ENTITY_MIN_COUNT:
            vocab[ent] = len(vocab)

    return vocab


class MemoryDataset(Dataset):
    def __init__(self, docs, metas, embeds, tokenizer, entity_vocab, max_len=MAX_SEQ_LEN):
        self.samples = []
        self.n_entities = len(entity_vocab)
        skipped = 0

        for doc, meta, emb in zip(docs, metas, embeds):
            if not doc or not meta:
                skipped += 1
                continue

            # Domain label
            dtype = meta.get("type", "general")
            domain = DOMAIN_MAP.get(dtype, 4)

            # Entity multi-label vector
            entity_labels = torch.zeros(self.n_entities, dtype=torch.float32)
            try:
                entities = json.loads(meta.get("entities", "[]"))
                for e in entities:
                    if e in entity_vocab:
                        entity_labels[entity_vocab[e]] = 1.0
            except (json.JSONDecodeError, TypeError):
                pass

            # Tokenize
            enc = tokenizer(doc, truncation=True, max_length=max_len,
                           padding="max_length", return_tensors="pt")
            input_ids = enc["input_ids"].squeeze(0)
            attention_mask = enc["attention_mask"].squeeze(0)

            # Embedding target
            emb_tensor = torch.tensor(emb, dtype=torch.float32)

            self.samples.append({
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "entity_labels": entity_labels,
                "domain": torch.tensor(domain, dtype=torch.long),
                "embedding": emb_tensor,
            })

        if skipped:
            print(f"  Skipped {skipped} entries (empty doc/meta)")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


# ---------------------------------------------------------------------------
# 2. Model
# ---------------------------------------------------------------------------

class FFNOnlyLayer(nn.Module):
    """Pythia layer with attention completely bypassed -- FFN-only."""
    def __init__(self, original_layer):
        super().__init__()
        self.post_attention_layernorm = original_layer.post_attention_layernorm
        self.mlp = original_layer.mlp
        self.use_parallel_residual = original_layer.use_parallel_residual

    def forward(self, x):
        if self.use_parallel_residual:
            return x + self.mlp(self.post_attention_layernorm(x))
        else:
            return x + self.mlp(self.post_attention_layernorm(x))


class NeuronMultiTask(nn.Module):
    def __init__(self, base_model_name=MODEL_NAME, n_layers=N_LAYERS, n_entities=0):
        super().__init__()
        base = AutoModelForCausalLM.from_pretrained(base_model_name)

        self.embed = base.gpt_neox.embed_in
        self.layers = nn.ModuleList([
            FFNOnlyLayer(layer) for layer in base.gpt_neox.layers[:n_layers]
        ])
        self.final_ln = base.gpt_neox.final_layer_norm

        hidden_dim = base.config.hidden_size  # 768

        # Local context via 1D convolution (cheap cross-token features)
        # Captures trigram patterns without attention
        self.conv_block = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.GELU(),
        )

        # Task heads
        self.domain_head = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, 5),
        )

        self.embed_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 384),
        )

        # Multi-label entity classifier
        self.entity_head = nn.Sequential(
            nn.Linear(hidden_dim, 512),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(512, n_entities),
        )

    def get_backbone_params(self):
        """Original Pythia weights (freeze in Stage 1)."""
        return list(self.embed.parameters()) + \
               list(self.layers.parameters()) + \
               list(self.final_ln.parameters())

    def get_head_params(self):
        """New layers: conv block + task heads (always trainable)."""
        return list(self.conv_block.parameters()) + \
               list(self.domain_head.parameters()) + \
               list(self.embed_head.parameters()) + \
               list(self.entity_head.parameters())

    def forward(self, input_ids, attention_mask=None):
        x = self.embed(input_ids)
        for layer in self.layers:
            x = layer(x)
        x = self.final_ln(x)

        # Apply conv block for local context: [B, S, D] -> [B, D, S] -> conv -> [B, S, D]
        x_conv = self.conv_block(x.transpose(1, 2)).transpose(1, 2)
        x = x + x_conv  # residual

        # Pooled representation for all heads
        if attention_mask is not None:
            mask = attention_mask.unsqueeze(-1).float()
            pooled = (x * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
        else:
            pooled = x.mean(dim=1)

        entity_logits = self.entity_head(pooled)  # [B, n_entities]
        domain_logits = self.domain_head(pooled)   # [B, 5]
        embedding = self.embed_head(pooled)         # [B, 384]
        # Note: no L2 norm here -- let the loss handle alignment

        return entity_logits, domain_logits, embedding


# ---------------------------------------------------------------------------
# 3. Training
# ---------------------------------------------------------------------------

def cosine_embedding_loss(pred, target):
    """1 - cosine_similarity, so 0 = perfect alignment."""
    cos = F.cosine_similarity(pred, target, dim=-1)
    return (1 - cos).mean()


def compute_loss(entity_logits, domain_logits, embedding,
                 entity_labels, domain_labels, embed_targets):
    """Multi-task loss: 0.3*entity_BCE + 0.2*domain_CE + 0.5*cosine_loss"""
    # Entity: binary cross-entropy with positive weighting
    pos_weight = torch.ones_like(entity_logits[0]) * 5.0
    entity_bce = F.binary_cross_entropy_with_logits(
        entity_logits, entity_labels, pos_weight=pos_weight
    )

    # Domain CE
    domain_ce = F.cross_entropy(domain_logits, domain_labels)

    # Embedding: cosine similarity loss
    embed_loss = cosine_embedding_loss(embedding, embed_targets)

    total = 0.3 * entity_bce + 0.2 * domain_ce + 0.5 * embed_loss
    return total, entity_bce.item(), domain_ce.item(), embed_loss.item()


def evaluate(model, loader, device, entity_vocab_inv=None, threshold=0.5):
    """Evaluate on validation set. Returns metrics dict."""
    model.eval()

    total_loss = 0
    total_entity_bce = 0
    total_domain_ce = 0
    total_embed_loss = 0
    n_batches = 0

    # Domain metrics
    domain_correct = 0
    domain_total = 0

    # Entity metrics
    entity_tp = 0
    entity_fp = 0
    entity_fn = 0

    # Embedding metrics
    cos_sims = []

    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            entity_labels = batch["entity_labels"].to(device)
            domain = batch["domain"].to(device)
            embed_target = batch["embedding"].to(device)

            entity_logits, domain_logits, embedding = model(input_ids, attention_mask)

            loss, ebce, dce, eloss = compute_loss(
                entity_logits, domain_logits, embedding,
                entity_labels, domain, embed_target
            )

            total_loss += loss.item()
            total_entity_bce += ebce
            total_domain_ce += dce
            total_embed_loss += eloss
            n_batches += 1

            # Domain accuracy
            preds = domain_logits.argmax(dim=-1)
            domain_correct += (preds == domain).sum().item()
            domain_total += domain.size(0)

            # Entity: threshold sigmoid outputs
            ent_preds = (torch.sigmoid(entity_logits) > threshold).float()
            ent_gold = entity_labels

            # Per-sample entity metrics
            tp = (ent_preds * ent_gold).sum().item()
            fp = (ent_preds * (1 - ent_gold)).sum().item()
            fn = ((1 - ent_preds) * ent_gold).sum().item()
            entity_tp += tp
            entity_fp += fp
            entity_fn += fn

            # Cosine similarity
            cos = F.cosine_similarity(embedding, embed_target, dim=-1)
            cos_sims.extend(cos.cpu().tolist())

    precision = entity_tp / max(entity_tp + entity_fp, 1)
    recall = entity_tp / max(entity_tp + entity_fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-9)
    total_extracted = entity_tp + entity_fp
    poison_rate = entity_fp / max(total_extracted, 1)

    metrics = {
        "val_loss": total_loss / max(n_batches, 1),
        "entity_bce": total_entity_bce / max(n_batches, 1),
        "domain_ce": total_domain_ce / max(n_batches, 1),
        "embed_loss": total_embed_loss / max(n_batches, 1),
        "domain_acc": domain_correct / max(domain_total, 1),
        "entity_precision": precision,
        "entity_recall": recall,
        "entity_f1": f1,
        "entity_poison": poison_rate,
        "embed_cosine": np.mean(cos_sims) if cos_sims else 0.0,
    }
    return metrics


def train_stage(model, train_loader, val_loader, optimizer, device,
                n_epochs, stage_name, patience=3):
    """Train for n_epochs with early stopping."""
    best_val_loss = float("inf")
    patience_counter = 0
    best_state = None

    for epoch in range(1, n_epochs + 1):
        model.train()
        total_loss = 0
        n_batches = 0
        t0 = time.time()

        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            entity_labels = batch["entity_labels"].to(device)
            domain = batch["domain"].to(device)
            embed_target = batch["embedding"].to(device)

            optimizer.zero_grad()
            entity_logits, domain_logits, embedding = model(input_ids, attention_mask)
            loss, _, _, _ = compute_loss(
                entity_logits, domain_logits, embedding,
                entity_labels, domain, embed_target
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        train_loss = total_loss / max(n_batches, 1)
        dt = time.time() - t0

        # Validate
        metrics = evaluate(model, val_loader, device)
        val_loss = metrics["val_loss"]

        print(f"  [{stage_name}] Epoch {epoch:2d} | "
              f"train={train_loss:.4f} val={val_loss:.4f} | "
              f"dom={metrics['domain_acc']:.3f} ent_f1={metrics['entity_f1']:.3f} "
              f"ent_p={metrics['entity_precision']:.3f} "
              f"cos={metrics['embed_cosine']:.3f} | {dt:.1f}s", flush=True)

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"  Early stopping at epoch {epoch}")
                break

    # Restore best
    if best_state is not None:
        model.load_state_dict(best_state)
        model.to(device)

    return best_val_loss


def main():
    print("=" * 70)
    print("PROJECT NEURON Phase 4: Fine-tune on Subconscious Memory Data")
    print("=" * 70)

    # Load data
    print("\n[1/5] Loading ChromaDB data...")
    docs, metas, embeds = load_chromadb_data()

    # Build entity vocabulary from ALL data before splitting
    print("\n  Building entity vocabulary...")
    entity_vocab = build_entity_vocab(metas)
    entity_vocab_inv = {v: k for k, v in entity_vocab.items()}
    n_entities = len(entity_vocab)
    print(f"  Entity vocab size: {n_entities} (min_count={ENTITY_MIN_COUNT})")

    # Load tokenizer
    print("\n[2/5] Loading tokenizer and building datasets...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Shuffle and split
    indices = list(range(len(docs)))
    random.shuffle(indices)

    n_train = min(2500, int(len(indices) * 0.8))
    n_val = len(indices) - n_train

    train_idx = indices[:n_train]
    val_idx = indices[n_train:]

    print(f"  Split: {n_train} train, {n_val} val")

    train_docs = [docs[i] for i in train_idx]
    train_metas = [metas[i] for i in train_idx]
    train_embeds = [embeds[i] for i in train_idx]

    val_docs = [docs[i] for i in val_idx]
    val_metas = [metas[i] for i in val_idx]
    val_embeds = [embeds[i] for i in val_idx]

    print("  Building train dataset...")
    train_ds = MemoryDataset(train_docs, train_metas, train_embeds, tokenizer, entity_vocab)
    print(f"  Train samples: {len(train_ds)}")

    print("  Building val dataset...")
    val_ds = MemoryDataset(val_docs, val_metas, val_embeds, tokenizer, entity_vocab)
    print(f"  Val samples: {len(val_ds)}")

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

    # Build model
    print("\n[3/5] Building NeuronMultiTask model...")
    model = NeuronMultiTask(n_entities=n_entities).float().to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    n_head_params = sum(p.numel() for p in model.get_head_params())
    n_backbone_params = sum(p.numel() for p in model.get_backbone_params())
    print(f"  Total params: {n_params:,}")
    print(f"  Backbone params: {n_backbone_params:,}")
    print(f"  Head params: {n_head_params:,}")
    print(f"  Device: {DEVICE}", flush=True)

    # Stage 1: Freeze backbone, train heads only
    print("\n[4/5] Stage 1: Train heads + conv (20 epochs)...")
    for p in model.get_backbone_params():
        p.requires_grad = False

    optimizer1 = torch.optim.AdamW(model.get_head_params(), lr=2e-3, weight_decay=0.01)
    train_stage(model, train_loader, val_loader, optimizer1, DEVICE,
                n_epochs=20, stage_name="Stage1", patience=7)

    # Stage 2: Unfreeze all, differential lr
    print("\n      Stage 2: Full fine-tune (40 epochs)...")
    for p in model.get_backbone_params():
        p.requires_grad = True

    optimizer2 = torch.optim.AdamW([
        {"params": model.get_backbone_params(), "lr": 5e-5},
        {"params": model.get_head_params(), "lr": 5e-4},
    ], weight_decay=0.01)

    train_stage(model, train_loader, val_loader, optimizer2, DEVICE,
                n_epochs=50, stage_name="Stage2", patience=10)

    # Final evaluation
    print("\n[5/5] Final evaluation on validation set...")
    metrics = evaluate(model, val_loader, DEVICE)

    # Sweep thresholds for best entity F1
    print("\n  Threshold sweep for entity extraction:")
    best_thresh = 0.5
    best_ent_f1 = metrics["entity_f1"]
    for thresh in [0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7]:
        m = evaluate(model, val_loader, DEVICE, threshold=thresh)
        print(f"    thresh={thresh:.2f}: F1={m['entity_f1']:.3f} P={m['entity_precision']:.3f} R={m['entity_recall']:.3f} poison={m['entity_poison']:.3f}")
        if m["entity_f1"] > best_ent_f1:
            best_ent_f1 = m["entity_f1"]
            best_thresh = thresh

    print(f"  Best threshold: {best_thresh} (F1={best_ent_f1:.3f})")
    metrics = evaluate(model, val_loader, DEVICE, threshold=best_thresh)

    # Per-class domain accuracy
    model.eval()
    class_correct = {i: 0 for i in range(5)}
    class_total = {i: 0 for i in range(5)}
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            domain = batch["domain"]
            _, domain_logits, _ = model(input_ids, attention_mask)
            preds = domain_logits.argmax(dim=-1).cpu()
            for p, g in zip(preds, domain):
                g_item = g.item()
                class_total[g_item] += 1
                if p.item() == g_item:
                    class_correct[g_item] += 1

    # Save model
    os.makedirs(SAVE_DIR, exist_ok=True)
    save_path = os.path.join(SAVE_DIR, "neuron_multitask.pt")
    torch.save({
        "model_state_dict": model.state_dict(),
        "config": {
            "base_model": MODEL_NAME,
            "n_layers": N_LAYERS,
            "hidden_dim": 768,
            "max_seq_len": MAX_SEQ_LEN,
            "domain_map": DOMAIN_MAP,
            "entity_vocab": entity_vocab,
            "entity_threshold": best_thresh,
        },
        "metrics": metrics,
    }, save_path)
    print(f"\n  Model saved to {save_path}")

    # Print structured results
    print("\n" + "=" * 70)
    print("NEURON PHASE 4 RESULTS")
    print("=" * 70)

    print(f"\n{'Metric':<30} {'Value':>10} {'Target':>10} {'Pass':>6}")
    print("-" * 60)

    # Entity extraction
    ent_f1 = metrics["entity_f1"]
    ent_poison = metrics["entity_poison"]
    print(f"{'Entity F1':<30} {ent_f1:>10.3f} {'>0.80':>10} {'YES' if ent_f1 > 0.8 else 'NO':>6}")
    print(f"{'Entity Precision':<30} {metrics['entity_precision']:>10.3f} {'--':>10} {'--':>6}")
    print(f"{'Entity Recall':<30} {metrics['entity_recall']:>10.3f} {'--':>10} {'--':>6}")
    print(f"{'Entity Poison Rate':<30} {ent_poison:>10.3f} {'<0.05':>10} {'YES' if ent_poison < 0.05 else 'NO':>6}")
    print(f"{'Entity Threshold':<30} {best_thresh:>10.2f}")

    # Domain classification
    dom_acc = metrics["domain_acc"]
    print(f"{'Domain Accuracy':<30} {dom_acc:>10.3f} {'>0.85':>10} {'YES' if dom_acc > 0.85 else 'NO':>6}")

    # Per-class
    for i, name in enumerate(DOMAIN_NAMES):
        if class_total[i] > 0:
            acc = class_correct[i] / class_total[i]
            print(f"  {name:<28} {acc:>10.3f} {'':>10} {class_total[i]:>5}n")

    # Embedding
    cos = metrics["embed_cosine"]
    print(f"{'Embedding Cosine Sim':<30} {cos:>10.3f} {'>0.85':>10} {'YES' if cos > 0.85 else 'NO':>6}")

    # Loss breakdown
    print(f"\n{'Val Loss (total)':<30} {metrics['val_loss']:>10.4f}")
    print(f"{'  Entity BCE':<30} {metrics['entity_bce']:>10.4f}")
    print(f"{'  Domain CE':<30} {metrics['domain_ce']:>10.4f}")
    print(f"{'  Embed Cosine Loss':<30} {metrics['embed_loss']:>10.4f}")

    print(f"\n{'Entity Vocab Size':<30} {n_entities:>10}")

    print("\n" + "=" * 70)
    print(f"Model: {SAVE_DIR}/neuron_multitask.pt")
    print(f"Params: {n_params:,} total ({n_backbone_params:,} backbone + {n_head_params:,} heads)")
    print("=" * 70)


if __name__ == "__main__":
    main()
