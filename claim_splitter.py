#!/usr/bin/env python3
"""
Claim splitter: conversation text -> claim-level segments.

Pure regex + heuristics. No model. Each output segment should
correspond to roughly one atomic memory.

Copyright 2026 Nick Lo. MIT License.
"""

import re


def split_claims(text):
    """Split conversation text into claim-level segments.

    Returns: list of {"text": str, "speaker": "Human"|"Assistant"}
    """
    # Step 1: Split into speaker blocks
    blocks = split_speaker_blocks(text)

    # Step 2: Split each block into claims
    claims = []
    for block in blocks:
        speaker = block['speaker']
        segments = split_block_into_claims(block['text'])
        for seg in segments:
            seg = seg.strip()
            if len(seg) > 10:  # skip tiny fragments
                claims.append({'text': seg, 'speaker': speaker})

    return claims


def split_speaker_blocks(text):
    """Split text into Human/Assistant blocks."""
    blocks = []
    # Match "Human:", "H:", "A:", "Assistant:" at start of line or text
    pattern = r'(?:^|\n)\s*(Human|H|A|Assistant)\s*:\s*'
    parts = re.split(pattern, text, flags=re.IGNORECASE)

    # parts alternates: [preamble, speaker, content, speaker, content, ...]
    if len(parts) < 3:
        # No speaker tags found, treat as single block
        return [{'speaker': 'Human', 'text': text.strip()}]

    i = 1  # skip preamble
    while i + 1 < len(parts):
        speaker_raw = parts[i].strip()
        content = parts[i + 1].strip()

        if speaker_raw.lower() in ('human', 'h'):
            speaker = 'Human'
        else:
            speaker = 'Assistant'

        if content:
            blocks.append({'speaker': speaker, 'text': content})
        i += 2

    return blocks


def split_block_into_claims(text):
    """Split a speaker block into claim-level segments."""
    segments = []

    # Step 1: Split on explicit list/bullet structure
    # Numbered lists: "1) ...", "1. ...", "1: ..."
    # Bullet: "- ..."
    lines = text.split('\n')
    current = []

    for line in lines:
        stripped = line.strip()
        if not stripped:
            if current:
                segments.append(' '.join(current))
                current = []
            continue

        # Check if line starts a new list item
        is_list = bool(re.match(r'^(?:\d+[.):]|\-|\*|•)\s', stripped))
        if is_list and current:
            segments.append(' '.join(current))
            current = [stripped]
        else:
            current.append(stripped)

    if current:
        segments.append(' '.join(current))

    # Step 2: Split long segments on sentence boundaries
    refined = []
    for seg in segments:
        if len(seg) < 120:
            refined.append(seg)
            continue
        sents = split_sentences(seg)
        refined.extend(merge_short_sentences(sents))

    # Step 3: Explode comma-separated lists of facts
    exploded = []
    for seg in refined:
        sub = explode_comma_lists(seg)
        exploded.extend(sub)

    # Step 4: Split on colon-data patterns (e.g., "Result: 73 to 37")
    final = []
    for seg in exploded:
        sub = split_on_colon_data(seg)
        final.extend(sub)

    return final


def split_sentences(text):
    """Split text into sentences, keeping numbers attached."""
    # Split on period + space + capital letter
    # But NOT on abbreviations or decimals (e.g., 135.9, tok/s., etc.)
    # Strategy: split on ". " then check if next char is uppercase
    parts = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)

    # Also split on " — " (em dash used as separator)
    refined = []
    for p in parts:
        sub = re.split(r'\s+—\s+', p)
        refined.extend(sub)

    return [s.strip() for s in refined if s.strip()]


def merge_short_sentences(sentences):
    """Merge very short sentences with their neighbors."""
    if len(sentences) <= 1:
        return sentences

    merged = []
    i = 0
    while i < len(sentences):
        s = sentences[i]
        # If short (<50 chars) and not the last, merge with next
        if len(s) < 50 and i + 1 < len(sentences):
            merged.append(s + ' ' + sentences[i + 1])
            i += 2
        else:
            merged.append(s)
            i += 1

    return merged


def explode_comma_lists(text):
    """Explode comma/semicolon-separated lists of independent facts.

    Detects patterns like:
      "SiLU Easy (Mode 25 proven), No bias Easy, GQA Easy, RoPE Easy"
      "C CPU ops alone: 42.2 to 46.2 (+9.5%), slashed CPU..."

    Only explodes if 3+ items match a "Name Descriptor" pattern
    separated by commas or semicolons.
    """
    # Pattern 1: "Label Descriptor" lists
    # e.g., "SiLU Easy, No bias Easy, GQA Easy"
    # Each item: word(s) + descriptor word (Easy/Medium/Hard/proven/etc.)
    # or "Name: value" items
    label_desc = re.findall(
        r'([A-Z][A-Za-z0-9_/\- ]{1,30}?)\s+(Easy|Medium|Hard|proven|DONE|PASS|FAIL|blocked|dead|ready)',
        text, re.IGNORECASE)
    if len(label_desc) >= 3:
        items = []
        # Split on comma, keeping parenthetical context
        parts = re.split(r',\s*(?=[A-Z])', text)
        for p in parts:
            p = p.strip().rstrip('.')
            if len(p) > 5:
                items.append(p)
        if len(items) >= 3:
            return items

    # Pattern 2: "Label: data" items separated by periods or commas
    # e.g., "C CPU ops alone: 42.2 to 46.2. Cross-layer fusion: ..."
    colon_items = re.findall(r'[A-Z][^.:]{2,40}:\s*[^.]{5,}', text)
    if len(colon_items) >= 3:
        return [c.strip() for c in colon_items]

    # Pattern 3: Split on " + " separators (e.g., "RMSNorm + RoPE + GQA")
    if ' + ' in text and text.count(' + ') >= 2:
        # Only if the items around + are short labels
        parts = text.split(' + ')
        if all(len(p.strip()) < 60 for p in parts):
            return [p.strip() for p in parts if len(p.strip()) > 3]

    return [text]


def split_on_colon_data(text):
    """Split segments that contain colon-separated data points.

    E.g., "C CPU ops alone: 42.2 to 46.2 (+9.5%), slashed CPU from 2.69ms to 0.46ms"
    becomes a separate claim.

    Only split if the segment is long enough and has multiple colon-data items.
    """
    if len(text) < 100:
        return [text]

    # Split on sentence boundaries within data-heavy segments
    # "Result: 73 to 37 dispatches/token. 28.9 to 135.9 tok/s (4.7x speedup)."
    # Split on ". " followed by a number or capital
    sents = re.split(r'(?<=[.)])\s+(?=[\d(A-Z])', text)
    if len(sents) > 1:
        return merge_short_sentences([s.strip() for s in sents if s.strip()])

    # Split on ". " for long segments
    if len(text) > 200:
        sents = split_sentences(text)
        if len(sents) > 1:
            return merge_short_sentences(sents)

    return [text]


# ===================================================================
# Test / CLI
# ===================================================================

if __name__ == "__main__":
    import sys
    import os

    if len(sys.argv) > 1:
        path = sys.argv[1]
    else:
        path = os.path.expanduser(
            "~/Desktop/cowork/vault/subconscious/chunks/chunk_A1.txt")

    with open(path) as f:
        text = f.read()

    claims = split_claims(text)
    print(f"Input: {len(text)} chars")
    print(f"Output: {len(claims)} claims\n")
    for i, c in enumerate(claims):
        speaker = c['speaker']
        txt = c['text'][:100]
        print(f"  [{i:2d}] ({speaker:>9}) {txt}...")
