#!/usr/bin/env python3
"""
Integration patch for q3_spec_decode_server.py.

Adds ANE 1B drafter as a third draft source alongside n-gram and CPU 1B.
The ANE drafter runs during the GPU verify cycle on independent silicon.

To integrate, add to Engine.__init__():
    from ane_drafter import ANEDrafter
    self.ane_drafter = ANEDrafter()

And modify the generate loop to fire ANE drafts during GPU verify:

    # ── CPU K=1 + ANE K=1: both draft during verify ──
    cpu_draft = self.cpu_drafter.draft_one()
    ane_draft = self.ane_drafter.draft_one()

    # Feed both drafts + last_tok to 70B in ONE batch verify
    # [last_tok, cpu_draft, ane_draft] or tree structure
    verify_mx = mx.array([[last_tok, cpu_draft]])
    # ... standard verify logic ...

    # Sync ANE drafter state with accepted tokens
    if target_token == cpu_draft:
        self.ane_drafter.feed(cpu_draft)
        self.ane_drafter.feed(bonus)
    else:
        self.ane_drafter.correct(1, target_token)

The key insight: CPU and ANE draft SIMULTANEOUSLY during the ~94ms GPU
verify cycle. The batch verification plateau means checking both drafts
costs the same as checking one.

Timeline per cycle:
    t=0ms:    GPU starts verifying previous drafts
    t=0ms:    CPU drafts token A (10ms)
    t=0ms:    ANE drafts token B (27ms)
    t=10ms:   CPU draft A ready
    t=27ms:   ANE draft B ready
    t=94ms:   GPU verify complete
    t=94ms:   Feed [A, B] to next GPU verify batch

Both A and B are ready before GPU finishes. Free additional candidates.
"""

# This file documents the integration approach.
# The actual build requires llama_loader.py (agent building it).
# Once llama_loader.py is ready, run:
#
#   python ane_drafter.py          # Standalone benchmark
#   python spec_decode_ane_test.py # Integration test with 70B
