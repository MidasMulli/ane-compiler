"""
Paged ANE Dispatcher for models exceeding the 128-op concurrent limit.

Splits ops into batches, loads each batch when needed.
Used for 8B+ models where 168 ops > 128 limit.
"""

import os
import sys
import time
import subprocess
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

PIPE_TOOL = os.path.join(os.path.dirname(__file__), 'tests', 'ane_standalone_pipe')


class PagedDispatcher:
    """Dispatcher that pages ops in batches under the kext limit.
    
    Loads batch 1 (layers 0-23 + lm_head), runs those layers.
    Unloads, loads batch 2 (layers 24-31), runs those layers.
    
    Slow reload (~30s per batch swap), but enables full 32-layer 8B.
    For a background agent processing a queue, the reload cost is amortized.
    """
    
    def __init__(self, compiled, n_layers, max_ops=125):
        self.compiled = {k: v for k, v in compiled.items()
                         if not k.startswith('_') and len(v) == 3}
        self.n_layers = n_layers
        self.max_ops = max_ops
        
        # Split into batches
        ops_per_layer = 5  # qkv, o, gate, up, down
        lm_head_ops = len([k for k in self.compiled if 'lm_head' in k])
        layers_per_batch = (max_ops - lm_head_ops) // ops_per_layer
        
        self.batches = []
        for batch_start in range(0, n_layers, layers_per_batch):
            batch_end = min(batch_start + layers_per_batch, n_layers)
            batch_ops = {}
            for k, v in self.compiled.items():
                if 'lm_head' in k:
                    batch_ops[k] = v
                elif k.startswith('L'):
                    try:
                        layer_num = int(k.split('_')[0][1:])
                        if batch_start <= layer_num < batch_end:
                            batch_ops[k] = v
                    except:
                        pass
            self.batches.append({
                'start': batch_start,
                'end': batch_end,
                'ops': batch_ops,
            })
        
        self.current_batch = -1
        self.dispatcher = None
        
    def _load_batch(self, batch_idx):
        """Load a specific batch of ops."""
        if self.current_batch == batch_idx:
            return
        
        if self.dispatcher:
            self.dispatcher.stop()
        
        from generate import ANEDispatcher
        batch = self.batches[batch_idx]
        self.dispatcher = ANEDispatcher(batch['ops'], quiet=True)
        self.dispatcher.start()
        self.current_batch = batch_idx
        
    def dispatch(self, op_name, input_fp16):
        """Dispatch an op, loading the right batch if needed."""
        # Find which batch has this op
        if self.dispatcher and op_name in self.batches[self.current_batch]['ops']:
            return self.dispatcher.dispatch(op_name, input_fp16)
        
        # Need to find and load the right batch
        for i, batch in enumerate(self.batches):
            if op_name in batch['ops']:
                self._load_batch(i)
                return self.dispatcher.dispatch(op_name, input_fp16)
        
        raise KeyError(f"Op {op_name} not found in any batch")
    
    def ensure_batch_for_layer(self, layer_idx):
        """Pre-load the batch containing a specific layer."""
        for i, batch in enumerate(self.batches):
            if batch['start'] <= layer_idx < batch['end']:
                self._load_batch(i)
                return
    
    def stop(self):
        if self.dispatcher:
            self.dispatcher.stop()
            self.dispatcher = None
    
    @property
    def n_batches(self):
        return len(self.batches)
    
    def batch_info(self):
        return [(b['start'], b['end'], len(b['ops'])) for b in self.batches]
