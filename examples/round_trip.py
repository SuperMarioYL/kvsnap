"""Programmatic round-trip: save a KV cache to disk and restore it.

This is the shape an integrator embeds in their own serving harness — the
``kvsnap`` CLI and ``kvsnap serve`` wrapper are thin shells over these calls.
"""

from pathlib import Path

from kvsnap import checkpoint, restore
from kvsnap.backend_vllm import InMemoryBackend, make_mock_snapshot

# 1. extract a snapshot from your engine and persist it
backend = InMemoryBackend(make_mock_snapshot(num_layers=4, num_blocks_used=8))
checkpoint.save(backend, session="agent-42", base_dir="./.kvsnap")

# 2. (process restarts here — backend memory is gone)

# 3. on a fresh engine, verify + inject, skipping re-prefill
fresh = InMemoryBackend()
restore.restore(fresh, session="agent-42", base_dir="./.kvsnap")
print("restored:", fresh.snapshot.arch.num_layers, "layers OK")
