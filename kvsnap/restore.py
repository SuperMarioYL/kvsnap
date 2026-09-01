"""m2 — deserialize a KV cache from disk and inject it into a fresh engine.

``restore`` is the other half of the loop: on restart the operator boots a
fresh vLLM process (or, for testing, a fresh :class:`InMemoryBackend`) and
``restore`` loads the checkpoint from disk, verifies it, and pushes the
tensors back into the engine's KV-cache slots so the prefix does not have to be
re-prefilled.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from . import config
from .backend_vllm import KVCacheBackend
from .storage import KVSnapshot, read_checkpoint
from .verify import VerifyResult, verify_checkpoint

__all__ = ["restore", "load"]


def load(
    session: str = config.DEFAULT_SESSION,
    base_dir: str | Path | None = None,
) -> KVSnapshot:
    """Read a checkpoint into a CPU :class:`KVSnapshot` without injecting.

    Useful for ``kvsnap restore --mock`` (round-trip demo) and inspection.
    """
    return read_checkpoint(config.session_dir(base_dir, session))


def restore(
    backend: KVCacheBackend,
    session: str = config.DEFAULT_SESSION,
    base_dir: str | Path | None = None,
    verify: bool = True,
) -> VerifyResult:
    """Verify then inject ``session`` into ``backend``.

    Verification runs before injection so a corrupted checkpoint never reaches
    the engine — a single bad blob would otherwise silently misalign attention.
    Returns the :class:`VerifyResult` so ``serve`` can log the outcome.
    """
    dest = config.session_dir(base_dir, session)
    if verify:
        result = verify_checkpoint(dest)
        if not result.ok:
            raise result.to_error()
    snapshot = read_checkpoint(dest)
    backend.inject(snapshot)
    return VerifyResult(
        ok=True, session=session, checked=len(snapshot.layer_k) * 2 + 1, errors=[]
    )
