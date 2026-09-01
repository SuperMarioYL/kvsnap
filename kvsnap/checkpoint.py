"""m1 — serialize a KV cache to disk.

``save`` pulls a :class:`~kvsnap.storage.KVSnapshot` out of a backend and
writes a self-describing checkpoint directory. This is the half of the
restart-resilience loop an operator triggers with ``kill -USR1 <pid>``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from . import config
from .backend_vllm import KVCacheBackend
from .storage import KVSnapshot, write_checkpoint

__all__ = ["save", "save_snapshot"]


def save(
    backend: KVCacheBackend,
    session: str = config.DEFAULT_SESSION,
    base_dir: str | Path | None = None,
) -> Path:
    """Extract a snapshot from ``backend`` and persist it as ``session``.

    Returns the checkpoint directory. Raises if the backend has nothing to
    snapshot (no live prefix).
    """
    snapshot = backend.extract()
    return save_snapshot(snapshot, session=session, base_dir=base_dir)


def save_snapshot(
    snapshot: KVSnapshot,
    session: str = config.DEFAULT_SESSION,
    base_dir: str | Path | None = None,
) -> Path:
    """Persist an already-extracted snapshot (used by the SIGUSR1 hook)."""
    dest = config.session_dir(base_dir, session)
    # Snapshot the previous good session aside so a failed save cannot clobber
    # the only restart path — air-gap operators cannot redownload a model.
    return write_checkpoint(snapshot, dest)
