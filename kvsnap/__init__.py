"""KVSnap — GLM-5.3 1M-token KV-cache checkpoint & restore for 信创 air-gap.

The package owns a portable on-disk KV-cache checkpoint format and the
save/restore/verify pipeline around it. The real vLLM integration
(:mod:`kvsnap.serve`, :class:`kvsnap.backend_vllm.VLLMBackend`) imports vLLM
lazily; the format core (:mod:`kvsnap.storage`, :mod:`kvsnap.verify`,
:mod:`kvsnap.checkpoint`, :mod:`kvsnap.restore`) runs without a GPU.
"""

from __future__ import annotations

from . import config
from .backend_vllm import InMemoryBackend, VLLMBackend, make_mock_snapshot
from .checkpoint import save, save_snapshot
from .restore import load
from .storage import (
    ArchFingerprint,
    BlobRef,
    CheckpointError,
    KVSnapshot,
    Manifest,
    load_manifest,
    read_checkpoint,
    write_checkpoint,
)
from .verify import VerifyError, VerifyResult, verify_checkpoint

__version__ = config.VERSION

# Note: the restore() callable lives in kvsnap.restore (the module), reached as
# `from kvsnap import restore; restore.restore(...)` or `from kvsnap.restore
# import restore`. We deliberately do NOT re-bind a top-level `restore` here,
# because that would shadow the `kvsnap.restore` submodule for
# `from kvsnap import restore`.

__all__ = [
    "__version__",
    "config",
    "save",
    "save_snapshot",
    "load",
    "read_checkpoint",
    "write_checkpoint",
    "load_manifest",
    "verify_checkpoint",
    "VerifyResult",
    "VerifyError",
    "KVSnapshot",
    "Manifest",
    "ArchFingerprint",
    "BlobRef",
    "CheckpointError",
    "VLLMBackend",
    "InMemoryBackend",
    "make_mock_snapshot",
]
