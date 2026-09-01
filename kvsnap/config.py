"""Defaults, paths, and constants for KVSnap.

Centralises the on-disk format magic, default session layout, and vLLM
compatibility surface so the rest of the package stays free of magic values.
"""

from __future__ import annotations

from pathlib import Path

__all__ = [
    "VERSION",
    "FORMAT_MAGIC",
    "FORMAT_VERSION",
    "DEFAULT_KVSNAP_DIR",
    "DEFAULT_SESSION",
    "DEFAULT_BLOCK_SIZE",
    "MANIFEST_NAME",
    "BLOCK_MAP_NAME",
    "BLOB_GLOB",
    "SUPPORTED_DTYPES",
    "session_dir",
    "manifest_path",
    "block_map_path",
    "VLLM_TARGET_VERSION",
]

#: Semantic version — single source for the release tag (mirrors VERSION file).
VERSION = "0.1.0"

#: On-disk format identifier. Bump when the layout is intentionally breaking.
FORMAT_MAGIC = "KVS1"
FORMAT_VERSION = 1

#: Default checkpoint root (relative to the vLLM process cwd unless overridden).
DEFAULT_KVSNAP_DIR = ".kvsnap"
DEFAULT_SESSION = "default"

#: vLLM paged-attention block size (tokens per block). GLM-5.3 on vLLM uses 16.
DEFAULT_BLOCK_SIZE = 16

MANIFEST_NAME = "manifest.yaml"
BLOCK_MAP_NAME = "block_map.bin"
BLOB_GLOB = "*.bin"

#: torch dtypes we can losslessly (de)serialize. KV cache is typically bfloat16.
SUPPORTED_DTYPES = ("float32", "float16", "bfloat16")

#: The vLLM internal surface this backend is pinned to. vLLM internals drift
#: across releases; ``serve.py`` refuses to run against an unpinned engine and
#: prints this constant so the operator knows which wheel to install.
VLLM_TARGET_VERSION = "0.6.x"


def session_dir(base: str | Path | None, session: str) -> Path:
    """Resolve a session's checkpoint directory under ``base``.

    ``base`` defaults to ``./.kvsnap`` so checkpoints land next to the process
    that produced them — the air-gap operator's working directory.
    """
    root = Path(base) if base else Path(DEFAULT_KVSNAP_DIR)
    return root / session


def manifest_path(base: str | Path | None, session: str) -> Path:
    return session_dir(base, session) / MANIFEST_NAME


def block_map_path(base: str | Path | None, session: str) -> Path:
    return session_dir(base, session) / BLOCK_MAP_NAME
