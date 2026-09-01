"""m3 — integrity verification.

Every blob carries an xxhash64 in the manifest; :func:`verify_checkpoint`
recomputes each one and validates the manifest structure so a torn write, a
disk error, or a tampered blob is caught *before* restore injects bad bytes
into the engine. This is the air-gap operator's "can I trust this checkpoint"
answer — there is no cloud to re-fetch the model from.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any

import xxhash

from . import config
from .storage import CheckpointError, load_manifest

__all__ = ["VerifyResult", "VerifyError", "verify_checkpoint", "verify_blob"]


class VerifyError(CheckpointError):
    """A checkpoint failed verification (corruption or structural drift)."""


@dataclasses.dataclass
class VerifyResult:
    ok: bool
    session: str
    checked: int
    errors: list[str]

    def to_error(self) -> VerifyError:
        return VerifyError(
            f"checkpoint '{self.session}' failed verification: "
            + "; ".join(self.errors)
        )


def verify_blob(
    path: str | Path, offset: int, length: int, expected_xxhash64: str
) -> bool:
    """Recompute the xxhash64 of a blob slice and compare to ``expected``."""
    p = Path(path)
    if not p.exists():
        return False
    raw = p.read_bytes()
    return xxhash.xxh64(raw[offset : offset + length]).hexdigest() == expected_xxhash64


def verify_checkpoint(
    dest_dir: str | Path, *, session: str = ""
) -> VerifyResult:
    """Validate the manifest and every blob's xxhash64.

    Returns a :class:`VerifyResult`; ``ok`` is False if anything is off. The
    manifest is checked first (magic, format version, blob count vs arch), then
    each blob is re-hashed — the order means a structural failure reports
    without touching disk for every blob.
    """
    dest = Path(dest_dir)
    if not dest.is_dir():
        return VerifyResult(False, session, 0, [f"{dest} is not a directory"])
    if not session:
        session = dest.name

    errors: list[str] = []

    try:
        manifest = load_manifest(dest)
    except CheckpointError as exc:
        return VerifyResult(False, session, 0, [str(exc)])

    # structural checks
    if manifest.magic != config.FORMAT_MAGIC:
        errors.append(
            f"magic mismatch: expected {config.FORMAT_MAGIC}, "
            f"got {manifest.magic}"
        )
    if manifest.format_version != config.FORMAT_VERSION:
        errors.append(
            f"format version {manifest.format_version} != "
            f"{config.FORMAT_VERSION}"
        )

    arch = manifest.arch
    # every layer needs exactly one K and one V blob
    k_layers = {b.layer for b in manifest.blobs if b.kind == "k"}
    v_layers = {b.layer for b in manifest.blobs if b.kind == "v"}
    block_maps = [b for b in manifest.blobs if b.kind == "block_map"]
    if k_layers != set(range(arch.num_layers)):
        errors.append(
            f"K blobs cover layers {sorted(k_layers)}, "
            f"expected 0..{arch.num_layers - 1}"
        )
    if v_layers != set(range(arch.num_layers)):
        errors.append(
            f"V blobs cover layers {sorted(v_layers)}, "
            f"expected 0..{arch.num_layers - 1}"
        )
    if len(block_maps) != 1:
        errors.append(
            f"expected exactly 1 block_map blob, found {len(block_maps)}"
        )

    # per-blob xxhash64 recompute — this is the corruption detector
    checked = 0
    for ref in manifest.blobs:
        blob = dest / ref.path
        if not blob.exists():
            errors.append(f"missing blob: {ref.path}")
            continue
        raw = blob.read_bytes()[ref.offset : ref.offset + ref.length]
        actual = xxhash.xxh64(raw).hexdigest()
        checked += 1
        if actual != ref.xxhash64:
            errors.append(
                f"hash mismatch for {ref.path}: "
                f"manifest {ref.xxhash64} != disk {actual}"
            )
        # size sanity
        if len(raw) != ref.length:
            errors.append(
                f"size mismatch for {ref.path}: "
                f"manifest {ref.length} != disk {len(raw)}"
            )

    ok = not errors
    return VerifyResult(ok=ok, session=session, checked=checked, errors=errors)
