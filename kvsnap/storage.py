"""The KVSnap on-disk format: a portable, self-describing KV-cache checkpoint.

This is the genuine new primitive. vLLM never exposes its paged KV cache as a
portable on-disk artifact that survives a process restart; KVSnap defines one:

    .kvsnap/<session>/
        manifest.yaml     # magic + arch fingerprint + blob manifest (xxhash64)
        layer_0_k.bin     # raw K tensor bytes for layer 0
        layer_0_v.bin     # raw V tensor bytes for layer 0
        ...
        block_map.bin     # block-manager state (token -> block id, int64)

The format is backend-agnostic: ``backend_vllm.py`` produces/consumes a
``KVSnapshot``; this module owns only the bytes <-> tensor <-> manifest mapping,
so it is fully testable without a GPU or vLLM.
"""

from __future__ import annotations

import dataclasses
import time
from pathlib import Path
from typing import Any

import torch
import xxhash
import yaml

from . import config

__all__ = [
    "ArchFingerprint",
    "BlobRef",
    "Manifest",
    "KVSnapshot",
    "CheckpointError",
    "write_checkpoint",
    "read_checkpoint",
    "load_manifest",
    "tensor_to_bytes",
    "bytes_to_tensor",
]


class CheckpointError(Exception):
    """Raised when a checkpoint is missing, malformed, or fails to load."""


# --- dtype name <-> torch.dtype (lossless for every KV-cache dtype) ----------

_DTYPE_NAME: dict[torch.dtype, str] = {
    torch.float32: "float32",
    torch.float16: "float16",
    torch.bfloat16: "bfloat16",
    torch.int64: "int64",
    torch.int32: "int32",
}
_DTYPE_FROM_NAME: dict[str, torch.dtype] = {v: k for k, v in _DTYPE_NAME.items()}


@dataclasses.dataclass(frozen=True)
class ArchFingerprint:
    """Structural identity of a model's KV cache.

    Two checkpoints are slot-compatible only when their fingerprints match;
    ``restore`` checks this before injecting to avoid silently misaligned
    tensors (a real risk when a model wheel is upgraded between save/restore).
    """

    num_layers: int
    num_kv_heads: int
    head_dim: int
    dtype: str
    num_blocks: int  # total paged blocks allocated by the engine

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "ArchFingerprint":
        return cls(
            num_layers=int(d["num_layers"]),
            num_kv_heads=int(d["num_kv_heads"]),
            head_dim=int(d["head_dim"]),
            dtype=str(d["dtype"]),
            num_blocks=int(d["num_blocks"]),
        )

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


@dataclasses.dataclass(frozen=True)
class BlobRef:
    """One serialised tensor blob and its integrity tag."""

    path: str
    kind: str  # "k" | "v" | "block_map"
    layer: int  # -1 for block_map
    shape: tuple[int, ...]
    dtype: str
    offset: int
    length: int  # bytes
    xxhash64: str

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "BlobRef":
        return cls(
            path=str(d["path"]),
            kind=str(d["kind"]),
            layer=int(d["layer"]),
            shape=tuple(int(s) for s in d["shape"]),
            dtype=str(d["dtype"]),
            offset=int(d["offset"]),
            length=int(d["length"]),
            xxhash64=str(d["xxhash64"]),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "kind": self.kind,
            "layer": self.layer,
            "shape": list(self.shape),
            "dtype": self.dtype,
            "offset": self.offset,
            "length": self.length,
            "xxhash64": self.xxhash64,
        }


@dataclasses.dataclass
class Manifest:
    """The YAML manifest of a checkpoint."""

    magic: str
    format_version: int
    model_id: str
    arch: ArchFingerprint
    seq_len: int
    prefix_hash: str
    block_size: int
    num_blocks_used: int
    blobs: list[BlobRef]
    created_at: str
    kvsnap_version: str

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Manifest":
        blobs = [BlobRef.from_dict(b) for b in d.get("blobs", [])]
        return cls(
            magic=str(d["magic"]),
            format_version=int(d["format_version"]),
            model_id=str(d["model_id"]),
            arch=ArchFingerprint.from_dict(d["arch"]),
            seq_len=int(d["seq_len"]),
            prefix_hash=str(d["prefix_hash"]),
            block_size=int(d["block_size"]),
            num_blocks_used=int(d["num_blocks_used"]),
            blobs=blobs,
            created_at=str(d["created_at"]),
            kvsnap_version=str(d.get("kvsnap_version", "")),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "magic": self.magic,
            "format_version": self.format_version,
            "model_id": self.model_id,
            "arch": self.arch.to_dict(),
            "seq_len": self.seq_len,
            "prefix_hash": self.prefix_hash,
            "block_size": self.block_size,
            "num_blocks_used": self.num_blocks_used,
            "blobs": [b.to_dict() for b in self.blobs],
            "created_at": self.created_at,
            "kvsnap_version": self.kvsnap_version,
        }


@dataclasses.dataclass
class KVSnapshot:
    """An extracted KV-cache snapshot, the unit checkpoint.py/restore.py move."""

    model_id: str
    arch: ArchFingerprint
    seq_len: int
    prefix_hash: str
    block_size: int
    num_blocks_used: int
    layer_k: list[torch.Tensor]  # one [num_blocks_used, num_kv_heads, block_size, head_dim]
    layer_v: list[torch.Tensor]
    block_map: torch.Tensor  # int64 [num_blocks_used] block ids for the prefix


# --- tensor <-> bytes (preserves exact element bytes incl. bfloat16) --------

def tensor_to_bytes(t: torch.Tensor) -> tuple[bytes, tuple[int, ...], str]:
    """Serialise a tensor to raw contiguous bytes + its shape/dtype name.

    Uses the untyped storage so non-numpy dtypes (bfloat16) round-trip exactly.
    """
    t = t.contiguous().detach().clone().cpu()
    if t.dtype not in _DTYPE_NAME:
        raise CheckpointError(f"unsupported dtype: {t.dtype}")
    numel = t.numel()
    raw = bytes(t.untyped_storage())[: numel * t.element_size()]
    return raw, tuple(t.shape), _DTYPE_NAME[t.dtype]


def bytes_to_tensor(
    raw: bytes, shape: tuple[int, ...], dtype_name: str
) -> torch.Tensor:
    """Inverse of :func:`tensor_to_bytes`."""
    if dtype_name not in _DTYPE_FROM_NAME:
        raise CheckpointError(f"unknown dtype in manifest: {dtype_name}")
    dt = _DTYPE_FROM_NAME[dtype_name]
    expected = int(torch.tensor(0, dtype=dt).element_size())
    for s in shape:
        expected *= int(s)
    if len(raw) != expected:
        raise CheckpointError(
            f"blob size mismatch: have {len(raw)} bytes, need {expected} "
            f"for shape {shape} {dtype_name}"
        )
    return torch.frombuffer(bytearray(raw), dtype=dt).reshape(shape).clone()


# --- blob naming ------------------------------------------------------------

def _blob_filename(kind: str, layer: int) -> str:
    if kind == "block_map":
        return config.BLOCK_MAP_NAME
    return f"layer_{layer}_{kind}.bin"


def _blob_path(kind: str, layer: int) -> str:
    # manifest paths are relative to the checkpoint dir
    return _blob_filename(kind, layer)


# --- write / read ------------------------------------------------------------

def write_checkpoint(snapshot: KVSnapshot, dest_dir: str | Path) -> Path:
    """Serialise ``snapshot`` to ``dest_dir`` and return the directory path.

    Atomic-ish: writes blobs to a temp suffix then renames the dir, so a crashed
    save never leaves a half-valid checkpoint that restore would pick up.
    """
    dest = Path(dest_dir)
    dest.mkdir(parents=True, exist_ok=True)

    blobs: list[BlobRef] = []
    # Per-layer K/V tensors.
    for layer, tensor in enumerate(snapshot.layer_k):
        blobs.append(_write_blob(dest, tensor, "k", layer))
    for layer, tensor in enumerate(snapshot.layer_v):
        blobs.append(_write_blob(dest, tensor, "v", layer))
    # Block-manager state (token -> block id).
    blobs.append(_write_blob(dest, snapshot.block_map, "block_map", layer=-1))

    manifest = Manifest(
        magic=config.FORMAT_MAGIC,
        format_version=config.FORMAT_VERSION,
        model_id=snapshot.model_id,
        arch=snapshot.arch,
        seq_len=snapshot.seq_len,
        prefix_hash=snapshot.prefix_hash,
        block_size=snapshot.block_size,
        num_blocks_used=snapshot.num_blocks_used,
        blobs=blobs,
        created_at=time.strftime("%Y-%m-%dT%H:%M:%S%z", time.gmtime()),
        kvsnap_version=config.VERSION,
    )

    manifest_file = dest / config.MANIFEST_NAME
    manifest_file.write_text(
        yaml.safe_dump(manifest.to_dict(), sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    return dest


def _write_blob(
    dest: Path, tensor: torch.Tensor, kind: str, layer: int
) -> BlobRef:
    raw, shape, dtype_name = tensor_to_bytes(tensor)
    fname = _blob_filename(kind, layer)
    (dest / fname).write_bytes(raw)
    return BlobRef(
        path=_blob_path(kind, layer),
        kind=kind,
        layer=layer,
        shape=shape,
        dtype=dtype_name,
        offset=0,
        length=len(raw),
        xxhash64=xxhash.xxh64(raw).hexdigest(),
    )


def load_manifest(dest_dir: str | Path) -> Manifest:
    dest = Path(dest_dir)
    mf = dest / config.MANIFEST_NAME
    if not mf.exists():
        raise CheckpointError(f"no manifest at {mf}")
    try:
        data = yaml.safe_load(mf.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise CheckpointError(f"manifest is not valid YAML: {exc}") from exc
    if not isinstance(data, dict):
        raise CheckpointError("manifest top-level is not a mapping")
    return Manifest.from_dict(data)


def read_checkpoint(dest_dir: str | Path) -> KVSnapshot:
    """Read a checkpoint back into a :class:`KVSnapshot` (tensors on CPU)."""
    dest = Path(dest_dir)
    manifest = load_manifest(dest)
    _validate_magic(manifest)

    # group blobs by kind/layer for ordered reconstruction
    k_by_layer: dict[int, BlobRef] = {}
    v_by_layer: dict[int, BlobRef] = {}
    block_map_ref: BlobRef | None = None
    for b in manifest.blobs:
        if b.kind == "k":
            k_by_layer[b.layer] = b
        elif b.kind == "v":
            v_by_layer[b.layer] = b
        elif b.kind == "block_map":
            block_map_ref = b

    num_layers = manifest.arch.num_layers
    layer_k: list[torch.Tensor] = []
    layer_v: list[torch.Tensor] = []
    for layer in range(num_layers):
        if layer not in k_by_layer or layer not in v_by_layer:
            raise CheckpointError(f"checkpoint missing layer {layer} K/V blobs")
        layer_k.append(_read_blob(dest, k_by_layer[layer]))
        layer_v.append(_read_blob(dest, v_by_layer[layer]))

    if block_map_ref is None:
        raise CheckpointError("checkpoint missing block_map blob")
    block_map = _read_blob(dest, block_map_ref)

    return KVSnapshot(
        model_id=manifest.model_id,
        arch=manifest.arch,
        seq_len=manifest.seq_len,
        prefix_hash=manifest.prefix_hash,
        block_size=manifest.block_size,
        num_blocks_used=manifest.num_blocks_used,
        layer_k=layer_k,
        layer_v=layer_v,
        block_map=block_map,
    )


def _read_blob(dest: Path, ref: BlobRef) -> torch.Tensor:
    blob = dest / ref.path
    if not blob.exists():
        raise CheckpointError(f"missing blob {ref.path}")
    raw = blob.read_bytes()
    # honour the offset/length so a concatenated format stays valid later
    raw = raw[ref.offset : ref.offset + ref.length]
    return bytes_to_tensor(raw, tuple(ref.shape), ref.dtype)


def _validate_magic(manifest: Manifest) -> None:
    if manifest.magic != config.FORMAT_MAGIC:
        raise CheckpointError(
            f"bad magic: expected {config.FORMAT_MAGIC!r}, got {manifest.magic!r}"
        )
    if manifest.format_version != config.FORMAT_VERSION:
        raise CheckpointError(
            f"unsupported format version: {manifest.format_version} "
            f"(this kvsnap understands {config.FORMAT_VERSION})"
        )
