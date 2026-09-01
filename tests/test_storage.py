"""Tests for the on-disk format (storage.py) — the testable core.

These run without a GPU or vLLM: they build small torch tensors (including
bfloat16, the real KV-cache dtype) and assert the bytes <-> tensor <-> manifest
round-trip is exact and self-describing.
"""

from __future__ import annotations

import torch
import xxhash
import yaml

import kvsnap
from kvsnap import config, storage
from kvsnap.backend_vllm import make_mock_snapshot


def _small_snapshot(dtype=torch.bfloat16, num_layers=2, num_blocks=4):
    return make_mock_snapshot(
        model_id="test-glm",
        num_layers=num_layers,
        num_kv_heads=4,
        head_dim=64,
        block_size=16,
        num_blocks_used=num_blocks,
        dtype=dtype,
    )


def test_tensor_bytes_roundtrip_all_kv_dtypes():
    """bfloat16 / float16 / float32 must survive bytes round-trip exactly."""
    for dt in (torch.float32, torch.float16, torch.bfloat16):
        a = torch.randn(3, 4, 5, dtype=dt)
        raw, shape, name = storage.tensor_to_bytes(a)
        b = storage.bytes_to_tensor(raw, shape, name)
        assert torch.equal(a, b), f"{dt} did not round-trip"
        assert name in config.SUPPORTED_DTYPES


def test_write_then_read_roundtrips_tensors(tmp_path):
    snap = _small_snapshot()
    dest = storage.write_checkpoint(snap, tmp_path / "s1")

    # manifest is a real YAML file with the magic + arch fingerprint
    mf = (dest / config.MANIFEST_NAME).read_text(encoding="utf-8")
    data = yaml.safe_load(mf)
    assert data["magic"] == config.FORMAT_MAGIC
    assert data["format_version"] == config.FORMAT_VERSION
    assert data["model_id"] == "test-glm"
    assert data["arch"]["num_layers"] == 2
    assert data["arch"]["head_dim"] == 64

    # blobs exist and are non-empty
    blob_files = sorted(dest.glob("*.bin"))
    assert len(blob_files) == 2 * 2 + 1  # 2 layers * (k+v) + block_map
    assert all(f.stat().st_size > 0 for f in blob_files)

    # tensors come back bit-identical
    back = storage.read_checkpoint(dest)
    assert back.arch == snap.arch
    assert back.seq_len == snap.seq_len
    assert back.prefix_hash == snap.prefix_hash
    for k_orig, k_back in zip(snap.layer_k, back.layer_k):
        assert torch.equal(k_orig, k_back.to(k_orig.dtype))
    for v_orig, v_back in zip(snap.layer_v, back.layer_v):
        assert torch.equal(v_orig, v_back.to(v_orig.dtype))
    assert torch.equal(snap.block_map, back.block_map)


def test_manifest_blob_xxhash_matches_disk(tmp_path):
    snap = _small_snapshot()
    dest = storage.write_checkpoint(snap, tmp_path / "s2")
    manifest = storage.load_manifest(dest)
    for ref in manifest.blobs:
        raw = (dest / ref.path).read_bytes()[ref.offset : ref.offset + ref.length]
        assert xxhash.xxh64(raw).hexdigest() == ref.xxhash64


def test_load_manifest_missing_dir_raises(tmp_path):
    try:
        storage.load_manifest(tmp_path / "nope")
        assert False, "expected CheckpointError"
    except kvsnap.CheckpointError:
        pass


def test_read_checkpoint_bad_magic_raises(tmp_path):
    snap = _small_snapshot()
    dest = storage.write_checkpoint(snap, tmp_path / "bad")
    mf = dest / config.MANIFEST_NAME
    data = yaml.safe_load(mf.read_text(encoding="utf-8"))
    data["magic"] = "XXXX"
    mf.write_text(yaml.safe_dump(data), encoding="utf-8")
    try:
        storage.read_checkpoint(dest)
        assert False, "expected CheckpointError for bad magic"
    except kvsnap.CheckpointError as exc:
        assert "bad magic" in str(exc)


def test_int64_block_map_roundtrips(tmp_path):
    """block_map is int64 (token->block id), not a float KV dtype."""
    snap = _small_snapshot()
    assert snap.block_map.dtype == torch.int64
    back = storage.read_checkpoint(storage.write_checkpoint(snap, tmp_path / "bm"))
    assert back.block_map.dtype == torch.int64
    assert torch.equal(snap.block_map, back.block_map)
