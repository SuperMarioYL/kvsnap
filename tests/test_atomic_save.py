"""fix-atomic-save — a failed save must never clobber the previous good checkpoint.

Reproduces the v0.1.0 defect red: ``write_checkpoint`` wrote blobs directly
into the live session dir, so a save dying mid-write left the session with a
torn mix of old manifest + new partial blobs. The atomic temp-dir + swap must
keep the previous checkpoint fully verifiable through any failure.
"""

from __future__ import annotations

import torch

from kvsnap import storage
from kvsnap.backend_vllm import make_mock_snapshot
from kvsnap.verify import verify_checkpoint


def _small_snapshot(tag: float, num_layers: int = 2):
    snap = make_mock_snapshot(
        model_id="test-glm",
        num_layers=num_layers,
        num_kv_heads=4,
        head_dim=64,
        block_size=16,
        num_blocks_used=4,
        dtype=torch.bfloat16,
    )
    # make the content identifiable so we can prove which snapshot survived
    snap.layer_k[0][0, 0, 0, 0] = tag
    return snap


def test_failed_save_preserves_previous_checkpoint(tmp_path, monkeypatch):
    dest = tmp_path / "default"
    good = _small_snapshot(tag=1.0)
    storage.write_checkpoint(good, dest)
    assert verify_checkpoint(dest, session="default").ok

    calls = {"n": 0}
    real_write_blob = storage._write_blob

    def flaky_write_blob(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] == 3:  # mid-write: past the K blobs, before the manifest
            raise OSError("disk full (injected)")
        return real_write_blob(*args, **kwargs)

    monkeypatch.setattr(storage, "_write_blob", flaky_write_blob)
    try:
        storage.write_checkpoint(_small_snapshot(tag=2.0), dest)
        raise AssertionError("injected OSError did not propagate")
    except OSError:
        pass

    # the live session is still the previous good checkpoint, fully verifiable
    result = verify_checkpoint(dest, session="default")
    assert result.ok, f"previous checkpoint corrupted: {result.errors}"
    reloaded = storage.read_checkpoint(dest)
    assert torch.equal(reloaded.layer_k[0], good.layer_k[0]), (
        "session holds the NEW (partial) snapshot instead of the old good one"
    )


def test_failed_save_leaves_no_partial_directories(tmp_path, monkeypatch):
    dest = tmp_path / "default"
    storage.write_checkpoint(_small_snapshot(tag=1.0), dest)

    real_write_blob = storage._write_blob

    def flaky_write_blob(dest_dir, tensor, kind, layer):
        if kind == "v":
            raise OSError("disk full (injected)")
        return real_write_blob(dest_dir, tensor, kind, layer)

    monkeypatch.setattr(storage, "_write_blob", flaky_write_blob)
    try:
        storage.write_checkpoint(_small_snapshot(tag=2.0), dest)
        raise AssertionError("injected OSError did not propagate")
    except OSError:
        pass

    siblings = sorted(p.name for p in tmp_path.iterdir())
    assert siblings == ["default"], (
        f"partial directories left behind: {siblings}"
    )


def test_successful_resave_swaps_cleanly(tmp_path):
    dest = tmp_path / "default"
    storage.write_checkpoint(_small_snapshot(tag=1.0), dest)
    newer = _small_snapshot(tag=2.0)
    storage.write_checkpoint(newer, dest)

    result = verify_checkpoint(dest, session="default")
    assert result.ok, f"re-saved checkpoint failed verification: {result.errors}"
    reloaded = storage.read_checkpoint(dest)
    assert torch.equal(reloaded.layer_k[0], newer.layer_k[0])

    siblings = sorted(p.name for p in tmp_path.iterdir())
    assert siblings == ["default"], (
        f"stale backup/temp dirs left behind: {siblings}"
    )
