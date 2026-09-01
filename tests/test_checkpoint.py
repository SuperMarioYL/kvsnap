"""Tests for checkpoint.py (m1) and restore.py (m2).

The m1 milestone is "save produces a valid checkpoint directory". The m2
milestone is "save -> kill -> restart -> restore round-trips exactly". Both are
covered against the real checkpoint/restore code path using
:class:`InMemoryBackend` (no GPU, no vLLM) so the format is exercised honestly.
"""

from __future__ import annotations

import torch
from click.testing import CliRunner

import kvsnap
from kvsnap import checkpoint
from kvsnap.restore import restore as do_restore
from kvsnap.backend_vllm import InMemoryBackend, make_mock_snapshot
from kvsnap.cli import cli


def test_m1_save_produces_valid_checkpoint(tmp_path):
    snap = make_mock_snapshot(num_layers=3, num_blocks_used=5)
    backend = InMemoryBackend(snap)
    dest = checkpoint.save(backend, session="s", base_dir=tmp_path)
    assert dest == tmp_path / "s"
    assert dest.is_dir()
    # valid = manifest present, magic correct, blob count matches arch
    manifest = kvsnap.load_manifest(dest)  # read_checkpoint loads via storage
    assert manifest.magic == "KVS1"
    assert manifest.arch.num_layers == 3
    k = [b for b in manifest.blobs if b.kind == "k"]
    v = [b for b in manifest.blobs if b.kind == "v"]
    assert len(k) == 3 and len(v) == 3
    # blobs non-empty
    assert all((dest / b.path).stat().st_size > 0 for b in manifest.blobs)


def test_m2_round_trip_save_then_restore_is_exact(tmp_path):
    """save -> read -> inject into a fresh backend keeps tensors identical."""
    original = make_mock_snapshot(num_layers=2, num_blocks_used=6)
    src_backend = InMemoryBackend(original)
    checkpoint.save(src_backend, session="rt", base_dir=tmp_path)

    # simulate process restart: brand-new backend, nothing in memory
    fresh = InMemoryBackend()
    result = do_restore(fresh, session="rt", base_dir=tmp_path)
    assert result.ok

    restored = fresh.snapshot
    assert restored is not None
    assert restored.arch == original.arch
    assert restored.seq_len == original.seq_len
    assert restored.prefix_hash == original.prefix_hash
    for a, b in zip(original.layer_k, restored.layer_k):
        assert torch.equal(a, b)
    for a, b in zip(original.layer_v, restored.layer_v):
        assert torch.equal(a, b)
    assert torch.equal(original.block_map, restored.block_map)


def test_restore_refuses_corrupt_checkpoint(tmp_path):
    """A corrupted blob must never reach the engine — restore verifies first."""
    snap = make_mock_snapshot(num_layers=1, num_blocks_used=2)
    backend = InMemoryBackend(snap)
    dest = checkpoint.save(backend, session="c", base_dir=tmp_path)

    # flip one byte in the first K blob
    k_blob = dest / "layer_0_k.bin"
    data = bytearray(k_blob.read_bytes())
    data[0] ^= 0xFF
    k_blob.write_bytes(bytes(data))

    try:
        do_restore(InMemoryBackend(), session="c", base_dir=tmp_path)
        assert False, "restore should have failed verification"
    except kvsnap.CheckpointError:
        pass


def test_cli_save_mock_writes_checkpoint(tmp_path):
    runner = CliRunner()
    res = runner.invoke(
        cli,
        ["save", "--mock", "--layers", "2", "--blocks", "4",
         "--dir", str(tmp_path)],
    )
    assert res.exit_code == 0, res.output
    assert (tmp_path / "default" / "manifest.yaml").exists()


def test_cli_restore_mock_round_trip(tmp_path):
    runner = CliRunner()
    runner.invoke(cli, ["save", "--mock", "--layers", "2", "--blocks", "4",
                        "--dir", str(tmp_path)])
    res = runner.invoke(cli, ["restore", "--mock", "--dir", str(tmp_path)])
    assert res.exit_code == 0, res.output
    assert "restored session" in res.output
