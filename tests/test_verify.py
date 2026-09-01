"""Tests for verify.py (m3) — integrity & corruption detection.

xxhash64 per blob, manifest validation, and tamper detection. These are the
air-gap operator's "can I trust this checkpoint" guardrails, so they must catch:
a flipped byte, a missing blob, a rewritten manifest magic, and a missing K/V
layer.
"""

from __future__ import annotations

import yaml

import kvsnap
from kvsnap import checkpoint, verify
from kvsnap.backend_vllm import InMemoryBackend, make_mock_snapshot
from kvsnap.storage import write_checkpoint


def _save(tmp_path, **kw):
    snap = make_mock_snapshot(num_layers=2, num_blocks_used=3, **kw)
    return write_checkpoint(snap, tmp_path / "v"), snap


def test_verify_passes_on_clean_checkpoint(tmp_path):
    dest, _ = _save(tmp_path)
    result = verify.verify_checkpoint(dest)
    assert result.ok
    # 2 layers * 2 (k,v) + 1 block_map
    assert result.checked == 2 * 2 + 1
    assert result.errors == []


def test_verify_detects_flipped_byte(tmp_path):
    dest, _ = _save(tmp_path)
    blob = dest / "layer_1_v.bin"
    data = bytearray(blob.read_bytes())
    data[5] ^= 0xFF
    blob.write_bytes(bytes(data))
    result = verify.verify_checkpoint(dest)
    assert not result.ok
    assert any("hash mismatch" in e for e in result.errors)


def test_verify_detects_missing_blob(tmp_path):
    dest, _ = _save(tmp_path)
    (dest / "layer_0_k.bin").unlink()
    result = verify.verify_checkpoint(dest)
    assert not result.ok
    assert any("missing blob" in e for e in result.errors)


def test_verify_detects_bad_magic(tmp_path):
    dest, _ = _save(tmp_path)
    mf = dest / "manifest.yaml"
    data = yaml.safe_load(mf.read_text(encoding="utf-8"))
    data["magic"] = "BAD1"
    mf.write_text(yaml.safe_dump(data), encoding="utf-8")
    result = verify.verify_checkpoint(dest)
    assert not result.ok
    assert any("magic mismatch" in e for e in result.errors)


def test_verify_detects_truncated_blob(tmp_path):
    dest, _ = _save(tmp_path)
    blob = dest / "layer_0_k.bin"
    blob.write_bytes(blob.read_bytes()[:-4])  # shrink by 4 bytes
    result = verify.verify_checkpoint(dest)
    assert not result.ok
    # both a size mismatch and a hash mismatch are legitimate signals
    assert any("mismatch" in e for e in result.errors)


def test_verify_blob_helper(tmp_path):
    dest, snap = _save(tmp_path)
    import xxhash
    ref = next(b for b in kvsnap.load_manifest(dest).blobs if b.kind == "k")
    raw = (dest / ref.path).read_bytes()
    assert verify.verify_blob(dest / ref.path, ref.offset, ref.length, ref.xxhash64)
    bad = xxhash.xxh64(raw[:-1]).hexdigest()
    assert not verify.verify_blob(dest / ref.path, ref.offset, ref.length, bad)


def test_cli_verify_command_exit_codes(tmp_path):
    from click.testing import CliRunner
    from kvsnap.cli import cli

    runner = CliRunner()
    dest, _ = _save(tmp_path)
    # clean → exit 0 (session is named "v", the dir _save wrote)
    res = runner.invoke(cli, ["verify", "--session", "v", "--dir", str(tmp_path)])
    assert res.exit_code == 0, res.output
    # corrupt → non-zero
    blob = dest / "layer_0_v.bin"
    blob.write_bytes(bytearray(blob.read_bytes()[::-1]))
    res = runner.invoke(cli, ["verify", "--session", "v", "--dir", str(tmp_path)])
    assert res.exit_code != 0
