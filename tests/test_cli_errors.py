"""fix-restore-clean-errors + fix-mock-geometry-validation.

Both are CLI contract holes on the operator's documented happy path: a bad
session name or an impossible ``--mock`` geometry must produce one clean
``Error: ...`` line with a non-zero exit, never a Python traceback, and must
never be reported as a successful save.
"""

from __future__ import annotations

from click.testing import CliRunner

import kvsnap
from kvsnap import checkpoint
from kvsnap.backend_vllm import InMemoryBackend, make_mock_snapshot
from kvsnap.cli import cli
from kvsnap.restore import restore as do_restore
from kvsnap.verify import verify_checkpoint


def _run(*args: str):
    return CliRunner().invoke(cli, list(args))


# --- restore must fail cleanly -------------------------------------------------


def test_cli_restore_missing_session_is_clean_error(tmp_path):
    res = _run("restore", "--mock", "--session", "nope", "--dir", str(tmp_path))
    assert res.exit_code != 0
    assert "Traceback" not in res.output, res.output
    assert "Error:" in res.output
    # the message must name the session, not report checkpoint ''
    assert "nope" in res.output


def test_cli_restore_corrupt_session_is_clean_error(tmp_path):
    snap = make_mock_snapshot(num_layers=1, num_blocks_used=2)
    dest = checkpoint.save(InMemoryBackend(snap), session="c", base_dir=tmp_path)
    blob = dest / "layer_0_k.bin"
    blob.write_bytes(bytearray(blob.read_bytes())[::-1])

    res = _run("restore", "--mock", "--session", "c", "--dir", str(tmp_path))
    assert res.exit_code != 0
    assert "Traceback" not in res.output, res.output
    assert "Error:" in res.output


def test_restore_verify_error_names_the_session(tmp_path):
    try:
        do_restore(InMemoryBackend(), session="ghost", base_dir=tmp_path)
        raise AssertionError("restore of a missing session must fail")
    except kvsnap.CheckpointError as exc:
        # the session label itself, not just the path it happened to build
        assert "checkpoint 'ghost'" in str(exc)


# --- mock geometry must be rejected, not crash ---------------------------------


def test_cli_save_mock_rejects_non_positive_layers(tmp_path):
    for value in ("-1", "0"):
        res = _run("save", "--mock", "--layers", value, "--dir", str(tmp_path))
        assert res.exit_code == 2, res.output
        assert "Traceback" not in res.output, res.output
        assert "--layers" in res.output


def test_cli_save_mock_rejects_non_positive_blocks(tmp_path):
    for value in ("-1", "0"):
        res = _run("save", "--mock", "--blocks", value, "--dir", str(tmp_path))
        assert res.exit_code == 2, res.output
        assert "Traceback" not in res.output, res.output
        assert "--blocks" in res.output


def test_cli_save_mock_rejects_non_positive_kv_heads_and_head_dim(tmp_path):
    for opt in ("--kv-heads", "--head-dim"):
        res = _run("save", "--mock", opt, "0", "--dir", str(tmp_path))
        assert res.exit_code == 2, res.output
        assert opt in res.output


def test_cli_save_mock_valid_geometry_still_saves(tmp_path):
    res = _run("save", "--mock", "--layers", "2", "--blocks", "3",
               "--dir", str(tmp_path))
    assert res.exit_code == 0, res.output
    result = verify_checkpoint(tmp_path / "default")
    assert result.ok, result.errors
    # 2 layers * 2 (k, v) + block_map
    assert result.checked == 5
