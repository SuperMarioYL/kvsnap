"""fix-serve-module-entrypoints + fix-serve-extra-vllm-args.

The README documents ``python -m kvsnap.serve --model ...`` and
``python -m kvsnap --version`` as primary commands; both silently misbehaved
on v0.1.0 (no ``__main__`` anywhere). Also covers ``_engine_kwargs``, which
must turn trailing vLLM flags into real engine kwargs instead of dropping
them.
"""

from __future__ import annotations

import subprocess
import sys

import kvsnap
from kvsnap.serve import _coerce_scalar, _engine_kwargs


def _run(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-m", *args],
        capture_output=True,
        text=True,
        timeout=120,
    )


def test_python_dash_m_kvsnap_version():
    result = _run("kvsnap", "--version")
    assert result.returncode == 0, result.stderr
    assert kvsnap.__version__ in result.stdout


def test_python_dash_m_kvsnap_serve_help():
    result = _run("kvsnap.serve", "--help")
    assert result.returncode == 0, result.stderr
    assert "Usage" in result.stdout


def test_python_dash_m_kvsnap_serve_model_help_passthrough():
    result = _run("kvsnap.serve", "--model", "x", "--help")
    assert result.returncode == 0, result.stderr
    assert "Usage" in result.stdout


def test_engine_kwargs_converts_flags_to_kwargs():
    assert _engine_kwargs(
        ["--gpu-memory-utilization", "0.9", "--max-num-seqs", "256"]
    ) == {"gpu_memory_utilization": 0.9, "max_num_seqs": 256}


def test_engine_kwargs_bare_flag_becomes_true():
    assert _engine_kwargs(["--enable-prefix-caching"]) == {
        "enable_prefix_caching": True
    }


def test_engine_kwargs_string_value_stays_string():
    assert _engine_kwargs(["--dtype", "bfloat16"]) == {"dtype": "bfloat16"}


def test_engine_kwargs_empty_is_empty():
    assert _engine_kwargs(None) == {}
    assert _engine_kwargs([]) == {}


def test_engine_kwargs_rejects_bare_positional():
    try:
        _engine_kwargs(["oops"])
        raise AssertionError("bare positional must fail loudly")
    except ValueError as exc:
        assert "oops" in str(exc)


def test_coerce_scalar():
    assert _coerce_scalar("0.9") == 0.9
    assert _coerce_scalar("256") == 256
    assert _coerce_scalar("true") is True
    assert _coerce_scalar("FALSE") is False
    assert _coerce_scalar("bfloat16") == "bfloat16"
