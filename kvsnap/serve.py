"""m3 — the ``kvsnap serve`` vLLM wrapper.

Wraps ``vllm serve`` so the operator gets checkpoint-on-signal and
restore-on-startup for free: instead of ``vllm serve --model glm-5.3-1m`` they
run ``python -m kvsnap.serve --model glm-5.3-1m``, send ``kill -USR1 <pid>``
before maintenance to checkpoint, then restart with ``--restore default`` to
resume with the full 1M-token context intact — no prefill.

vLLM is imported lazily; this module imports cleanly without it. The real
serve path only runs inside an air-gap box with a GPU and the vLLM wheel
installed (see :data:`config.VLLM_TARGET_VERSION`).
"""

from __future__ import annotations

import os
import signal
import threading
from pathlib import Path
from typing import Any

from . import config
from .checkpoint import save as do_save
from .restore import restore as do_restore
from .backend_vllm import VLLMBackend

__all__ = ["serve"]

# A single rich console per process is enough; no shared module needed.
from rich.console import Console

console = Console()


def serve(
    model: str,
    restore_session: str | None = None,
    session: str = config.DEFAULT_SESSION,
    base_dir: str | Path | None = None,
    host: str = "0.0.0.0",
    port: int = 8000,
    block_size: int = config.DEFAULT_BLOCK_SIZE,
    dtype: str = "bfloat16",
    extra_vllm_args: list[str] | None = None,
) -> None:
    """Start a vLLM OpenAI server with KVSnap checkpoint/restore wired in.

    Flow: build engine -> (restore on startup if asked) -> register SIGUSR1
    save hook -> serve until killed. SIGUSR1 checkpoints synchronously; the
    operator waits for the hook's "checkpoint complete" log line before the
    maintenance kill.
    """
    try:
        from vllm import LLM
    except ImportError as exc:  # pragma: no cover - exercised only on GPU box
        raise SystemExit(
            "kvsnap serve needs the `vllm` extra: pip install 'kvsnap[vllm]' "
            f"(target vLLM {config.VLLM_TARGET_VERSION})"
        ) from exc

    console.print(
        f"[cyan]kvsnap[/] starting vLLM for [bold]{model}[/] "
        f"(target vLLM {config.VLLM_TARGET_VERSION})..."
    )
    llm = LLM(model=model, dtype=dtype, block_size=block_size)
    backend = VLLMBackend(
        engine=llm,
        model_id=model,
        block_size=block_size,
        dtype=_to_torch_dtype(dtype),
    )

    if restore_session:
        console.print(
            f"[green]✓[/] restoring KV cache from session "
            f"[bold]{restore_session}[/] (skipping prefill)..."
        )
        do_restore(backend, session=restore_session, base_dir=base_dir, verify=True)
        console.print("[green]✓[/] restore complete — context intact")
    else:
        console.print(
            "[dim]no --restore: starting cold (no checkpoint loaded)[/]"
        )

    _install_sigusr1_hook(backend, session=session, base_dir=base_dir)
    console.print(
        f"[cyan]kvsnap[/] SIGUSR1 hook armed — "
        f"`kill -USR1 {os.getpid()}` to checkpoint, then restart with "
        f"`--restore {session}`."
    )

    _run_openai_server(llm, host=host, port=port, extra_args=extra_vllm_args)


def _install_sigusr1_hook(
    backend: VLLMBackend,
    session: str,
    base_dir: str | Path | None,
) -> None:
    """Arm SIGUSR1 to checkpoint synchronously (one writer, no partial writes)."""

    def _on_sigusr1(signum: int, frame: Any) -> None:
        console.print(
            f"[yellow]USR1 received[/] — checkpointing session "
            f"[bold]{session}[/]..."
        )
        try:
            dest = do_save(backend, session=session, base_dir=base_dir)
            console.print(
                f"[green]✓[/] checkpoint complete: {dest}"
            )
        except Exception as exc:  # noqa: BLE001 - must not crash the server
            console.print(f"[red]✗ checkpoint failed: {exc}[/]")

    # SIGUSR1 is POSIX; unavailable on Windows (air-gap targets are Linux).
    try:
        signal.signal(signal.SIGUSR1, _on_sigusr1)
    except (AttributeError, ValueError):  # pragma: no cover
        console.print("[red]SIGUSR1 unavailable on this platform[/]")


def _run_openai_server(
    llm: Any, host: str, port: int, extra_args: list[str] | None
) -> None:
    """Launch vLLM's OpenAI API server in-process over the loaded engine."""
    try:
        import uvicorn
        from vllm.entrypoints.openai.api_server import (
            make_app,
        )
    except ImportError as exc:  # pragma: no cover - GPU box only
        raise SystemExit(
            "vLLM OpenAI server components not available; install the full "
            f"vLLM wheel (target {config.VLLM_TARGET_VERSION})"
        ) from exc

    app = make_app(llm)
    uvicorn.run(app, host=host, port=port)


def _to_torch_dtype(name: str):
    import torch

    return {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }.get(name, torch.bfloat16)
