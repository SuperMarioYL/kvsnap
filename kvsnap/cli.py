"""The ``kvsnap`` command-line interface.

Commands operate on the on-disk checkpoint directory so an air-gap operator can
inspect, verify, and replay checkpoints without a running engine:

* ``kvsnap save --mock``   — write a checkpoint from a fake KV cache (demo/test)
* ``kvsnap restore --mock`` — load a checkpoint into an in-memory backend (demo)
* ``kvsnap list``           — list saved sessions
* ``kvsnap info``           — show a session's manifest
* ``kvsnap verify``        — xxhash64-integrity-check a session
* ``kvsnap serve``          — wrap ``vllm serve`` with SIGUSR1 save + --restore

The ``--mock`` flag exists so the format, verify, and round-trip path can be
exercised on any machine without a GPU. Production save/restore/serve happen
in-process inside a real vLLM engine (see :mod:`serve`).
"""

from __future__ import annotations

from pathlib import Path

import click
import torch
from rich.console import Console
from rich.table import Table

from . import __version__
from . import config
from .checkpoint import save as do_save
from .restore import load, restore as do_restore
from .verify import verify_checkpoint
from .backend_vllm import InMemoryBackend, make_mock_snapshot
from .storage import load_manifest

console = Console()

# shared option defaults ------------------------------------------------------
_dir_opt = click.option("--dir", "base_dir", default=None, envvar="KVSNAP_DIR",
                         help="Checkpoint root (default ./.kvsnap).")
_session_opt = click.option("--session", default=config.DEFAULT_SESSION,
                            help="Checkpoint session name.")


def _resolve_session_dir(base_dir, session) -> Path:
    return config.session_dir(base_dir, session)


@click.group()
@click.version_option(__version__, prog_name="kvsnap")
def cli() -> None:
    """KVSnap — GLM-5.3 KV-cache 落盘恢复 for 信创 air-gap deployments."""


# --- save --------------------------------------------------------------------

@cli.command()
@_session_opt
@_dir_opt
@click.option("--mock", is_flag=True, help="Use a fake KV cache (no vLLM/GPU).")
@click.option("--model", default="glm-5.3-1m", help="Model id for the checkpoint.")
@click.option("--layers", type=int, default=4, help="Mock: decoder layers.")
@click.option("--kv-heads", type=int, default=8, help="Mock: KV heads per layer.")
@click.option("--head-dim", type=int, default=128, help="Mock: head dim.")
@click.option("--blocks", type=int, default=8, help="Mock: blocks used.")
@click.option("--dtype", default="bfloat16",
              type=click.Choice(["bfloat16", "float16", "float32"]))
def save(session, base_dir, mock, model, layers, kv_heads, head_dim, blocks, dtype):
    """Checkpoint a KV cache to disk."""
    dt = _torch_dtype(dtype)
    if mock:
        snap = make_mock_snapshot(
            model_id=model, num_layers=layers, num_kv_heads=kv_heads,
            head_dim=head_dim, num_blocks_used=blocks, dtype=dt,
        )
        backend = InMemoryBackend(snap)
        dest = do_save(backend, session=session, base_dir=base_dir)
        _print_saved(dest, snap)
        return
    # production path needs a live vLLM engine — only available via `serve`
    raise click.UsageError(
        "Production `save` runs in-process via SIGUSR1 inside `kvsnap serve`; "
        "use `kvsnap save --mock` to test the format offline."
    )


# --- restore -----------------------------------------------------------------

@cli.command()
@_session_opt
@_dir_opt
@click.option("--mock", is_flag=True, help="Load into an in-memory backend (no vLLM/GPU).")
def restore(session, base_dir, mock):
    """Load a checkpoint back into a backend (verifies first)."""
    if not mock:
        raise click.UsageError(
            "Production `restore` runs on startup via `kvsnap serve --restore "
            "<session>`; use `kvsnap restore --mock` to replay offline."
        )
    backend = InMemoryBackend()
    result = do_restore(backend, session=session, base_dir=base_dir)
    snap = backend.snapshot
    console.print(
        f"[green]✓[/] restored session [bold]{session}[/] "
        f"({snap.arch.num_layers} layers, {snap.num_blocks_used} blocks, "
        f"{snap.seq_len} tokens) — verified {result.checked} blobs"
    )


# --- list --------------------------------------------------------------------

@cli.command(name="list")
@_dir_opt
def list_cmd(base_dir):
    """List saved checkpoint sessions."""
    root = Path(base_dir) if base_dir else Path(config.DEFAULT_KVSNAP_DIR)
    if not root.is_dir():
        console.print(f"[dim]no checkpoints at {root}[/]")
        return
    table = Table(title=f"KVSnap sessions @ {root}", show_header=True)
    table.add_column("session")
    table.add_column("model", style="cyan")
    table.add_column("layers", justify="right")
    table.add_column("seq_len", justify="right")
    table.add_column("created", style="dim")
    for session_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        try:
            m = load_manifest(session_dir)
        except Exception:
            continue
        table.add_row(
            session_dir.name, m.model_id, str(m.arch.num_layers),
            str(m.seq_len), m.created_at,
        )
    console.print(table)


# --- info --------------------------------------------------------------------

@cli.command()
@_session_opt
@_dir_opt
def info(session, base_dir):
    """Show a session's manifest."""
    dest = _resolve_session_dir(base_dir, session)
    try:
        m = load_manifest(dest)
    except Exception as exc:
        raise click.ClickException(str(exc))
    table = Table(title=f"KVSnap manifest — {session}", show_header=False)
    table.add_column("k", style="bold")
    table.add_column("v")
    for k, v in m.to_dict().items():
        if k == "blobs":
            table.add_row(k, f"{len(m.blobs)} blobs")
        elif k == "arch":
            table.add_row(k, ", ".join(f"{a}={b}" for a, b in m.arch.to_dict().items()))
        else:
            table.add_row(k, str(v))
    console.print(table)


# --- verify ------------------------------------------------------------------

@cli.command()
@_session_opt
@_dir_opt
def verify(session, base_dir):
    """Integrity-check a session (xxhash64 per blob)."""
    dest = _resolve_session_dir(base_dir, session)
    result = verify_checkpoint(dest, session=session)
    if result.ok:
        console.print(
            f"[green]✓[/] {session}: verified {result.checked} blobs, all hashes match"
        )
    else:
        console.print(f"[red]✗[/] {session}: verification failed")
        for err in result.errors:
            console.print(f"  [red]-[/] {err}")
        raise click.ClickException(f"verification failed for '{session}'")


# --- serve -------------------------------------------------------------------

@cli.command(context_settings={"ignore_unknown_options": True})
@click.option("--model", required=True, help="vLLM model id/name.")
@click.option("--restore", "restore_session", default=None,
              help="Session to restore on startup (skips prefill).")
@_session_opt
@_dir_opt
@click.option("--host", default="0.0.0.0")
@click.option("--port", type=int, default=8000)
@click.option("--block-size", type=int, default=config.DEFAULT_BLOCK_SIZE)
@click.option("--dtype", default="bfloat16",
              type=click.Choice(["bfloat16", "float16", "float32"]))
@click.argument("vllm_args", nargs=-1, type=click.UNPROCESSED)
def serve(model, restore_session, session, base_dir, host, port,
          block_size, dtype, vllm_args):
    """Wrap `vllm serve` with SIGUSR1 checkpoint + --restore."""
    from .serve import serve as run_serve
    run_serve(
        model=model,
        restore_session=restore_session,
        session=session,
        base_dir=base_dir,
        host=host,
        port=port,
        block_size=block_size,
        dtype=dtype,
        extra_vllm_args=list(vllm_args),
    )


def _torch_dtype(name: str) -> torch.dtype:
    return {"bfloat16": torch.bfloat16, "float16": torch.float16,
            "float32": torch.float32}[name]


def _print_saved(dest, snap) -> None:
    size = sum(f.stat().st_size for f in dest.glob("*.bin"))
    console.print(
        f"[green]✓[/] saved session [bold]{dest.name}[/] -> {dest}\n"
        f"  model={snap.model_id}  layers={snap.arch.num_layers}  "
        f"kv_heads={snap.arch.num_kv_heads}  head_dim={snap.arch.head_dim}\n"
        f"  seq_len={snap.seq_len}  blocks={snap.num_blocks_used}  "
        f"dtype={snap.arch.dtype}  on_disk={size / 1e6:.2f} MB"
    )


def main() -> None:
    cli()


if __name__ == "__main__":
    main()
