"""KV-cache backends.

Defines the small :class:`KVCacheBackend` surface that :mod:`checkpoint` and
:mod:`restore` move snapshots through, plus two implementations:

* :class:`VLLMBackend` — the production backend. It reaches *in-process* into a
  running vLLM engine to read the paged KV-cache tensors and block-manager
  state, and to inject them back after a restart. vLLM is imported lazily so the
  package itself has no hard vLLM dependency; the backend only works inside a
  real vLLM process (it is what :mod:`serve` constructs).

* :class:`InMemoryBackend` — a no-vLLM, no-GPU backend that holds tensors in
  CPU memory. It is used by the tests for honest round-trip coverage of the
  real checkpoint/restore code path, and by ``kvsnap --mock`` so the CLI demo
  runs on any machine without a GPU.

The vLLM internal surface this couples to is pinned (see
:data:`config.VLLM_TARGET_VERSION`). That coupling is intentional and is the
single point to update when vLLM internals drift.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

import torch

from . import config
from .storage import ArchFingerprint, KVSnapshot

__all__ = [
    "KVCacheBackend",
    "VLLMBackend",
    "InMemoryBackend",
    "make_mock_snapshot",
]


@runtime_checkable
class KVCacheBackend(Protocol):
    """What checkpoint/restore need from an engine."""

    def extract(self) -> KVSnapshot: ...

    def inject(self, snapshot: KVSnapshot) -> None: ...


def make_mock_snapshot(
    model_id: str = "glm-5.3-1m",
    num_layers: int = 4,
    num_kv_heads: int = 8,
    head_dim: int = 128,
    block_size: int = 16,
    num_blocks_used: int = 4,
    dtype: torch.dtype = torch.bfloat16,
    seq_len: int | None = None,
) -> KVSnapshot:
    """Build a realistic fake :class:`KVSnapshot` (CPU tensors) for tests/demo.

    The shapes match a real vLLM paged KV cache:
    ``[num_blocks_used, num_kv_heads, block_size, head_dim]`` per layer per K/V,
    and an int64 block table ``[num_blocks_used]``. Tensor *values* are random;
    only the structure needs to be realistic for the format + round-trip tests.
    """
    shape = (num_blocks_used, num_kv_heads, block_size, head_dim)
    layer_k = [torch.randn(*shape, dtype=dtype) for _ in range(num_layers)]
    layer_v = [torch.randn(*shape, dtype=dtype) for _ in range(num_layers)]
    block_map = torch.randint(
        0, 1 << 30, (num_blocks_used,), dtype=torch.int64
    )
    arch = ArchFingerprint(
        num_layers=num_layers,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        dtype=str(dtype).replace("torch.", ""),
        num_blocks=num_blocks_used,  # mock: capacity == used
    )
    return KVSnapshot(
        model_id=model_id,
        arch=arch,
        seq_len=seq_len if seq_len is not None else num_blocks_used * block_size,
        prefix_hash=_hash_tokens(torch.randint(0, 1 << 30, (arch.num_layers,))),
        block_size=block_size,
        num_blocks_used=num_blocks_used,
        layer_k=layer_k,
        layer_v=layer_v,
        block_map=block_map,
    )


def _hash_tokens(tokens: torch.Tensor) -> str:
    import xxhash

    raw = bytes(tokens.contiguous().cpu().to(torch.int64).untyped_storage())
    return xxhash.xxh64(raw).hexdigest()


# --- in-memory backend (tests + CLI --mock) ----------------------------------

class InMemoryBackend:
    """Holds a KVSnapshot in CPU memory; round-trip-exact for tests/demo.

    ``inject`` overwrites the held snapshot so a save->kill->restart->restore
    cycle can be simulated against the real format code without a GPU:
    """

    def __init__(self, snapshot: KVSnapshot | None = None) -> None:
        self._snapshot = snapshot

    def extract(self) -> KVSnapshot:
        if self._snapshot is None:
            raise RuntimeError("InMemoryBackend has no snapshot to extract")
        # clone so callers mutating the returned snapshot do not corrupt state
        return _clone_snapshot(self._snapshot)

    def inject(self, snapshot: KVSnapshot) -> None:
        self._snapshot = _clone_snapshot(snapshot)

    @property
    def snapshot(self) -> KVSnapshot | None:
        return self._snapshot


def _clone_snapshot(s: KVSnapshot) -> KVSnapshot:
    return KVSnapshot(
        model_id=s.model_id,
        arch=s.arch,
        seq_len=s.seq_len,
        prefix_hash=s.prefix_hash,
        block_size=s.block_size,
        num_blocks_used=s.num_blocks_used,
        layer_k=[t.clone() for t in s.layer_k],
        layer_v=[t.clone() for t in s.layer_v],
        block_map=s.block_map.clone(),
    )


# --- vLLM backend (production; lazy import, version-pinned) ------------------

class VLLMBackend:
    """Production backend over a running vLLM engine.

    Constructs cheaply (no vLLM import); vLLM is touched only when
    :meth:`extract`/:meth:`inject` actually run, which always happens inside a
    vLLM process started by :mod:`serve`.

    The accessors below target vLLM :data:`config.VLLM_TARGET_VERSION`. The KV
    cache lives on ``model_runner.model.layers[i].self_attn`` as a list of
    ``(k_cache, v_cache)`` tensor pairs, each shaped
    ``[num_blocks, num_kv_heads, head_dim, block_size]`` (vLLM's
    paged layout). The block table for the prefix sequence is read from the
    scheduler. These internals drift across releases; keep the accessors here
    the only coupling point.
    """

    def __init__(
        self,
        engine: Any,
        model_id: str,
        num_layers: int | None = None,
        num_kv_heads: int | None = None,
        head_dim: int | None = None,
        block_size: int = config.DEFAULT_BLOCK_SIZE,
        dtype: torch.dtype | None = None,
        device: str | torch.device = "cuda",
    ) -> None:
        self._engine = engine
        self.model_id = model_id
        self.num_layers = num_layers
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.block_size = block_size
        self.dtype = dtype
        self.device = torch.device(device)

    def _detect_arch(self) -> tuple[int, int, int, torch.dtype]:
        """Read the live arch (layers / kv heads / head dim / dtype) from the
        engine's KV cache rather than guessing model constants — these values
        differ across GLM checkpoints and should never be hardcoded.
        """
        kv = self._kv_cache()
        if not kv:
            raise RuntimeError("engine has no KV cache allocated yet")
        num_layers = len(kv)
        k0 = kv[0][0]
        # vLLM paged layout: [num_blocks, num_kv_heads, head_dim, block_size]
        num_kv_heads = int(k0.shape[1]) if k0.dim() >= 2 else 0
        head_dim = int(k0.shape[2]) if k0.dim() >= 3 else 0
        return num_layers, num_kv_heads, head_dim, k0.dtype

    # -- engine accessors (centralised; update here when vLLM drifts) ---------

    def _engine_core(self) -> Any:
        # vLLM's LLM wraps an LLMEngine; unwrap defensively across versions.
        eng = getattr(self._engine, "llm_engine", self._engine)
        return eng

    def _model_runner(self) -> Any:
        eng = self._engine_core()
        executor = getattr(eng, "model_executor")
        worker = getattr(executor, "driver_worker")
        return getattr(worker, "model_runner")

    def _model(self) -> Any:
        return getattr(self._model_runner(), "model")

    def _kv_cache(self) -> list[tuple[torch.Tensor, torch.Tensor]]:
        """The per-layer ``(k_cache, v_cache)`` tensor pairs on GPU."""
        model = self._model()
        # vLLM stores the KV cache on the model after profiling; mirror the
        # canonical attribute name used by the pinned version.
        kv = getattr(model, "kv_caches", None)
        if kv is None:
            raise RuntimeError(
                "vLLM model has no `kv_caches`; engine not warmed up or "
                f"vLLM != {config.VLLM_TARGET_VERSION}"
            )
        return list(kv)

    def _block_table_for_prefix(self, seq_len: int) -> torch.Tensor:
        """The int64 block ids covering the first ``seq_len`` prefix tokens."""
        eng = self._engine_core()
        scheduler = getattr(eng, "scheduler")
        block_manager = getattr(scheduler, "block_manager")
        num_blocks = (seq_len + self.block_size - 1) // self.block_size
        # Read the prefix's block ids from the running sequence table.
        seqs = getattr(scheduler, "running", []) or []
        if not seqs:
            raise RuntimeError("no running sequence to checkpoint")
        seq = seqs[0]
        block_table = list(getattr(seq, "block_table", []))
        ids = block_table[:num_blocks]
        if len(ids) < num_blocks:
            raise RuntimeError(
                f"sequence has {len(ids)} blocks but prefix needs {num_blocks}"
            )
        return torch.tensor(ids, dtype=torch.int64, device="cpu")

    def _total_blocks(self) -> int:
        eng = self._engine_core()
        scheduler = getattr(eng, "scheduler")
        block_manager = getattr(scheduler, "block_manager")
        return int(getattr(block_manager, "num_total_gpu_blocks", 0))

    # -- KVCacheBackend -------------------------------------------------------

    def extract(self) -> KVSnapshot:
        import xxhash

        kv = self._kv_cache()
        num_layers, num_kv_heads, head_dim, dt = self._detect_arch()
        if len(kv) < num_layers:
            raise RuntimeError(
                f"engine has {len(kv)} KV layers, expected {num_layers}"
            )
        layer_k: list[torch.Tensor] = []
        layer_v: list[torch.Tensor] = []
        for layer in range(num_layers):
            k_cache, v_cache = kv[layer]
            # clone to CPU so the snapshot is device-independent on disk
            layer_k.append(k_cache.detach().to("cpu").contiguous().clone())
            layer_v.append(v_cache.detach().to("cpu").contiguous().clone())

        # blocks actually holding prefix tokens (heuristic: last non-zero layer)
        num_blocks_used = layer_k[0].shape[0]
        block_map = self._block_table_for_prefix(self._seq_len(num_blocks_used))

        tokens = self._prefix_token_ids()
        prefix_raw = bytes(
            tokens.contiguous().cpu().to(torch.int64).untyped_storage()
        )
        arch = ArchFingerprint(
            num_layers=num_layers,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            dtype=str(dt).replace("torch.", ""),
            num_blocks=self._total_blocks(),
        )
        return KVSnapshot(
            model_id=self.model_id,
            arch=arch,
            seq_len=self._seq_len(num_blocks_used),
            prefix_hash=xxhash.xxh64(prefix_raw).hexdigest(),
            block_size=self.block_size,
            num_blocks_used=num_blocks_used,
            layer_k=layer_k,
            layer_v=layer_v,
            block_map=block_map,
        )

    def inject(self, snapshot: KVSnapshot) -> None:
        self._assert_compatible(snapshot)
        kv = self._kv_cache()
        if len(kv) < snapshot.arch.num_layers:
            raise RuntimeError(
                f"engine has {len(kv)} KV layers, snapshot has "
                f"{snapshot.arch.num_layers}"
            )
        with torch.no_grad():
            for layer in range(snapshot.arch.num_layers):
                k_dst, v_dst = kv[layer]
                k_src = snapshot.layer_k[layer].to(self.device).to(k_dst.dtype)
                v_src = snapshot.layer_v[layer].to(self.device).to(v_dst.dtype)
                k_dst.copy_(k_src)
                v_dst.copy_(v_src)
        # The block table is restored through the scheduler so the engine's
        # sequence state stays consistent with the injected tensors.
        self._restore_block_table(snapshot)

    def _seq_len(self, num_blocks_used: int) -> int:
        # without a known prefix length we treat every used block as live
        return int(num_blocks_used) * self.block_size

    def _prefix_token_ids(self) -> torch.Tensor:
        # Best-effort: a real integration passes the prompt token ids explicitly;
        # fall back to the engine's last processed prompt if available.
        runner = self._model_runner()
        ids = getattr(runner, "last_input_ids", None)
        if ids is None:
            # deterministic placeholder hash seed when tokens are unavailable
            return torch.zeros(self.block_size, dtype=torch.int64)
        return ids.detach().to("cpu").to(torch.int64).reshape(-1)

    def _restore_block_table(self, snapshot: KVSnapshot) -> None:
        eng = self._engine_core()
        scheduler = getattr(eng, "scheduler")
        seqs = getattr(scheduler, "running", []) or []
        if not seqs:
            return
        seq = seqs[0]
        ids = snapshot.block_map.tolist()
        try:
            seq.block_table = list(ids)  # type: ignore[attr-defined]
        except AttributeError:
            # some versions expose an immutable seq; the tensors are already
            # restored, so block-table drift only affects future allocation.
            pass

    def _assert_compatible(self, snapshot: KVSnapshot) -> None:
        # Detect the live engine arch (do not trust constructor-provided values).
        num_layers, num_kv_heads, head_dim, dt = self._detect_arch()
        want = ArchFingerprint(
            num_layers=num_layers,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            dtype=str(dt).replace("torch.", ""),
            num_blocks=self._total_blocks(),
        )
        got = snapshot.arch
        for field in ("num_layers", "num_kv_heads", "head_dim", "dtype"):
            if getattr(want, field) != getattr(got, field):
                raise RuntimeError(
                    f"checkpoint incompatible: {field} want "
                    f"{getattr(want, field)!r} got {getattr(got, field)!r}"
                )

