"""fix-extract-used-blocks — snapshot only the prefix's used blocks, not the pool.

Uses a fake engine mimicking the pinned vLLM surface (no vLLM import, no GPU):
``engine.llm_engine.scheduler.running[0].get_seqs()[0].block_table`` for the
prefix, ``...model_executor.driver_worker.model_runner.model.kv_caches`` for
the paged tensors. Reproduces the v0.1.0 defect red: a 3-of-8-block prefix
raised ``sequence has 3 blocks but prefix needs 8`` and serialised the whole
pool.
"""

from __future__ import annotations

import torch

from kvsnap.backend_vllm import VLLMBackend

NUM_KV_HEADS = 4
HEAD_DIM = 64
BLOCK_SIZE = 16
NUM_LAYERS = 2


class _Seq:
    def __init__(self, block_table: list[int]) -> None:
        self.block_table = list(block_table)

    def get_len(self) -> int:
        return len(self.block_table) * BLOCK_SIZE


class _SeqGroup:
    def __init__(self, seq: _Seq) -> None:
        self._seq = seq

    def get_seqs(self) -> list[_Seq]:
        return [self._seq]


class _BlockManager:
    def __init__(self, num_total_gpu_blocks: int) -> None:
        self.num_total_gpu_blocks = num_total_gpu_blocks


class _Scheduler:
    def __init__(self, seq: _Seq, num_total_gpu_blocks: int) -> None:
        self.running = [_SeqGroup(seq)]
        self.block_manager = _BlockManager(num_total_gpu_blocks)


class _Model:
    def __init__(self, kv: list[tuple[torch.Tensor, torch.Tensor]]) -> None:
        self.kv_caches = kv


class _ModelRunner:
    def __init__(self, model: _Model) -> None:
        self.model = model
        self.last_input_ids = torch.arange(
            len("prefix-token-ids") * 8, dtype=torch.int64
        )


class _Worker:
    def __init__(self, model_runner: _ModelRunner) -> None:
        self.model_runner = model_runner


class _Executor:
    def __init__(self, worker: _Worker) -> None:
        self.driver_worker = worker


class _EngineCore:
    def __init__(self, scheduler: _Scheduler, executor: _Executor) -> None:
        self.scheduler = scheduler
        self.model_executor = executor


class FakeEngine:
    """The vLLM engine surface VLLMBackend couples to, on CPU tensors."""

    def __init__(
        self,
        num_pool_blocks: int,
        block_table: list[int],
        seed: int = 0,
    ) -> None:
        gen = torch.Generator().manual_seed(seed)
        kv = []
        for _ in range(NUM_LAYERS):
            shape = (num_pool_blocks, NUM_KV_HEADS, BLOCK_SIZE, HEAD_DIM)
            k = torch.randn(*shape, dtype=torch.bfloat16, generator=gen)
            v = torch.randn(*shape, dtype=torch.bfloat16, generator=gen)
            kv.append((k, v))
        seq = _Seq(block_table)
        self.llm_engine = _EngineCore(
            _Scheduler(seq, num_pool_blocks),
            _Executor(_Worker(_ModelRunner(_Model(kv)))),
        )

    @property
    def kv(self) -> list[tuple[torch.Tensor, torch.Tensor]]:
        return self.llm_engine.model_executor.driver_worker.model_runner.model.kv_caches


def _backend(engine: FakeEngine) -> VLLMBackend:
    return VLLMBackend(
        engine=engine,
        model_id="test-glm",
        block_size=BLOCK_SIZE,
        dtype=torch.bfloat16,
        device="cpu",
    )


def test_extract_snapshots_only_used_blocks():
    engine = FakeEngine(num_pool_blocks=8, block_table=[0, 1, 2], seed=7)

    snapshot = _backend(engine).extract()

    assert snapshot.num_blocks_used == 3
    assert snapshot.seq_len == 3 * BLOCK_SIZE
    assert snapshot.block_map.tolist() == [0, 1, 2]
    for layer_k in snapshot.layer_k:
        assert layer_k.shape[0] == 3, "snapshot carried the whole pool"
    for layer_v in snapshot.layer_v:
        assert layer_v.shape[0] == 3
    # and the snapshot content is exactly the sequence's blocks
    for layer, (k_cache, _v_cache) in enumerate(engine.kv):
        assert torch.equal(snapshot.layer_k[layer], k_cache[:3])


def test_extract_handles_scattered_block_tables():
    engine = FakeEngine(num_pool_blocks=8, block_table=[5, 2, 7], seed=9)

    snapshot = _backend(engine).extract()

    assert snapshot.block_map.tolist() == [5, 2, 7]
    for layer, (k_cache, _v_cache) in enumerate(engine.kv):
        expected = k_cache.index_select(
            0, torch.tensor([5, 2, 7], dtype=torch.int64)
        )
        assert torch.equal(snapshot.layer_k[layer], expected)


def test_round_trip_into_differently_sized_pool():
    source = FakeEngine(num_pool_blocks=8, block_table=[0, 1, 2], seed=11)
    snapshot = _backend(source).extract()

    target = FakeEngine(num_pool_blocks=5, block_table=[0, 0, 0], seed=99)
    _backend(target).inject(snapshot)

    for layer, (k_cache, v_cache) in enumerate(target.kv):
        assert torch.equal(k_cache[:3], snapshot.layer_k[layer])
        assert torch.equal(v_cache[:3], snapshot.layer_v[layer])
    # the target sequence's block table was restored to the snapshot's ids
    seq = target.llm_engine.scheduler.running[0].get_seqs()[0]
    assert seq.block_table == [0, 1, 2]


def test_round_trip_scattered_blocks_stay_at_their_ids():
    block_table = [5, 2, 7]
    source = FakeEngine(num_pool_blocks=8, block_table=block_table, seed=13)
    snapshot = _backend(source).extract()

    target = FakeEngine(num_pool_blocks=9, block_table=[0], seed=99)
    _backend(target).inject(snapshot)

    ids = torch.tensor(block_table, dtype=torch.int64)
    for layer, (k_cache, _v_cache) in enumerate(target.kv):
        assert torch.equal(k_cache.index_select(0, ids), snapshot.layer_k[layer])


def test_inject_rejects_block_ids_beyond_the_pool():
    source = FakeEngine(num_pool_blocks=8, block_table=[0, 6], seed=17)
    snapshot = _backend(source).extract()

    small = FakeEngine(num_pool_blocks=4, block_table=[0], seed=99)
    try:
        _backend(small).inject(snapshot)
        raise AssertionError("block id 6 cannot exist in a 4-block pool")
    except RuntimeError as exc:
        assert "pool" in str(exc)
