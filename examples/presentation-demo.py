"""Round-trip seeded CPU tensors through the real checkpoint format."""
from tempfile import TemporaryDirectory
import json, torch
from kvsnap.backend_vllm import InMemoryBackend, make_mock_snapshot
from kvsnap.checkpoint import save
from kvsnap.restore import restore

torch.manual_seed(42)
original = make_mock_snapshot(model_id="synthetic-cpu", num_layers=2, num_kv_heads=2, head_dim=8, num_blocks_used=3)
with TemporaryDirectory(prefix="kvsnap-demo-") as folder:
    save(InMemoryBackend(original), session="sample", base_dir=folder)
    fresh = InMemoryBackend()
    result = restore(fresh, session="sample", base_dir=folder)
    restored = fresh.snapshot
    tensors_equal = all(torch.equal(a,b) for a,b in zip(original.layer_k + original.layer_v, restored.layer_k + restored.layer_v))
    block_map_equal = torch.equal(original.block_map, restored.block_map)
    assert tensors_equal and block_map_equal
    print(json.dumps({"backend":"CPU InMemoryBackend", "layers":restored.arch.num_layers,"blocks":restored.num_blocks_used,"verified_blobs":result.checked,"tensors_equal":tensors_equal,"block_map_equal":block_map_equal},indent=2))
