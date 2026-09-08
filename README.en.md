**English** | [简体中文](README.md)

<picture>
  <source media="(max-width: 640px) and (prefers-color-scheme: dark)" srcset="assets/presentation/hero-mobile-dark.svg">
  <source media="(max-width: 640px)" srcset="assets/presentation/hero-mobile-light.svg">
  <source media="(prefers-color-scheme: dark)" srcset="assets/presentation/hero-dark.svg">
  <img src="assets/presentation/hero-light.svg" width="1000" alt="Store KV tensors, block mappings and architecture metadata as a disk snapshot and verify content before restoration.">
</picture>

**Store KV tensors, block mappings and architecture metadata as a disk snapshot and verify content before restoration.**

`v0.1.0` · `Python 3.12+ / PyTorch` · [MIT](LICENSE)

[Website](https://kvsnap.lei6393.com) · [Demo record](docs/demo-results.json)

## Why use it

In-process cache state is hard to inspect independently: which tensors were saved, are they intact and can they be read back? KVSnap packages that state as a self-describing directory with a manifest, blobs and save, verify and restore interfaces.

## Architecture

<picture>
  <source media="(max-width: 640px) and (prefers-color-scheme: dark)" srcset="assets/presentation/architecture-mobile-dark.svg">
  <source media="(max-width: 640px)" srcset="assets/presentation/architecture-mobile-light.svg">
  <source media="(prefers-color-scheme: dark)" srcset="assets/presentation/architecture-dark.svg">
  <img src="assets/presentation/architecture-light.svg" width="1000" alt="KVCacheBackend exposes extract/inject. checkpoint writes a manifest, per-layer K/V blobs and block_map; verify recomputes xxhash64; restore verifies, reads tensors and injects them into a backend. InMemoryBackend supports CPU round-trip checks. VLLMBackend directly depends on engine internals and requires separate integration validation.">
</picture>

KVCacheBackend exposes extract/inject. checkpoint writes a manifest, per-layer K/V blobs and block_map; verify recomputes xxhash64; restore verifies, reads tensors and injects them into a backend. InMemoryBackend supports CPU round-trip checks. VLLMBackend directly depends on engine internals and requires separate integration validation.

Source entry points: [kvsnap/backend_vllm.py](kvsnap/backend_vllm.py) · [kvsnap/checkpoint.py](kvsnap/checkpoint.py) · [kvsnap/storage.py](kvsnap/storage.py) · [kvsnap/verify.py](kvsnap/verify.py) · [kvsnap/restore.py](kvsnap/restore.py) · [kvsnap/serve.py](kvsnap/serve.py) · [kvsnap/config.py](kvsnap/config.py)

## Install

Requires Python 3.12+, uv and PyTorch. The default source install runs the CPU example; actual vLLM integration additionally needs the corresponding engine and hardware.

```bash
git clone https://github.com/SuperMarioYL/kvsnap.git
cd kvsnap
uv venv --python 3.12
uv pip install --python .venv/bin/python -e .
```

## Quickstart

Seed=42 creates small synthetic CPU KV tensors, exercises the real disk format and integrity checks, then restores into a fresh backend and compares tensors. No model, GPU or million-token context is run.

```bash
.venv/bin/python examples/presentation-demo.py
```

Complete inputs and execution steps are included in the commands above and the [demo record](docs/demo-results.json).

## Usage

```bash
.venv/bin/kvsnap save --mock --layers 2 --blocks 3
.venv/bin/kvsnap list
.venv/bin/kvsnap info --session default
.venv/bin/kvsnap verify
.venv/bin/kvsnap restore --mock
```
See [round_trip.py](examples/round_trip.py) for Python use. A serving integration should call checkpoint.save at a consistent state boundary and restore into an engine with compatible allocated caches. Saving tensors alone does not restore scheduler request lifecycles.

## Recorded demo

<picture>
  <source media="(max-width: 640px) and (prefers-color-scheme: dark)" srcset="assets/presentation/process-mobile-dark.svg">
  <source media="(max-width: 640px)" srcset="assets/presentation/process-mobile-light.svg">
  <source media="(prefers-color-scheme: dark)" srcset="assets/presentation/process-dark.svg">
  <img src="assets/presentation/process-light.svg" width="1000" alt="Seed=42 creates small synthetic CPU KV tensors, exercises the real disk format and integrity checks, then restores into a fresh backend and compares tensors. No model, GPU or million-token context is run.">
</picture>

### Save, verify and read back

Two layers and three blocks produce five blobs; K/V tensors and the block map are equal.

```text
$ .venv/bin/python examples/presentation-demo.py
{
  "backend": "CPU InMemoryBackend",
  "layers": 2,
  "blocks": 3,
  "verified_blobs": 5,
  "tensors_equal": true,
  "block_map_equal": true
}
```

## Capabilities and integration

<picture>
  <source media="(max-width: 640px) and (prefers-color-scheme: dark)" srcset="assets/presentation/integrations-mobile-dark.svg">
  <source media="(max-width: 640px)" srcset="assets/presentation/integrations-mobile-light.svg">
  <source media="(prefers-color-scheme: dark)" srcset="assets/presentation/integrations-dark.svg">
  <img src="assets/presentation/integrations-light.svg" width="1000" alt="CLI list/info/verify operate on disk directories. save --mock and restore --mock exercise the CPU example; the Python API can be embedded in a serving process. The serve wrapper exposes SIGUSR1 saving and startup restoration, whose model and hardware compatibility cannot be inferred from the CPU demo.">
</picture>

CLI list/info/verify operate on disk directories. save --mock and restore --mock exercise the CPU example; the Python API can be embedded in a serving process. The serve wrapper exposes SIGUSR1 saving and startup restoration, whose model and hardware compatibility cannot be inferred from the CPU demo.



## Configuration

The default directory is `.kvsnap` and session is default. The manifest records model ID, layers, KV heads, head_dim, dtype, blocks, sequence length, prefix hash and each blob’s shape and xxhash64. config.py defines the format and vLLM target version. xxhash is a content check, not a signature or encryption.

## Roadmap and scope

CPU serialization and restoration are verified. Real-engine request-state restoration, cross-version compatibility, additional hardware, multi-node synchronization and automatic scheduling require further validation or implementation.

- GLM, Ascend, Hygon and specific vLLM versions were not validated; no second-scale restore or prefill savings are claimed.
- The adapter uses internal attributes and partially inferred state; inspect scheduler, prefix and block-table contracts before real serving use.

![Terminal recording](assets/demo.gif) · [Recording script](docs/demo.tape)

## License

[MIT](LICENSE)
