# Changelog

All notable changes to KVSnap are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] — 2026-09-01

First public release — the on-disk KV-cache checkpoint format and the full
save → restore → verify pipeline for GLM-5.3 in 信创 air-gap deployments.

### m1 — serialize KV cache
- New on-disk checkpoint format (magic `KVS1`) under `.kvsnap/<session>/`:
  `manifest.yaml` + per-layer `layer_*_k.bin` / `layer_*_v.bin` blobs +
  `block_map.bin` (token→block map, int64).
- `kvsnap.storage`: lossless tensor ↔ bytes for `float32` / `float16` /
  `bfloat16` (the real KV-cache dtype) via the untyped storage; manifest
  records the architecture fingerprint and a per-blob xxhash64.
- `kvsnap.checkpoint.save`: extract a snapshot from a backend and persist it.
- `kvsnap.backend_vllm.VLLMBackend`: in-process extraction of vLLM's paged KV
  cache + block-manager state (vLLM imported lazily, version-pinned 0.6.x).

### m2 — restore KV cache
- `kvsnap.restore.restore`: verify, then read blobs and inject into a fresh
  engine — `save → kill → restart → restore` yields bit-identical tensors,
  proving recovery rather than recomputation.
- Architecture-fingerprint check before injection: a model-wheel upgrade
  cannot silently misalign KV-cache slots.

### m3 — verify + serve wrapper
- `kvsnap.verify.verify_checkpoint`: per-blob xxhash64 recomputation, manifest
  structural validation, and corruption/tamper detection.
- `kvsnap.serve`: `vllm serve` wrapper with a `SIGUSR1` checkpoint hook and
  `--restore <session>` auto-load on startup.
- `kvsnap.cli`: `save`, `restore`, `list`, `info`, `verify`, `serve` commands
  (click); `--mock` runs the full format/verify/round-trip path offline.
- Bilingual README (Chinese primary + English sibling), animated hero/atlas
  SVG pairs, rendered demo GIF, CI (`test` / `demo` / `release`).

### Tests
- `tests/test_storage.py`: tensor round-trip across KV dtypes, manifest magic,
  blob xxhash, missing-dir and bad-magic errors, int64 block_map.
- `tests/test_checkpoint.py`: m1 valid-checkpoint, m2 exact round-trip,
  corrupt-checkpoint refusal, CLI save/restore.
- `tests/test_verify.py`: clean pass, flipped byte, missing blob, bad magic,
  truncated blob, `verify_blob` helper, CLI exit codes.

[0.1.0]: https://github.com/SuperMarioYL/kvsnap/releases/tag/v0.1.0
