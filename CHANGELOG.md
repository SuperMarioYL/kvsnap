# Changelog

All notable changes to KVSnap are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.3.0] — 2026-10-26

Fix-led release from the post-ship source review of v0.2.0: four defects on the
operator's documented happy path — a half-finished version bump, a raw
traceback from `restore --mock`, unvalidated mock geometry, and a release that
shipped no downloadable artifacts.

### fix — version surfaces
- The v0.2.0 bump updated `VERSION`, `kvsnap.config.VERSION`, `pyproject.toml`,
  `web/site.json` `content_version` and the changelog, but left the README
  badge (both languages), the site footer tag and the recorded demo version at
  `v0.1.0`. All four are now bumped, and `tests/test_version_lockstep.py`
  asserts every surface (README.md, README.en.md, `web/site.json` `footer.tag`,
  `docs/demo-results.json`, changelog head) — previously it pinned only four
  surfaces, so it passed while the others drifted.

### fix — clean CLI errors from `restore --mock`
- `kvsnap restore --mock` on a missing or corrupt session no longer dumps a
  Python traceback: the `CheckpointError` is converted to a single clean
  `Error: ...` line with a non-zero exit, matching `info` and `verify`.
- `kvsnap.restore.restore` now passes the session name into
  `verify_checkpoint`, so the failure message names the session instead of
  reporting `checkpoint ''`.

### fix — reject impossible mock geometry
- `kvsnap save --mock --layers -1` / `--blocks -1` used to crash with a raw
  `RuntimeError` from inside torch; `--layers 0` was worse — it wrote a
  checkpoint with no K/V blobs, reported it as saved, and `verify` accepted it.
  The geometry flags (`--layers`, `--kv-heads`, `--head-dim`, `--blocks`) are
  now validated as `>= 1` at the CLI boundary and fail with a usage error
  (exit 2).

### fix — release artifacts
- `.github/workflows/release.yml` built the wheel, sdist and checksums and then
  only stored them as an expiring workflow artifact, so every release page had
  zero downloadable assets (`gh release view v0.2.0 --json assets` → `[]`). The
  workflow now attaches `dist/*` to the release created for the tag.

## [0.2.0] — 2026-09-18

Fix-led release from the post-ship source review: four reproduced defects on
the save/serve happy path, plus a version-lockstep guard.

### fix — atomic save
- `kvsnap.storage.write_checkpoint` now writes the whole checkpoint (blobs +
  manifest) into a sibling temp dir, fsyncs, then swaps it into place while
  keeping the previous session as `<session>.last-good` until the swap lands.
  A save that dies mid-write (disk full, power loss) can no longer clobber
  the only good checkpoint — previously the live session was overwritten
  blob-by-blob in place.

### fix — snapshot only the used blocks
- `VLLMBackend.extract` derives `num_blocks_used` from the running sequence's
  block table and copies exactly those blocks (in table order), instead of
  serialising the entire allocated pool. v0.1.0 failed with
  `sequence has N blocks but prefix needs M` on any prefix that didn't fill
  the pool, and wrote GBs of dead blocks when it didn't fail.
- `VLLMBackend.inject` writes each snapshot block back to its physical block
  id (`index_copy_`), so a checkpoint restores into a differently sized pool.
- `seq_len` now prefers the sequence's own token count (`get_len()`) over
  block-count arithmetic.

### fix — documented entrypoints
- `python -m kvsnap --version` and `python -m kvsnap.serve --model ...` now
  run as documented (new `kvsnap/__main__.py` + a `__main__` guard in
  `serve.py`); both previously exited silently doing nothing.

### fix — stop dropping trailing vLLM flags
- `kvsnap serve --gpu-memory-utilization 0.9 ...` trailing flags are now
  converted into real engine kwargs (`_engine_kwargs`) and passed to the
  `LLM(...)` constructor; unknown flags fail loudly inside vLLM instead of
  being silently eaten.

### quality — version lockstep
- `tests/test_version_lockstep.py` pins `VERSION`, `kvsnap.config.VERSION`,
  `pyproject.toml`, and `web/site.json` to the same version so no future bump
  can land half-done.

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

[0.3.0]: https://github.com/SuperMarioYL/kvsnap/releases/tag/v0.3.0
[0.2.0]: https://github.com/SuperMarioYL/kvsnap/releases/tag/v0.2.0
[0.1.0]: https://github.com/SuperMarioYL/kvsnap/releases/tag/v0.1.0
