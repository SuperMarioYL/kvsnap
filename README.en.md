<div align="right"><sub><b>English</b> | [简体中文](./README.md)</sub></div>

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="./assets/hero-dark.svg">
  <source media="(prefers-color-scheme: light)" srcset="./assets/hero-light.svg">
  <img src="./assets/hero-light.svg" width="880" alt="KVSnap — GLM-5.3 KV-cache checkpoint and restore">
</picture>

<p align="center"><sub>Persists and restores GLM-5.3's 1M-token KV cache across restarts for 信创 (China-domestic) air-gap operators — resume, don't re-prefill.</sub></p>

<p align="center"><strong>GLM-5.3 loses its million-token KV cache on every restart in an air-gap? KVSnap checkpoints to disk and restores in seconds, skipping hours of prefill.</strong></p>

<p align="center">
  <a href="./LICENSE"><img src="https://img.shields.io/github/license/SuperMarioYL/kvsnap?color=blue" alt="License: MIT"></a>
  <a href="https://github.com/SuperMarioYL/kvsnap/releases"><img src="https://img.shields.io/github/v/release/SuperMarioYL/kvsnap" alt="Latest release"></a>
  <a href="https://github.com/SuperMarioYL/kvsnap/actions/workflows/test.yml"><img src="https://img.shields.io/github/actions/workflow/status/SuperMarioYL/kvsnap/test.yml?branch=main&label=CI" alt="CI"></a>
  <img src="https://img.shields.io/badge/Python-3.12+-blue?logo=python&logoColor=white" alt="Python 3.12+">
  <img src="https://img.shields.io/badge/GLM--5.3-1M_KV_cache-5E5CE6" alt="GLM-5.3 1M KV cache">
</p>

<h2><img src="https://api.iconify.design/tabler:topology-star-3.svg?color=%230071E3&width=24" height="22" align="absmiddle" alt=""> Architecture</h2>

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="./assets/atlas-dark.svg">
  <source media="(prefers-color-scheme: light)" srcset="./assets/atlas-light.svg">
  <img src="./assets/atlas-light.svg" width="880" alt="Architecture: vLLM Engine → KVSnap → Disk checkpoint">
</picture>

KVSnap runs **in-process** inside vLLM, reading the engine's paged KV-cache tensors and block-manager state directly over Python and serializing them into a self-describing on-disk checkpoint; after restart it injects the same bytes back into the fresh engine, skipping the entire prefill. No microservices, no separate daemon. The CLI (`list` / `info` / `verify`) operates on the checkpoint directory on disk and never touches the GPU.

<h2><img src="https://api.iconify.design/tabler:bulb.svg?color=%230071E3&width=24" height="22" align="absmiddle" alt=""> How it works</h2>

**Why this exists.** In a 信创 air-gap, operators run GLM-5.3 agents with million-token contexts: every maintenance window, driver update, or crash wipes out the 80GB+ of accumulated attention state. There is no cloud to fall back to, vLLM does not serialize the KV cache to disk for this model, and re-processing 1M tokens of prefill on domestic GPUs (昇腾 910 / 海光 DCU) takes tens of minutes to hours. KVSnap turns "restart loses everything" into "restart resumes".

**The core primitive** — a portable, self-describing KV-cache checkpoint format (vLLM, LMCache, and no current tool expose one as a cross-restart on-disk artifact):

```
.kvsnap/<session>/
  manifest.yaml      # magic=KVS1 + arch fingerprint + blob manifest (xxhash64)
  layer_0_k.bin      # layer-0 K tensor raw bytes (bfloat16 round-trips losslessly)
  layer_0_v.bin      # layer-0 V tensor raw bytes
  ...
  block_map.bin      # block-manager state (token→block map, int64)
```

`manifest.yaml` records the model architecture fingerprint (layers / KV heads / head_dim / dtype / total blocks), sequence length, the xxhash64 of the token prefix, and each blob's shape, byte length, and xxhash64. Before injecting, `restore` checks the architecture fingerprint — a model-wheel upgrade cannot silently misalign. `verify` recomputes every blob's xxhash64, so a torn write, disk error, or tampered blob is caught before it reaches the engine.

<h2><img src="https://api.iconify.design/tabler:rocket.svg?color=%230071E3&width=24" height="22" align="absmiddle" alt=""> Install</h2>

```bash
pip install kvsnap
```

For production you also need vLLM (target [vLLM 0.6.x](https://github.com/vllm-project/vllm)):

```bash
pip install 'kvsnap[vllm]'
```

> In an air-gap machine room with no public internet, install offline from wheels: `pip install --no-index --find-links=./wheels kvsnap`.

<h2><img src="https://api.iconify.design/tabler:rocket.svg?color=%230071E3&width=24" height="22" align="absmiddle" alt=""> Quick start</h2>

Three commands, fully offline, exercise the whole save→verify→restore chain (`--mock` uses a fake KV cache — no GPU / vLLM needed):

```bash
kvsnap save --mock --layers 3 --blocks 8     # write a checkpoint to ./.kvsnap/default/
kvsnap verify                                 # recompute every blob's xxhash64
kvsnap restore --mock                         # inject into a fresh backend, tensors bit-identical
```

<details><summary>sample output</summary>

```
✓ saved session default -> .kvsnap/default
  model=glm-5.3-1m  layers=3  kv_heads=8  head_dim=128
  seq_len=128  blocks=8  dtype=bfloat16  on_disk=1.05 MB
✓ default: verified 7 blobs, all hashes match
✓ restored session default (3 layers, 8 blocks, 128 tokens) — verified 7 blobs
```
</details>

<h2><img src="https://api.iconify.design/tabler:terminal-2.svg?color=%230071E3&width=24" height="22" align="absmiddle" alt=""> Usage</h2>

The full production happy path — run `kvsnap` instead of bare `vllm serve`:

```bash
# 1. start vLLM under kvsnap (SIGUSR1 checkpoint hook is armed)
python -m kvsnap.serve --model glm-5.3-1m

# 2. before maintenance, send SIGUSR1; the process checkpoints to ./.kvsnap/default/
kill -USR1 <pid>

# 3. restart with --restore, skipping hours of prefill, context intact
python -m kvsnap.serve --model glm-5.3-1m --restore default
```

On-disk checkpoint inspection (no running engine needed, runs on any machine):

```bash
kvsnap list                      # list all sessions
kvsnap info --session default    # show manifest: arch fingerprint + blob list
kvsnap verify                    # integrity check (corruption/tamper reported)
```

See [`examples/`](./examples) for more.

<h2><img src="https://api.iconify.design/tabler:photo.svg?color=%230071E3&width=24" height="22" align="absmiddle" alt=""> Demo</h2>

<p align="center"><img src="./assets/demo.gif" width="880" alt="KVSnap demo: save → list → info → verify → restore"></p>

<p align="center"><sub>Offline <code>--mock</code> path (no GPU). Production uses <code>kill -USR1</code> to save and <code>--restore</code> to resume.</sub></p>

<h2><img src="https://api.iconify.design/tabler:currency-yuan.svg?color=%230071E3&width=24" height="22" align="absmiddle" alt=""> Enterprise</h2>

The OSS core (`save` / `restore` / `verify`) is free forever under MIT. The enterprise license targets 信创 / state-owned operators and is priced per deployment site:

| Capability | OSS | Enterprise |
|---|:---:|:---:|
| Single-node save / restore / verify | ✓ | ✓ |
| Multi-node cache sync (distribute one checkpoint across an air-gap cluster) | — | ✓ |
| 信创 audit logging (traceable data operations) | — | ✓ |
| 昇腾 910 / 海光 DCU GPU adaptation | — | ✓ |
| On-site integration & training | — | ✓ |

Pricing: **¥50,000–200,000 / year / site**. A single-site annual license includes email + 企业微信 (WeCom) support; ¥200,000 covers multi-site with on-site integration and priority response. A lighter entry point is the **¥5,000 / month** support package (企业微信 / Alipay business pay) before committing to an annual contract.

Billing goes through **bank transfer + VAT special invoice** (信创 / government customers require a formal contract, 发票, and corporate-account payment — no credit cards or Stripe). Trial flow: validate OSS in the air-gap lab → hit the multi-node / audit requirement → contact via the 企业微信 group or email in the README → 30-day enterprise trial → validate in the cluster → quote → contract → invoice. To reach us, open an [issue](https://github.com/SuperMarioYL/kvsnap/issues) tagged `enterprise`.

<h2><img src="https://api.iconify.design/tabler:help.svg?color=%230071E3&width=24" height="22" align="absmiddle" alt=""> FAQ</h2>

**Doesn't vLLM already have prefix caching?** Prefix caching is cross-request in-memory reuse — it does not cross restarts and does not touch disk. KVSnap solves disk recovery after a process restart — a different problem.

**Why not just resend the context?** Pre-filling 1M tokens on 信创 hardware (昇腾 910) takes tens of minutes to hours. KVSnap compresses recovery to seconds.

**Doesn't LMCache already manage KV cache?** LMCache does cross-request in-memory KV cache sharing (a serving optimization) — it does not persist to disk across restarts and is not built for the air-gap case. KVSnap is disk persistence + restart recovery.

**Won't vLLM absorb this?** Possibly. But 信创 audit/compliance and domestic-GPU adaptation are things upstream won't prioritize — that is KVSnap's differentiation.

**Why not just save the KV cache to a file?** The KV cache is a paged tensor on the GPU; serializing and restoring it correctly requires understanding vLLM's block manager — it is not plain file I/O. KVSnap bundles the tensors, block map, architecture fingerprint, and xxhash64 integrity into one self-describing format.

**Can `save` be run as a standalone CLI?** Production `save` is triggered in-process via SIGUSR1 (`kill -USR1 <pid>`), not as a standalone command. `kvsnap save --mock` exists only for offline format validation and demos.

<h2><img src="https://api.iconify.design/tabler:map-2.svg?color=%230071E3&width=24" height="22" align="absmiddle" alt=""> Roadmap</h2>

- [x] **m1 — serialize**: extract KV-cache tensors via the vLLM backend and write a self-describing checkpoint (manifest + blobs + block_map) to disk
- [x] **m2 — restore**: read blobs from disk and inject into a fresh engine; save→kill→restart→restore yields bit-identical tensors, proving recovery (not recomputation)
- [x] **m3 — verify + serve wrapper**: xxhash64/blob integrity, `kvsnap serve` (SIGUSR1 save + `--restore` auto-load), CLI polish, bilingual README
- [ ] **v0.2** — multi-model support (Qwen3-Plus, DeepSeek-V3), 昇腾 / 海光 GPU adaptation, multi-node cache sync, 信创 audit logging, automatic checkpoint scheduling

<h2><img src="https://api.iconify.design/tabler:share.svg?color=%230071E3&width=24" height="22" align="absmiddle" alt=""> Share</h2>

```
KVSnap — disk checkpoint & restore for GLM-5.3's 1M-token KV cache, built for 信创 air-gap. Restart loses the whole context? SIGUSR1 to save, seconds to restore, skip hours of prefill. https://github.com/SuperMarioYL/kvsnap
```

<h2><img src="https://api.iconify.design/tabler:users.svg?color=%230071E3&width=24" height="22" align="absmiddle" alt=""> Contributing</h2>

Issues and PRs welcome: file bugs or feature requests in [Issues](https://github.com/SuperMarioYL/kvsnap/issues), and run the [test.yml](./.github/workflows/test.yml) suite before opening a PR. For enterprise / 信创 adaptation needs, open an issue tagged `enterprise`. A Gitee mirror serves the 信创 procurement audience.

<p align="center"><sub><a href="./LICENSE">MIT</a> © 2026 SuperMarioYL</sub></p>
