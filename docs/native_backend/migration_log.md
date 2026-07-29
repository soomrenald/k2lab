# Native backend migration log

## 2026-07-28 — Phase 0 / Gate 1 candidate

Branch: `feature/native-k2-backend`

Completed:

- read the native-backend migration specification in full;
- inventoried tracked desktop code and the exact pinned `k2core` source;
- searched direct ComfyUI/custom-node imports and indirect process, environment,
  filesystem, model-discovery and documentation assumptions;
- traced generation, image-edit and face-refinement calls from QML to the disposable
  ComfyUI worker;
- confirmed that this repository contains no RunPod entry point or deployment setup;
- proposed backend/service/module boundaries and immutable shared schemas;
- created the dependency inventory, parity matrix, test plan, risk register, known
  differences and rollback plan.

Validation before documentation edits:

```text
QT_QPA_PLATFORM=offscreen .venv/bin/python -m pytest -q
171 passed, 2 skipped, 6 subtests passed in 3.40s
```

Production behavior changes: none.
Dependencies/lockfiles changed: none.
Parity runs: none; native backend is not implemented.
Performance runs: none.
Cloud/GPU spend: none.

Gate status: awaiting explicit Gate 1 approval. Model-loading implementation is prohibited
until approval.

## 2026-07-28 — Gate 1 approved / Gate 2 started

The user explicitly approved Gate 1 with the recommended scope, reference, licensing,
fixture-storage, and RunPod-deferral decisions. Gate 2 implementation began in the shared
`k2core` package. Native model loading remains out of scope.

## 2026-07-28 — Gate 2 candidate

Pinned shared core commit: `e35a01d7f381764a7b3069aaeef13f3aa4c83b1f`

Completed:

- added immutable shared requests/results and structured errors;
- added the backend lifecycle, explicit selector, ComfyUI adapter, and unsupported native
  placeholder;
- routed desktop worker load/generate/edit/refine operations through the adapter;
- retained ComfyUI as the unset and explicit default;
- verified explicit native failure without fallback;
- updated and fetched the locked shared-core dependency;
- passed core and K2Lab suites;
- passed one real local ROCm clean-generation smoke through `ComfyUIBackend`.

Production default: unchanged (`comfyui`).
Native inference: not implemented.
Cloud/GPU spend: no cloud spend; one local 512×512 eight-step smoke.
Gate status: awaiting explicit Gate 2 approval.

## 2026-07-28 — Gate 2 approved / Phase 1 started

The user explicitly approved Gate 2. Phase 1 began with the approved constraints:

- add only the K2Lab-owned registry, hash/header validation, and legacy scanner;
- keep the registry usable with arbitrary absolute paths and symlinks;
- do not copy, alter, map, or load model tensor payloads;
- do not change the worker, ComfyUI backend, backend default, or generation behavior;
- stop at Gate 3.

## 2026-07-28 — Gate 3 candidate

Pinned shared core commit: `221b358a4cecc9b40a794e7591e2ca43cd092b9e`

Completed:

- added a versioned TOML model registry with required component hashes;
- added strict existence, safetensors, architecture-fingerprint, shape, and hash checks;
- added symlink resolution and case-insensitive duplicate-name rejection;
- added a read-only legacy ComfyUI scanner that emits registry TOML to stdout;
- added a standalone validator that does not construct desktop settings or start Qt;
- validated both locally installed Krea2 variants plus the shared encoder and VAE;
- passed the full core and K2Lab suites.

Production default: unchanged (`comfyui`).
Runtime/model-loading changes: none.
GPU work: none.
Model copies or mutations: none.
Gate status: awaiting explicit Gate 3 approval.

## 2026-07-28 — Standing approval / Gate 4 complete

The user approved Gate 3 and all later migration gates. This removes approval pauses but
does not remove evidence, parity, rollback, sequencing, or safety requirements. RunPod,
cloud spend, deployment, and removal of the ComfyUI fallback remain outside that standing
authorization.

Pinned shared core commit: `52458eae1e2431f47e8b382f4e1e0e268a9e999a`

Completed:

- added strict native safetensors loading without importing ComfyUI;
- limited strict loading to the exact approved Turbo, encoder, and VAE identities;
- added reviewed component key namespaces and static architecture configuration;
- preserved source quantization metadata and dtypes by default;
- added explicit dtype conversion and CPU/CUDA/ROCm device policy;
- reported hashes, tensors, parameters, storage, dtypes, mappings, and load time;
- added deterministic partial-failure cleanup and unload;
- connected native registry/model selection to the existing worker protocol;
- retained CPU staging for `auto` so all components are not forced into 16 GiB VRAM;
- passed a real worker load and three local ROCm load/unload cycles.

Production default: unchanged (`comfyui`).
Native generation: unsupported.
Model writes: none.
Gate 4 status: approved under standing authorization; Phase 3 may proceed.

## 2026-07-29 — Phase 3 started

Pinned shared core checkpoint: `9a61edfca3e1480cabd9be97509aef3adda4516c`

Completed so far:

- added registry ownership and deterministic hashing for standalone tokenizer assets;
- made the legacy scanner locate the reference Qwen tokenizer without making it a
  hard-coded runtime path;
- implemented the exact Krea2 system/user/assistant prompt template and conditioned-token
  boundary;
- implemented the shifted 10,000-point Flux schedule and simple scheduler selection;
- implemented CFG-1 Euler flow integration, deterministic Torch noise, and checkpoint
  callbacks;
- proved exact 50-token identity and float32 sigma identity against the current ComfyUI
  reference for the synthetic Gate 2 prompt.

Still required before Gate 5 can be marked complete: executable Qwen hidden-state taps,
Krea2 transformer forward execution, VAE decoding, native image output, and latent/image
parity. Native generation remains unsupported in the backend until those are complete.

## 2026-07-29 — Native Qwen text milestone

Pinned shared core checkpoint: `f35f30fae2fe70c822b52a115a6e00b57e854e1e`

Completed:

- built the reviewed text-only Qwen3-VL-4B graph without importing ComfyUI;
- strictly mapped 650 executable tensors, including 252 scaled-FP8 linear layers;
- explicitly rejected unknown quantization markers and ignored only the reviewed
  315-tensor vision tower that clean Krea2 text conditioning does not execute;
- preserved the exact 12 pre-layer taps and the intentional absence of final
  normalization on those intermediate states;
- matched the Comfy reference conditioning shape `(1, 20, 30720)` with cosine
  similarity `1.0`, mean absolute error `3.96e-6`, and maximum error `2.90e-4`;
- produced byte-identical repeated native encodes in one process;
- released all model allocations from VRAM after unload (the ROCm context retained
  its normal approximately 32 MiB baseline).

Production default remains `comfyui`; native generation remains unavailable while
the transformer and VAE execution milestones are incomplete.

## 2026-07-29 — Native Krea2 transformer milestone

Pinned shared core checkpoint: `7c0dcfd5da045d3ed6ce1bfeeb45266795650f10`

Completed:

- used Diffusers 0.39's upstream `Krea2Transformer2DModel` rather than copying the
  Comfy implementation;
- strictly mapped all 686 checkpoint tensors and the flattened per-block modulation
  tables onto the upstream graph;
- preserved all 256 scaled-FP8 layers, including the checkpoint's 96 full-precision
  and 160 fixed-scale FP8 activation-matmul policies;
- retained FP32 RMSNorm execution and explicit repeated grouped-query K/V semantics;
- implemented 5D latent validation, 2×2 packing/unpacking, padding/cropping, 3-axis
  position IDs, timestep handling, and all-valid mask elision;
- matched a local 512×512, sigma-1 Comfy velocity checkpoint with cosine similarity
  `0.999693`, mean absolute error `0.0266`, RMSE `0.0346`, and maximum error `0.203`;
- produced byte-identical repeated native transformer outputs in one process;
- released all model allocations from VRAM after unload, leaving only the normal ROCm
  context baseline.

The product default remains `comfyui`. Native generation remains unavailable until
VAE decode and complete multi-step/image parity are implemented.

## 2026-07-29 — Native Krea2 VAE milestone

Pinned shared core checkpoint: `5f65090c2b3a066464c8224f77d3720060f955d1`

Completed:

- used Diffusers 0.39's upstream `AutoencoderKLQwenImage` graph without importing or
  copying ComfyUI runtime code;
- strictly mapped all 194 checkpoint tensors and all 126,892,531 parameters, with no
  missing, unexpected, shape-mismatched, or unmaterialized tensors;
- implemented the exact 16-channel Krea/Wan latent mean/std conversion before decode;
- validated a deterministic normalized latent against the current ComfyUI VAE at
  128×128 output resolution with cosine similarity `0.9999896`, mean absolute error
  `0.00134`, RMSE `0.00194`, and maximum error `0.01852`;
- produced byte-identical repeated native decodes in one process;
- released model allocations after unload, leaving only the normal approximately
  34 MiB ROCm context allocation.

The product default remains `comfyui`. Native generation remains unavailable until
the clean end-to-end path and Gate 5 parity/reliability evidence are complete.

## 2026-07-29 — Native clean-generation orchestrator milestone

Pinned shared core checkpoint: `04d5b7115791739a5e091b517125250fe6dcb70b`

Completed:

- connected the native prompt encoder, explicit per-request CPU RNG, simple sigma
  schedule, Euler integration, transformer, latent normalization, VAE, PNG output,
  progress, diagnostics, cancellation checks, and structured errors;
- staged text encoder, transformer, and VAE execution sequentially so the path runs
  on the supported local 16 GiB ROCm device;
- exposed only clean text-to-image capability and explicitly rejected negative prompts,
  partial denoise, regional controls, LoRAs, projector controls, and post-upscale;
- completed a 512×512 two-step end-to-end generation and clean unload;
- compared the native image with an identical current-Comfy request at cosine
  `0.999104`, mean absolute pixel error `0.01048`, RMSE `0.03350`, and PSNR
  `29.50 dB`.

The product default remains `comfyui`, and native remains developer-only. Gate 5 is not
complete until the approved eight-step golden fixture, cancellation, malformed-input,
dimension, sequential-generation, cleanup, desktop, and supported RunPod soak evidence
is recorded.

## 2026-07-29 — Gate 5 local golden candidate

Pinned shared core checkpoint: `bc653e493d2378b166d1bcb968afe7dd492f3efa`

Completed:

- added reusable decoded-image parity measurements in shared core;
- stored the synthetic 512×512 eight-step request, environment, component identities,
  thresholds, measurements, timings, and external artifact hashes as a versioned fixture;
- measured native-versus-Comfy image cosine `0.9998963`, MAE `0.0056209`, RMSE
  `0.0113998`, and PSNR `38.8621 dB`;
- proved pixel-exact native repeated-seed output in two runs;
- proved cooperative step-boundary cancellation and cleanup on local ROCm;
- passed 163 core tests and 181 K2Lab tests, with the existing two intentional skips.

Follow-up local Gate 5 evidence:

- passed square, portrait, landscape, invalid-alignment, and malformed-request matrices;
- completed ten sequential 512×512 eight-step runs with one pixel hash and zero
  allocated/reserved VRAM growth;
- completed a real desktop worker bootstrap/load/generate protocol smoke with pixel-exact
  output and the command correlation ID preserved;
- corrected the worker's native phase labels so prompt encoding and VAE decode no longer
  appear as denoising step 0/0.

All locally actionable Gate 5 checks now pass. Gate 5 remains incomplete only because its
100-generation supported-RunPod soak is blocked by deferred infrastructure scope/spend.

## 2026-07-29 — Native ordinary LoRA milestone

Pinned shared core checkpoint: `39bce9c749c1daae603987228f9e53ada4707de7`

Completed:

- added strict standard safetensors LoRA parsing for A/B and down/up conventions;
- implemented non-destructive per-forward adapter deltas with rank, alpha, positive,
  zero, and negative strength handling;
- validates every target/key/shape before changing the executable graph and rejects
  regional scope, bare parameters, DoRA, LoKr, incomplete pairs, and partial application;
- applies multiple adapters in declared order and reports hashes, tensors, targets,
  ranks, strengths, bytes, status, and unmatched keys;
- proved on the real local transformer that positive/negative adapters change output,
  zero strength and disable/re-enable return pixel-exact base output, multiple adapters
  remain ordered, and no VRAM allocation remains above the normal floor;
- passed 168 core tests, 2 intentional skips, and 6 subtests.

Gate 6 remains incomplete pending standard-LoRA cross-backend parity and the separate LoKr
format used by several existing K2Lab adapters. The product default remains `comfyui`.

## 2026-07-29 — Native direct LoKr milestone

Pinned shared core checkpoint: `5ae1d9b29652ba08213aa469fb76045fdd7a5462`

Completed:

- added strict parsing for the direct `lokr_w1`/`lokr_w2` linear format used by current
  K2Lab adapters while continuing to reject decomposed, Tucker, DoRA, incomplete, mixed,
  and partially applicable files;
- implemented factorized Kronecker-product forward deltas without materializing merged
  matrices, and matched an explicit `torch.kron` reference within `9.54e-7`;
- parsed the existing `realism_engine_krea2_v3.1` adapter with all 768 tensors mapped to
  256 targets and 1,562,320,896 adapter bytes;
- completed a real two-step 512×512 worker generation at strength `0.1` on the local
  16 GiB ROCm device, with full LoKr inspection metadata in the result;
- passed 169 core tests, 2 intentional skips, and 6 subtests.

Gate 6 remains incomplete pending the final versioned cross-backend parity fixture and
repeat-enable matrix. The product default remains `comfyui`.

## 2026-07-29 — Gate 6 native ordinary-LoRA parity

Pinned shared core checkpoint: `5ae1d9b29652ba08213aa469fb76045fdd7a5462`

Completed:

- passed the complete no/one/multiple/zero/negative/missing/incompatible/re-enable matrix;
- versioned an eight-step standard-LoRA cross-backend fixture with cosine `0.9998517`,
  MAE `0.0071454`, RMSE `0.0138494`, and PSNR `37.1714 dB`;
- reviewed the standard-LoRA native and Comfy outputs as visually equivalent;
- separately validated the existing 256-target realism direct-LoKr adapter end to end;
- retained non-destructive per-generation deltas and strict no-partial-application rules.

Gate 6 passes. The product default remains `comfyui`, and native remains developer-only
while regional prompting and later phases are implemented.
