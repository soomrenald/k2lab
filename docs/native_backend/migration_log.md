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

## 2026-07-29 — Gate 7 native regional prompting

Pinned shared core checkpoint: `1c571a8d5415612cf1a41a7c848bcc6f76e75290`

Completed:

- bridged the existing K2Lab-owned regional compiler and exact chunked attention
  controller into the native Diffusers transformer;
- added explicit normalized-coordinate conversion and reusable mask, latent-grid,
  token-assignment, and overlap visual diagnostics;
- passed the one/multiple/overlap/edge/full/tiny/invalid/aspect/global interaction matrix;
- matched the reference compiled prompt, 173 tokens, token spans, and 224/16 attention
  call counts in an eight-step two-region generation;
- measured native-versus-Comfy cosine `0.9988357`, MAE `0.0109999`, and RMSE `0.0363296`,
  with visually equivalent placement and content;
- proved the no-region path remains pixel-exact to the Gate 5 native golden output.

Gate 7 passes with the approved scaled-FP8 image difference. The product default remains
`comfyui`, and native remains developer-only.

## 2026-07-29 — Gate 8 native regional LoRA

Pinned shared core checkpoint: `76dc7119d80218d7874448d0b181cedccf22b0dc`

Completed:

- installed LoRA deltas inside native transformer linear forwards with exact job-local
  text/image gates for combined, text-refiner, layerwise-text, and projector streams;
- preserved ordered multiple-adapter composition, global-plus-regional scope, explicit
  standard K/V omission, and current character-identity Q/K/V behavior;
- added opt-in Q/K/V, hidden, attention-output, MLP-output, residual, token, and mask
  instrumentation plus bounded per-step LoRA-delta attention adaptation;
- measured zero delta outside every exercised route and proved instrumentation on/off
  and the explicitly disabled regional path are pixel-exact;
- completed representative, mixed-scope, and character-identity cross-backend fixtures
  with visually equivalent outputs and no unexplained artifacts;
- measured the representative native path at 16.34 seconds versus 82.66 seconds for
  current Comfy low-VRAM execution and lower isolated process RSS;
- passed 182 core tests, 2 intentional skips, and 12 subtests.

Gate 8 passes with the approved scaled-FP8 image difference. Native remains developer-only
and `comfyui` remains the product default.

## 2026-07-29 — Gate 9 native image editing

Pinned shared core checkpoint: `8f7ec5c53401643000cce1db12fb6be65d61d2ff`

Completed:

- implemented deterministic VAE source encoding, latent normalization, partial shifted-flow
  schedules, explicit noise, masked Euler sampling, VAE decode, and bounded compositing;
- connected regional reference/edit clauses, prompt emphases, regional LoRAs, attention
  adaptation, progress, cancellation checks, structured errors, and output metadata;
- made strength zero a pixel-exact no-model passthrough in both native and Comfy backends;
- retained source metadata unless explicitly replaced by the current request;
- passed zero/low/medium/full, same/resized/unaligned-aspect, whole/regional/masked,
  regional-LoRA, determinism, boundary, malformed-source, and recovery cases;
- measured representative native-versus-Comfy support cosine `0.9996564`, MAE
  `0.0079054`, RMSE `0.0174867`, and PSNR `35.1458 dB`;
- preserved every pixel outside each regional composite support;
- passed 188 core tests, 2 intentional skips, and 12 subtests.

Gate 9 passes with the approved scaled-FP8 difference and documented full-denoise
finite-feather behavior. Native remains developer-only and `comfyui` remains the default.

## 2026-07-29 — Gate 10 native memory and device management

Pinned shared core checkpoint: `237fd23dc4a578e9d1a095fac0587d4d6bdf88e4`

Completed:

- replaced Comfy memory ownership for the native path with explicit K2 device planning,
  allocator telemetry, sequential component execution, cleanup, optional VAE tiling,
  and one configured safe VAE OOM fallback;
- validated ten same-process load/unload cycles and five deterministic sequential
  generations without persistent allocator growth;
- classified a deliberate preflight OOM before GPU work, recovered with a fresh valid
  request, and cancelled an OOM-prone workload after one denoising step;
- cleared BLAS workspaces and request traceback/tensor references so final unload
  consistently reaches zero allocated and reserved bytes;
- completed local ROCm BF16, explicit FP16, mixed-lane FP8 weight, tiled VAE, CPU VAE,
  clean-generation, and image-edit smoke tests;
- passed the required NVIDIA A40 matrix with ten exact-cleanup load cycles, five
  deterministic sequential generations, preflight OOM, immediate recovery, one-step
  cancellation, explicit VAE tiling, and explicit FP16 compute;
- passed 200 core tests, 2 intentional skips, and 14 subtests.

Gate 10 passes. No 80 GB device was available, and the release-level 100-job soak
remains outstanding. Native stays developer-only and `comfyui` remains the product
default while Phase 9 integration begins.

## 2026-07-29 — Gate 11 desktop and RunPod integration

Pinned shared core checkpoint: `237fd23dc4a578e9d1a095fac0587d4d6bdf88e4`

Completed:

- added server-configured RunPod backend selection, capabilities, job/event metadata,
  explicit native model bindings, pre-GPU capability rejection, and container startup
  validation while retaining ComfyUI as the default;
- routed native RunPod generation and image editing through the same k2core schemas and
  backend used by desktop, with exact tested runtime pins and structured errors;
- added persistent prompt-safe agent/worker diagnostics and distinct startup,
  generation, worker-disconnect, proxy, provider-timeout, and pod-failure classes;
- passed a byte-identical canonical request fixture through desktop and RunPod worker
  entrypoints;
- completed a live A40 direct-worker run and persistent job-service run with pixel-exact
  RGB output;
- proved immediate and post-reconnect duplicate submissions return the same durable job
  and output IDs without new GPU work;
- recovered completed state, all 20 events, an empty resumed cursor page, output
  inventory, and PNG backend/correlation metadata after manager reconstruction;
- ended both live probes at 0 MiB GPU memory and an estimated incremental compute cost
  below $0.02;
- passed 305 RunPod tests, 15 intentional skips, and 16 subtests.

Gate 11 passes. The updated container has not yet been built/published as a clean release
candidate, and native remains developer-only with `comfyui` as the product default.

## 2026-07-29 — Gate 12 release-readiness evidence

Pinned shared core checkpoint: `237fd23dc4a578e9d1a095fac0587d4d6bdf88e4`

RunPod source checkpoint:
`79837482e458ef216ba3d990b134fd9a0a4d6ab9`

Completed:

- added a resumable, workload-validated native soak probe and ran 100 sequential
  512×512 eight-step Euler/simple jobs on one NVIDIA A40;
- produced one exact RGB pixel hash across all 100 jobs while keeping the model resident;
- measured zero first-to-last window median growth for both NVIDIA used memory and
  worker RSS;
- shut down the worker in 1.52 seconds and returned the GPU from 379 MiB to 0 MiB;
- measured 9.69-second mean, 9.32-second p50, and 11.60-second p95 generation time;
- completed the 999.25-second session for approximately $0.1221 at $0.44/hour;
- added workspace-owned tokenizer handling, third-party notice inventories, and a
  native-only container/workflow candidate that contains no ComfyUI install;
- added a session-only experimental desktop selector, one-click ComfyUI fallback, and
  allowlisted prompt-safe issue-report ZIP without changing saved settings or the
  environment-controlled default;
- retained `comfyui` as the product default and preserved both rollback environment
  variables.

The 100-job soak passes. Overall release readiness remains blocked until the native-only
image is published and booted on a RunPod GPU, licensing/provenance decisions are
recorded, representative outputs receive human approval, and the release rollback drill
passes.

## 2026-07-29 — Gate 12 clean native image checkpoint

RunPod source checkpoint:
`ef555b37b9781b93f1985e6b46baaf582cffad99`

Completed:

- exported the exact web runtime closure from `uv.lock` and enforced every artifact hash
  during the image build;
- built the native-only image from the approved immutable CUDA base with no ComfyUI tree;
- verified pinned imports and a clean `pip check`;
- booted the agent against an empty temporary workspace with native as the image default;
- received authenticated `ready` health with container, agent, and storage stages true;
- passed 315 RunPod tests, 15 intentional skips, 16 subtests, and Ruff.

The initial pinned scan found four fixed HIGH issues in development artifacts and pip's
embedded inventory. RunPod checkpoint `8aff782` narrowed the runtime copy and removed
unneeded packaging tools after a successful `pip check`. The rebuilt digest then passed
with zero HIGH, zero CRITICAL, and zero secrets, and pinned Syft emitted an SPDX 2.3
SBOM. The local image is not a published release candidate. GPU boot, clean desktop
integration, rollback drill, human output approval, and licensing decisions remain
release blockers.

## 2026-07-29 — Gate 12 paired A40 memory checkpoint

RunPod source checkpoint:
`f8b2184045247902ad11be6cf4c2f4feee6431c0`

The bounded paired-backend probe ran the identical canonical 512×512, eight-step request
and model hashes through ComfyUI and native on the same A40 and Python/Torch environment.
At 100 ms sampling, ComfyUI peaked at 18,889 MiB and native at 13,653 MiB. The 0.7228
ratio passes the 1.15 release ceiling, and both workers cleaned up to 0 MiB. Total probe
time was 58.76 seconds for approximately $0.0072.

RunPod checkpoint `ff569c4` adds the explicit release-candidate mechanism: ordinary
pull-request/manual image validation remains non-publishing, while an approved
`native-v*` tag publishes the native image under a distinct tag in the preserved public
GHCR workspace package, validates the pushed digest, emits the SBOM, and signs the digest
with GitHub OIDC.

## 2026-07-29 — Gate 12 signed release-candidate checkpoint

RunPod release source:
`4b091c54162fc689833b5115f78e47b1955525cb`

RunPod publication-evidence checkpoint:
`3b13c46`

The first immutable tag, `native-v0.4.0-rc.1`, published an image but its workflow
stopped before scanning and signing because the embedded authenticated-health command
had invalid Python indentation. The tag and failed run are retained as immutable
evidence and must not be deployed. RunPod checkpoint `4b091c5` fixes the command and
adds a regression test that compiles the exact embedded program.

The replacement `native-v0.4.0-rc.2` workflow passed build, pushed-digest validation,
no-ComfyUI inspection, pinned imports and runtime lock checks, empty-workspace
authenticated health, zero-HIGH/zero-CRITICAL Trivy policy, SPDX generation, and GitHub
OIDC signing. The immutable candidate is
`ghcr.io/soomrenald/k2lab-runpod-workspace@sha256:7662f6440bd4e2a1f6059876c042df98a1e00284c89c35e6aaec3aa446be856f`.
Independent Cosign verification matched the exact tag workflow identity, trusted GitHub
Actions issuer, certificate chain, claims, and transparency-log entry.

Publication and signing now pass. Fresh RunPod GPU boot, native generation, failure
recovery, swap-back generation on the preserved ComfyUI digest, human output approval,
and licensing/provenance decisions remain release blockers.

## 2026-07-29 — Gate 12 fresh GPU candidate and rollback acceptance

RunPod acceptance-evidence checkpoint:
`59de1c5`

The signed RC2 digest booted on a fresh disposable secure-cloud NVIDIA A40 at $0.44/hour.
Authenticated health reported `ready`; the approved tokenizer and exact transformer,
text-encoder, and VAE hashes were installed. A deliberately unsupported post-upscale
request returned `native_feature_unsupported` with HTTP 409, and the following valid
256×256 two-step native request completed with output SHA-256
`f50c865a3a5d9d70fb13317c8dbb554f81da6b0f52ca6ed17a81c24af5e5c7b5`.

The same Pod was reset in place to preserved ComfyUI version `0.3.0`, immutable index
digest
`ghcr.io/soomrenald/k2lab-runpod-workspace@sha256:19652733039379d1ef47cd3279e6b266b802c7a68a1c380173221a2d8ace6435`.
Its 50 GB `/workspace` volume retained the hash-verified inventory, authenticated health
returned `ready`, and a ComfyUI generation completed with output SHA-256
`695744d1adb6dfa5f8af258d6d4f184deff4ea2eee47c060f9a60a985765d391`.
The cleanup guard permanently deleted the disposable Pod and Pod volume. The passing
drill took 791.42 seconds and approximately $0.0967 compute cost.

Desktop checkpoint `c2b157c` was built as a wheel and installed into a new Python 3.12
virtual environment outside the repository `.venv`, with empty model directories and no
visible ComfyUI Python package. The installed QML loaded; unset backend selection,
session-only native selection, unsupported-capability gating, prompt-safe issue-report
creation, and one-click non-persistent ComfyUI fallback all passed. This exercise did not
repeat native GPU generation in the temporary desktop environment; the signed clean
RunPod candidate and prior desktop GPU gates provide that execution evidence.

The fresh candidate boot, structured failure recovery, native generation, image-swap
rollback, post-rollback generation, and destructive cleanup checks now pass. Gate 12
overall remains blocked by representative-output human approval, K2Lab/k2core license
decisions, and the Krea license-acceptance/content-filtering/mirror strategy. The exact
installed FantasyPortrait detector hash matches the official Apache-2.0 repository, so
its source-provenance gap is closed; the converted Qwen text-encoder derivation/notice
chain still requires confirmation before redistribution. Native remains opt-in and
`comfyui` remains the default.

## 2026-07-29 — Gate 12 representative-output human review

The owner accepted the native-to-ComfyUI visual match across the contact sheet and
accepted the regional-LoRA example's visible red-vase/blue-vase separation. Clean
generation and regional LoRA therefore pass representative human review.

The owner rejected the ordinary-LoRA and image-edit examples as demonstrations of
successful functionality. The ordinary fixture used a deliberately weak synthetic
single-target adapter and showed no useful perceptual effect. The image-edit fixture
requested that the right vase become emerald-green glass, but the displayed output did
not visibly perform that edit. Backend similarity does not convert either result into a
feature pass. Gate 12 remains blocked until both examples are remediated, regenerated,
and approved in a new human review.

Corrective candidates were then generated locally. The ordinary-LoRA candidate replaces
the single-target synthetic adapter with a real 256-target standard LoRA and shows a
material same-seed change while retaining close native/Comfy parity. The image-edit
candidate replaces the original two-step, 0.5-denoise request and competing blue-vase
reference clause with eight steps, 0.75 denoise, a box covering the whole object, and an
explicit replacement prompt; both backends visibly render the complete right vase
emerald green. The owner approved both corrective rows on 2026-07-30. Representative
human output review is now complete; the original rejection remains recorded as the
reason the acceptance evidence was replaced.

## 2026-07-30 — Gate 12 license and distribution-policy closure

Current licensed k2core checkpoint:
`903166f756614b13c0add0196fb5705206370dc3`

RunPod licensed-source and image-packaging checkpoint:
`cc905bb`

Completed:

- licensed K2Lab and k2core first-party source under Apache-2.0;
- pinned the desktop package to the licensed k2core checkpoint without changing the
  executable GPU evidence checkpoint;
- approved noncommercial open-source distribution with no bundled or redistributed
  model weights, prepopulated cache, project-operated mirror, or unattended model
  acquisition, while retaining explicit operator-requested upstream downloads;
- made model acquisition and upstream-term acceptance the deployment operator's
  responsibility;
- limited the tested deployment scope to private, single-operator workspaces with
  operator prompt/output review;
- retained the converted Qwen FP8 notice-chain gap as a blocker for future artifact
  redistribution, not for source-only distribution; and
- kept native opt-in, ComfyUI as the default, and the existing rollback path intact.

The RunPod closure adds its Apache-2.0 license and model-use policy to source and the
native image, advances both workspace Dockerfiles and the lockfile to the licensed
k2core checkpoint, and preserves explicit operator-requested downloads from authorized
providers without bundling weights or operating a project mirror.

All Gate 12 functional, representative-output, clean-build, security, signing,
fresh-GPU, rollback, licensing, and approved-scope policy requirements are complete.
Gate 12 passes for this scope. Public/shared inference, commercial operation,
project-operated mirrors, unattended model acquisition, or model redistribution
requires a new review.
