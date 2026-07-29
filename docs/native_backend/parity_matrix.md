# Native backend parity matrix

Status values are `PASS`, `PASS WITH APPROVED DIFFERENCE`, `FAIL`, `NOT IMPLEMENTED`, and
`BLOCKED`. Phase 0 does not claim native parity.

Reference baseline:

- K2Lab: `add2a38`
- k2core: `a82b0b32a891e19eac5c5f6e35f8a9bfb715f9dc`
- observed local ComfyUI: `285a98944c397a4a81f15ac63d69fa3dbc0a27b9`
- CPU suite: 171 passed, 2 intentionally skipped, 6 subtests passed
- GPU reference fixtures: not captured for this migration

| Capability / case | Current ComfyUI behavior | Native status | Required evidence / gate |
| --- | --- | --- | --- |
| Backend-neutral schema round-trip | Loose worker dictionaries only | NOT IMPLEMENTED | Exact request/result/metadata round-trip; Gate 2 |
| Default backend remains ComfyUI | Only direct ComfyUI runtime exists | NOT IMPLEMENTED | Unset/`comfyui` selection tests and log evidence; Gate 2 |
| Clean Krea2 Turbo text-to-image | Supported, CFG 1.0 | NOT IMPLEMENTED | Golden embedding/noise/sigma/latent/image fixture; Gate 5 |
| Krea2 Raw / non-Turbo | Architecture selectable; production semantics unclear | BLOCKED | Product decision and reference fixture |
| Seed repeatability | Seed passed to `comfy.sample.prepare_noise` and `sample` | NOT IMPLEMENTED | Same-backend repeats plus cross-backend initial-noise comparison; Gate 5 |
| Dimensions / 16-pixel alignment | Supported | NOT IMPLEMENTED | Dimension matrix and latent-shape checkpoints; Gate 5 |
| Sampler/scheduler combinations | UI exposes copied ComfyUI registries | BLOCKED | Approve required pair list, then sigma/scheduler parity; Gate 5 |
| Progress | Per denoising step | NOT IMPLEMENTED | Ordered events and final completion; Gate 5 |
| Cancellation | Desktop terminates disposable worker | NOT IMPLEMENTED | Response envelope, cleanup, and behavioral-difference review; Gate 5 |
| Ordinary LoRA | Unfused ComfyUI adapter patches | NOT IMPLEMENTED | No/one/multiple/zero/negative/error/deterministic-order fixtures; Gate 6 |
| Regional prompting | Unified prompt and optimized-attention override | NOT IMPLEMENTED | Region/overlap/role/token/mask/intermediate attention fixtures; Gate 7 |
| Regional LoRA | Unfused text/image delta gates with unsafe target omission | NOT IMPLEMENTED | Target/delta/leakage/global-plus-regional fixtures; Gate 8 |
| Prompt emphasis | Attention-logit boost by resolved token span | NOT IMPLEMENTED | Occurrence/token-span and spatial-scope fixtures; Gate 7/8 |
| Projector presets / custom vector | Global or token-selective 1x12 delta | NOT IMPLEMENTED | Projector output delta and identity protection fixtures; Gate 8 |
| LoRA-delta attention adaptation | Bounded next-step regional scale | NOT IMPLEMENTED | Per-step statistics/scale fixtures; Gate 8 |
| Image editing | VAE source encode, denoise mask, exact exterior composite | NOT IMPLEMENTED | Strength/aspect/mask/boundary/source-pixel fixtures; Gate 9 |
| Face detection | ONNX detector wrapper | NOT IMPLEMENTED | Asset/provider/box fixtures and license resolution; Gate 9/11 |
| Face refinement | Crop-local img2img and composite | NOT IMPLEMENTED | Crop assignment, seed, LoRA, blend, failure fixtures; Gate 9 |
| Post-upscale | Pillow or ComfyUI tiled model upscale | NOT IMPLEMENTED | Size, seam, metadata, memory fixtures; Gate 10 |
| PNG/project metadata | Supported by current runtime | NOT IMPLEMENTED | Key/type/value equality and backward round-trip; Gate 2 onward |
| Model registry independent of ComfyUI | Exact arbitrary paths work, defaults/discovery are ComfyUI-shaped | NOT IMPLEMENTED | Hash/symlink/duplicate/missing/legacy scan tests; Gate 3 |
| Strict component loading | ComfyUI loader plus K2 header manifest | NOT IMPLEMENTED | Key/shape/count/dtype/unload reports; Gate 4 |
| CUDA | Supported through selected ComfyUI environment | NOT IMPLEMENTED | Approved GPU matrix and soak tests; Gates 5/10 |
| ROCm | Supported through selected ComfyUI environment | NOT IMPLEMENTED | Local smoke, attention backend, FP8 and memory evidence; Gates 5/10 |
| Desktop entry point | Supported | NOT IMPLEMENTED | Identical shared-schema fixture through UI service; Gate 11 |
| RunPod entry point | No implementation in this repository | BLOCKED | Scope/repository, job service, persistence/reconnect tests; Gate 11 |
| Structured error taxonomy | Plain exception type/message events | NOT IMPLEMENTED | Every category serialized with correlation and retry fields; Gate 2 onward |
| Durable correlation ID | Per-command UUID only | NOT IMPLEMENTED | Same ID across UI/service/worker/output; Gate 2/11 |

No row may move from `NOT IMPLEMENTED` to `PASS` without a stored report linked to the
exact fixture, component hashes, software versions, hardware, and comparison thresholds.
