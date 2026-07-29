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
| Backend-neutral schema round-trip | Typed schemas now wrap the legacy worker dictionaries | PASS | Exact successful result contract and request delegation tests; Gate 2 |
| Default backend remains ComfyUI | Unset and explicit `comfyui` select the adapter | PASS | Selector, worker integration, rollback tests and real worker log; Gate 2 |
| Clean Krea2 Turbo text-to-image | Supported, CFG 1.0 | PASS WITH APPROVED DIFFERENCE | Local 512×512 eight-step golden fixture: image cosine 0.9998963, MAE 0.0056209, pixel-exact native repeats; Gate 5 reliability work remains |
| Krea2 Raw / non-Turbo | Architecture selectable; production semantics unclear | BLOCKED | Product decision and reference fixture |
| Seed repeatability | Seed passed to `comfy.sample.prepare_noise` and `sample` | PASS | Explicit CPU generator, exact initial-noise contract, and pixel-exact repeated native golden runs; Gate 5 |
| Dimensions / 16-pixel alignment | Supported | PASS | Square, portrait, landscape, and invalid-alignment latent-shape matrix; 512×512 local execution |
| Sampler/scheduler combinations | UI exposes copied ComfyUI registries | PASS | Approved first pair Euler/simple has exact sigma construction; every other pair is rejected explicitly by native capabilities |
| Progress | Per denoising step | PASS | Ordered text, eight diffusion, and VAE events observed in golden runs; Gate 5 |
| Cancellation | Desktop terminates disposable worker | PASS WITH APPROVED DIFFERENCE | Native cooperatively stops before the next step and cleans up; current Comfy desktop terminates its disposable worker |
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
| Model registry independent of ComfyUI | Versioned TOML accepts arbitrary paths; legacy discovery is opt-in | PASS | Eight required registry cases, full suites, and read-only validation of both installed Krea2 sets; Gate 3 |
| Strict component loading | ComfyUI loader plus K2 header manifest | PASS | Exact approved hashes, 2,097/2,097 mapped tensors, parameter/dtype/device reports, and three stable ROCm unload cycles; Gate 4 |
| CUDA | Supported through selected ComfyUI environment | NOT IMPLEMENTED | Approved GPU matrix and soak tests; Gates 5/10 |
| ROCm | Supported through selected ComfyUI environment | PASS | Local 16 GiB scaled-FP8 eight-step golden, repeat, cancellation, and cleanup evidence; wider Gate 10 matrix remains |
| Desktop entry point | Supported | NOT IMPLEMENTED | Identical shared-schema fixture through UI service; Gate 11 |
| RunPod entry point | No implementation in this repository | BLOCKED | Scope/repository, job service, persistence/reconnect tests; Gate 11 |
| Structured error taxonomy | Legacy fields retained with additive structured error payload | PASS | Category conversion and explicit unsupported-native worker evidence; Gate 2 |
| Durable correlation ID | Per-command UUID only | NOT IMPLEMENTED | Same ID across UI/service/worker/output; Gate 2/11 |

No row may move from `NOT IMPLEMENTED` to `PASS` without a stored report linked to the
exact fixture, component hashes, software versions, hardware, and comparison thresholds.
