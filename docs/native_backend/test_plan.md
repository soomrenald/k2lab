# Native backend test and parity plan

## Goals

The harness must characterize the current ComfyUI path before native implementation,
compare both backends through the same immutable request, and retain enough intermediate
evidence to localize divergence. It must never silently fall back to ComfyUI.

## Existing baseline

On the Phase 0 branch:

```text
QT_QPA_PLATFORM=offscreen .venv/bin/python -m pytest -q
171 passed, 2 skipped, 6 subtests passed
```

The skipped tests intentionally defer Torch work to the configured GPU worker. This is a
contract/unit baseline, not generation parity or performance evidence.

## Fixture layout

```text
tests/fixtures/parity/
  clean_t2i/
  turbo_t2i/
  ordinary_lora/
  multi_lora/
  regional_prompt/
  regional_lora/
  projector/
  prompt_emphasis/
  image_edit/
  face_refinement/
  cancellation/
  metadata/
```

Each case should have:

```text
request.json
environment.json
components.json
expected_metadata.json
reference.png
checkpoints/
  prompt_tokens.*
  prompt_embeddings.*
  initial_noise.*
  sigmas.*
  step-N-transformer-input.*
  step-N-transformer-output.*
  step-N-scheduler-output.*
  final_latents.*
report.json
```

`components.json` records SHA-256, byte size, safetensors metadata, architecture and role
for transformer, encoder, VAE, every LoRA, detector, and upscaler. `environment.json`
records backend/revision, Python, Torch, driver/runtime, GPU, dtype, attention backend,
allocator settings, OS, and K2Lab/k2core revisions. Prompts must be synthetic or explicitly
approved for fixture storage.

Large tensors and model files must use an approved artifact store with immutable hashes;
do not add them to Git by default.

## Harness design

1. Deserialize one `GenerationRequest`.
2. Reject unsupported capabilities before GPU loading.
3. Run the explicitly named backend; never fallback.
4. Capture structured timings, memory, metadata and selected intermediate checkpoints.
5. Release the backend in `finally`.
6. Compare against the fixture using per-checkpoint rules.
7. Emit one of the five parity statuses plus machine-readable measurements.

Reference capture must occur through the first `ComfyUIBackend`, not by calling old and
new request builders separately. This proves that both backends receive the same schema.

Checkpoint hooks must be disabled by default and must record their timing/memory overhead.
Performance runs must not enable checkpoint capture.

## Comparison rules

Thresholds are intentionally not invented in Phase 0. Gate owners must approve numeric
tolerances after repeated reference runs establish same-backend variability on each
supported hardware/runtime class.

| Evidence | Initial rule |
| --- | --- |
| Request and metadata | Exact canonical JSON equality except declared volatile fields |
| Model/component identity | Exact hash equality |
| Token IDs, region order, masks, seed, sigma/timestep sequence | Exact equality |
| Initial noise | Exact equality when generator/device permits; otherwise investigated, not waived |
| Embeddings and intermediate tensors | Absolute/relative tolerance derived from repeated reference runs, reported with max/mean/RMS error |
| Final latents | Tensor tolerance plus cosine/RMS similarity |
| Image | Exact size/mode; perceptual metrics and difference images; human review where required |
| Exterior protected edit pixels | Exact equality where current contract promises it |
| Cancellation | No more than one denoising step plus measured cleanup time |
| Performance | Native clean ≤20% slower, regional ≤25% slower, peak VRAM ≤15% higher |
| Soak | No persistent memory growth over 100 jobs |

No tolerance may hide a semantic difference in token spans, region ownership, LoRA target
selection, scheduler choice, denoise strength, seed, or metadata.

## Test layers by phase

### Gate 2: abstraction

- schema boundary values and canonical serialization;
- `K2LAB_BACKEND` unset, `comfyui`, `native`, and invalid;
- default and logs prove ComfyUI selection;
- `ComfyUIBackend` characterization against current worker results;
- unsupported native features fail explicitly;
- structured errors, progress, cancellation token, correlation ID;
- no UI-visible change and project/PNG metadata regression suite.

### Gate 3: registry

- arbitrary absolute paths and symlinks;
- legacy ComfyUI scanner;
- missing/unreadable/wrong-format/wrong-architecture/wrong-shape files;
- duplicate IDs and hash mismatch;
- no model file mutation or copy.

### Gates 4–5: loading and clean generation

- strict keys, shapes, parameter count, dtype, device and unload;
- repeated load/unload and VRAM leak;
- token/embedding/noise/sigma/transformer/scheduler/VAE checkpoints;
- seeds, dimensions, malformed request, progress and cancellation;
- ten sequential generations and desktop smoke;
- 100-job supported-RunPod release soak: passed on NVIDIA A40 at 512×512, eight
  Euler/simple steps, with pixel-exact output, zero median GPU/RSS growth, and terminal
  cleanup to 0 MiB; see the Gate 12 report.
- clean native-only image: local immutable-base build and signed RC2 publication passed
  no-ComfyUI assertion, pinned import smoke, `pip check`, agent boot, Docker health,
  authenticated health, remediated HIGH/CRITICAL scan, SPDX SBOM, and independent Cosign
  verification; fresh RunPod GPU boot and clean desktop integration remain blocked.
- paired identical-workload A40 peak memory: ComfyUI 18,889 MiB, native 13,653 MiB,
  native-to-ComfyUI ratio 0.7228 against the 1.15 ceiling, with both workers returning to
  0 MiB.

### Gates 6–8: LoRA and regions

- no/one/multiple/zero/negative LoRA, order, missing and incompatible targets;
- ordinary enable/disable without stale deltas;
- one/multiple/overlap/edge/full/tiny/invalid region and aspect ratios;
- global plus regional prompting/LoRA;
- Q/K/V target behavior, unsafe omission, token deltas and leakage diagnostics;
- projector and prompt emphasis characterization;
- no-region and disabled-LoRA baselines remain unchanged.

### Gates 9–11: editing, devices, products

- edit strength 0/low/medium/full, padded sizes and aspect ratios;
- denoise mask, exact protected exterior, boundary diagnostics;
- face detection/refinement where included;
- CUDA, ROCm, CPU-policy and supported dtype matrix;
- intentional OOM, one safe fallback, recovery and cancellation;
- identical serialized request through desktop and RunPod;
- timeout classification, disconnect/reconnect, completed-output recovery and no duplicate
  execution.

## Required reports at every gate

Every gate report must record changed files, automated tests, manual runs, parity statuses,
performance measurements, known differences, unresolved risks, exact rollback, and the
recommended next step. A missing GPU or fixture is `BLOCKED` or `NOT IMPLEMENTED`, never a
pass.
