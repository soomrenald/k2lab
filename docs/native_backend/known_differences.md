# Known differences and unknowns

## Phase 0

There is no native implementation, so there are no approved output differences and no
parity claims.

The following current-state differences affect migration planning:

- The specification's target has desktop and RunPod call one `k2core` service. This
  repository has a desktop-only QProcess worker and no RunPod implementation.
- The specification recommends a single `InferenceBackend` lifecycle. The pinned
  `k2core` already has narrower image/frame/video protocols and `ComfyKreaBackend`, but the
  desktop bypasses them and calls `ComfyBaselineRuntime` directly.
- The specification calls for validated shared requests/results. The current desktop and
  worker exchange mutable JSON dictionaries.
- The target requires cooperative cancellation between denoising steps. Current desktop
  cancellation terminates the entire disposable worker process.
- The target requires precise structured errors. Current worker events expose a message
  and Python exception type.
- The target requires a backend in metadata and a cross-system correlation ID. Current
  PNG metadata does not identify a backend, and command UUIDs are not persisted as job
  correlation IDs.
- Arbitrary exact model paths are already accepted, but default discovery, worker
  selection, face detector discovery, and documentation assume a ComfyUI tree.
- Current UI sampler/scheduler options mirror broad ComfyUI registries. The native
  implementation is required to implement only K2Lab-needed combinations; that subset is
  not yet approved.
- Current documentation says the generation path is eight-step, CFG-free Turbo even
  though Raw-architecture checkpoints can be selected. Raw parity scope is unresolved.

## Difference approval record

### KD-01 — Local ROCm scaled-FP8 clean-generation arithmetic

1. Fixture: `clean-t2i-krea2-turbo-teapot-512`; clean Krea2 Turbo text-to-image.
2. Measurement: image cosine `0.9998963`, MAE `0.0056209`, RMSE `0.0113998`,
   maximum error `0.2980392`, and PSNR `38.8621 dB`; native repeats are pixel-exact.
3. Exact cross-implementation pixels are impractical because the native upstream graph
   and Comfy graph accumulate small scaled-FP8 rounding differences across 28 blocks.
   Tokens, RNG, sigmas, Euler semantics, dimensions, and latent normalization match.
4. Human review found no meaningful semantic or composition difference in the synthetic
   fixture.
5. Approved under the user's standing authorization for Gate 5 on 2026-07-29, subject to
   the explicit thresholds in the versioned fixture. This does not approve Gate 5 while
   its reliability checklist remains incomplete.
6. Rollback keeps `comfyui` selected or reverts the native orchestrator and executable
   milestones. No fallback was removed.

### KD-02 — Local full-denoise finite-feather image-edit boundary

1. Fixture: `image-edit-krea2-two-vases-512`; regional source-latent editing.
2. Measurement: medium native/Comfy support cosine `0.9996564`, MAE `0.0079054`, and
   zero changed exterior pixels. The full-denoise stress case uses a 16-pixel composite
   feather; the normal comparison uses 48 pixels.
3. Sigma-1 local editing reconstructs the masked area independently, so a finite
   rectangular composite cannot guarantee semantic continuity for every prompt.
4. Low/medium edits are visually continuous. Full denoise can show the finite support,
   especially when the feather is deliberately reduced; exterior pixels remain exact.
5. Approved under the user's standing Gate 9 authorization on 2026-07-29, with the
   stress/default outputs and control mitigations recorded in the versioned fixture.
6. Rollback keeps `comfyui` selected or reverts the Gate 9 core and K2Lab commits. The
   Comfy fallback and its matching mask semantics remain available.

Future entries must include:

1. affected fixture and feature;
2. measured comparison;
3. reason exact parity is impractical or undesirable;
4. user-visible effect;
5. approving person/date/gate;
6. rollback impact.

## Gate 10 validation boundary

- Local ROCm device ownership, BF16/FP16 compute, mixed FP8 weight loading, tiling,
  explicit CPU VAE, OOM classification, cancellation, recovery, and cleanup pass.
- NVIDIA A40 CUDA correctness, memory telemetry, determinism, tiling, FP16, OOM,
  cancellation, recovery, repeated loading, and exact terminal cleanup pass.
- No 80 GB device was available.

## Gate 11 validation boundary

- Desktop and RunPod worker entrypoints parse the same byte-identical fixture into the
  shared k2core schema. Live RunPod direct-worker and persistent job-service runs produce
  pixel-exact RGB output.
- RunPod native selection is server-only and developer-only. Unsupported pose,
  projector, post-upscale, and face-refinement controls fail explicitly before GPU work.
- The feature branch was tested against the pinned current RunPod image through an
  ephemeral checkout. A clean build, publication, and boot of the updated container is
  still required for release readiness.
- Durable idempotency, event cursor reconnect, completed-output retention, correlation
  metadata, cancellation, and precise timeout/disconnect classification pass.

## Gate 12 release-readiness boundary

- The release-workload 100-job native A40 soak passes with pixel-exact output, zero
  median GPU/RSS growth, and terminal cleanup to 0 MiB.
- Paired identical-workload A40 sampling passes the peak-memory threshold: native used
  72.28% of the ComfyUI lifecycle peak against a 115% ceiling.
- A clean native-only image has been built and its agent booted locally from the approved
  immutable CUDA base. Its remediated digest passes the pinned HIGH/CRITICAL scan and has
  an SPDX SBOM. Signed RC2 was published, booted on a fresh RunPod A40, generated
  natively after a structured failure-recovery check, and swapped in place to the
  preserved ComfyUI image; rollback health, retained inventory, ComfyUI generation, and
  destructive cleanup passed. A clean installed desktop wheel also passed QML packaging,
  session-only native selection, capability gating, prompt-safe reporting, and
  non-persistent ComfyUI fallback. GPU generation was not repeated in that temporary
  desktop environment.
- Dependency notices are recorded, K2Lab and k2core are Apache-2.0, and the exact
  FantasyPortrait detector hash matches the official Apache-2.0 artifact. The approved
  model policy distributes no weights and limits tested deployment to private,
  operator-reviewed workspaces. The unresolved converted-Qwen notice chain therefore
  blocks future redistribution of that artifact, not source-only distribution. Human
  review accepted clean generation and regional LoRA, initially rejected perceptually
  ineffective ordinary-LoRA and image-edit examples, then approved the corrective
  real-LoRA and full-object eight-step edit candidates on 2026-07-30.
- Native remains opt-in and `comfyui` remains the product default.
