# Gate 10 native memory and device-management report

Date: 2026-07-29

Gate status: **BLOCKED — EXTERNAL CUDA HARDWARE REQUIRED**

The K2Lab-owned device manager and every locally feasible Gate 10 case pass. The
specification also requires an A40 48 GB validation; no A40 or authorized cloud job is
available in the current environment. An 80 GB GPU is likewise unavailable. Per the
migration safety rules, Phase 9 does not begin while this gate is blocked.

The native backend remains developer-only and the product default remains `comfyui`.

## Pinned implementation

- k2core: `237fd23dc4a578e9d1a095fac0587d4d6bdf88e4`
- K2Lab before this report: `4fc87b3`
- local accelerator: AMD Radeon Graphics, ROCm, 16 GiB

## Completed implementation

- A standalone K2-owned manager resolves CUDA/ROCm/explicit-CPU placement without
  importing ComfyUI memory-management globals.
- Transformer, text encoder, and VAE execute sequentially from CPU-staged checkpoint
  tensors. `cpu_offload` also stages explicitly accelerator-targeted components on CPU.
- The manager records current and peak allocator use, free/total VRAM, available/total
  RAM, requested reserve thresholds, resolved devices, backend, and device name.
- BF16 is the preserved default compute lane. Explicit FP16 converts only BF16
  parameters and buffers; FP8 matrices and FP32 scales/norm lanes remain unchanged.
- Explicit FP8 weight policy preserves the checkpoint's required mixed FP8/BF16/FP32
  lanes. Unsupported weight or compute conversions fail during planning.
- Optional VAE tiling and an explicitly configured, one-request-wide VAE OOM retry are
  supported. There is no retry loop.
- Failure paths clear request tensor references and tracebacks. Cleanup collects Python
  objects, clears ROCm/CUDA BLAS workspaces, and releases cached allocator blocks.
- The desktop worker forwards `cpu_offload` and `vae_tiling`, and flattens nested
  allocator telemetry into the existing UI result contract.

## Gate 10 validation matrix

| Required case | Status | Evidence |
| --- | --- | --- |
| Load/unload stress | PASS | 10 same-process cycles; every load and unload ended at 0 allocated / 0 reserved bytes |
| Sequential generation soak | PASS (local) | Five same-process 256×256 two-step jobs; identical pixel hash and stable per-job current/peak memory |
| Intentional OOM | PASS | 20 GiB reserve on a 16 GiB device failed before GPU work with retry-safe `OutOfMemoryError` |
| Cancellation during OOM-prone work | PASS | 512×512 eight-step request cancelled after step 1; structured cancellation and full unload cleanup |
| Recovery after failure | PASS | Fresh valid request completed immediately after preflight OOM; allocator returned to 0/0 |
| A40 48 GB | BLOCKED | No A40 or authorized paid-cloud execution environment is available |
| 80 GB GPU if available | NOT AVAILABLE | No 80 GB device is available in the current environment |
| Local ROCm smoke | PASS | BF16, FP16, FP8-weight policy, VAE tiling, CPU VAE, generation, image edit, cancellation, and cleanup exercised |

The release-level 100-job soak remains future release evidence. Gate 10's local
same-process soak found no growth across five generations, while ten component
load/unload cycles ended at exact allocator zero.

## Measured local results

Five identical 256×256 two-step BF16 jobs all produced pixel SHA-256
`c85e1d1bc39859d9e67b7d9a3cab56b858d5c6806480c6f37d94462f501c7211`.
Each job reported:

- completion: 113,704,960 bytes allocated; 117,440,512 reserved;
- peak: 13,689,917,952 bytes allocated; 14,023,655,424 reserved;
- OOM fallback: unused;
- final unload: 0 bytes allocated; 0 reserved.

The explicit FP16/FP8-policy 256×256 one-step job produced pixel SHA-256
`886c5ff5f3cf109ab4eb3262ffb0b813d9014f3b7b5fa131921d2b5dd57d5d54`,
peaked at 13,689,819,648 allocated and 14,023,655,424 reserved bytes, used no fallback,
and ended at 0/0 after unload.

A real 512×512 edit peaked at 13,766,470,656 allocated and 14,092,861,440 reserved
bytes. Explicit VAE tiling also completed. A mixed placement run kept the VAE on CPU
while text and transformer execution remained on ROCm and completed successfully.

The cancellation probe retained only 6,291,456 inactive reserved bytes immediately
after failure cleanup and reached 0/0 on unload. The earlier 113 MB post-job allocator
residue was traced to BLAS workspaces; explicit workspace cleanup removed it, with no
live tensors found by the ownership probe.

## Automated verification

- k2core: 200 passed, 2 intentional environment skips, 14 subtests.
- K2Lab: 184 passed, 2 intentional environment skips, 6 subtests.
- Static checks: Ruff clean and `git diff --check` clean.

## Known differences and unresolved risks

- CUDA behavior, A40 memory envelope, and performance are unverified.
- No 80 GB accelerator evidence exists.
- The release-level 100-job soak is not yet complete.
- FP16 is a real executable option but is not claimed pixel-identical to BF16; both are
  deterministic in the exercised request.
- Automatic placement deliberately refuses CPU when no accelerator is present. CPU
  execution must be selected explicitly.
- Native projector controls, face refinement, and post-upscale remain explicitly
  unsupported and are not enabled in the UI.

## Rollback

Keep `K2LAB_BACKEND=comfyui`, or revert the K2Lab Gate 10 pin/report commit. In k2core,
revert `237fd23`, `0f08d6e`, `799fc28`, `d171392`, `26bd6a2`, and `7ce1b4f` in reverse
chronological order. The ComfyUI production path and default are unchanged.

## Required unblock

Provide an A40 48 GB execution target or authorize a bounded cloud validation after
confirming model files, hashes, container image, disk size, GPU class, expected runtime,
cost exposure, and exact test scope. If an 80 GB target is available at that time, run
the same matrix there. Only then can Gate 10 be reconsidered and Phase 9 begin.
