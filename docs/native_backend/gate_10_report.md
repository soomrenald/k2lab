# Gate 10 native memory and device-management report

Date: 2026-07-29

Gate status: **PASS**

The K2Lab-owned device manager passes the local ROCm and required NVIDIA A40 validation
matrices. No 80 GB GPU was available, so the specification's conditional 80 GB case was
not run.

The native backend remains developer-only and the product default remains `comfyui`.

## Pinned implementation

- k2core: `237fd23dc4a578e9d1a095fac0587d4d6bdf88e4`
- K2Lab before this report: `4fc87b3`
- local accelerator: AMD Radeon Graphics, ROCm, 16 GiB
- CUDA accelerator: NVIDIA A40, CUDA 12.8, 47,697,690,624 bytes
- versioned A40 evidence:
  `tests/fixtures/parity/device/gate10_a40_48gb.json`

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
| Load/unload stress | PASS | Ten same-process cycles on ROCm and ten on A40; every terminal unload ended at 0 allocated / 0 reserved bytes |
| Sequential generation soak | PASS | Five same-process jobs on each device; pixel-exact repeats and stable per-job current/peak memory |
| Intentional OOM | PASS | Oversized reserve failed before GPU work with retry-safe `OutOfMemoryError` on both devices |
| Cancellation during OOM-prone work | PASS | 512×512 eight-step request cancelled after step 1 on both devices; structured cancellation and full unload cleanup |
| Recovery after failure | PASS | Fresh valid request completed immediately after preflight OOM on both devices; allocator returned to 0/0 |
| A40 48 GB | PASS | Exact approved artifacts; stress, soak, OOM, recovery, cancellation, tiling, FP16, and cleanup matrix |
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

## Measured A40 results

The A40 matrix used K2Lab RunPod image version `0.3.0`, PyTorch `2.9.1+cu128`,
Diffusers `0.39.0`, Transformers `5.14.1`, and the exact approved transformer, text
encoder, VAE, and tokenizer hashes. The approved scaled-FP8 transformer was downloaded
from immutable Comfy-Org/Krea-2 revision
`67751d8bd092b59c4398ad05634d605f1d8b6a2f` and verified before inference.

Ten load/unload cycles ended at exact allocator zero. Five 256×256 two-step jobs all
produced pixel SHA-256
`f8f38f76571947fdc7422e499b03cb8c143d35dcda7250472e530534f775c69e`.
Every job reported:

- completion: 10,027,008 bytes allocated; 27,262,976 reserved;
- peak: 13,591,729,152 bytes allocated; 13,910,409,216 reserved;
- OOM fallback: unused;
- final unload: 0 bytes allocated; 0 reserved.

A 60 GiB reserve produced `OutOfMemoryError` with `retry_safe=true` and
`gpu_work_started=false`; allocator use remained 0/0. The immediate recovery job passed.
The 512×512 cancellation workload stopped after step 1, reported
`gpu_work_started=true`, retained no active allocation after cleanup, and reached 0/0
on unload. Explicit VAE tiling and FP16 compute with mixed FP8 weights both completed
without fallback and reached 0/0.

The full matrix took 374.84 seconds. The published A40 rate was $0.44/hour and the
predeclared validation ceiling was $0.33; this excludes prior pod uptime and storage.
Terminal `nvidia-smi` reported 0 MiB GPU memory in use.

## Automated verification

- k2core: 200 passed, 2 intentional environment skips, 14 subtests.
- K2Lab: 187 passed, 2 intentional environment skips, 6 subtests.
- Static checks: Ruff clean and `git diff --check` clean.

## Known differences and unresolved risks

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

## Recommended next step

Proceed to Phase 9 desktop and RunPod integration. Keep native developer-only and
`comfyui` as the product default until the remaining integration and release gates pass.
