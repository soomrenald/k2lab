# Gate 5 native clean-generation report

Date: 2026-07-29

Gate status: **NOT COMPLETE**

Local clean-generation parity status: **PASS WITH APPROVED DIFFERENCE**

The standing gate authorization permits work to continue, but does not turn missing
reliability or infrastructure evidence into a pass. Native remains developer-only and
the product default remains `comfyui`.

## Pinned implementation

- K2Lab before this report: `4c5cfca581719c0f2af04d8c988dd466ad34f947`
- k2core clean generation: `04d5b7115791739a5e091b517125250fe6dcb70b`
- k2core parity utility: `bc653e493d2378b166d1bcb968afe7dd492f3efa`
- ComfyUI reference: `285a98944c397a4a81f15ac63d69fa3dbc0a27b9`

The versioned fixture is
`tests/fixtures/parity/clean_t2i/krea2_turbo_teapot_512.json`. Large PNG and tensor
artifacts remain outside Git and are identified by SHA-256.

## Local golden request

Synthetic prompt: “A small red ceramic teapot on a plain white table, studio lighting.”

| Field | Value |
| --- | --- |
| Size | 512×512 |
| Steps | 8 |
| Seed | 20260729 |
| Sampler / scheduler | Euler / simple |
| CFG / denoise | 1.0 / 1.0 |
| GPU | AMD Radeon Graphics, 17,095,983,104 bytes |
| Torch / HIP | 2.10.0+rocm7.1 / 7.1.25424 |
| Diffusers / Transformers | 0.39.0 / 5.14.1 |

All three component hashes and the tokenizer directory hash match the approved registry.
Initial noise uses an explicit CPU generator; the nine-value sigma schedule matches the
current Comfy reference exactly.

## Image parity

Decoded RGB pixels were compared independently of volatile PNG timestamp and correlation
metadata.

| Measurement | Observed | Acceptance |
| --- | ---: | ---: |
| Size / mode | exact 512×512 RGB | exact |
| Cosine | 0.9998963 | at least 0.999 |
| Mean absolute error | 0.0056209 | at most 0.015 |
| RMSE | 0.0113998 | at most 0.04 |
| Maximum absolute error | 0.2980392 | at most 0.5 |
| PSNR | 38.8621 dB | at least 30 dB |

The difference is attributed to the measured accumulation of small scaled-FP8
transformer arithmetic differences. Prompt tokens, seed/noise, sigma order, Euler update,
latent normalization, output dimensions, and semantic content match. Human review found
the images visually equivalent.

## Repeatability, cancellation, and cleanup

- Two native eight-step runs in one loaded pipeline produced byte-identical decoded RGB
  pixels, SHA-256 `b04a747c3b92720d30e0ee592465dfb0a8b869d764e9c058ba0baa0f96e45b87`.
- Cancellation requested from the step-1 callback stopped before step 2 and returned a
  structured `CancellationError` with the original correlation ID.
- Cancellation cleanup left 120,537,600 allocated bytes before final pipeline unload.
- Final unload left 113,246,208 allocated/reserved bytes, the observed Diffusers/ROCm
  context floor.
- Native peak allocated memory was 13,917,537,792 bytes.
- Native generation took 18.10 seconds cold and 11.87 seconds warm after one load.
- The identical current Comfy low-VRAM reference took 78.27 seconds. This local
  measurement is not a CUDA or RunPod performance claim.

## Gate checklist

| Requirement | Status | Evidence / remaining work |
| --- | --- | --- |
| Golden fixture parity | PASS WITH APPROVED DIFFERENCE | Versioned fixture and metrics above |
| Repeated-seed consistency | PASS | Two pixel-exact native repeats |
| Dimension tests | NOT IMPLEMENTED | 128×128 VAE and 512×512 end-to-end exercised; matrix remains |
| Cancellation | PASS | Cancel after step 1, structured error, cleanup measured |
| Malformed input handling | PARTIAL | Shared schema and unsupported-control tests pass; expand matrix |
| VRAM cleanup | PASS | Cancellation and final unload measurements above |
| 10 sequential generations | NOT IMPLEMENTED | Still required |
| 100-generation supported RunPod soak | BLOCKED | RunPod scope/spend remain deferred |
| Desktop smoke | NOT IMPLEMENTED | Shared worker integration still required |

`PARTIAL` in the checklist is descriptive progress, not a parity status. It does not count
as a Gate 5 pass.

## Automated tests

- k2core: 163 passed, 2 intentionally skipped, 6 subtests passed.
- K2Lab: 181 passed, 2 intentionally skipped, 6 subtests passed.

## Rollback

Revert the K2Lab fixture/report/pin commit, then revert k2core parity utility commit
`bc653e4`, clean-orchestrator commit `04d5b71`, and earlier native executable milestones as
needed. The normal backend remains ComfyUI throughout.
