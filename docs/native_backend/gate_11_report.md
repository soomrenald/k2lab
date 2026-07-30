# Gate 11 desktop and RunPod integration report

Date: 2026-07-29

Gate status: **PASS**

Desktop and RunPod now route the same canonical request fixture through
`k2core.inference.GenerationRequest` and the native backend. The production default
remains `comfyui`, and native selection remains a server/developer setting.

## Pinned implementation

- k2core: `237fd23dc4a578e9d1a095fac0587d4d6bdf88e4`
- desktop fixture checkpoint: `9f459dd`
- RunPod integration checkpoint: `87e1c72e8e9138b28b87a9f716e5262d2111812d`
- canonical fixture SHA-256:
  `472aa82fc8bbbd6ef65d2d5601e8ed0da9ce0d7652d7243e0acee706840a12c1`
- versioned evidence:
  `tests/fixtures/parity/integration/gate11_desktop_runpod.json`

## Completed implementation

- RunPod accepts `comfyui` or `native` only from server configuration, reports the
  selection in capabilities, jobs, events, and worker results, and keeps ComfyUI as the
  default.
- Native RunPod jobs require explicit content-addressed transformer, text encoder, VAE,
  and tokenizer bindings. Unsupported pose, pose-adapter, projector, post-upscale, and
  face-refinement work fails before worker/GPU startup.
- The RunPod worker loads the exact pinned k2core implementation and uses its
  `PipelineConfig`, `GenerationRequest`, `ImageEditRequest`, progress, result, and
  structured error contracts. The existing ComfyUI implementation remains intact.
- The container pins the A40-tested native runtime versions and validates native imports
  and tokenizer availability at startup.
- Agent and worker diagnostics persist on the workspace volume. Normal logs record
  backend, component hashes, placement/progress summaries, seed, scheduler, sampler,
  LoRA/region counts, output, errors, and cleanup without logging full prompts.
- Startup/model-load timeout, generation timeout, premature worker disconnection, agent
  proxy timeout classes, provider timeout, and provider resource failure remain
  distinguishable.
- Existing durable command IDs, job/event files, cursors, output inventory, cancellation,
  and isolated worker termination are preserved.

## Identical fixture validation

The desktop and RunPod repositories contain byte-identical
`tests/fixtures/gate11/native_clean_generation.json` files. Both worker entrypoint tests
construct the shared `GenerationRequest` and verify correlation ID, prompt, seed,
Euler/simple selection, region, and prompt emphasis.

The A40 live worker probe used RunPod commit `bd084e8` plus pinned core `237fd23`.
The persistent agent probe used RunPod commit `87e1c72`. Both produced the same RGB
pixel SHA-256:

`eab1f3d13a3684716671777e1e560ee8b23d44d59a62d0f4fcd584f2fe908434`

The PNG file hashes intentionally differ because the service path adds its durable
project/job metadata and uses a different correlation ID. Pixel content is exact.

## Live RunPod job-service validation

The A40 service probe:

- completed in 31.88 seconds;
- returned the same job ID for the immediate duplicate submission;
- persisted 20 events and resumed from the terminal cursor with zero duplicate events;
- reconstructed `JobManager` and recovered the job as `completed`;
- resolved the retained output by the same opaque output file ID;
- retained backend, correlation ID, seed, sampler, and scheduler in PNG metadata.

Repeating the same command after reconstruction returned in 0.21 seconds with the same
job ID and output file ID and did not perform GPU inference. Terminal `nvidia-smi`
reported 0 MiB.

The direct worker probe completed in 60.98 seconds with 18 lifecycle/progress events,
`ready`, and `complete`. The two billable probes totaled 92.85 seconds, approximately
$0.0114 at the published $0.44/hour A40 rate.

## Automated verification

- k2core: 200 passed, 2 intentional environment skips, 14 subtests.
- desktop K2Lab: 191 passed, 2 intentional environment skips, 6 subtests.
- RunPod: 305 passed, 15 intentional environment/live-test skips, 16 subtests.
- Static checks: Ruff clean and `git diff --check` clean.

Regression coverage includes duplicate submission, retained outputs, reconnectable event
cursors, manager reconstruction, cancellation, resident-worker release, safe structured
failures, precise agent proxy timeouts, startup timeout, generation timeout, worker
disconnect, provider timeout reconciliation, and provider resource loss.

## Known differences and remaining release work

- The Gate 11 live probe ran the feature-branch source in an ephemeral checkout against
  the current pinned RunPod image/runtime. The updated container manifest has not yet
  been built, published, and booted as a release-candidate image.
- Native pose controls, projector conditioning, post-upscale, and face refinement remain
  explicitly unsupported and capability-gated.
- The production web UI does not expose native selection; this remains deliberate until
  release approval.
- The release-level 100-job soak, clean-install image validation, license review, and
  rollback drill remain later release work.

## Rollback

Keep `K2LAB_INFERENCE_BACKEND=comfyui` in RunPod and `K2LAB_BACKEND=comfyui` on desktop.
Revert the RunPod feature-branch checkpoints in reverse order, or revert desktop
checkpoint `9f459dd`. No ComfyUI behavior or default was removed.

## Recommended next step

Proceed to the remaining migration documentation, dependency/license, clean-image,
performance, and release-readiness work. Native must remain opt-in until those checks
pass.
