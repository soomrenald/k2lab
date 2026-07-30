# Gate 12 release-readiness report

Date: 2026-07-29

100-job soak status: **PASS**

Overall release readiness: **BLOCKED**

The release-workload native soak passed on one NVIDIA A40. The native backend remains
developer-only, `comfyui` remains the default, and the existing ComfyUI path remains
available. This report does not approve a release: the clean native-only image has not
yet been built and booted, first-party and model licensing decisions remain open, and
representative output still requires human approval.

## Pinned implementation and evidence

- k2core: `237fd23dc4a578e9d1a095fac0587d4d6bdf88e4`
- desktop checkpoint before this evidence update: `65782bc`
- RunPod source: `79837482e458ef216ba3d990b134fd9a0a4d6ab9`
- canonical request fixture SHA-256:
  `472aa82fc8bbbd6ef65d2d5601e8ed0da9ce0d7652d7243e0acee706840a12c1`
- full resumable state SHA-256:
  `91882f8d1665e1ea0cb8cc6f06e90f1f1b81a22d0049ce9e616612ef54f46e10`
- compact versioned evidence:
  `tests/fixtures/parity/device/gate12_a40_100_job_soak.json`

The remote state is 132,353 bytes and remains at
`/workspace/k2lab/state/gate12-native-soak-a40.json`. It contains every iteration,
timing, image hash, and memory sample. The Git fixture contains the release-relevant
summary and the raw-state digest without storing pod-specific output paths for all 100
images.

## Workload and environment

The resident native worker generated 100 sequential 512×512 images at eight Euler/simple
steps with seed 424242 and the Gate 11 canonical synthetic request. Models remained loaded
between jobs. Every component was content-addressed:

- transformer:
  `eb4dd8c612cfd10f64f25b057e6e6bbcb5737c94a7372177e456dbf7579502f1`
- text encoder:
  `54bd5144df0bbc25dd6ccadfcb826b521445a1b06ae5a42570bdd2974ca87094`
- VAE:
  `a70580f0213e67967ee9c95f05bb400e8fb08307e017a924bf3441223e023d1f`
- tokenizer:
  `9362730d7f1fe82e277f363f2294f30edb2bb81b5c67b0d1b83813a5ac21f34d`

The pod exposed one NVIDIA A40 with 46,068 MiB, driver 580.159.04, CUDA 12.8,
PyTorch 2.9.1+cu128, Diffusers 0.39.0, Transformers 5.14.1, Safetensors 0.8.0,
and Python 3.12.3.

## Results

| Check | Result |
| --- | --- |
| Completed jobs | 100/100 |
| Determinism | PASS; one RGB pixel hash across all jobs |
| Pixel SHA-256 | `351a407e207cf64a6831772a8941b94bfea08d2abe351b7f8faaf6fc13fe8bb7` |
| Warm GPU growth | PASS; 379 MiB first-window median and 379 MiB last-window median |
| Worker RSS growth | PASS; 19,729,321,984-byte first and last medians |
| Peak allocator use | 13,687,942,656 bytes allocated; 13,994,295,296 bytes reserved |
| Cleanup | PASS; 0 MiB before, 0 MiB after, 1.52 seconds |
| Generation time | 9.69-second mean; 9.32-second p50; 11.60-second p95 |
| Phase means | text 4.04 s; transformer 4.63 s; VAE 0.37 s; output 0.14 s |
| Session / cost | 999.25 seconds; approximately $0.1221 at $0.44/hour |

Each PNG has different embedded job metadata and therefore a distinct file hash, while
decoded RGB pixels are exact across all 100 outputs.

The earlier controlled local clean-generation fixture measured native warm generation at
11.87 seconds versus 78.27 seconds for ComfyUI, so the available identical-job evidence
does not show a severe total-time regression. The A40 run did not capture a matching
ComfyUI peak-memory baseline; the specification's 15% cross-backend VRAM threshold
therefore remains unresolved rather than being inferred from unlike measurements.

## Additional implementation completed

The paired RunPod branch now includes:

- a resumable, bounded soak probe that validates the workload before resuming;
- workspace-owned tokenizer uploads and readiness checks;
- a clean native-only Dockerfile candidate with one native Python environment and no
  ComfyUI clone or install;
- a pull-request/manual validation workflow that never publishes an image, requires an
  immutable base digest, asserts that ComfyUI is absent, runs imports/tests, scans with
  Trivy, and emits an SPDX bill of materials;
- third-party notices for desktop, k2core, and RunPod, plus automated drift checks.

The Dockerfile and workflow are source-level candidates only. No immutable base digest
was available locally, so no clean image has been built, booted, or promoted.

## Automated verification

- k2core: 202 passed, 2 intentional environment skips;
- desktop K2Lab after this evidence update: 198 passed, 2 intentional environment
  skips, 6 subtests;
- RunPod before the soak-probe-only commits: 310 passed, 15 intentional
  environment/live-test skips, 16 subtests;
- focused RunPod soak-probe contract tests after those commits: 4 passed;
- RunPod frontend typecheck, contract tests, and production build passed;
- Ruff and `git diff --check` passed in all changed repositories.

## Remaining release blockers

- Build and boot the native-only image from an approved immutable base digest, then run
  clean-install desktop/RunPod smoke, failure recovery, and rollback checks in that image.
- Capture an identical-workload ComfyUI A40 peak-VRAM baseline or explicitly revise the
  15% memory threshold.
- Obtain human approval for representative native outputs.
- Decide and record K2Lab and k2core first-party licenses, model redistribution terms,
  and face-detector provenance. The notice inventory is evidence, not legal approval.
- Add the user-facing experimental selector, one-click fallback, and issue-report bundle
  only after the clean-image and release decisions are complete.
- Preserve prior image tags and dependency locks and perform the documented rollback
  drill. Do not make native the default during this work.

## Rollback

Set `K2LAB_BACKEND=comfyui` on desktop and
`K2LAB_INFERENCE_BACKEND=comfyui` on RunPod. Revert the feature-branch checkpoints in
reverse order if source rollback is required. No database migration, user setting,
model path, or output schema was changed irreversibly, and no ComfyUI support was
removed.

## Recommended next step

Resolve an approved immutable base-image digest and build the native-only candidate.
Boot it as a non-publishing release candidate, run the clean-install validation matrix,
then perform the rollback drill. Keep native opt-in until the remaining blockers are
closed.
