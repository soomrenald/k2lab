# Gate 12 release-readiness report

Date: 2026-07-29

100-job soak status: **PASS**

Overall release readiness: **BLOCKED**

The release-workload native soak passed on one NVIDIA A40. A clean native-only image was
also built and booted locally from the approved immutable CUDA base. The native backend
remains developer-only, `comfyui` remains the default, and the existing ComfyUI path
remains available. This report does not approve a release: the clean image has not been
published or booted on a RunPod GPU, first-party and model licensing decisions remain
open, and representative output still requires human approval.

## Pinned implementation and evidence

- k2core: `237fd23dc4a578e9d1a095fac0587d4d6bdf88e4`
- desktop checkpoint before this evidence update: `65782bc`
- RunPod soak source: `79837482e458ef216ba3d990b134fd9a0a4d6ab9`
- RunPod clean-image source: `8aff7822a61526222b77adbc482edb2c429bfaa6`
- RunPod release-workflow source: `84e23282dda3c3cb28e06c49edc1e68f4b209267`
- canonical request fixture SHA-256:
  `472aa82fc8bbbd6ef65d2d5601e8ed0da9ce0d7652d7243e0acee706840a12c1`
- full resumable state SHA-256:
  `91882f8d1665e1ea0cb8cc6f06e90f1f1b81a22d0049ce9e616612ef54f46e10`
- compact versioned evidence:
  `tests/fixtures/parity/device/gate12_a40_100_job_soak.json`
- clean-image build/boot evidence:
  `tests/fixtures/parity/integration/gate12_clean_native_image.json`

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
11.87 seconds versus 78.27 seconds for ComfyUI. A later paired A40 probe used the same
canonical request, model hashes, Python/Torch environment, and 100 ms NVIDIA memory
sampling for both backends. ComfyUI peaked at 18,889 MiB and native at 13,653 MiB, a
native-to-ComfyUI ratio of 0.7228. This passes the specification's maximum ratio of 1.15.
Both workers returned the GPU from an initial 0 MiB to terminal 0 MiB.

The paired probe took 58.76 seconds and approximately $0.0072. Its compact evidence is
`tests/fixtures/parity/device/gate12_a40_backend_vram.json`; the 4,660-byte full state
remains at `/workspace/k2lab/state/gate12-backend-vram-a40-rerun.json` with SHA-256
`f5539db28a1473ba7a0a29df6f1bc1d28bb5d60ec42b6928835bf94cae8198c6`.

## Additional implementation completed

The paired RunPod branch now includes:

- a resumable, bounded soak probe that validates the workload before resuming;
- workspace-owned tokenizer uploads and readiness checks;
- a clean native-only Dockerfile candidate with one native Python environment and no
  ComfyUI clone or install;
- a pull-request/manual validation workflow that never publishes an image, plus an
  explicit `native-v*` release-candidate path that publishes the content-addressed
  native image to GHCR and signs its digest with GitHub OIDC;
- immutable-base, no-ComfyUI, import, dependency, authenticated boot, Trivy, and SPDX
  checks that validate the local or pushed digest before release completion;
- third-party notices for desktop, k2core, and RunPod, plus automated drift checks.

The desktop branch now also includes a session-only **Native K2 (experimental)**
selector, one-click ComfyUI fallback, capability re-gating in both directions, and a
prompt-safe issue-report ZIP. The selector does not persist or change the environment;
restart continues to use `K2LAB_BACKEND`, whose unset default is ComfyUI.

The clean Dockerfile was built locally from
`docker.io/nvidia/cuda@sha256:ac55d124da4882b497f732d8dfd9a702d5447a5f29d08d56da6f64f0a1eb34bc`.
Its final local image ID and manifest-list digest are
`sha256:2364e8ed1d75a66a951d35797b6dd80afb2c4ec86370e31471c678eeb3e0a094`;
the image is 7,252,176,589 bytes. The build uses one Python environment, installs the
web dependency closure from the checked-in `uv.lock` export with hash enforcement, and
installs the local package with `--no-deps`. It runs `pip check`, then removes pip,
setuptools, and wheel because they are not runtime dependencies.

Local clean-image validation confirmed:

- no `/opt/ComfyUI` tree or ComfyUI-named directory under `/opt`;
- native is the image default and `/opt/k2lab-venv/bin/python` is the worker;
- pinned Torch, Diffusers, Transformers, Safetensors, FastAPI, Uvicorn, and k2core
  imports succeed;
- `pip check` reports no broken requirements;
- development `node_modules` and runtime pip/setuptools/wheel are absent;
- the agent boots from an empty temporary `/workspace`, passes Docker health, and returns
  authenticated status `ready` with container, agent, and storage readiness true.

Model and worker readiness were false as expected because the temporary smoke workspace
contained no model weights.

The first pinned Trivy 0.70.0 scan correctly failed with four fixed HIGH findings: two
from an accidentally copied development TypeScript binary and two from pip's embedded
package inventory. The image was narrowed to runtime sources, entrypoint, and notices;
`node_modules` was excluded; and pip/setuptools/wheel were removed only after the
successful dependency check. The rebuilt digest passed the same policy with zero HIGH,
zero CRITICAL, and zero secrets. The 731,482-byte Trivy JSON has SHA-256
`a893fe202d19319278d2fc22d01f9d6c512079d5bb50f95f55daa2b8033a4a5c`.

Pinned Syft 1.42.3 emitted a 5,467,463-byte SPDX 2.3 JSON with 243 packages, 6,041
files, and 7,107 relationships. Its SHA-256 is
`5397252aeeec5e98a2c563f845229e48b85a2dc192f730d3ed7c83bea5595c2c`.
Both raw local artifacts remain in `/tmp/k2lab-gate12-native-evidence`; the checked-in
compact fixture records their hashes and summary. The image was not published, promoted,
or executed on the RunPod GPU.

## Automated verification

- k2core: 202 passed, 2 intentional environment skips;
- desktop K2Lab after this evidence update: 205 passed, 2 intentional environment
  skips, 6 subtests;
- RunPod after the release-candidate workflow: 316 passed, 15 intentional
  environment/live-test skips, 16 subtests;
- RunPod frontend typecheck, contract tests, and production build passed;
- Ruff and `git diff --check` passed in all changed repositories.

## Remaining release blockers

- Create an approved `native-v*` release-candidate tag, allow its workflow to publish and
  sign the content-addressed native-only candidate, and boot that digest on a RunPod GPU
  for clean-install desktop/RunPod smoke, failure recovery, and rollback checks.
- Obtain human approval for representative native outputs.
- Decide and record K2Lab and k2core first-party licenses, model redistribution terms,
  and face-detector provenance. The notice inventory is evidence, not legal approval.
- Exercise the experimental selector, one-click fallback, and issue-report bundle in
  the clean release-candidate image.
- Preserve prior image tags and dependency locks and perform the documented rollback
  drill. Do not make native the default during this work.

## Rollback

Set `K2LAB_BACKEND=comfyui` on desktop and
`K2LAB_INFERENCE_BACKEND=comfyui` on RunPod. Revert the feature-branch checkpoints in
reverse order if source rollback is required. No database migration, user setting,
model path, or output schema was changed irreversibly, and no ComfyUI support was
removed.

The current production rollback target remains available at immutable version `0.3.0`,
index digest
`ghcr.io/soomrenald/k2lab-runpod-workspace@sha256:19652733039379d1ef47cd3279e6b266b802c7a68a1c380173221a2d8ace6435`.
Its linux/amd64 manifest and attestation resolve in GHCR. The unset/explicit ComfyUI
selection and session fallback matrix passes 28 targeted tests, and the Gate 1–11
coordination tags resolve remotely.

These are rollback prerequisites, not a completed image-swap drill. The reviewed native
image is intentionally local-only, so RunPod cannot switch to its immutable registry
digest and then back to the preserved ComfyUI digest. The compact blocked record is
`tests/fixtures/parity/integration/gate12_rollback_readiness.json`.

## Recommended next step

Create an approved `native-v*` tag to publish and sign the exact reviewed candidate. Boot
that digest on a RunPod GPU, run the clean-install validation matrix, and perform the
rollback drill. Keep native opt-in until the remaining blockers are closed.
