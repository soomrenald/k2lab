# Native backend migration risk register

Probability and impact are qualitative Phase 0 estimates. Owners are proposed roles, not
assignments.

| ID | Risk | Probability / impact | Mitigation and detection | Gate / owner |
| --- | --- | --- | --- | --- |
| R01 | Current reference is not reproducible because ComfyUI revision, environment and model hashes are not enforced | High / Critical | Capture exact runtime/component manifests before fixtures; reject incomplete reference runs | Gate 1–2 / release owner |
| R02 | Backend abstraction changes current UI or output semantics | Medium / Critical | Characterization tests around current request dictionaries, metadata and worker events; default remains ComfyUI | Gate 2 / desktop + core |
| R03 | `k2core` lives in a separate repository and consumer pins can drift | High / High | Phase branches and tests in `k2core`; update `pyproject.toml` and `uv.lock` together; record both commits in reports | Every gate / core |
| R04 | Native Krea model/key mapping silently omits or initializes weights | Medium / Critical | Strict loading default; exact key/shape/count/dtype reports; no unexplained keys | Gate 4 / model loading |
| R05 | Tokenization or prompt embeddings differ and invalidate all regional token spans | High / Critical | Exact token fixtures and embedding checkpoints before sampling work | Gate 4–5 / text encoder |
| R06 | Noise, sigma, scheduler or CFG semantics differ despite similar-looking output | High / Critical | Exact RNG/schedule fixtures and per-step latent comparison; implement only approved pairs | Gate 5 / sampling |
| R07 | FP8 format/optimization differs across CUDA and ROCm | High / High | Record weight/compute dtype and kernel backend; hardware matrix; BF16 reference path where feasible | Gate 4–5, 10 / loader + devices |
| R08 | Replacing ModelPatcher changes unfused LoRA strength/order or mutates base FP8 weights | High / Critical | Base-weight hashes, per-target delta tests, deterministic adapter order, unload/reload tests | Gate 6 / adapters |
| R09 | Regional LoRA or attention leaks identity/content outside boxes | Medium / Critical | Q/K/V/hidden/attention diagnostics, outside-gate RMS, visual overlap matrix, human review | Gate 7–8 / regional controls |
| R10 | Current “unsafe target omission” semantics are misunderstood | Medium / Critical | Characterize exact allowed/skipped target sets from representative LoRAs; stop rather than broaden targeting | Gate 6–8 / adapters |
| R11 | Image-edit native mask/latent scaling causes boundary artifacts or changes protected pixels | High / High | Exact exterior-pixel assertions, latent-mask checkpoints, difference images and visual review | Gate 9 / editing |
| R12 | Memory/offload rewrite leaks VRAM or repeatedly retries paid GPU work | High / Critical | Explicit device plan, `finally` cleanup, one opt-in retry maximum, load/unload and 100-job soak | Gate 10 / runtime |
| R13 | Cancellation behavior changes from process kill to cooperative token and leaves state alive | Medium / High | Test cancellation at load/encode/each denoise/decode; verify state and memory cleanup | Gate 5/10 / service |
| R14 | Broad ComfyUI sampler UI promises more than the native scope can support | High / High | Approve a required combination list; capability-driven controls with explicit explanation | Gate 1/5 / product |
| R15 | RunPod behavior diverges because no current RunPod implementation exists here | High / Critical | Identify scope before coding; use shared schemas/service; durable job/correlation/output store tests | Gate 1/11 / platform |
| R16 | Private prompts leak through parity artifacts or structured logs | Medium / High | Synthetic fixtures, prompt redaction by default, opt-in warning, artifact access policy | Gate 2 / security |
| R17 | GPL-3.0 ComfyUI code is copied into a differently licensed product | Medium / Critical | No vendoring before formal review; prefer upstream libraries and clean-room behavior specs | Before code copying / legal |
| R18 | K2Lab/`k2core` have no declared license, preventing compatibility conclusions | High / High | Establish repository licenses and third-party policy before vendoring/distribution changes | Gate 1 / owner + legal |
| R19 | Face detector model provenance/license is unclear | Medium / High | Record source, hash, model license and redistribution terms; require explicit configured asset | Gate 3/9 / legal + model registry |
| R20 | Golden tensors/images are too large or hardware-sensitive for Git | High / Medium | Approved immutable artifact store, hashes in Git, same-backend variability study | Gate 2 / test infrastructure |
| R21 | Existing QML/hidden-Widgets architecture causes duplicate backend business logic | High / High | Add one application service behind `MainWindow`; do not reimplement in QML or RunPod | Gate 2/11 / application |
| R22 | Error conversion hides actionable ComfyUI/native failure causes | Medium / High | Typed categories with original detail/cause, retry/GPU-started fields, category tests | Gate 2 onward / service |

Gate 12 update: R12's required 100-job native A40 soak passed with zero median GPU and
worker-RSS growth and terminal cleanup to 0 MiB. The clean native-only image also built
and booted locally without ComfyUI; its remediated digest passes the pinned vulnerability
policy and has a hashed SPDX SBOM. The paired identical-workload A40 comparison also
passes at a 0.7228 native-to-ComfyUI peak ratio. R12 remains open only for a
published-candidate RunPod GPU boot and release rollback drill. R17–R19 also remain open:
notice inventories do not replace first-party, dependency, or model-license decisions.

## Highest-priority Gate 1 decisions

1. Establish the exact production reference environment and hashes.
2. Approve Turbo/Raw and sampler/scheduler scope.
3. Identify the RunPod codebase and deployment owner.
4. Establish K2Lab/`k2core` licensing and the no-vendoring posture.
5. Approve storage and privacy rules for golden artifacts.
