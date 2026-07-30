# Native backend migration rollback

## Phase 0

Phase 0 changes documentation only. No runtime, dependency, configuration, UI, worker,
model, output, or user-data behavior changes.

To leave the migration work and return to the production branch:

```bash
git switch main
```

To inspect or revert the Phase 0 documentation commit after it is created:

```bash
git log --oneline feature/native-k2-backend
git revert <phase-0-commit>
```

Do not delete untracked prompt JSON files or the untracked migration specification while
rolling back. They are user-owned and are not part of the Phase 0 commit.

## Required rollback contract for Gate 2 and later

- An unset `K2LAB_BACKEND` must select `comfyui`.
- `K2LAB_BACKEND=comfyui` must explicitly select the reference backend.
- An unsupported or failed native request must not silently fallback.
- Reverting the backend-selection/core-pin commits must restore the prior worker protocol
  without changing project/config/PNG schemas.
- Previous container tags, lockfiles and component manifests must remain available.
- A native failure must not mutate ComfyUI model files or persistent backend state.

Before passing Gate 2, verify rollback by running the complete test suite and at least one
approved ComfyUI generation after explicitly selecting `comfyui`.

## Approval-gate tags

The desktop repository is the coordination repository for approval records. Its annotated
`native-backend-gate-NN` tags point to the commit that records each passed gate; the gate
report at that commit records the exact coupled k2core and RunPod checkpoints.

| Passed gate | Coordination tag | Desktop commit |
| --- | --- | --- |
| 1 | `native-backend-gate-01` | `b64d640` |
| 2 | `native-backend-gate-02` | `649ca8b` |
| 3 | `native-backend-gate-03` | `acff502` |
| 4 | `native-backend-gate-04` | `7e1d79f` |
| 5 | `native-backend-gate-05` | `f7d3cc6` |
| 6 | `native-backend-gate-06` | `520a6ca` |
| 7 | `native-backend-gate-07` | `e9f86e4` |
| 8 | `native-backend-gate-08` | `40cfb3d` |
| 9 | `native-backend-gate-09` | `bcdd3f3` |
| 10 | `native-backend-gate-10` | `1e31c4f` |
| 11 | `native-backend-gate-11` | `58ec367` |

Gate 12 is not tagged as passed. Its current later checkpoints include desktop `13a9fff`,
RunPod release source `4b091c5`, and RunPod publication/GPU-acceptance evidence
`59de1c5`; the original clean-image source is `8aff782`. Overall release readiness
remains blocked.

To return to an earlier passed gate without rewriting history, create a branch from its
tag. To remove later feature-branch changes while retaining history, revert later
checkpoints in reverse order. Do not force-move a gate tag, delete a prior container tag,
or delete the preserved dependency lockfiles.

## Gate 12 rollback-drill status

The preserved ComfyUI production target is version `0.3.0` at immutable index digest
`ghcr.io/soomrenald/k2lab-runpod-workspace@sha256:19652733039379d1ef47cd3279e6b266b802c7a68a1c380173221a2d8ace6435`.
The index, linux/amd64 manifest, and attestation resolve in GHCR. Desktop unset/explicit
ComfyUI routing and the session fallback pass their targeted tests.

The reviewed candidate is published and signed at immutable digest
`ghcr.io/soomrenald/k2lab-runpod-workspace@sha256:7662f6440bd4e2a1f6059876c042df98a1e00284c89c35e6aaec3aa446be856f`.
The full drill passed on a fresh disposable NVIDIA A40. RC2 booted with authenticated
health, recovered from the expected structured native rejection, and generated
successfully. The same Pod was then reset in place to the preserved ComfyUI image while
retaining `/workspace`; authenticated rollback health, hash-verified model inventory,
and ComfyUI generation passed. The cleanup guard permanently deleted the Pod and its
volume. Compact prompt-safe evidence is recorded in
`tests/fixtures/parity/integration/gate12_rollback_readiness.json` and at RunPod
checkpoint `59de1c5`.

This verifies image rollback; it does not approve the release or change the product
default. Until the remaining Gate 12 blockers close, keep desktop
`K2LAB_BACKEND=comfyui` and RunPod `K2LAB_INFERENCE_BACKEND=comfyui` available as the
immediate runtime rollback.
