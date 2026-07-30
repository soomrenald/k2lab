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

Gate 12 is not tagged as passed. Its current evidence checkpoints are desktop `db95c34`
and RunPod `ef555b3`; overall release readiness remains blocked.

To return to an earlier passed gate without rewriting history, create a branch from its
tag. To remove later feature-branch changes while retaining history, revert later
checkpoints in reverse order. Do not force-move a gate tag, delete a prior container tag,
or delete the preserved dependency lockfiles.
