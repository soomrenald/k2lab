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
