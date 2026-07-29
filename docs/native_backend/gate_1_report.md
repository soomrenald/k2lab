# Approval Gate 1 report

Status: **APPROVED 2026-07-28**
Scope completed: Phase 0 only
Branch: `feature/native-k2-backend`

## Approval record

The user explicitly approved Gate 1 with the recommended decisions:

- current configured ComfyUI plus exact component hashes is the initial reference;
- Krea2 Turbo with `euler`/`simple` is the first native target, without removing any
  existing ComfyUI option;
- Raw and additional native sampler combinations are deferred until characterized;
- RunPod integration is deferred until its product repository is brought into scope;
- no ComfyUI source will be vendored; use upstream libraries or clean-room behavior;
- golden prompts will be synthetic, with large artifacts kept outside Git;
- Gate 2 may add schemas, structured errors, `ComfyUIBackend`, and
  `K2LAB_BACKEND=comfyui` as the default;
- native model loading remains prohibited before its later approval gate.

## Completed work

- Full direct and indirect ComfyUI/custom-node dependency inventory.
- Current generation, image-edit, and face-refinement execution paths.
- Desktop/RunPod divergence assessment.
- K2Lab-specific logic ownership map.
- Proposed backend interface, application-service boundary, module layout, shared schemas,
  and implementation order.
- Initial parity matrix and parity-test plan.
- Risk, licensing, unknowns, known-differences, and rollback records.

Detailed deliverables:

- [Architecture and execution path](architecture.md)
- [Dependency inventory](dependency_inventory.md)
- [Parity matrix](parity_matrix.md)
- [Parity test plan](test_plan.md)
- [Risk register](risk_register.md)
- [Known differences](known_differences.md)
- [Rollback](rollback.md)
- [Migration log](migration_log.md)

## Changed files

Only the nine documentation files under `docs/native_backend/` were added. There are no
production code, configuration, UI, dependency, lockfile, test, model, or output changes.
The untracked migration specification and untracked prompt JSON files remain untouched and
are excluded from the planned commit.

## Test results

Pre-documentation baseline:

```text
QT_QPA_PLATFORM=offscreen .venv/bin/python -m pytest -q
171 passed, 2 skipped, 6 subtests passed in 3.40s
```

The two skips explicitly defer Torch behavior to the configured ComfyUI GPU worker.
Documentation validation and final diff checks are recorded in the branch handoff/commit.

## Parity and performance results

- Native parity: `NOT IMPLEMENTED`; no native backend exists.
- GPU golden reference capture: not run.
- Performance: not measured.
- Cloud/GPU spend: none.

No parity or performance pass is claimed at Gate 1.

## Principal findings

1. Runtime inference is implemented in the separately pinned `k2core` revision, while many
   same-named modules in this repository are compatibility re-exports.
2. The desktop bypasses the existing general `k2core.backends` protocols and invokes
   `ComfyBaselineRuntime` directly through loose JSON dictionaries.
3. The deepest coupling is to ComfyUI model loading, sampling, memory management,
   `ModelPatcher`, weight-adapter injection, and the optimized-attention override—not to
   node graphs or `folder_paths`.
4. K2Lab regional prompting, regional LoRA routes, edit compositing, projector behavior,
   and metadata are already mostly shared-domain logic and must remain backend-neutral.
5. Arbitrary exact model paths work today, but worker discovery, defaults, legacy face
   detector discovery, and documentation assume a ComfyUI installation.
6. No RunPod handler, job service, container manifest, or reconnectable job store exists
   in this repository.

## Known differences and unresolved risks

There is no native output to compare. The blocking Gate 1 decisions are:

- exact production ComfyUI revision, model/component hashes, and supported hardware;
- whether Raw/non-Turbo is in scope alongside the current eight-step CFG-free Turbo path;
- the required sampler/scheduler subset;
- the RunPod codebase and ownership boundary;
- K2Lab and `k2core` licensing (neither checkout declares a license);
- whether GPL-3.0 ComfyUI vendoring is prohibited pending legal review;
- provenance/redistribution permission for `face_det.onnx`;
- privacy and storage policy for large golden images/intermediate tensors.

The full risk list and mitigations are in [risk_register.md](risk_register.md).

## Rollback

Phase 0 is documentation-only:

```bash
git switch main
```

After the Phase 0 commit exists, its documentation can be removed with a normal
`git revert <phase-0-commit>`. No model or user data is touched.

## Recommended next step

Approve or amend Gate 1 and resolve the blocking scope/reference/license questions. After
explicit approval, begin Gate 2 in the `k2core` repository:

1. add immutable shared schemas and structured errors;
2. wrap the unchanged runtime in `ComfyUIBackend`;
3. add explicit `K2LAB_BACKEND` selection with `comfyui` as the default;
4. characterize current requests/results before routing the desktop through the wrapper;
5. capture golden ComfyUI fixtures before any native loader work.

Per the migration specification, do not start native model loading before this gate is
explicitly approved.
