# Native backend migration log

## 2026-07-28 — Phase 0 / Gate 1 candidate

Branch: `feature/native-k2-backend`

Completed:

- read the native-backend migration specification in full;
- inventoried tracked desktop code and the exact pinned `k2core` source;
- searched direct ComfyUI/custom-node imports and indirect process, environment,
  filesystem, model-discovery and documentation assumptions;
- traced generation, image-edit and face-refinement calls from QML to the disposable
  ComfyUI worker;
- confirmed that this repository contains no RunPod entry point or deployment setup;
- proposed backend/service/module boundaries and immutable shared schemas;
- created the dependency inventory, parity matrix, test plan, risk register, known
  differences and rollback plan.

Validation before documentation edits:

```text
QT_QPA_PLATFORM=offscreen .venv/bin/python -m pytest -q
171 passed, 2 skipped, 6 subtests passed in 3.40s
```

Production behavior changes: none.
Dependencies/lockfiles changed: none.
Parity runs: none; native backend is not implemented.
Performance runs: none.
Cloud/GPU spend: none.

Gate status: awaiting explicit Gate 1 approval. Model-loading implementation is prohibited
until approval.

## 2026-07-28 — Gate 1 approved / Gate 2 started

The user explicitly approved Gate 1 with the recommended scope, reference, licensing,
fixture-storage, and RunPod-deferral decisions. Gate 2 implementation began in the shared
`k2core` package. Native model loading remains out of scope.

## 2026-07-28 — Gate 2 candidate

Pinned shared core commit: `e35a01d7f381764a7b3069aaeef13f3aa4c83b1f`

Completed:

- added immutable shared requests/results and structured errors;
- added the backend lifecycle, explicit selector, ComfyUI adapter, and unsupported native
  placeholder;
- routed desktop worker load/generate/edit/refine operations through the adapter;
- retained ComfyUI as the unset and explicit default;
- verified explicit native failure without fallback;
- updated and fetched the locked shared-core dependency;
- passed core and K2Lab suites;
- passed one real local ROCm clean-generation smoke through `ComfyUIBackend`.

Production default: unchanged (`comfyui`).
Native inference: not implemented.
Cloud/GPU spend: no cloud spend; one local 512×512 eight-step smoke.
Gate status: awaiting explicit Gate 2 approval.
