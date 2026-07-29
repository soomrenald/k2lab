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

## 2026-07-28 — Gate 2 approved / Phase 1 started

The user explicitly approved Gate 2. Phase 1 began with the approved constraints:

- add only the K2Lab-owned registry, hash/header validation, and legacy scanner;
- keep the registry usable with arbitrary absolute paths and symlinks;
- do not copy, alter, map, or load model tensor payloads;
- do not change the worker, ComfyUI backend, backend default, or generation behavior;
- stop at Gate 3.

## 2026-07-28 — Gate 3 candidate

Pinned shared core commit: `221b358a4cecc9b40a794e7591e2ca43cd092b9e`

Completed:

- added a versioned TOML model registry with required component hashes;
- added strict existence, safetensors, architecture-fingerprint, shape, and hash checks;
- added symlink resolution and case-insensitive duplicate-name rejection;
- added a read-only legacy ComfyUI scanner that emits registry TOML to stdout;
- added a standalone validator that does not construct desktop settings or start Qt;
- validated both locally installed Krea2 variants plus the shared encoder and VAE;
- passed the full core and K2Lab suites.

Production default: unchanged (`comfyui`).
Runtime/model-loading changes: none.
GPU work: none.
Model copies or mutations: none.
Gate status: awaiting explicit Gate 3 approval.

## 2026-07-28 — Standing approval / Gate 4 complete

The user approved Gate 3 and all later migration gates. This removes approval pauses but
does not remove evidence, parity, rollback, sequencing, or safety requirements. RunPod,
cloud spend, deployment, and removal of the ComfyUI fallback remain outside that standing
authorization.

Pinned shared core commit: `52458eae1e2431f47e8b382f4e1e0e268a9e999a`

Completed:

- added strict native safetensors loading without importing ComfyUI;
- limited strict loading to the exact approved Turbo, encoder, and VAE identities;
- added reviewed component key namespaces and static architecture configuration;
- preserved source quantization metadata and dtypes by default;
- added explicit dtype conversion and CPU/CUDA/ROCm device policy;
- reported hashes, tensors, parameters, storage, dtypes, mappings, and load time;
- added deterministic partial-failure cleanup and unload;
- connected native registry/model selection to the existing worker protocol;
- retained CPU staging for `auto` so all components are not forced into 16 GiB VRAM;
- passed a real worker load and three local ROCm load/unload cycles.

Production default: unchanged (`comfyui`).
Native generation: unsupported.
Model writes: none.
Gate 4 status: approved under standing authorization; Phase 3 may proceed.
