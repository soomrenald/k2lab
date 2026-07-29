# Approval Gate 2 report

Status: **APPROVED**

The user explicitly approved Gate 2 on 2026-07-28. Phase 1 began with the recommended
scope below; native loading and generation remain prohibited until their later gates.

## Completed work

Pinned shared `k2core` commit
`e35a01d7f381764a7b3069aaeef13f3aa4c83b1f` (including foundational commit
`aac0699c511790f49bf6b4addf692ad337173c79`) adds:

- immutable, validated pipeline, device, generation, image-edit, face-refinement, LoRA,
  progress, and result schemas;
- the complete structured inference error taxonomy and legacy-error conversion;
- the migration-facing `InferenceBackend` lifecycle protocol;
- a behavior-preserving `ComfyUIBackend` over the unchanged `ComfyBaselineRuntime`;
- a developer-only `NativeK2Backend` placeholder with no supported capabilities;
- explicit `K2LAB_BACKEND` parsing, logging, validation, and no silent fallback;
- contract tests for default selection, explicit rollback, schemas, delegation, progress,
  error behavior, and successful-result compatibility.

K2Lab now:

- pins the exact core commit above in both `pyproject.toml` and `uv.lock`;
- selects the backend once when the worker starts;
- loads and runs generation, image editing, and face refinement through
  `ComfyUIBackend`;
- preserves the existing success-event/result dictionaries;
- adds the structured error object alongside the legacy exception type on failures;
- retains the existing disposable worker, UI request builder, ComfyUI runtime, model
  loading, sampling, LoRA, regional control, image-edit, output, and cleanup behavior.

No native model registry, loader, tensor execution, sampling, or generation was
implemented.

## Changed files

### k2core

```text
src/k2core/backends/__init__.py
src/k2core/backends/comfyui.py
src/k2core/backends/native.py
src/k2core/inference/__init__.py
src/k2core/inference/backend.py
src/k2core/inference/errors.py
src/k2core/inference/schemas.py
src/k2core/inference/selection.py
tests/test_inference_contracts.py
```

### K2Lab

```text
pyproject.toml
uv.lock
src/k2_region_lab/worker/entrypoint.py
tests/test_worker_backend_selection.py
docs/native_backend/gate_1_report.md
docs/native_backend/gate_2_report.md
docs/native_backend/migration_log.md
docs/native_backend/parity_matrix.md
```

## Automated tests

`k2core`:

```text
127 passed, 2 skipped, 6 subtests passed
ruff check: passed
```

K2Lab with the fetched, locked core commit:

```text
175 passed, 2 skipped, 6 subtests passed
targeted worker selection/rollback tests: 4 passed
ruff check on changed Python files: passed
git diff --check: passed
```

The expected skips still defer Torch tensor work to the configured GPU-worker
environment.

## Local GPU smoke

One synthetic reference request ran through the real worker and `ComfyUIBackend`:

| Field | Value |
| --- | --- |
| Prompt | synthetic red ceramic teapot studio fixture |
| Size | 512×512 |
| Steps | 8 |
| Sampler / scheduler | `euler` / `simple` |
| Seed | `20260728` |
| Backend | `comfyui`, explicitly selected and recorded in worker log |
| Accelerator | ROCm 7.1, Torch 2.10.0, AMD Radeon Graphics, 16 GiB |
| Result | PASS |
| Runtime | 93.35 seconds, including current text encode/denoise/decode path |
| OOM fallback | not used |
| Output hash | `3f9ae9d66ff63ccae2aa2283624a4f10882764d924367ec500aeafad5cc2bb31` |

The temporary image and tensor manifests are under `/tmp/k2lab-native-gate2/` and are not
committed. Visual inspection confirmed a valid image matching the synthetic prompt.

Exact component hashes:

| Component | SHA-256 |
| --- | --- |
| `krea2_turbo_fp8_scaled.safetensors` | `eb4dd8c612cfd10f64f25b057e6e6bbcb5737c94a7372177e456dbf7579502f1` |
| `qwen3vl_4b_fp8_scaled.safetensors` | `54bd5144df0bbc25dd6ccadfcb826b521445a1b06ae5a42570bdd2974ca87094` |
| `qwen_image_vae.safetensors` | `a70580f0213e67967ee9c95f05bb400e8fb08307e017a924bf3441223e023d1f` |

The first sandboxed probe could not see `/dev/kfd`; the identical approved local run with
GPU-device access passed. ComfyUI emitted its existing warning that
`expandable_segments` is unsupported on this ROCm platform; generation was unaffected.

## Gate criteria

| Criterion | Evidence | Status |
| --- | --- | --- |
| Existing suites pass | Core and K2Lab results above | PASS |
| Current generation through `ComfyUIBackend` | Real local GPU smoke | PASS |
| No observable UI change | Full desktop/QML regression suite; no UI/QML files changed | PASS |
| No success-output schema regression | Adapter contract and worker integration assert exact legacy dictionaries | PASS |
| Default remains ComfyUI | Unset-selector tests and selector implementation | PASS |
| Backend selection logged | Unit assertion and real worker log entry | PASS |
| Rollback via flag | Unset and explicit `K2LAB_BACKEND=comfyui` worker tests | PASS |
| Native unsupported behavior is explicit | `native` fails before runtime load; no fallback | PASS |

## Parity and performance

- Reference ComfyUI abstraction parity: **PASS** for the synthetic clean-generation smoke
  and exact successful result/event contract.
- Native generation: **NOT IMPLEMENTED**.
- Cross-backend image/tensor parity: **NOT IMPLEMENTED**.
- Comparative performance: not applicable; only the unchanged reference path exists.
- Cloud spend: none.

## Known differences and risks

- Failure events now include an additive structured `error` object while retaining
  `exception_type` and the existing top-level message.
- Standalone `encode_image` and `decode_latents` lifecycle methods report unsupported;
  current desktop functionality uses image editing through `generate`.
- Cancellation remains current disposable-worker termination. Cooperative cancellation is
  deferred to the native sampling phase.
- The native placeholder is developer-only, has an empty capability set, and cannot load.
- Backend identity is present in the typed result and logs but is not injected into legacy
  PNG metadata yet, avoiding a Gate 2 success-schema change.
- No current regional, LoRA, edit, or face GPU fixture was rerun in this gate. Their
  runtime delegates are covered by exact argument-mapping tests and the unchanged full
  regression suite; golden GPU capture remains the next test-harness work.
- The K2Lab/`k2core` licensing and face-detector provenance risks from Gate 1 remain open.

## Rollback

Runtime fallback does not require a code change:

```bash
K2LAB_BACKEND=comfyui k2lab
```

The unset selector has the same behavior. To remove Gate 2 code, revert this report's
containing K2Lab commit and `k2core` commit
`e35a01d7f381764a7b3069aaeef13f3aa4c83b1f`; the prior K2Lab dependency pin is
`a82b0b32a891e19eac5c5f6e35f8a9bfb715f9dc`.

## Approved next step

After explicit Gate 2 approval, begin Phase 1 only:

1. add a K2Lab-owned model registry with absolute paths and hashes;
2. add a read-only legacy ComfyUI scanner;
3. validate missing files, architecture, shapes, symlinks, duplicates, and hash mismatch;
4. make no model-loading or generation changes;
5. stop at Gate 3 with discovery/validation evidence.
