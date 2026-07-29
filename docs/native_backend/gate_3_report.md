# Approval Gate 3 report

Status: **AWAITING EXPLICIT APPROVAL**

## Scope and outcome

Phase 1 is complete. K2Lab can discover existing ComfyUI model files, emit a standalone
K2Lab registry, and validate that registry without loading tensor payloads or generating.
The worker and both backend implementations were not changed.

Pinned shared `k2core` commit:
`221b358a4cecc9b40a794e7591e2ca43cd092b9e` (including registry foundation
`600243507f2c596223b8bb4e269e34774fc9633a`)

## Implemented behavior

- versioned TOML registry with one or more named Krea2 model sets;
- required absolute/resolved component paths and exact SHA-256 identities;
- paths independent of any ComfyUI directory convention;
- symlink support without copying the target;
- strict safetensors header, required-key, tensor-shape, and layer-count validation;
- case-insensitive duplicate model-name rejection;
- hash caching when model sets share the encoder or VAE;
- read-only legacy ComfyUI discovery and TOML output;
- standalone validation before desktop settings, logging, Qt, workers, or backends.

The registry is not connected to `NativeK2Backend.load`. Native loading remains
unsupported and is reserved for Phase 2 after Gate 3 approval.

## Changed files

### k2core

```text
src/k2core/model/__init__.py
src/k2core/model/manifests.py
src/k2core/model/registry.py
tests/test_model_registry.py
```

### K2Lab

```text
pyproject.toml
uv.lock
src/k2_region_lab/app.py
tests/test_model_registry_cli.py
docs/native_backend/gate_2_report.md
docs/native_backend/gate_3_report.md
docs/native_backend/migration_log.md
docs/native_backend/model_registry.md
docs/native_backend/parity_matrix.md
```

No worker, runtime, backend, desktop controller, QML, sampler, output, or model file was
changed.

## Required test cases

| Case | Result |
| --- | --- |
| Valid model registry and TOML round trip | PASS |
| Missing model file | PASS — reported without loading or crashing |
| Wrong architecture | PASS — rejected |
| Wrong tensor shape | PASS — exact mismatch reported from header |
| Symlink resolution | PASS — configured and resolved paths retained |
| Duplicate model names | PASS — rejected case-insensitively |
| Model hash mismatch | PASS — expected and observed hashes reported |
| Legacy ComfyUI path import | PASS — valid registry emitted, source files unchanged |

The scanner regression test compares source modification times and SHA-256 values before
and after scanning.

## Automated validation

`k2core`:

```text
ruff check src tests: passed
136 passed, 2 skipped, 6 subtests passed
```

K2Lab with the exact fetched core pin:

```text
ruff check src tests: passed
178 passed, 2 skipped, 6 subtests passed
git diff --check: passed
```

The two expected skips continue to defer Torch regional tensor work to the configured GPU
worker environment.

## Read-only local model proof

The scanner discovered two complete local sets:

| Component | SHA-256 |
| --- | --- |
| `krea2_raw_int8_convrot.safetensors` | `5585a4a38c4bcfb6fde2d480a4aa6edf7f665721ebde56d30662c35a45f5fa5c` |
| `krea2_turbo_fp8_scaled.safetensors` | `eb4dd8c612cfd10f64f25b057e6e6bbcb5737c94a7372177e456dbf7579502f1` |
| `qwen3vl_4b_fp8_scaled.safetensors` | `54bd5144df0bbc25dd6ccadfcb826b521445a1b06ae5a42570bdd2974ca87094` |
| `qwen_image_vae.safetensors` | `a70580f0213e67967ee9c95f05bb400e8fb08307e017a924bf3441223e023d1f` |

Both model sets and all six component references reported `PASS`. Shared component files
were hashed once per operation. The generated registry is temporary at
`/tmp/k2lab-gate3-registry.toml` and is not committed.

This proof started no Qt application, worker, backend, Torch runtime, or GPU operation. It
performed no generation and wrote nothing under the model directories.

## Existing functionality and parity

- `K2LAB_BACKEND` remains unset/default `comfyui`.
- No generation call path changed after the approved Gate 2 adapter.
- The full application suite and worker-selection/rollback coverage pass.
- The Gate 2 real ROCm ComfyUI smoke remains the reference; Phase 1 changed no file on
  that path.
- Native generation remains `NOT IMPLEMENTED`; no cross-backend parity or performance
  claim is made.

Registry discovery/validation parity status: **PASS**.

## Known differences and risks

- Header fingerprints validate the known Krea2 transformer, Qwen3-VL encoder, and Qwen
  image VAE layouts; they are not a substitute for strict state-dict loading at Gate 4.
- Hashing large files is intentionally I/O intensive. It is explicit CLI work, caches
  shared files within one operation, and has no runtime-generation cost.
- The legacy name scanner may require exact-file configuration for unusually named valid
  artifacts.
- Registry source provenance is descriptive; signature/trust policy is not implemented.
- All Gate 2 known differences and the licensing/provenance risks remain open.

## Rollback

Runtime rollback remains:

```bash
K2LAB_BACKEND=comfyui k2lab
```

That is already the default. Phase 1 is not consulted by generation. To remove Phase 1,
revert this report's containing K2Lab commit and core commit
`221b358a4cecc9b40a794e7591e2ca43cd092b9e`; restore the prior core pin
`e35a01d7f381764a7b3069aaeef13f3aa4c83b1f`.

## Approval required

Approve Gate 3 before Phase 2 begins. The recommended next step is strict native component
loading only: state-dict mapping, dtype/device placement, parameter-count comparison,
load/unload verification, and Gate 4 reporting. Clean generation remains prohibited until
Gate 4 is approved.
