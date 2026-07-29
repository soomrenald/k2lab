# Approval Gate 4 report

Status: **APPROVED UNDER STANDING GATE AUTHORIZATION**

## Scope and outcome

Phase 2 strict native component loading is complete for the exact approved Krea2 Turbo
reference set. The loader uses only PyTorch and safetensors, imports no ComfyUI module,
and does not expose generation yet.

Pinned shared `k2core` commit:
`52458eae1e2431f47e8b382f4e1e0e268a9e999a`

## Loading contract

- `PipelineConfig` may carry one immutable `RegisteredModel`.
- Native loading verifies registry hashes and header architecture before tensor access.
- Strict mode accepts only the three reviewed reference component hashes.
- Every safetensors key is identity-mapped into a K2Lab-owned component container.
- Unknown top-level namespaces fail before tensor loading.
- Source dtype and quantization metadata are preserved when weight dtype is `auto`.
- Explicit BF16, FP16, FP32, and FP8 conversion policies are supported where Torch does.
- `auto` stages on CPU; explicit `cuda` also addresses ROCm through Torch.
- Partial loads clear already-created tensor mappings before propagating an error.
- Unload clears every state dictionary, runs collection, and releases the Torch cache.

Executable model forward definitions remain Phase 3 work.

## Strict mapping report

| Component | Tensors mapped | Missing | Unexpected | Parameters | Storage bytes |
| --- | ---: | ---: | ---: | ---: | ---: |
| Krea2 Turbo transformer | 686 / 686 | 0 | 0 | 12,820,073,292 | 13,141,645,464 |
| Qwen3-VL-4B encoder | 1,217 / 1,217 | 0 | 0 | 4,437,832,188 | 5,242,332,912 |
| Qwen image VAE | 194 / 194 | 0 | 0 | 126,892,531 | 253,785,062 |
| Total | 2,097 / 2,097 | 0 | 0 | 17,384,798,011 | 18,637,763,438 |

The reported counts include quantization scale/marker tensors because they are required
state-dict entries, not silently discarded auxiliaries.

Exact hashes remain:

| Component | SHA-256 |
| --- | --- |
| Transformer | `eb4dd8c612cfd10f64f25b057e6e6bbcb5737c94a7372177e456dbf7579502f1` |
| Text encoder | `54bd5144df0bbc25dd6ccadfcb826b521445a1b06ae5a42570bdd2974ca87094` |
| VAE | `a70580f0213e67967ee9c95f05bb400e8fb08307e017a924bf3441223e023d1f` |

## Device, timing, and unload proof

Local environment: Torch 2.10.0+rocm7.1, AMD Radeon Graphics.

The transformer and encoder were staged on CPU and the 254 MB VAE was explicitly placed
on the ROCm device for three cycles:

| Cycle | Total load | GPU allocated loaded | GPU allocated unloaded | RSS unloaded |
| --- | ---: | ---: | ---: | ---: |
| 1 | 7.733 s | 254,438,400 B | 0 B | 813,480 KiB |
| 2 | 7.781 s | 254,438,400 B | 0 B | 813,756 KiB |
| 3 | 7.786 s | 254,438,400 B | 0 B | 813,876 KiB |

Every cycle reported strict matches for all three components and released the pipeline.
Post-unload RSS changed by 396 KiB across the three cycles, with no persistent GPU
allocation. The unrelated platform warning about a missing
`/opt/amdgpu/share/libdrm/amdgpu.ids` file did not affect Torch device access.

A second smoke passed through `k2_region_lab.worker.bootstrap` with
`K2LAB_BACKEND=native`, the pinned installed core, the registry/model selector, and the
normal `load_model` protocol. It emitted `ready` with the same mapping report, then shut
down cleanly.

## Automated tests

`k2core`:

```text
ruff check src tests: passed
141 passed, 2 skipped, 6 subtests passed
```

K2Lab with the exact fetched core pin:

```text
uv lock --check: passed
ruff check src tests: passed
179 passed, 2 skipped, 6 subtests passed
git diff --check: passed
```

New tests cover strict identity mapping, unsupported identities, unexpected keys, dtype
conversion, CPU default staging, parameter/storage reporting, unload, registry-required
backend loading, configuration parsing, and native worker selection without constructing
the ComfyUI runtime.

## Existing functionality

- The unset backend remains `comfyui`.
- Explicit `K2LAB_BACKEND=comfyui` remains the rollback path.
- Registry settings are ignored by the ComfyUI backend.
- No ComfyUI runtime, generation, UI, sampler, scheduler, output, LoRA, regional, edit, or
  face behavior changed.
- Native generation continues to fail explicitly as unsupported.

## Known differences and risks

- Native components are verified state dictionaries, not executable modules yet.
- Strict loading intentionally rejects Krea2 Raw and any repacked/updated hashes until a
  full mapping review is added.
- CPU safetensors staging is mmap-backed, so component load timing excludes touching every
  weight page. Hash verification reads every source byte and dominates the total time.
- Global dtype conversion of quantized reference files would materially increase memory;
  it is implemented and unit-tested but was not run over the complete 18.6 GB set.
- The loader does not yet implement device offload during forward execution.

## Rollback

The default and emergency rollback remain:

```bash
K2LAB_BACKEND=comfyui k2lab
```

To remove Phase 2, revert this report's containing K2Lab commit and core commit
`52458eae1e2431f47e8b382f4e1e0e268a9e999a`, then restore the previous core pin
`221b358a4cecc9b40a794e7591e2ca43cd092b9e`.

## Next phase

Standing approval permits Phase 3: executable Krea2 Turbo text encoding, transformer,
Euler/simple schedule, VAE decoding, deterministic checkpoints, and clean text-to-image
parity. LoRA, regional prompting, editing, and face refinement remain excluded until
their specified phases.
