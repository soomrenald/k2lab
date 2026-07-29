# Native backend architecture proposal

Status: Phase 0 proposal; not approved for implementation
Inventory baseline: K2Lab `add2a38`, `k2core`
`a82b0b32a891e19eac5c5f6e35f8a9bfb715f9dc`
Observed local ComfyUI checkout: `285a98944c397a4a81f15ac63d69fa3dbc0a27b9`

The ComfyUI commit above identifies the checkout inspected for this inventory. K2Lab does
not currently pin or verify that commit, so it is not yet a reproducible runtime
requirement.

## Current ownership boundary

This repository owns the desktop application and worker orchestration. Most modules under
`src/k2_region_lab` that appear to contain inference-domain logic are compatibility
re-exports from the separately versioned `k2core` package. `pyproject.toml` and `uv.lock`
pin the exact `k2core` revision listed above.

```text
Qt Quick UI
  -> QmlWorkspaceController
  -> hidden Qt Widgets MainWindow (current state and request builder)
  -> ExternalWorkerClient
  -> selected ComfyUI Python via QProcess
  -> k2_region_lab.worker.bootstrap
  -> k2_region_lab.worker.entrypoint
  -> k2core.worker.runtime.ComfyBaselineRuntime
  -> ComfyUI model loading, sampling, LoRA, attention hooks, VAE, memory manager
  -> PNG plus line-delimited JSON worker events
```

The worker is disposable for generation, image editing, and face refinement. Cancellation
terminates that process rather than passing a cooperative token through the denoising
loop. Model discovery and safetensors-header validation can run without ComfyUI, but model
loading and inference cannot.

### Current generation call path

1. QML invokes `QmlWorkspaceController.runActive()` in
   `src/k2_region_lab/qml/controller.py:1374`.
2. The controller calls the compatible `MainWindow` action. `MainWindow._generate_baseline`
   (`src/k2_region_lab/desktop/main_window.py:5361`) resolves the seed, canvas geometry,
   regional controls, prompt emphases, projector settings, LoRA routes, memory policy,
   model paths, output rules, and embedded project document into an unvalidated dictionary.
3. `MainWindow._advance_pending_generation`
   (`src/k2_region_lab/desktop/main_window.py:5300`) creates a fresh worker, then performs
   probe, model validation, model load, and dispatch in order.
4. `ExternalWorkerClient.start`
   (`src/k2_region_lab/desktop/worker_client.py:41`) launches the configured ComfyUI
   interpreter. It injects this repository and the ComfyUI root into `PYTHONPATH` and
   identifies the pinned desktop-installed `k2core` package via
   `K2LAB_K2CORE_PACKAGE`.
5. `k2_region_lab.worker.bootstrap` imports only that pinned `k2core` package into the GPU
   process. Torch, NumPy, Pillow, ONNX Runtime, and ComfyUI continue to resolve from the
   selected GPU environment.
6. `k2_region_lab.worker.entrypoint` reads line-delimited commands. It calls
   `probe_runtime`, `validate_model_artifacts`, `ComfyBaselineRuntime.load`, and finally
   `ComfyBaselineRuntime.generate`.
7. `ComfyBaselineRuntime` loads the transformer, Qwen encoder, and VAE through ComfyUI;
   compiles K2Lab's unified prompt, projector delta, regional LoRA routes, and spatial
   attention override; calls `comfy.sample.sample`; decodes with the ComfyUI VAE wrapper;
   and writes K2Lab PNG metadata.
8. JSON progress/result events return over stdout. The desktop displays the result and
   waits for the disposable process to exit, releasing its GPU and system-memory state.

Image editing and face refinement use the same bootstrap/load sequence but dispatch
`EDIT_IMAGE` and `REFINE_FACES`. Editing VAE-encodes the complete padded source, supplies a
ComfyUI denoise mask, restores source pixels outside K2Lab's composite mask, and retains
the same regional prompt/LoRA hooks. Face refinement detects or accepts face crops, runs
crop-level img2img, and composites each crop.

## Desktop and RunPod divergence

No RunPod code exists in this repository at this baseline. Searches found no RunPod
handler, serverless job schema, HTTP/API entry point, container manifest, reconnectable
job store, or deployment configuration. The current line-delimited worker protocol is
desktop-specific:

- it accepts local filesystem paths rather than asset identifiers;
- request dictionaries are constructed from Qt controls;
- cancellation kills a child process;
- output completion is observed through QProcess;
- correlation is a per-command UUID, not a durable job ID;
- completed output recovery assumes a shared local filesystem;
- errors are an exception type and message, not the structured error taxonomy required by
  the migration specification.

`k2core.backends` already contains general image/frame backend protocols and a
`ComfyKreaBackend` adapter, but the desktop worker does not call that adapter. It calls
`ComfyBaselineRuntime` directly. The existing adapter accepts loose mappings and does not
cover the full desktop request surface. It is useful prior art, not Gate 2 completion.

The RunPod integration must be added only after the schemas and application service below
exist. RunPod and desktop must serialize the same request type and call the same service;
they must not adapt independently to `ComfyBaselineRuntime`.

## K2Lab-specific logic that must remain backend-neutral

The following logic is product behavior, not a ComfyUI replacement detail:

| Behavior | Current authority at pinned `k2core` revision |
| --- | --- |
| Region geometry and immutable layout | `k2core.regions.geometry`, `k2core.regions.layout` |
| Unified regional prompt text, roles, token spans, soft fields, overlap order, emphasis | `k2core.regional_prompting` |
| Cross-modal text/image attention permissions and late-step relaxation | `k2core.spatial_attention` |
| Regional/global LoRA route compilation and unsafe target omission | `k2core.regional_lora` |
| AI Toolkit/Krea LoRA key normalization and strict coverage reports | `k2core.lora.compatibility` |
| Unfused composite adapter masks, delta statistics, and adaptation feedback | `k2core.worker.runtime:614-1031` |
| Twelve-value Krea text-fusion projector and identity-token protection | `k2core.projector` and `k2core.worker.runtime:729-864` |
| Source padding, latent/composite edit masks, edit conditioning, exact exterior restore | `k2core.image_edit` |
| Face assignment, crop geometry, detector wrapper, and compositing | `k2core.face_detail` |
| Project schema and PNG project round-trip | `k2core.project` |
| Filename and output metadata rules | `k2core.output` plus `k2core.worker.runtime` |
| Memory policy thresholds and one-retry OOM policy | `k2core.memory` plus `k2core.worker.runtime` |

The native backend may consume these results, but it must not clone or reinterpret them.
Transformer integration points currently live in `ComfyBaselineRuntime` and must be moved
behind a backend-owned execution adapter without changing their semantics.

## Proposed module boundaries

Changes to the shared package belong in the `k2core` repository, followed by a reviewed
pin update in this repository. The `.venv` or `.uv-cache` copies are not editable source.

```text
k2core/
  inference/
    backend.py          # InferenceBackend protocol and lifecycle contracts
    capabilities.py     # immutable capability declarations
    errors.py           # structured application error hierarchy
    schemas.py          # validated immutable requests/results
    service.py          # selection, logging, correlation, cleanup
  backends/
    comfyui.py          # only module allowed to expose ComfyUI objects
    native/
      backend.py        # NativeK2Backend, unsupported at first
      registry.py       # Phase 1 model registry and legacy scanner
      loader.py         # Phase 2 component loading
      sampling.py       # Phase 3 RNG, schedules, denoising
      adapters.py       # Phase 4 and 6 LoRA execution
      attention.py      # Phase 5 and 6 Krea control points
      editing.py        # Phase 7 latent/edit path
      devices.py        # Phase 8 memory and placement
  parity/
    fixtures.py
    checkpoints.py
    compare.py
    report.py
```

Desktop should gain one application service that converts workspace state to a
`GenerationRequest`, calls the selected backend, and converts structured progress/results
back to UI state. The worker transport may continue initially, but it should transport
the shared schema rather than Qt-shaped dictionaries. A future RunPod handler must call
that same service.

Only `k2core.backends.comfyui` may import `comfy.*`. K2Lab-specific region, prompt, edit,
and metadata modules must remain importable and unit-testable without ComfyUI or Torch.

## Proposed backend interface

```python
class InferenceBackend(Protocol):
    @property
    def backend_id(self) -> str: ...

    def load(self, config: PipelineConfig) -> LoadedPipeline: ...

    def generate(
        self,
        request: GenerationRequest,
        *,
        progress: ProgressCallback | None = None,
        cancellation: CancellationToken | None = None,
    ) -> GenerationResult: ...

    def encode_image(self, request: ImageEncodeRequest) -> LatentResult: ...
    def decode_latents(self, request: LatentDecodeRequest) -> ImageResult: ...
    def unload(self) -> None: ...
    def capabilities(self) -> BackendCapabilities: ...
```

`ComfyUIBackend` must be implemented first as an adapter around the unchanged
`ComfyBaselineRuntime`. `NativeK2Backend` should initially report every inference feature
as unsupported. Backend selection belongs in the service:

```text
K2LAB_BACKEND unset or "comfyui" -> ComfyUIBackend
K2LAB_BACKEND "native"           -> NativeK2Backend (developer-only)
anything else                    -> ConfigurationError; never silent fallback
```

The selected backend must be logged and included in output/job metadata. Native
capabilities must drive disabled controls; no control may be accepted and ignored.

## Proposed shared schemas

Use frozen, slotted dataclasses with explicit validation throughout `k2core`. This matches
the existing shared package and avoids introducing a validation framework only for part of
the boundary. Enums should be string-valued for stable JSON serialization.

| Schema | Required content |
| --- | --- |
| `PipelineConfig` | model ID/architecture, component references and hashes, device and dtype policies, strict loading, attention implementation |
| `ModelComponent` | role, absolute path, SHA-256, format, expected architecture, optional legacy origin |
| `DevicePlan` | transformer/text/VAE devices, compute and weight dtype, offload, tiling, safe fallback |
| `GenerationRequest` | correlation ID, prompt, dimensions, steps, seed, CFG, sampler, scheduler, sigma policy, denoise, mode, regions, LoRAs, projector, edit input, output policy |
| `PromptSpec` | global text, negative text, token emphases, private-prompt logging consent |
| `RegionSpec` | stable ID/name, pixel box, normalized box, enabled flag, priority, role, prompt, face-identity prompt |
| `RegionMaskSpec` | region ID, mask source, interpolation, overlap/clip policy |
| `LoraSpec` | stable ID, component hash/path, strength, deterministic order, global/region scope, routing mode, trigger |
| `ImageEditSpec` | source asset/path, source hash, edit-entire-image flag, denoise strength, latent/composite feather, retention policy |
| `OutputSpec` | directory/asset sink, filename prefix, image format, metadata and intermediate-capture policy |
| `GenerationResult` | correlation ID, backend ID/version, output assets, seed, dimensions, timing, peak memory, warnings, metadata |
| `ProgressEvent` | correlation ID, phase, step/total, fraction, memory snapshot, non-private detail |
| `BackendCapabilities` | named features plus supported modes, samplers, schedulers, dtypes, devices, limits, diagnostic version |
| `ImageEncodeRequest`, `LatentResult` | image source/hash, resize/pad policy, device/dtype, latent tensor descriptor and scaling |
| `LatentDecodeRequest`, `ImageResult` | latent descriptor, VAE/config identity, tiling policy, image asset and metadata |
| `StructuredError` | category, summary, technical detail, backend, phase, remediation, retry safety, GPU-started flag, correlation ID |

No schema may contain a ComfyUI `ModelPatcher`, conditioning tuple, latent dictionary, or
VAE object. Tensor-bearing internal results need an opaque K2Lab-owned tensor descriptor
or private implementation object; their serialized form must be explicit.

Validation must preserve current constraints: 16-pixel-aligned generation dimensions,
Turbo CFG of 1.0, current seed range and seed modes, ordered sampler/scheduler names,
half-open boxes, deterministic region/LoRA order, and exact image-edit protection rules.

## Proposed implementation order

1. Gate 1 approval of this inventory and open questions.
2. In `k2core`, add shared schemas, structured errors, backend selection, and a
   `ComfyUIBackend` wrapper with characterization tests. Do not route the UI yet.
3. Capture golden ComfyUI fixtures through the wrapper, including intermediate hooks.
4. Route the existing worker through the wrapper with `comfyui` as the hard default; pass
   Gate 2.
5. Implement native registry/scanner only; pass Gate 3.
6. Continue one numbered phase and approval gate at a time.
7. Add RunPod only after the shared request/service boundary is stable.

## Open questions requiring approval or investigation

- Which exact ComfyUI revision(s), Krea model files, hashes, and hardware form the
  production reference? The local checkout is not a sufficient release definition.
- Does “Krea2 Raw” need parity now? Current UI documentation says the active path is
  eight-step, CFG-free Turbo, while the setup accepts Raw-architecture files.
- Which sampler/scheduler pairs are advertised and actually used in production fixtures?
  The UI exposes the full ordered ComfyUI registry, broader than the spec's “required
  combinations only” rule.
- What is the authoritative license for K2Lab and `k2core`? Neither repository checkout
  contains a license declaration at this baseline.
- May any GPL-3.0 ComfyUI code be vendored? The recommended answer is no pending formal
  license review; use clean-room behavior-level implementations or upstream libraries.
- What is the provenance and redistribution permission for `face_det.onnx`? Its containing
  FantasyPortrait tree has Apache-2.0 text, but the model artifact itself needs an explicit
  source/model-license record.
- Where will canonical golden images and intermediate tensors live? They are likely too
  large for ordinary Git and may contain private prompts.
- What RunPod repository/service is in scope, or must it be created from this repository?
- What native tensor checkpoint mechanism can observe Q/K/V and scheduler state without
  affecting timing or memory enough to invalidate performance measurements?
- Is whole-process cancellation an approved ComfyUI reference behavior while native uses a
  cooperative token between steps?

No model-loading implementation should begin until these questions and Gate 1 are
explicitly reviewed.
