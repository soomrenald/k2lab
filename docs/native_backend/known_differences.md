# Known differences and unknowns

## Phase 0

There is no native implementation, so there are no approved output differences and no
parity claims.

The following current-state differences affect migration planning:

- The specification's target has desktop and RunPod call one `k2core` service. This
  repository has a desktop-only QProcess worker and no RunPod implementation.
- The specification recommends a single `InferenceBackend` lifecycle. The pinned
  `k2core` already has narrower image/frame/video protocols and `ComfyKreaBackend`, but the
  desktop bypasses them and calls `ComfyBaselineRuntime` directly.
- The specification calls for validated shared requests/results. The current desktop and
  worker exchange mutable JSON dictionaries.
- The target requires cooperative cancellation between denoising steps. Current desktop
  cancellation terminates the entire disposable worker process.
- The target requires precise structured errors. Current worker events expose a message
  and Python exception type.
- The target requires a backend in metadata and a cross-system correlation ID. Current
  PNG metadata does not identify a backend, and command UUIDs are not persisted as job
  correlation IDs.
- Arbitrary exact model paths are already accepted, but default discovery, worker
  selection, face detector discovery, and documentation assume a ComfyUI tree.
- Current UI sampler/scheduler options mirror broad ComfyUI registries. The native
  implementation is required to implement only K2Lab-needed combinations; that subset is
  not yet approved.
- Current documentation says the generation path is eight-step, CFG-free Turbo even
  though Raw-architecture checkpoints can be selected. Raw parity scope is unresolved.

## Difference approval record

No difference is approved. Add an entry here only with:

1. affected fixture and feature;
2. measured comparison;
3. reason exact parity is impractical or undesirable;
4. user-visible effect;
5. approving person/date/gate;
6. rollback impact.
