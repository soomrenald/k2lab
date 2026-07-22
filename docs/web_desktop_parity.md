# Web/Desktop functional parity contract

The RunPod web application is a remote execution surface for the same K2 Region Lab workflows
as the local Qt Quick application. Except for the explicit exclusions below, a project that can
be configured and run locally must be configurable, inspectable, saved, restored, and run from
the browser without editing JSON by hand.

"Parity" means equivalent behavior and project/worker payloads. It does not require identical
widget placement or local filesystem controls where the browser uses workspace assets instead.

## Explicit exclusions

The browser does not need to expose:

- memory-policy selection;
- accelerator diagnostics;
- model tensor validation;
- LoRA compatibility inspection;
- live GPU VRAM, system RAM, or GPU-activity telemetry.

These are the only approved desktop omissions. Explicit worker-memory release remains required.

## Acceptance matrix

| Workflow | Required browser behavior | Acceptance evidence |
| --- | --- | --- |
| Project lifecycle | New, Open project JSON, Import application PNG metadata, Save, and Save As | Round-trip a version-18 project without losing any field; import `k2lab_project` from a generated PNG |
| Project recovery | Restore prompts, regions, roles/order, emphasis, LoRAs and routing, sampling, edit, face, projector, and upscale settings | Golden complex-project hydration/serialization test |
| Canvas source | Load/upload, replace, and clear a generation reference; choose a cloud input or output for edit/face work | UI test plus submitted opaque input file ID |
| Comparison | Source, Result, and adjustable comparison view | Component behavior test |
| Regions | Create, select, move, eight-direction resize, delete, rename, enable, reorder front/back, and choose Auto/Subject/Background | Version-18 serialized geometry, priority, and role assertions |
| Prompt editing | Global, regional, reference, edit-target, and face-identity prompts with overflow scrollbars | Live-state and serializer assertions |
| Unified prompt | Exact shared compiler, subject/background organization, subject fill, relationship text, character triggers, and preview | Server golden test against `compile_regional_prompt_plan` |
| Phrase emphasis | Select exact global/regional phrase, occurrence, strength, validation, removal, save, and restore | Round-trip and invalid-match tests |
| LoRAs | Add/remove/activate, strength, generation/reference/edit routing, global or multi-region scope, standard or character-identity mode, and trigger phrase | Round-trip plus worker-payload tests |
| Generation | Dimensions, steps, sampler, scheduler, Fixed/Random/Increment seed, batch count, regional controls, late relaxation, LoRA-delta adaptation, projector, and post-upscale | Control-bound and submitted-project tests |
| Image edit | Reference and target layers, source restoration, prompts, sampling, denoise, latent/composite feather, reference retention, identity preservation, whole-image edit, regional controls, LoRAs, emphasis, and reference projector | Restored-project and worker-payload tests |
| Face refinement | Choose source/latest first pass, detect faces, show numbered boxes and scores, select any/all/none, draw multiple polygon lassos, undo/clear lassos, and submit selection plus paths | Detection API and refinement-request tests |
| Model setup | Inventory and select diffusion model, text encoder, VAE, face detector, LoRAs, and upscaler using persistent workspace assets; show missing/ready state | Agent capability/setup response and UI selection tests |
| Output behavior | Configurable safe filename prefix; outputs and project metadata persist on workspace storage and remain selectable/downloadable | Worker payload and inventory tests |
| Run control | Start each mode, cancel the active/batched run, and explicitly release worker GPU/system memory | Agent cancellation/release contract tests |
| Events | Scrollable chronological log covering lifecycle, transfers, validation errors, worker progress, completion, and cancellation | Bounded 1,000-entry ring; oldest entries discarded after the maximum |
| Installation | One-command loopback launcher, stored non-secret configuration, encrypted RunPod credential, persistent workspace, clear readiness/errors, and no manual API calls | Launcher/control-plane smoke test and documented first/subsequent run |

## Shared-authority rules

- The canonical project schema remains `k2-region-lab-project` version 18.
- The Python project parser validates every browser run before submission.
- Unified prompt text is compiled only by the shared Python compiler.
- Browser control options and bounds must be covered by contract tests against the Python
  project validators; silent TypeScript-only drift is a release blocker.
- A visible control with no handler is a failing feature, not a placeholder implementation.
- RunPod image publication remains blocked until every non-excluded row above is implemented and
  the user confirms the browser workflow.
