# K2 Region Lab Current-State Handoff

Snapshot date: 2026-07-20  
Code baseline: commit `631978e` (`Preview region resizing during drag`)  
Project schema: `k2-region-lab-project`, version 18

This document is the short operational handoff for resuming work on K2 Region Lab. The
README remains the detailed feature and installation reference, the engineering reference
documents the regional-control design, and `docs/runpod_web_workspace_spec.md` is the
authoritative specification for the deferred web/RunPod product.

## 1. Current product state

K2 Region Lab is a functioning local Linux/Python 3.12 research application for Krea 2
generation, regional prompting, regional LoRA routing, image editing, and face refinement.
The default desktop is now a PySide6 Qt Quick/QML workspace. The older Qt Widgets UI is
still available with `k2lab --legacy-widgets` and, importantly, still owns much of the
application state and business logic behind the QML adapter.

The present UI is acceptable for continued local testing, but it is not the final
deployment UI. No RunPod web implementation has been started; only its detailed design has
been committed.

### User-visible workspace

- A left rail selects Generate, Edit, or Faces.
- A large central canvas displays the source and result, supports source/result comparison,
  and displays editable region overlays.
- A fixed contextual inspector holds Prompt, Regions, LoRAs, and Advanced controls.
- A bottom action bar keeps status, memory telemetry, progress, Stop, and the mode-specific
  run action visible.
- The gear button opens a dedicated matching setup window. It includes runtime, model,
  memory, output, worker, validation, and accelerator controls only; generation and editing
  settings are not duplicated there.
- Setup changes are staged. **Apply settings** commits them. Closing with dirty values asks
  whether to apply, discard, or cancel; **Close window** does not close the main app.

### Region editing behavior

- Region names are also listed beside the image through the region strip, so a region can
  be selected without hitting a small box on the canvas.
- Clicking an unselected region selects it. Dragging inside the selected region moves it.
- All four corners and all four edges resize the selected region.
- Resize geometry updates continuously during pointer motion and is committed on release.
- Regions remain constrained to the painted image area and have a 16-pixel minimum size.
- Generation regions, image-edit reference regions, and image-edit target regions are
  distinct semantic layers.

### LoRA controls

- LoRAs can be loaded, activated/deactivated without losing configuration, and removed.
- Deactivation remembers the prior nonzero strength and restores it when reactivated.
- Strength and other numeric controls use a shared `ValueSlider`: the track is draggable,
  the value is directly editable, and both representations remain synchronized.
- A LoRA can be global or assigned to one or more regions. Standard and character-identity
  routing remain available where applicable.
- Associated image metadata no longer produces duplicate LoRA rows when a matching LoRA is
  already in the library.

### Source/result comparison

The comparison mode supports Source, Result, and Compare. In Compare, the slider controls
the width of a clipped result layer over the source. Commit `547e184` removed conflicting
full-parent anchors that previously caused the result to cover the source at every slider
position.

## 2. Current architecture

| Area | Current authority | Notes |
| --- | --- | --- |
| Desktop entry point | `src/k2_region_lab/app.py` | QML by default; `--legacy-widgets` shows the old UI. |
| QML shell | `src/k2_region_lab/qml/ui/Main.qml` | Main workspace and mode switching. |
| QML components | `src/k2_region_lab/qml/ui/components/` | Canvas, inspector, region strip, and editable slider. |
| Setup window | `src/k2_region_lab/qml/ui/SetupWindow.qml` | Staged setup-only configuration. |
| QML adapter | `src/k2_region_lab/qml/controller.py` | Bridges QML to the hidden compatibility `MainWindow`. |
| State/business logic | `src/k2_region_lab/desktop/main_window.py` | Still the main compatibility owner; do not remove casually. |
| Project persistence | `src/k2_region_lab/project.py` | JSON and embedded PNG project metadata; schema version 18. |
| Image-edit helpers | `src/k2_region_lab/image_edit.py` | Source rules, masks, conditioning regions, and compositing. |
| GPU execution | `src/k2_region_lab/worker/runtime.py` | Generation, routed LoRAs, regional attention, editing, faces. |
| Worker transport | `src/k2_region_lab/worker/` | Disposable line-delimited external worker protocol. |
| Regional prompting | `src/k2_region_lab/regional_prompting.py` | Unified prompt and token-span compilation. |
| Regional attention | `src/k2_region_lab/spatial_attention.py` | Text/image attention gating and spatial fields. |
| Regional LoRAs | `src/k2_region_lab/regional_lora.py` | Unfused delta route compilation and masks. |

The QML migration is deliberately incremental: `app.py` constructs an unshown Widgets
`MainWindow`, wraps it in `QmlWorkspaceController`, and exposes that controller to QML.
Future refactoring should extract independent service/state objects from `MainWindow`
before removing it. Reimplementing logic separately in QML would create divergent project
and worker behavior.

## 3. Image-edit implementation

Image editing is a separate mode between generation and face refinement. The central run
button changes to **Run image edit**; the regular generation tab is not used.

### Loading and restoration

When an image is loaded, the application looks first for embedded `k2lab_project` PNG
metadata and then for an exact `<image-stem>.k2lab.json` sidecar. If found, it restores:

- original global and regional prompts and region boxes;
- prompt emphases and projector settings;
- LoRA files, strengths, activation/routing, and regional scopes;
- sampler, scheduler, step count, denoise controls, and seed;
- the fixed-seed behavior used for the source generation.

The restored data appears on the **Reference layer**. Newly drawn edit boxes live on the
separate **Edit targets** layer. Both can be modified. Loading a same-sized replacement can
retain the layout; accepting a different size clears geometry and edit-region assignments.
The canvas source always remains the originally loaded image, even after an edit result is
produced.

### Prompt semantics

- **Edit instruction** is the operation-wide edit intent. For local edits it is combined
  with each active target-region prompt; for explicit whole-image editing it becomes the
  global conditioning prompt.
- **Describe the edit inside this box** is local to one edit target and supplies the
  object/content instruction for that box.
- The original reference regions preserve the source subject/layout ownership during
  denoising. The original global scene prompt remains stored and visible, but it is omitted
  from active edit conditioning because it can conflict with removal or replacement.

### Denoising and preservation

- The complete source is VAE-encoded without rescaling; only right/bottom alignment padding
  is added when needed.
- Local edit boxes produce a union denoise mask with a latent transition collar. At every
  denoising step, latent samples outside the mask are clamped back to the source latent.
- The final decoded candidate is composited over the original with a narrower feather.
  Pixels outside final mask support are copied from the source exactly.
- Full-canvas image/text attention remains available so the edited area can blend with its
  surroundings; regional prompts and regional LoRA deltas retain their existing gates.
- **Edit entire image** intentionally removes outside-area protection. It is capable of
  broad face, body, and scene drift and should not be the default preservation workflow.

Current conservative defaults are fixed source seed, low denoise, reference retention 1.0,
latent feather 64 px, composite feather 48 px, preserve identity enabled, and whole-image
editing disabled.

### GPU validation evidence and unresolved quality risk

fixtures using the real ComfyUI worker and writes comparisons, amplified difference maps,
boundary crops, and metrics beneath `outputs/gpu-tests/`.

The recorded localized tests at denoise 0.35 and 0.70 report zero changed pixels outside
final mask support (`outside_mask_exact: true`). Approximately 7.5% of the 768×768 image
changed because the edit plus feather occupied that part of the canvas. This verifies exact
outside-mask pixel preservation, not semantic prompt quality.

The recorded whole-image tests changed approximately 99.5–99.9% of pixels. Their average
delta is smaller at denoise 0.35 than 0.70, but they confirm that global editing is not an
identity-preserving operation. Prompt adherence, identity retention inside an edit box, and
the subjective visibility of transition artifacts still require visual GPU evaluation.
Do not describe the image-edit quality problem as fully solved merely because the outside
pixel metric passes.

Machine-specific GPU validation command:

```bash
/opt/ComfyUI/venv_rocm7/bin/python \
```

Useful flags include `--cases localized`, `--cases global`, `--denoise`,
`--reference-retention`, `--latent-feather`, `--composite-feather`, and
`--omit-reference-global`.

## 4. Regional generation behavior worth preserving

- Generation uses one unified Qwen scene prompt and records token spans for each region.
- Subject boxes are hard cross-modal permissions for subject text. Image-to-image attention
  remains continuous to avoid rectangular scene seams.
- Overlap priority follows the draggable front-to-back region order.
- Regional LoRAs use unfused forward deltas; base FP8 weights are not rewritten.
- Global and regional LoRAs may coexist in a composite target. Regional target support is
  intentionally restricted where a projection cannot be localized safely.
- “Adapt from regional LoRA delta” measures routed delta response and scales the next-step
  spatial attention bias within bounded limits. It does not change the LoRA gate itself.
- Every generation uses a disposable worker so model and LoRA allocations are returned to
  the OS after completion or cancellation.

`README.md` before changing these contracts.

## 5. Tests and current baseline

Create/install the lightweight desktop environment with Python 3.12, then run QML tests
offscreen:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
QT_QPA_PLATFORM=offscreen .venv/bin/python -m pytest -q \
  tests/test_qml_workspace.py tests/test_desktop_smoke.py
```

Current focused GUI baseline: **46 passed**.

The complete suite was also run at this snapshot:

```bash
QT_QPA_PLATFORM=offscreen .venv/bin/python -m pytest -q
```

Observed result: **152 passed, 6 failed, 2 skipped, 6 subtests passed**. The six failures
were not introduced or repaired as part of the final GUI work:

1. Three face-detector provider tests use bare `Mock` sessions, while production session
   loading now validates `get_inputs()` and `get_outputs()`; the mocks have no sized input
   or output lists.
2. Two regional-LoRA runtime tests use a `FakeModel` without the newer
   `get_attachment()` method needed for projector bypass composition.
3. The external-worker protocol test cannot create its debug log at
   `~/.local/share/k2-region-lab/logs/worker-debug.log` in the restricted test sandbox.

These should be triaged explicitly on resumption. The first two appear to be test-double
contract drift, while the worker failure is environment/path related; that assessment has
not yet been converted into fixes. The two skipped tests intentionally defer Torch work to
the configured ComfyUI worker environment.

Before committing a desktop/QML change, the proven quick validation set is:

```bash
QT_QPA_PLATFORM=offscreen .venv/bin/python -m pytest -q \
  tests/test_qml_workspace.py tests/test_desktop_smoke.py
.venv/bin/python -m ruff check tests/test_qml_workspace.py
git diff --check
```

## 6. Recent commit landmarks

| Commit | Purpose |
| --- | --- |
| `631978e` | Live visual region resizing for every edge and corner. |
| `547e184` | Correct comparison-slider clipping geometry. |
| `d63d45b` | Prevent duplicate LoRAs when loading image reference metadata. |
| `1f1de60` | Synchronize setup numeric inputs and sliders in both directions. |
| `b1b1931` | Dedicated staged QML setup window with apply/discard behavior. |
| `165182f` | Improved selection, movement, edge resizing, region strip, and source retention. |
| `7d52ee2` | LoRA activation/removal and editable slider controls. |
| `093ec16` | Introduced the Qt Quick workspace interface. |
| `9170f1f` | Added the future RunPod web workspace specification. |
| `d60e10d` | Restored saved image-edit controls and fixed seed behavior. |
| `1dd6490` | Excluded conflicting source-global text from edit conditioning. |
| `13ef835` | Added the real-GPU image-edit validation harness. |
| `38c345f` | Preserved reference routing across repeated edits. |
| `da0e3be` | Added latent denoising constraints and protected compositing. |
| `4b72dab` | Added distinct reference and edit-target layers. |
| `c350308` | Restored image-associated project state. |
| `6b0e4d9` | Initial regional image-edit workspace. |

Use `git show <commit>` for design context rather than reconstructing these changes from
the final files alone.

## 7. Deferred RunPod/web product

The future deployment is specified in `docs/runpod_web_workspace_spec.md`. Phase one is a
persistent-Pod mode using the user's own RunPod account/API key and a stopped/restarted Pod
whose regular `/workspace` volume persists. Phase two adds a choice between that mode and a
portable workspace backed by network storage and ephemeral GPU selection.

The specification includes GPU-priority discovery, lifecycle/cost safety, credential
handling, container and storage sizing, prebuilt automatic runtime setup, local uploads,
direct Civitai/Hugging Face downloads, resumability, agent APIs, security boundaries, and
the phase-two migration seam. Do not begin implementation from conversation memory; treat
that file as the source of truth.

## 8. Recommended resumption order

1. Read this file, `README.md`, and the relevant regional engineering-reference section.
2. Confirm the current branch and inspect changes since `631978e`.
3. Preserve untracked/user-owned files. At this snapshot, `prompts/test4.json` and
   `prompts/testfive.json` are intentionally untracked and must not be modified or committed
   without explicit direction.
4. Re-run the 46-test focused QML/desktop suite.
5. Triage the six complete-suite failures before treating CI as clean.
6. For image-edit work, inspect the latest `outputs/gpu-tests/*/validation_report.json`
   reports and comparison images, then establish a new visual acceptance case before
   changing masks, conditioning, or feather values.
7. Keep each logical change in its own commit. This has been the requested workflow and
   makes regression comparison practical.

## 9. Known limitations and next decision points

- The QML frontend still depends on a hidden Widgets `MainWindow`; architectural cleanup is
  incomplete.
- Image editing guarantees exact retained pixels outside local composite support, but edit
  adherence and seamless identity-preserving synthesis inside/near the box are still
  experimental.
- Whole-image edit mode is intrinsically high drift with the current img2img pipeline.
- The current GUI is a strong local testing interface, not the planned install-free web
  deployment.
- GPU correctness and image quality cannot be established by the dependency-light unit
  suite. Use the actual configured ComfyUI Python environment and retain generated evidence.
- RunPod lifecycle, storage, uploads, provider downloads, and remote jobs are specified but
  unimplemented.

The safest next image-edit research step is a fixed-seed visual matrix over denoise,
latent feather, composite feather, and edit prompt wording, scored separately for outside
pixel exactness, boundary visibility, edit adherence, and identity retention. The safest
next architecture step is extracting project/workspace state from the hidden `MainWindow`
behind stable interfaces before sharing it with a web client.
