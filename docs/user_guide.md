# K2Lab User Guide

K2Lab is a local desktop workspace for Krea 2 image generation, regional prompting,
regional LoRA routing, image editing, and face refinement. It uses the models and
GPU-enabled Python environment from an existing ComfyUI installation.

K2Lab does not include or download model weights. Before configuring Krea or Qwen
assets, review the [model-use policy](../MODEL_USE_POLICY.md) and its linked upstream
terms. The desktop and RunPod configurations are intended as private,
operator-reviewed workspaces; do not expose them as public/shared inference services
without safeguards appropriate to the current Krea license and Acceptable Use Policy.

This guide covers the default Qt Quick interface. The older interface is still
available with `k2lab --legacy-widgets`, but its layout differs from the screenshots
and control names described here.

## 1. Before you begin

K2Lab requires:

- Linux and Python 3.12;
- a working ComfyUI checkout;
- a CUDA- or ROCm-enabled Python environment belonging to that ComfyUI checkout;
- a Krea 2 transformer, Qwen text encoder, and compatible VAE;
- enough GPU and system memory for the selected canvas size and memory policy.

Install and launch K2Lab:

```bash
git clone https://github.com/soomrenald/k2lab.git
cd k2lab
./scripts/install.sh
.venv/bin/k2lab
```

K2Lab's desktop environment should remain separate from the GPU worker environment.
Do not install another Torch build into K2Lab's `.venv`. If ComfyUI is not in a
standard location, select its checkout and Python interpreter in the setup window.

## 2. First-run setup

Select the gear button in the lower-left corner to open **Runtime & model setup**.
Changes in this window are staged until you select **Apply settings**.

### Runtime

- **ComfyUI checkout** is the root directory of the ComfyUI installation.
- **GPU worker Python** is the Python executable from ComfyUI's CUDA or ROCm
  environment. Leave it on **Auto** when K2Lab can discover the correct environment.

Keep the environment path itself rather than resolving a virtual-environment symlink.
Python uses that location to find the environment's `pyvenv.cfg`.

### Models

Select **Discover** to scan the configured model directories. You can then choose a
discovered Krea checkpoint or set exact files for:

- the Krea transformer;
- the Qwen text encoder;
- the VAE;
- the optional face detector.

A blank exact-file field means automatic discovery. Model weights are not included
with K2Lab.

### Memory

Choose the profile nearest to the GPU size. The named profiles are safety floors:
their reserved VRAM and minimum free system RAM can be increased, but not reduced.
Use **Custom / any GPU** when the hardware does not fit a named profile.

- **Keep VRAM free** controls how much GPU memory K2Lab tries to reserve.
- **Minimum free RAM** stops work before system memory becomes critically low.
- **Decode with CPU VAE** reduces VAE pressure on the GPU at the cost of speed and
  additional system-memory use.
- **Retry once after OOM** permits one conservative recovery attempt after an
  out-of-memory failure.

For an 8–12 GB GPU, start with the corresponding low-memory profile and a smaller
canvas. For a 16 GB GPU, **Safe 16 GB** is the default. A profile cannot compensate
for an incompatible Torch or ComfyUI installation.

### Output

Choose the output directory and filename prefix. Generated, edited, and refined
images are saved as PNG files with project and execution metadata.

### Validate the setup

Use the setup actions in this order:

1. **Discover**
2. **Start worker**
3. **Validate**
4. **Load model**

The status table reports the worker, accelerator, model components, and memory
state. Use **Diagnose** if the accelerator probe fails. Use **Release GPU memory**
only when a failed K2Lab worker remains alive; it does not terminate ComfyUI or
unrelated GPU applications.

After the settings are correct, select **Apply settings**. Closing the setup window
with unapplied changes offers **Apply settings**, **Discard**, or **Cancel**.

### Experimental backend and support bundle

ComfyUI remains the recommended/default backend. The **Experimental backend** section
can select **Native K2 (experimental)** for the current application session. Changing
the selection stops the isolated worker and clears its loaded-model state; it does not
modify projects, model files, configuration, or the `K2LAB_BACKEND` environment
variable. Restarting K2Lab therefore returns to the configured environment selection,
which defaults to ComfyUI.

Native never falls back silently during a job. Unsupported face refinement, projector,
and post-upscale controls are disabled with an explanation. Select **Use ComfyUI
fallback** to restore those controls for the next worker.

Select **Create issue report** to save a ZIP below the application data directory's
`issue-reports/` folder. The bundle contains version, platform, backend, capability,
model-hash, placement, parity, and memory summaries. It intentionally excludes full
prompts, environment variables, logs, user paths, projects, images, and LoRA names.
Review `issue-report.json` before sharing the bundle; attach logs or projects separately
only after checking them for private content.

## 3. Workspace tour

The left rail selects one of three independent modes:

- **Generate** creates an image from prompts, regions, and generation-mode LoRAs.
- **Edit** modifies a loaded image using edit targets and edit-mode LoRAs.
- **Faces** detects and refines selected faces in an existing image.

Generate and Edit use separate controls and LoRA assignments. Generation prompts,
regions, and LoRA routes are not silently applied to image editing; the source
layout shown in Edit mode is reference-only. Faces deliberately reuses the subject
regions and regional LoRAs associated with its source image.

The main areas are:

- the top project bar: **New**, **Open**, **Import PNG**, **Save**, and **Save as**;
- the center canvas and its region boxes;
- the right inspector: **Prompt**, **Regions**, **LoRAs**, and **Advanced**;
- the bottom bar: status, memory, progress, **Stop**, and the active run action;
- the Events button in the lower-left rail for logs and live GPU/RAM telemetry.

Use the inspector toggle above the canvas to gain more canvas space.

## 4. Generate an image

### A basic generation

1. Select **Generate**.
2. Enter a complete scene description in **Prompt → Global prompt**.
3. Open **Advanced** and choose the canvas size, sampler, scheduler, steps, and seed.
4. Select **Generate** in the bottom-right corner.
5. Watch progress and memory in the bottom bar or open **Event history** for details.

Each run uses a disposable worker. When the run finishes or is stopped, that worker
exits and releases its model, LoRA, and Python allocations.

### Seeds and batches

**Seed behavior** controls what happens after each run:

- **Fixed** repeats the same seed.
- **Random** selects a new seed.
- **Increment** advances the seed predictably.

Enable **Run generation in batch mode** to run the configured number of generations.
Outputs are saved separately.

### Add regional prompts

1. Select **Draw region** above the canvas or **+ Draw** in the Regions tab.
2. Drag on the canvas to create a box.
3. Select the box and give it a unique name.
4. Choose its role:
   - **Auto** treats wide bands as backgrounds and narrower boxes as subjects;
   - **Subject target** is for a person or object;
   - **Background band** is for broad scene areas.
5. Enter the region prompt and keep **Enabled** selected.
6. Repeat for other subjects or background areas.

Drag a box to move it and drag its corner handles to resize it. Use **Delete
selected** or the Regions tab's **Delete** action to remove it.

The Regions list is also the front-to-back depth order. Use **Forward** and
**Backward** when boxes overlap. The higher region receives priority in shared
subject areas.

Keep the global prompt scene-wide. Put subject-specific identity, clothing, pose,
or object details in the corresponding regional prompt. Avoid duplicating
conflicting descriptions between the global and regional prompts.

### Regional-attention controls

The normal starting point is to leave **Use unified spatial prompting** enabled.
The remaining controls tune how strongly prompts follow their boxes:

- **Inside boost** increases attention inside a region.
- **Outside penalty** discourages a region's prompt outside its box.
- **Spatial falloff** controls how gradually soft placement guidance changes near
  the boundary.
- **Separate overlapping subject targets** reduces subject collapse in overlaps.
- **Make subjects fill their boxes** treats a subject box as its desired visible
  extent rather than only its center.
- **Relax spatial guidance during late steps** reduces placement pressure while
  fine details form. **Late scale** is the final guidance multiplier.

These controls guide attention; they are not pixel masks. Image-to-image attention
continues across the canvas so region boundaries do not become visible seams.

Use **Preview unified prompt…** to inspect the scene caption K2Lab will send to the
text encoder.

### Phrase emphasis

To emphasize a word or phrase:

1. Highlight its exact text in the global or selected-region prompt.
2. Set a strength under **Prompt → Phrase emphasis**.
3. Select the corresponding emphasis action.

Start around `0.30–0.60`. Values at or above `1.00` can dominate the composition.
If the prompt text changes and no longer contains the exact phrase, the saved
emphasis is marked invalid until it is removed or the text is restored.

## 5. Use LoRAs

Open **LoRAs** and select **+ Add LoRA** to add a safetensors file to the library.
Each card has its own active switch, strength, scope, regions, and routing mode.

### Scope

- **Global** applies the LoRA scene-wide.
- **Add selected** assigns the LoRA to the currently selected region.
- A LoRA may be assigned to multiple regions.
- Multiple LoRAs may share a region.

Use **Inspect Krea compatibility** before relying on a new file. A file can load
successfully while exposing no compatible Krea targets.

### Routing

**Standard regional** is the normal box-scoped route. It applies compatible text
and image deltas to the assigned regional clauses and image-token cells.

**Character identity (face)** is intended for trained character LoRAs. Assign the
LoRA to a subject region and enter its exact training trigger. K2Lab inserts the
trigger into that region's identity anchor automatically; it does not need to be
repeated in the visible prompt. Character routing is unavailable for Global scope.

LoRA strength `1.0` means the full saved adapter strength and can be aggressive.
Reduce it when colors, identity, or style overwhelm the prompt.

### Mode isolation

Generate and Edit maintain separate LoRA assignments. Switching to Edit does not
activate generation-mode routes. In Edit mode, assign a LoRA while **Edit targets**
is selected if it should affect the edit. LoRAs visible with the reference/source
layout remain reference information and are not edit conditioning.

## 6. Edit an image

### Create a boxed edit

1. Select **Edit**.
2. Select **Load image** and choose a PNG, JPEG, or WebP image.
3. Select **Edit targets**.
4. Draw one or more edit regions.
5. Select each region and enter what should change in its prompt.
6. Enter an optional scene-wide **Edit instruction**.
7. Configure edit-only LoRAs while **Edit targets** is active.
8. Set denoise and feather controls under **Advanced**.
9. Select **Run image edit**.

The edit worker receives only the edit instruction, edit-target prompts, and
edit-mode LoRAs. A prompt restored from the source image is available for inspection
under **Source layout**, but it is not sent as edit conditioning. This prevents an
old source description—such as the original color—from fighting the requested edit.

Use **Edit entire image** only when broad scene changes are intentional. Without
that option, at least one edit box should be used.

### Denoise and feather controls

- **Denoise** controls how far the edit may depart from the source. Lower values
  preserve structure and identity; higher values allow stronger changes.
- **Latent feather** is the transition collar used during denoising. It lets the
  model blend structure around an edit instead of meeting a hard latent boundary.
  It may influence pixels near the box while the final composite remains protected.
- **Composite feather** is the narrower pixel-space blend used after decoding. It
  determines how smoothly the edited result is blended over the original.

Pixels outside final composite support are copied exactly from the source. A useful
starting point for subtle texture or detail work is low denoise with the defaults of
64 px latent feather and 48 px composite feather. Obvious color, material, or object
replacement usually needs the normal eight-step workload and a higher denoise value;
try `0.6`–`0.8`, then adjust downward if too much structure changes.

The edit box is a hard permission boundary. Include every part that should change: a
box below a vase rim, above a shoe sole, or inside an object's silhouette preserves the
excluded pixels even when the prompt asks to replace the entire object. If restored
source-region text contradicts the requested change, lower **Reference description
retention**; this trades source-description/identity retention for stronger prompt
adherence.

If a boundary is visible, increase feathering moderately or enlarge the box to
include contextual pixels. If too much of the source changes, lower denoise, reduce
the feathers, or use a tighter edit box.

The regional-attention options in Edit mode apply only to edit targets. **Relax
spatial guidance during late steps** has its own checkbox and can be disabled for
comparison runs.

### Compare source and result

After an edit completes, use the control at the lower-left of the canvas:

- **Source** shows the original;
- **Result** shows the edited image;
- **Compare** overlays a movable split.

In Compare mode, drag the slider to move the split continuously and inspect edges,
identity changes, and unintended edits.

### Restored image state

When a PNG contains K2Lab metadata—or an exact matching `.k2lab.json` sidecar—the
editor can restore its prompts, boxes, LoRA library, sampling settings, projector
state, and seed for inspection. Restored generation data remains isolated from the
edit worker.

Loading a same-sized replacement image preserves the edit layout. Accepting an
image with different dimensions clears edit boxes and their LoRA assignments because
the old coordinates no longer match.

## 7. Refine faces

Face refinement is a separate operation and never runs automatically after
generation.

1. Select **Faces**.
2. Load a source image, or select **Use latest first pass**.
3. Select **Detect faces**.
4. Review the numbered detections and select the faces to refine.
5. Adjust the conservative defaults only when needed.
6. Select **Refine selected faces**.

Selected detections are green; excluded detections are orange. K2Lab matches faces
to nearby subject regions and uses the LoRAs assigned to that region. Global LoRAs
are omitted from face refinement.

Important controls:

- **Denoise** changes how strongly the crop is regenerated.
- **Crop size** is the working resolution sent to Krea, not the detected face size.
- **Padding** includes more surrounding context.
- **Edge feather** softens the crop boundary.
- **Regional LoRA scale** multiplies the assigned regional LoRA strengths.
- **Detector threshold** trades missed faces against false detections.
- **Blend** controls how much of the refined pixel difference is applied.
- **Detector device** chooses Auto, CPU, or NVIDIA CUDA.

Start with a 512 px crop and the default low denoise. If the face deforms, reduce
denoise first, then regional LoRA scale, then blend. Changing the detector threshold
or device invalidates current detections, so run **Detect faces** again.

Face detection requires the compatible `face_det.onnx` model. NVIDIA CUDA detection
also requires `onnxruntime-gpu` in the worker environment. AMD/ROCm systems should
use the CPU provider with `onnxruntime`.

## 8. Projects, imported PNGs, and metadata

- **New** starts a new project.
- **Open** loads a saved K2Lab project JSON.
- **Save** updates the current project file.
- **Save as** writes a new project file.
- **Import PNG** restores a project snapshot embedded in a K2Lab PNG.

Imported PNG state has no JSON save destination until you use **Save** or **Save
as**. Project JSON stores source paths and settings, not the source image pixels.
Moving or deleting a referenced source image leaves its settings available but
disables editing until an image is loaded again.

K2Lab PNGs contain a compact project snapshot under the `k2lab_project` text key.
They also record execution details such as prompts, effective regions, LoRA routes,
sampling controls, and face-refinement reports. This makes a PNG the most convenient
way to recover the settings that produced it.

## 9. Upscaling

Generation can optionally run **Post-upscale after releasing Krea VRAM**:

- **CPU Lanczos** needs no model and produces an exact 2× or 4× resize.
- **Neural model (tiled GPU)** uses an ESRGAN/Real-ESRGAN-compatible model and can
  recover learned detail.

Upscaling happens after Krea, its LoRAs, and the VAE have been released from the
accelerator. It is a scene-wide output stage, not another denoising pass.

## 10. Troubleshooting

### Generate or Edit is disabled

Open setup and confirm that the worker has started, all required artifacts validate,
and the model is loaded. Image editing also requires a loaded source image. Face
refinement requires at least one selected detection.

### The worker reports CPU-only Torch

The selected worker Python is not the CUDA/ROCm environment used by ComfyUI. Select
the correct interpreter in setup and run **Diagnose**.

### A model is not discovered

Confirm the ComfyUI root and model directories. Select an exact transformer, text
encoder, or VAE file when several compatible files share a directory.

### A LoRA appears to do nothing

Confirm that it is active, has a nonzero strength, and is assigned to Global or the
intended region in the active mode. Then run **Inspect Krea compatibility**.

### An edit ignores a requested color

Confirm the instruction is attached to an Edit target rather than the Source
layout. Disable or reduce edit-mode LoRAs that overpower the instruction, especially
at strength `1.0`. Raise denoise gradually if the source color remains too strongly
preserved.

### An edit changes too much

Use a boxed edit, lower denoise, tighten the box, and reduce edit-mode LoRA strength.
Make sure **Edit entire image** is disabled.

### A boxed edit has a visible seam

Include enough context inside the box and increase latent or composite feather
moderately. Latent feather helps the denoising transition; composite feather smooths
the final pixel blend.

### The application runs out of memory

Reduce canvas or face crop size, select a smaller-GPU memory policy, enable CPU VAE,
keep OOM recovery enabled, and close unrelated GPU applications. Use **Release GPU
memory** only for a stranded K2Lab worker.

### More diagnostic detail is needed

Open **Event history** for worker messages and live resource telemetry. Launch with
debug logging when persistent logs are needed:

```bash
DEBUG=1 k2lab
```

Logs are written under `~/.local/share/k2-region-lab/logs/` unless
`K2LAB_DATA_DIR` selects another application-data directory.

For a bounded first report, open Setup and select **Create issue report**. That bundle
does not include the debug logs or full prompts.
