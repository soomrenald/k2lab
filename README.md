# K2 Region Lab

K2 Region Lab is a local PySide6 research application for generic pixel-space control of Krea 2. A region is one shared spatial domain for prompt routing, unfused LoRA delta gating, influence measurement, and attention tuning.

The implementation is at the foundation milestone. It currently provides:

- discovery and safetensors-header validation for local Krea 2 components;
- exact output-pixel to 16×16 Krea image-token geometry;
- area-fraction box rasterization and immutable spatial layouts;
- a PySide6 desktop shell with movable, corner-resizable, and deletable pixel-space boxes;
- editable, unique region names propagated through region and LoRA scope controls;
- an explicit Auto/Subject/Background role selector on every region;
- a LoRA file browser and loaded-LoRA library with Global or multi-region scope assignment;
- per-LoRA strength, Krea key diagnostics, and global or strict regional application;
- JSON project save/load for prompts, generation settings, boxes, names, LoRAs, and runtime paths;
- a typed worker protocol with isolated baseline GPU execution;
- a configurable external ROCm worker using the existing ComfyUI interpreter;
- full transformer, Qwen, and VAE tensor manifests with Krea-specific shape validation;
- fixed-seed Krea 2 Turbo baseline generation with progress events and PNG metadata;
- optional 2×/4× post-upscaling after the Krea GPU state has been released;
- selectable 16 GB memory policies, live VRAM/RAM telemetry, boundary offload, and OOM retry;
- confirmed cleanup of current-user K2 workers without terminating unrelated ROCm applications;
- in-app ROCm diagnostics with device permissions, runtime identity, and remediation hints;
- dependency-light unit tests for the geometry and artifact-discovery contracts.

The configured default model locations are:

```text
~/ComfyUI/models/diffusion_models/krea2_turbo_fp8_scaled.safetensors
~/ComfyUI/models/text_encoders/qwen3vl_4b_fp8_scaled.safetensors
~/ComfyUI/models/vae/qwen_image_vae.safetensors
```

## Development

The model execution environment targets Python 3.12. The geometry and discovery tests intentionally use only the standard library so they can run before GPU dependencies are installed:

```bash
PYTHONPATH=src python -m unittest discover -s tests -v
```

Inspect configured model artifacts without launching Qt:

```bash
PYTHONPATH=src python -m k2_region_lab --check-models
```

After installing the desktop dependencies, launch the application with:

```bash
k2lab
```

Do not resolve or replace the ComfyUI worker interpreter symlink: its venv path is required so Python finds the ROCm environment's `pyvenv.cfg`. The application preserves this path automatically.

The desktop starts its GPU worker with `~/ComfyUI/venv_rocm/bin/python` by default. Override the runtime without changing GUI dependencies using:

```text
K2LAB_WORKER_PYTHON
K2LAB_COMFYUI_ROOT
K2LAB_AUTO_START_WORKER
K2LAB_MEMORY_POLICY
K2LAB_RESERVE_VRAM_GB
K2LAB_MINIMUM_SYSTEM_RAM_GB
K2LAB_CPU_VAE
K2LAB_OOM_RECOVERY
K2LAB_OUTPUT_DIRECTORY
K2LAB_FILENAME_PREFIX
```

The right-side settings pane has separate **Model & memory** and **Generation & spatial** tabs so only one control group is visible at a time. Use **Validate tensors** before **Load Krea 2 baseline**. Validation reads only safetensors headers and writes complete manifests under the configured K2 Lab data directory. After loading, **Generate image** runs an eight-step Euler/simple Turbo pass by default and displays the saved image behind the editable region boxes. The generation tab provides an output-folder browser and editable filename prefix; both are saved in project JSON. The unified-prompt preview is a resizable, selectable-text dialog. The event viewer follows new messages only while its scrollbar is already at the latest event.

The compact monitor beside Events reads Linux DRM/sysfs and `/proc/meminfo` directly once per second, showing current GPU VRAM, system RAM, GPU activity, and a two-minute VRAM/RAM history without trying to embed `nvtop`. Seed behavior can be **Fixed**, **Random**, or **Increment** and is stored in the project. Every generation uses a disposable worker: after the final image is saved, that process exits so ROCm allocations, model weights, LoRA tensors, and Python heap memory are returned to the OS. The next **Generate image** click automatically starts, probes, validates, and loads a fresh worker before dispatching the request. **Stop generation** terminates the same isolated worker early while leaving the desktop and unsaved configuration open.

The **Use unified spatial prompting** backend compiles the global prompt and every enabled regional prompt into one scene-wide Qwen caption. It adds natural normalized positions, sizes, and relationships, records the text-token span belonging to every regional clause, and maps each half-open pixel box onto Krea's 16-pixel image-token grid. During each of Krea's single-stream transformer blocks, a bidirectional soft attention bias links regional text spans to nearby image tokens while leaving global text, text-to-text, and image-to-image attention unchanged. Because masked ROCm SDPA falls back to a memory-heavy math kernel on the target GPU, the override computes the exact biased softmax in bounded query chunks without materializing a complete per-head score matrix.

Every box has an **Auto**, **Subject target**, or **Background band** selector. Auto treats boxes spanning at least 70% of the canvas width as background and narrower boxes as subjects, which migrates existing projects without manual edits. Background bands keep full strength throughout the box and use a softer outside penalty. With **Make subjects fill their boxes** enabled, the box means the desired visible extent rather than only a center location: the unified caption includes explicit boundary coordinates, an image-relative size/framing description, and a minimal-empty-margin instruction, while attention remains strong through the box edges. Disable it to recover the older position-only center-weighted behavior. Overlapping subjects can still compete for soft ownership of image tokens so two subjects are less likely to collapse into the same location. Placement and size guidance remain strongest through the first half of denoising and can relax during late detail refinement.

**Inside boost**, **Outside penalty**, and **Spatial falloff** separately control the field. The overlap-competition and late-step relaxation checkboxes can be disabled for comparison runs. The GUI can preview the compiled prompt, and the prompt, effective roles, token spans, pixel boxes, settings, and attention-call count are stored in image metadata. The `krea-unified-spatial-attention-v3` backend still uses one global denoising pass; its boxes are stronger placement and scale targets, not hard masks. Regional negative text remains persisted, but Turbo's CFG-free path has no negative branch to route; localized negative conditioning is reserved for Raw/CFG support.

Each loaded LoRA has a saved model-strength control and an **Inspect selected LoRA…** action. Inspection runs in the loaded worker, reports exact Krea transformer target coverage, and catches files that only appear to load because their namespaces do not match. K2 Lab aligns the AI Toolkit `diffusion_model.blocks…` and `diffusion_model.txtfusion…` conventions used by Krea LoRAs with the namespace exposed by the active ComfyUI worker. Compatible LoRAs use unfused forward adapters, so base FP8 weights are never rewritten. A Global route enables all text and image lanes. A regional route enables only the regional clause's text-token span and the 16×16 image-token cells intersecting its assigned half-open pixel box; its delta is zero on all other lanes. One LoRA may use the union of several named boxes, several LoRAs may share a box, and global and regional adapters can run together through one composite layer hook. Boundary cells are weighted by their actual pixel overlap. Strength, all 256 target matches, route coverage, and measured text/image delta RMS are recorded in PNG metadata; the reported outside-gate delta is exactly zero by construction.

The prompt-attention field and LoRA gate intentionally have different semantics. Prompt placement can feather outside a box so the model composes a coherent scene. LoRA parameter deltas do not use that falloff. Information can still propagate from an enabled lane through the transformer's normal self-attention, but no LoRA projection is evaluated into an unassigned output lane. Delta-magnitude-driven attention tuning is the next control stage built on these measurements.

Before VAE decode, the worker explicitly offloads the denoising transformer and releases routed adapter hooks outside PyTorch inference mode. This both frees the additional VRAM held by multiple LoRAs and prevents ComfyUI's quantized FP8 parameter reconstruction from receiving inference tensors during the VAE memory handoff.

**Post-upscale after releasing Krea VRAM** is a scene-wide output stage, not another denoising pass. The worker decodes the original image, copies it to CPU memory, unloads Krea, its routed LoRAs, and the VAE from the accelerator, and empties the ROCm cache before upscaling. The built-in **CPU Lanczos** option works without additional weights and produces an exact 2× or 4× output, but it cannot invent detail absent from the 1024×1024 render. For learned detail recovery, choose **Neural model (tiled GPU)** and browse to an ESRGAN/Real-ESRGAN-compatible `.pth`, `.pt`, or `.safetensors` file. Neural inference uses overlapping tiles with automatic OOM tile reduction and keeps its assembled output in system RAM. The selected method, model, base size, and final size are stored in project JSON and PNG metadata.

If the accelerator probe fails, **Diagnose accelerator…** appears below the status. It restarts the worker with a clean environment and reports the interpreter, Torch/ROCm versions, device-file access, visibility variables, initialization errors, and suggested fixes without closing the application.

Use **Release K2 GPU memory…** if a failed run leaves a K2 worker holding VRAM. The confirmation dialog lists every matching current-user K2 worker PID, then stops only those processes. It deliberately does not terminate ComfyUI or other ROCm applications.

Launch with `DEBUG=1 k2lab` to write bounded rotating logs under `~/.local/share/k2-region-lab/logs/` (or the configured `K2LAB_DATA_DIR`). The desktop and GPU worker use separate `desktop-debug.log` and `worker-debug.log` files.

The original local stack uses PyTorch 2.9.1 with ROCm 6.4. Because scaled FP8 execution is native only on ROCm 6.5 or newer, the worker automatically uses ComfyUI's low-VRAM fallback on ROCm 6.4. **Safe 16 GB** is the default policy: neither its 4 GiB VRAM floor nor its 14 GiB available-system-RAM floor can be reduced by an older saved project, and it reports memory at each generation boundary and denoising step. If free VRAM crosses the critical floor between denoising steps, the worker stops before the next allocation and makes its single deterministic retry with a 5 GiB reserve and CPU VAE decode. Memory controls are locked while a model is loaded so the active worker configuration remains explicit. The worker also enables PyTorch expandable allocator segments and ROCm's experimental AOTriton attention backend when the environment does not explicitly configure them. A 1024×1024 eight-step run does not fit reliably on the tested 16 GB GPU under ROCm 6.4's BF16 dequantization fallback. The same baseline completed on PyTorch 2.10.0 with ROCm 7.1 and native scaled FP8/AOTriton, keeping about 2.8 GiB free during denoising. GPU VAE decode may exhaust its regular allocation and use ComfyUI's tiled fallback; K2 Lab keeps that fallback inside PyTorch inference mode for PyTorch 2.10 compatibility. Use a smaller canvas on ROCm 6.4 or select the ROCm 7.1 worker with `K2LAB_WORKER_PYTHON=~/ComfyUI/venv_rocm7/bin/python k2lab`. This launch-time selection takes precedence over an older worker path stored in an opened project.

