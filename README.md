# K2 Region Lab

K2 Region Lab is a local PySide6 research application for generic pixel-space control of Krea 2. A region is one shared spatial domain for prompt routing, unfused LoRA delta gating, influence measurement, and attention tuning.

## Installation

K2 Region Lab targets Linux, Python 3.12, and either NVIDIA CUDA or AMD ROCm. It uses a separate lightweight desktop environment while launching model work through an existing GPU-enabled ComfyUI Python environment. Model weights are not included. Runtime and model selection are accelerator-neutral; the selected ComfyUI environment determines whether Torch uses CUDA or ROCm.

Prerequisites:

- a current ComfyUI checkout with Krea 2 support;
- a Python 3.12 ComfyUI environment with a working CUDA or ROCm PyTorch build (`torch.cuda.is_available()` must return `True`); automatic selection checks common `.venv`, `venv`, and ROCm environment names, and the GUI can select any other interpreter;
- the Krea 2 Turbo transformer, Qwen text encoder, and VAE listed below.

Clone the repository and install the desktop application in its own environment:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

If ComfyUI or its GPU environment is somewhere else, configure both paths before launching:

```bash
export K2LAB_COMFYUI_ROOT=/path/to/ComfyUI
export K2LAB_WORKER_PYTHON=/path/to/ComfyUI/venv/bin/python
k2lab
```

With the default paths, simply run:

```bash
k2lab
```

For an NVIDIA installation, point K2 Lab at the Python interpreter from a CUDA-enabled ComfyUI environment. Nothing needs to be installed into the lightweight desktop environment beyond the normal K2 Lab dependencies:

```bash
export K2LAB_COMFYUI_ROOT=/path/to/ComfyUI
export K2LAB_WORKER_PYTHON=/path/to/ComfyUI/.venv/bin/python
K2LAB_MEMORY_POLICY=large_24gb k2lab
```

Use the environment created by ComfyUI's platform-appropriate installation instructions; do not install a second Torch build into K2 Lab's desktop `.venv`. The accelerator diagnostic reports whether the selected worker is using CUDA, ROCm, or a CPU-only Torch build.

Face refinement additionally requires an ONNX Runtime package in the ComfyUI worker environment and FantasyPortrait's `face_det.onnx` under `custom_nodes/ComfyUI-WanVideoWrapper/fantasyportrait/models/`. Install exactly one provider package for the machine:

```bash
# AMD/ROCm or CPU-only detector execution
/path/to/ComfyUI/GPU_ENV/bin/python -m pip install onnxruntime

# NVIDIA CUDA detector execution
/path/to/ComfyUI/GPU_ENV/bin/python -m pip install onnxruntime-gpu
```

Do not install both distributions in the same environment because they provide the same `onnxruntime` module. The Face refinement **Detector device** control defaults to **Auto**, which prefers `CUDAExecutionProvider` when the NVIDIA package exposes it and otherwise uses `CPUExecutionProvider`. **CPU** is the portable AMD/ROCm choice; **NVIDIA CUDA** requires a working CUDA ONNX Runtime installation and fails explicitly instead of silently changing providers.

The implementation is at the foundation milestone. It currently provides:

- discovery and safetensors-header validation for local Krea 2 components;
- exact output-pixel to 16×16 Krea image-token geometry;
- area-fraction box rasterization and immutable spatial layouts;
- a PySide6 desktop shell with movable, corner-resizable, and deletable pixel-space boxes;
- drag-ordered front-to-back region depth with explicit overlap/occlusion prompting;
- editable, unique region names propagated through region and LoRA scope controls;
- an explicit Auto/Subject/Background role selector on every region;
- a LoRA file browser and loaded-LoRA library with Global or multi-region scope assignment;
- per-LoRA strength, Krea key diagnostics, and global or strict regional application;
- JSON project save/load for prompts, generation settings, boxes, names, LoRAs, and runtime paths;
- a typed worker protocol with isolated baseline GPU execution;
- a configurable external CUDA or ROCm worker using the existing ComfyUI interpreter;
- full transformer, Qwen, and VAE tensor manifests with Krea-specific shape validation;
- Krea 2 Turbo baseline generation with the current standard ComfyUI sampler and scheduler choices, progress events, and PNG metadata;
- optional automatic face-crop refinement with per-region character LoRAs;
- optional 2×/4× post-upscaling after the Krea GPU state has been released;
- GPU-size memory profiles plus fully custom VRAM/RAM guards, live VRAM/RAM telemetry, boundary offload, and proportional OOM retry;
- confirmed cleanup of current-user K2 workers without terminating unrelated GPU applications;
- in-app CUDA/ROCm diagnostics with device permissions, runtime identity, and remediation hints;
- dependency-light unit tests for the geometry and artifact-discovery contracts.

The checked-in [`k2_region_lab.toml`](k2_region_lab.toml) file contains the default
paths and generation settings. Its default model locations are:

```text
~/ComfyUI/models/diffusion_models/krea2_turbo_fp8_scaled.safetensors
~/ComfyUI/models/text_encoders/qwen3vl_4b_fp8_scaled.safetensors
~/ComfyUI/models/vae/qwen_image_vae.safetensors
```

## Configuration

K2 Region Lab loads the first available TOML configuration from the following
locations:

1. the file named by `K2LAB_CONFIG_FILE`;
2. `k2_region_lab.toml` in the current working directory;
3. `~/.config/k2-region-lab/config.toml`.

Environment variables override values from the selected file. Relative paths in a
TOML file are resolved relative to that file, so the checked-in `output_directory =
"outputs"` continues to point at this checkout's output folder regardless of the
shell's working directory. Blank entries under `[models]` retain automatic model
discovery; set an entry to an exact file when several compatible models share a
directory.

The `[paths]` section configures the ComfyUI checkout, GPU-worker interpreter,
application data and output folders, and the transformer, text encoder, VAE, LoRA,
and upscaler model directories. `[models]` optionally pins the exact transformer,
text encoder, VAE, face detector, and default upscaler files. `[runtime]` contains
worker startup and memory defaults. `[generation]` contains the new-project canvas,
step, sampler, scheduler, seed, seed behavior, and filename defaults. Edit the
checked-in file for this installation, or copy it to the user configuration path
for a configuration shared by multiple checkouts. Set `worker_python = "auto"` for
vendor-neutral environment discovery, or provide an exact CUDA/ROCm interpreter.

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

Do not resolve or replace the ComfyUI worker interpreter symlink: its venv path is required so Python finds that CUDA or ROCm environment's `pyvenv.cfg`. The application preserves this path automatically.

The desktop automatically selects a common GPU-enabled environment under the configured ComfyUI checkout. Override the runtime without changing GUI dependencies using:

```text
K2LAB_CONFIG_FILE
K2LAB_WORKER_PYTHON
K2LAB_COMFYUI_ROOT
K2LAB_DATA_DIR
K2LAB_AUTO_START_WORKER
K2LAB_MEMORY_POLICY
K2LAB_RESERVE_VRAM_GB
K2LAB_MINIMUM_SYSTEM_RAM_GB
K2LAB_CPU_VAE
K2LAB_OOM_RECOVERY
K2LAB_OUTPUT_DIRECTORY
K2LAB_FILENAME_PREFIX
K2_TURBO_DIR
K2_TEXT_ENCODER_DIR
K2_VAE_DIR
K2_LORA_DIR
K2_UPSCALE_MODEL_DIR
K2_TURBO_MODEL
K2_TEXT_ENCODER_MODEL
K2_VAE_MODEL
K2_FACE_DETECTOR_MODEL
K2_UPSCALE_MODEL
```

Memory settings can be selected in **Model & memory** or set before launch. A policy is a safety floor: its reserve and system-RAM values may be raised in the UI, but not lowered. **Custom / any GPU** removes the hardware-size assumptions and permits a 0.5–128 GiB VRAM reserve and a 4–256 GiB available-RAM guard.

| Policy | Suggested hardware | VRAM kept free | Minimum available RAM | CPU VAE |
| --- | --- | ---: | ---: | --- |
| `low_8gb` | 8 GB GPU | 1 GiB | 24 GiB | Yes |
| `safe_12gb` | 12 GB GPU | 2 GiB | 18 GiB | Yes |
| `safe_16gb` | 16 GB GPU; unchanged default | 4 GiB | 14 GiB | No |
| `balanced` | General manual starting point | 3 GiB | 12 GiB | No |
| `performance` | More model residency | 2 GiB | 12 GiB | No |
| `large_24gb` | 24 GB or larger GPU | 3 GiB | 12 GiB | No |
| `custom` | Any unlisted GPU/RAM combination | User-set | User-set | User-set |
| `emergency` | Maximum offload on the original setup | 5.5 GiB | 16 GiB | Yes |

These controls make allocation behavior tunable; they cannot make an unsupported Torch/ComfyUI build or an arbitrarily large render fit a particular card. On smaller GPUs, start with `low_8gb` or `safe_12gb`, keep OOM recovery enabled, and reduce the canvas from 1024×1024 if necessary. Smaller-VRAM profiles require more system RAM because ComfyUI offloads more weights to the CPU. The custom policy is also useful for cards between the named sizes.

The right-side **Model and generation settings** pane contains **Model & memory**, **Generation & spatial**, **LoRA library & scope**, **Token emphasis**, and **Projector** tabs, so LoRA setup no longer occupies a separate dock. **Model & memory** can select the ComfyUI checkout and its CUDA- or ROCm-enabled Python interpreter. The **Krea checkpoint** selector lists every `krea*.safetensors` transformer in the configured ComfyUI diffusion-model directory; selecting one pins its exact path in worker payloads and saved projects. The transformer, text encoder, and VAE also retain **Choose…** and **Auto** controls for selecting files outside the list or returning to compatible name-based discovery. The face detector is selectable too; it must retain the compatible NanoDet input/output architecture even though ONNX execution can use CPU or NVIDIA CUDA. Use **Validate tensors** before **Load selected Krea 2 model**. Validation reads only safetensors headers and writes complete manifests under the configured K2 Lab data directory. Changing the runtime or primary model selection stops an idle worker so the next load cannot accidentally retain the previous weights. The Raw and Turbo checkpoints share the architecture targeted by regional attention and routed LoRAs, but the current generation path remains the eight-step, CFG-free Turbo path; selecting Raw alone does not add Raw's recommended full-step CFG sampling. A Raw-to-Turbo distillation LoRA can instead be loaded at strength 1.0 for this Turbo path, but it must remain **Global** because regionalizing the distillation delta would create a hybrid checkpoint rather than Turbo behavior. After loading, **Generate image** displays the saved image behind the editable region boxes. **Sampler** selects the denoising integration algorithm and **Scheduler** selects its noise/sigma schedule; both dropdowns reproduce the ordered `KSampler.SAMPLERS` and `KSampler.SCHEDULERS` registries from current ComfyUI. The worker validates the selection against the configured ComfyUI installation before sampling. Both values are saved in project JSON, embedded PNG project metadata, and separate PNG text fields. Project Open/Save dialogs start in the checkout's `prompts/` folder, and new generations default to the sibling `outputs/` folder. The generation tab provides an output-folder browser and editable filename prefix; both are saved in project JSON. The unified-prompt preview is a resizable, selectable-text dialog. The event viewer follows new messages only while its scrollbar is already at the latest event.

Drag the inner right edge of the left prompt pane or the inner left edge of the right settings pane to resize it. The two side panes are independent: resizing the right pane takes space from or returns space to the center workspace without changing the left pane. Drag the top edge of the bottom Events pane upward to make Events taller and shrink everything above it, or downward to restore height to the upper panes. The bottom pane owns both lower corners and therefore resizes the complete upper row consistently. Each dock can also be floated, closed, and restored from the checkable **View** menu; **View → Restore default pane layout** docks and shows all panes again.

The **Projector** tab controls Krea's 12-column `txtfusion.projector` delta. It provides the `FilterBypass2`, `FilterBypass3`, `skc3vo`, and `z0jglf` reference presets, twelve editable vector fields, and one multiplier that scales the entire vector. The control is off by default and is saved in project JSON and PNG metadata. Each subject region also has a separate **Face identity prompt** for the character trigger and stable face/hair description. **Face identity protection** scales the projector delta only on that field's exact Qwen token span: `0` applies the complete preset, while `1` retains the baseline projector mixture for those identity tokens. Body, pose, action, and scene tokens continue receiving the complete preset, and there is no image-space exclusion mask. K2 Lab installs this token-selective projector delta before regional LoRA hooks, preserving regional LoRA routing unchanged.

The compact monitor beside Events reads AMD VRAM counters from Linux DRM/sysfs, NVIDIA counters from `nvidia-smi`, and system memory from `/proc/meminfo` once per second. It shows current GPU VRAM, system RAM, GPU activity, and a two-minute VRAM/RAM history without embedding another monitor. Seed behavior can be **Fixed**, **Random**, or **Increment** and is stored in the project. Every generation uses a disposable worker: after the final image is saved, that process exits so GPU allocations, model weights, LoRA tensors, and Python heap memory are returned to the OS. The next **Generate image** click automatically starts, probes, validates, and loads a fresh worker before dispatching the request. **Stop generation** terminates the same isolated worker early while leaving the desktop and unsaved configuration open.

The **Use unified spatial prompting** backend compiles the global prompt and every enabled regional prompt into one scene-wide Qwen caption. The region list is a draggable front-to-back stack: the top row's clause comes first, and when two subject boxes overlap the caption explicitly keeps both in the shared area while placing the higher row in front and naturally occluding the lower row behind it. Broad background bands are excluded from this object-occlusion rule. The compiler adds qualitative positions, framing, and relationships without putting numeric coordinates into subject prose, records the text-token span belonging to every regional clause, and maps each half-open pixel box onto Krea's 16-pixel image-token grid. The two Krea text-refiner blocks keep each subject clause private. In the single-stream blocks, image tokens outside a subject box cannot attend that subject's text, and global or other-subject text cannot read image keys owned by the box. Image-to-image attention is deliberately left continuous across the canvas so region ownership cannot draw a rectangular transition into the scene. Overlapping image tokens belong to the first/highest-priority subject for cross-modal routing. Soft attention strength still shapes placement inside the permitted text/image pairs. The override computes the exact routed softmax in bounded query chunks without materializing a complete per-head score matrix.

Every box has an **Auto**, **Subject target**, or **Background band** selector. Auto treats boxes spanning at least 70% of the canvas width as background and narrower boxes as subjects, which migrates existing projects without manual edits. Background bands keep full strength throughout the box and use a softer outside penalty. With **Make subjects fill their boxes** enabled, the box means the desired visible extent rather than only a center location: the unified caption includes qualitative position and image-relative size/framing descriptions plus a minimal-empty-margin instruction, while attention remains strong through the box edges. Numeric coordinates are intentionally omitted from prompt prose so the model does not render them as guides or annotations. Disable subject fill to recover the older position-only center-weighted behavior. Overlapping subjects can still compete for soft ownership of image tokens so two subjects are less likely to collapse into the same location. Placement and size guidance remain strongest through the first half of denoising and can relax during late detail refinement.

**Inside boost**, **Outside penalty**, and **Spatial falloff** separately control the field. The overlap-competition and late-step relaxation controls can be disabled for comparison runs; when relaxation is enabled, **Late-step spatial scale** selects the final multiplier from 0.00 to 1.00. **Adapt spatial guidance from regional LoRA delta** is opt-in: it compares each regional LoRA's routed text/image delta magnitude with that LoRA's first denoising step, then applies a bounded 0.50–1.50 multiplier to the associated region's next-step spatial-attention bias. It never changes the regional LoRA gate, and global LoRAs do not participate. The GUI can preview the compiled prompt, and the prompt, effective roles, token spans, pixel boxes, settings, depth order, main-stream and text-refiner attention-call counts, and final delta-adaptation scales are stored in image metadata. The `krea-unified-spatial-attention-v6` backend uses one global denoising pass. Subject boxes are hard permissions for cross-modal subject-text access; image-to-image attention is not partitioned and there is no image compositing mask. Region-negative controls and serialization were removed because the Turbo CFG-free execution path cannot apply them. Older project files containing `negative_prompt` keys still open, but those unused values are discarded.

The **Token emphasis** tab lets you highlight a complete word or phrase in the global or selected-region editor and add an attention-logit boost. K2 Lab resolves the selection to its Qwen token span after building the unified prompt. Global selections are reinforced across the canvas; regional selections are reinforced through that region's permitted image tokens and soft strength field. Start at `0.30–0.60`; boosts at or above `1.00` can make a concept unnaturally dominant. Emphases round-trip through project JSON and their resolved token spans are stored in output metadata.

Each loaded LoRA has a saved model-strength control and an **Inspect selected LoRA…** action. Inspection runs in the loaded worker, reports exact Krea transformer target coverage, and catches files that only appear to load because their namespaces do not match. K2 Lab aligns the AI Toolkit `diffusion_model.blocks…` and `diffusion_model.txtfusion…` conventions used by Krea LoRAs with the namespace exposed by the active ComfyUI worker. Compatible LoRAs use unfused forward adapters, so base FP8 weights are never rewritten. A Global route enables all text and image lanes. A **Standard regional** route gates text-fusion deltas to the complete assigned regional clause and gates compatible main-stream deltas to the 16×16 image-token cells intersecting the assigned half-open pixel boxes. The regional attention partition prevents that modified clause from becoming shared scene conditioning and prevents image tokens outside the box from consuming it. Main-stream attention key/value targets are still omitted because those projections are consumed by every query before output gating; query, gate, attention-output, and MLP deltas remain available inside the box. A text-fusion-only LoRA is therefore genuinely applied instead of being silently reported as active with zero installed targets. One LoRA may use the union of several named boxes, several LoRAs may share a box, and global and regional adapters can run together through one composite layer hook. Boundary cells are included when they intersect the pixel box. Compatibility coverage, locality-skipped targets, route coverage, text-partition calls, and measured delta RMS are recorded in PNG metadata.

Regional character LoRAs can use **Character identity (face)** routing. Set the exact training trigger (for example `lface`) after assigning the LoRA to one or more regions. K2 Lab adds an explicit person/face identity anchor to each assigned unified-prompt clause while retaining the adapter's normal text-side coverage over the complete regional clause; its image-side delta remains inside the character box. Standard routing remains available per loaded instance, so duplicate instances of the same safetensors file may use different modes, regions, triggers, and strengths. Character identity routing is deliberately unavailable for Global scope because it requires a character box; returning the LoRA to Global restores Standard routing. The detected-crop face-refinement pass reuses the same identity anchor.

The prompt-attention field and LoRA gate intentionally have different semantics. Background placement may feather outside its box. Subject prompt and text-fusion LoRA conditioning is limited to image tokens intersecting the subject box, while the soft field controls relative attention strength among permitted pairs. Standard regional main-stream parameter deltas also use only the strict box mask. **Character identity (face)** adds its dedicated regional trigger anchor on top of the same text partition; the separate face-detail pass remains crop-local.

The separate **Image editing** workspace sits between the generation canvas and face refinement. Load a PNG, JPEG, or WebP image, draw one or more rectangular edit regions, and give each region its own prompt. The edit workspace has independent prompts, ordered boxes, sampling controls, and LoRA scopes; loaded LoRA files and their strengths remain shared with generation. In the LoRA tab, switch to the Image editing workspace to assign the selected LoRA as Disabled, Global, or to one or more edit regions. New and migrated projects leave LoRAs disabled for editing until explicitly assigned.

Image editing VAE-encodes the complete source, pads only the right and bottom edges to Krea's 16-pixel alignment without rescaling, and runs one img2img denoising pass through the same unified regional prompt compiler, attention partition, overlap priority, and routed-LoRA hooks as baseline generation. **Denoise** controls how far the result may depart from the source. When the global edit prompt is non-empty, the complete image may change and regional prompts specialize their boxes. When the global edit prompt is blank, at least one regional prompt is required and the decoded candidate is composited over the normalized source with a single feathered union of all active boxes. Pixels outside that bounded feather support remain unchanged; overlapping regions are not blended twice. A globally scoped edit LoRA does not by itself permit changes outside boxes when the global prompt is blank.

Edit state, including the source path, source dimensions, prompts, boxes, workflow-specific LoRA routes, denoise, and feather settings, is stored in project JSON and embedded into edited PNGs. Source pixels are not embedded in project files. If a saved source path is missing, the boxes and settings remain available but editing stays disabled until the image is reloaded. Loading a same-sized replacement preserves the edit layout; accepting a different-sized replacement clears boxes and edit-region LoRA assignments. Edited outputs are always PNGs, carry `k2lab_mode=krea2_regional_image_edit`, and can be used directly as the face-refinement source.

The separate **Face refinement** workspace never runs as part of baseline generation. It opens the newest non-refined output by default, can load any first-pass PNG, and displays the source and refined result side by side. Face refinement is an explicit two-stage operation: choose **Detect faces** first, review the numbered boxes over the source preview, then check one or more faces and run refinement. Selected boxes are green; detected but excluded boxes are orange. A small CPU NanoDet model detects faces left-to-right, matches each face to a nearby current subject region, crops a padded square around it, VAE-encodes that crop, and runs a low-denoise Krea pass using only the LoRAs assigned to that region. When present, the region's Face identity prompt becomes the refinement prompt instead of extracting body/action text from the ordinary region prompt. Global LoRAs are deliberately omitted, so a scene/style LoRA such as SNOfS cannot compete inside the face pass. There is no diffusion mask: the whole crop is refined, its pixel delta is blended over the source, and a smooth edge feather removes the crop boundary. Each run uses a fresh disposable worker and saves beside the source with `face_refined` in the filename.

**Detector threshold** is the minimum face confidence. Lowering it finds subtler, smaller, or less frontal faces but may introduce false positives; the default is 0.15 because the review-and-selection stage makes those harmless. Changing the threshold or detector device invalidates the displayed candidates, so detect again. **Detector device** selects Auto, portable CPU, or NVIDIA CUDA execution for the same architecture-specific NanoDet model. **Crop working resolution** is the square resolution Krea receives, not the size of the detected face box. Keep it at 512 initially; 256 can discard facial structure, while 768/1024 cost more memory and time. **Crop padding** controls how much source area surrounds the detected face. For deformation or invented features, reduce denoise first, then regional LoRA scale, then refined-pixel blend. The conservative defaults are 8 steps, 0.15 denoise, a 512 px crop, 0.50× of each regional LoRA's saved strength, and a 0.50 pixel-delta blend. All detection boxes, selected indices, provider, crop boxes, effective strengths, prompts, seeds, and applied LoRA reports are stored in the refined PNG metadata. The detector uses a compatible `face_det.onnx` bundled with FantasyPortrait under the configured ComfyUI root.

Every PNG produced by the PySide application contains a compact copy of the complete K2 project JSON under the `k2lab_project` PNG text key, including canvas, prompts, regions, LoRA routes and strengths, generation controls, face-detail controls, upscaler settings, and runtime paths. **File → Import image…** (`Ctrl+Shift+O`) restores that snapshot as a project and uses the selected PNG as the generation canvas and face-refinement source; the imported state has no JSON save target until it is saved as a project. Baseline outputs also retain the existing execution summaries. A refined output preserves the source PNG's text metadata, replaces the embedded project snapshot with the settings used for that refinement, and adds the complete `face_detail` detection/refinement report. This metadata can be inspected with Pillow through `Image.open(path).info` without a sidecar file.

Before VAE decode, the worker explicitly offloads the denoising transformer and releases routed adapter hooks outside PyTorch inference mode. This both frees the additional VRAM held by multiple LoRAs and prevents ComfyUI's quantized FP8 parameter reconstruction from receiving inference tensors during the VAE memory handoff.

**Post-upscale after releasing Krea VRAM** is a scene-wide output stage, not another denoising pass. The worker decodes the original image, copies it to CPU memory, unloads Krea, its routed LoRAs, and the VAE from the accelerator, and empties the GPU cache before upscaling. The built-in **CPU Lanczos** option works without additional weights and produces an exact 2× or 4× output, but it cannot invent detail absent from the 1024×1024 render. For learned detail recovery, choose **Neural model (tiled GPU)** and browse to an ESRGAN/Real-ESRGAN-compatible `.pth`, `.pt`, or `.safetensors` file. Neural inference uses overlapping tiles with automatic OOM tile reduction and keeps its assembled output in system RAM. The selected method, model, base size, and final size are stored in project JSON and PNG metadata.

If the accelerator probe fails, **Diagnose accelerator…** appears below the status. It restarts the worker with a clean environment and reports the interpreter, Torch/CUDA/ROCm versions, backend-specific device-file access, visibility variables, initialization errors, and suggested fixes without closing the application.

Use **Release K2 GPU memory…** if a failed run leaves a K2 worker holding VRAM. The confirmation dialog lists every matching current-user K2 worker PID, then stops only those processes. It deliberately does not terminate ComfyUI or other GPU applications.

Launch with `DEBUG=1 k2lab` to write bounded rotating logs under `~/.local/share/k2-region-lab/logs/` (or the configured `K2LAB_DATA_DIR`). The desktop and GPU worker use separate `desktop-debug.log` and `worker-debug.log` files.

The original local stack uses PyTorch 2.9.1 with ROCm 6.4. Because scaled FP8 execution is native only on ROCm 6.5 or newer, the worker automatically uses ComfyUI's low-VRAM fallback on ROCm 6.4. CUDA workers enable the same native FP8 model option on NVIDIA compute capability 8.9 or 9.x-and-newer devices; older CUDA devices retain ComfyUI's compatible fallback. **Safe 16 GB** remains the default policy: neither its 4 GiB VRAM floor nor its 14 GiB available-system-RAM floor can be reduced by an older saved project, and it reports memory at each generation boundary and denoising step. If free VRAM crosses the critical floor between denoising steps, the worker stops before the next allocation and makes one deterministic retry with CPU VAE decode and a reserve increase proportional to the detected GPU capacity; the original 16 GiB setup still moves from 4 GiB to 5 GiB. Memory controls are locked while a model is loaded so the active worker configuration remains explicit. The worker also enables PyTorch expandable allocator segments and ROCm's experimental AOTriton attention backend when the environment does not explicitly configure them. A 1024×1024 eight-step run does not fit reliably on the tested 16 GB GPU under ROCm 6.4's BF16 dequantization fallback. The same baseline completed on PyTorch 2.10.0 with ROCm 7.1 and native scaled FP8/AOTriton, keeping about 2.8 GiB free during denoising. GPU VAE decode may exhaust its regular allocation and use ComfyUI's tiled fallback; K2 Lab keeps that fallback inside PyTorch inference mode for PyTorch 2.10 compatibility. Auto-discovery still selects that ROCm 7.1 environment on the original installation, while CUDA installations normally resolve `.venv` or `venv`; the GUI and `K2LAB_WORKER_PYTHON` can select any other compatible environment. The application-owned worker choice takes precedence over paths stored by older projects.

