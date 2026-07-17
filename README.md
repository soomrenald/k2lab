# K2 Region Lab

K2 Region Lab is a local PySide6 research application for generic pixel-space control of Krea 2. A region is one shared spatial domain for prompt routing, unfused LoRA delta gating, influence measurement, and attention tuning.

The implementation is at the foundation milestone. It currently provides:

- discovery and safetensors-header validation for local Krea 2 components;
- exact output-pixel to 16×16 Krea image-token geometry;
- area-fraction box rasterization and immutable spatial layouts;
- a PySide6 desktop shell with movable, corner-resizable, and deletable pixel-space boxes;
- editable, unique region names propagated through region and LoRA scope controls;
- a LoRA file browser and loaded-LoRA library with Global or multi-region scope assignment;
- JSON project save/load for prompts, generation settings, boxes, names, LoRAs, and runtime paths;
- a typed worker protocol with isolated baseline GPU execution;
- a configurable external ROCm worker using the existing ComfyUI interpreter;
- full transformer, Qwen, and VAE tensor manifests with Krea-specific shape validation;
- fixed-seed Krea 2 Turbo baseline generation with progress events and PNG metadata;
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

Use **Validate tensors** before **Load Krea 2 baseline**. Validation reads only safetensors headers and writes complete manifests under the configured K2 Lab data directory. After loading, **Generate baseline** runs an eight-step Euler/simple Turbo pass by default and displays the saved image behind the editable region boxes. The model panel provides an output-folder browser and editable filename prefix; both are saved in project JSON. The event viewer follows new messages only while its scrollbar is already at the latest event.

If the accelerator probe fails, **Diagnose accelerator…** appears below the status. It restarts the worker with a clean environment and reports the interpreter, Torch/ROCm versions, device-file access, visibility variables, initialization errors, and suggested fixes without closing the application.

Use **Release K2 GPU memory…** if a failed run leaves a K2 worker holding VRAM. The confirmation dialog lists every matching current-user K2 worker PID, then stops only those processes. It deliberately does not terminate ComfyUI or other ROCm applications.

Launch with `DEBUG=1 k2lab` to write bounded rotating logs under `~/.local/share/k2-region-lab/logs/` (or the configured `K2LAB_DATA_DIR`). The desktop and GPU worker use separate `desktop-debug.log` and `worker-debug.log` files.

The original local stack uses PyTorch 2.9.1 with ROCm 6.4. Because scaled FP8 execution is native only on ROCm 6.5 or newer, the worker automatically uses ComfyUI's low-VRAM fallback on ROCm 6.4. **Safe 16 GB** is the default policy: neither its 4 GiB VRAM floor nor its 14 GiB available-system-RAM floor can be reduced by an older saved project, and it reports memory at each generation boundary and denoising step. If free VRAM crosses the critical floor between denoising steps, the worker stops before the next allocation and makes its single deterministic retry with a 5 GiB reserve and CPU VAE decode. Memory controls are locked while a model is loaded so the active worker configuration remains explicit. The worker also enables PyTorch expandable allocator segments and ROCm's experimental AOTriton attention backend when the environment does not explicitly configure them. A 1024×1024 eight-step run does not fit reliably on the tested 16 GB GPU under ROCm 6.4's BF16 dequantization fallback. The same baseline completed on PyTorch 2.10.0 with ROCm 7.1 and native scaled FP8/AOTriton, keeping about 2.8 GiB free during denoising. GPU VAE decode may exhaust its regular allocation and use ComfyUI's tiled fallback; K2 Lab keeps that fallback inside PyTorch inference mode for PyTorch 2.10 compatibility. Use a smaller canvas on ROCm 6.4 or select the ROCm 7.1 worker with `K2LAB_WORKER_PYTHON=~/ComfyUI/venv_rocm7/bin/python k2lab`. This launch-time selection takes precedence over an older worker path stored in an opened project.

