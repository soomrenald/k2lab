# K2 Region Lab

K2 Region Lab is a local PySide6 research application for generic pixel-space control of Krea 2. A region is one shared spatial domain for prompt routing, unfused LoRA delta gating, influence measurement, and attention tuning.

The implementation is at the foundation milestone. It currently provides:

- discovery and safetensors-header validation for local Krea 2 components;
- exact output-pixel to 16×16 Krea image-token geometry;
- area-fraction box rasterization and immutable spatial layouts;
- a PySide6 desktop shell with movable, corner-resizable, and deletable pixel-space boxes;
- a LoRA file browser and loaded-LoRA library with Global or multi-region scope assignment;
- a typed worker protocol with isolated baseline GPU execution;
- a configurable external ROCm worker using the existing ComfyUI interpreter;
- full transformer, Qwen, and VAE tensor manifests with Krea-specific shape validation;
- fixed-seed Krea 2 Turbo baseline generation with progress events and PNG metadata;
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

The desktop starts its GPU worker with `~/ComfyUI/venv_rocm/bin/python` by default. Override the runtime without changing GUI dependencies using:

```text
K2LAB_WORKER_PYTHON
K2LAB_COMFYUI_ROOT
K2LAB_AUTO_START_WORKER
K2LAB_RESERVE_VRAM_GB
```

Use **Validate tensors** before **Load Krea 2 baseline**. Validation reads only safetensors headers and writes complete manifests under the configured K2 Lab data directory. After loading, **Generate baseline** runs an eight-step Euler/simple Turbo pass by default and displays the saved image behind the editable region boxes.

The tested local stack uses PyTorch 2.9.1 with ROCm 6.4. Because scaled FP8 execution is native only on ROCm 6.5 or newer, the worker automatically uses ComfyUI's low-VRAM fallback on ROCm 6.4 and reserves 2 GiB of VRAM by default. Change the reserve with `K2LAB_RESERVE_VRAM_GB` if another desktop workload needs a different margin.

