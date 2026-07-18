# K2 Region Lab for ComfyUI

This custom-node pack reproduces K2 Region Lab's regional workflow in ComfyUI while preserving the original bare regional-LoRA nodes. It has two authoring styles:

- composable bare nodes for normal graph construction;
- `K2 Region Lab App`, an in-node editor with the same region, prompt, emphasis, projector, and LoRA-scope controls as the PySide6 application.

The pack uses native ComfyUI socket types whenever one already exists: `MODEL`, `CLIP`, `VAE`, `CONDITIONING`, `LATENT`, `IMAGE`, `MASK`, `UPSCALE_MODEL`, plus detector-produced `SEGS` or `BOUNDING_BOX`. It does not duplicate checkpoint, CLIP, VAE, latent, detector, or upscale-model loaders.

## Installation

From the K2 Region Lab checkout, link the package into ComfyUI and restart ComfyUI:

```bash
  /path/to/ComfyUI/custom_nodes/krea_region_lora
```

If an older standalone `krea_region_lora` directory is already present, move it aside before creating the link. The original six node IDs are retained, so saved workflows continue to resolve.

## Full app workflow

1. Load Krea 2 using the appropriate native ComfyUI model, CLIP, and VAE loaders.
2. Connect `CLIP` to `K2 Region Lab App`.
3. Draw boxes on the Regions canvas. Select each box to edit its unique name, role, priority, face identity prompt, regional prompt, and regional negative prompt. Rows are front-to-back and can be reordered.
4. Configure spatial guidance, phrase emphasis, the 12-value projector, and LoRA slot assignments in the app tabs.
5. Create up to four `K2 LoRA Reference` nodes, connect them to the app, and assign each slot to named regions in the app. Use native `Load LoRA` before the K2 sampler for a global LoRA.
6. Connect the app's layout and conditionings to `K2 Region Lab Sampler`, together with native `MODEL` and `LATENT` inputs.
7. Decode with the native VAE decoder.
8. Optionally send the image through `K2 Face Detailer`, using `SEGS` or `BOUNDING_BOX` from a detector node, and then `K2 Post Upscaler`.

`K2 Region Lab App` stores its complete state in `project_json`. Its JSON importer also accepts desktop `.k2lab.json` documents, so region layouts and generation controls can move between the two UIs.

## Bare nodes

### Region and prompt authoring

- `K2 Region From Mask` turns a native `MASK` plus native positive/negative conditionings into a named region.
- `K2 Region Stack` combines up to eight region specs with native global conditionings.
- `K2 Region Editor` is the compact draw/label editor without the full settings tabs.
- `K2 Compile Regional Prompts` encodes editor text through a connected native `CLIP`.
- `K2 Prompt Emphasis` emits ComfyUI weighted-prompt syntax for a selected phrase and occurrence; feed the result to native `CLIP Text Encode`.
- `K2 Regional Controls` exposes inside strength, outside penalty, feather, subject competition/fill, late relaxation, and LoRA-delta adaptation controls.
- `K2 Projector Controls` and `K2 Apply Projector` configure and patch Krea's `txtfusion.projector.weight`.

### LoRA routing and sampling

- `K2 LoRA Reference` selects a LoRA from ComfyUI's native LoRA folder and stores strength, schedule, standard/character-identity mode, and trigger.
- `K2 Apply LoRA To Regions` receives that LoRA reference and assigns it to one or more named regions. It can be chained to form a stack of any practical length.
- `K2 Region Lab Sampler` compiles the regional conditioning, applies the projector before LoRA routing, and runs the regional sampler.

The original compatibility nodes remain available:

- `K2 BBox To Regional Mask`
- `K2 Regional Character LoRA`
- `K2 Regional LoRA Stack 3`
- `K2 Regional Layer LoRA Apply`
- `K2 Regional Attention LoRA Sampler`
- `K2 Regional Decode Composite`

The strict adapter sampler keeps a base trajectory, computes complete guided LoRA branch deltas, resolves overlaps, and pins everything outside the region union back to the base trajectory after every step. When a model has no `k2_regional_velocity_predictor`, the layer-injection fallback applies LoRA activation deltas only to masked attention-output/MLP token streams and can final-pin the latent outside the union.

### Face detailer

`K2 Face Detailer` is a separate crop-based img2img node. It accepts native `IMAGE`, `MODEL`, `VAE`, and conditionings, plus either detector-produced `SEGS` (xyxy) or `BOUNDING_BOX` (selectable xywh/xyxy). It:

- assigns each detected face to the highest-priority containing subject region;
- uses that region's prompt conditionings;
- applies only that region's LoRAs to the face crop, scaled by Regional LoRA Scale;
- samples at 256/512/768/1024 working resolution;
- composites the result using the same crop padding, feather, and refined-pixel blend controls as the desktop GUI.

The node deliberately does not contain a detector. Connect a detector node already present in the ComfyUI graph.

### Post-upscaler

`K2 Post Upscaler` is separate from sampling and face detailing. CPU Lanczos produces an exact 2x/4x result without model weights. Neural mode accepts `UPSCALE_MODEL` from ComfyUI's native `Load Upscale Model`, performs tiled inference with OOM tile reduction, and resizes to the requested exact final scale when the model's native scale differs.

## Testing

Use the ComfyUI environment so Torch and the Comfy APIs are available:

```bash
/path/to/ComfyUI/venv/bin/python run_tests.py
```

The test suite covers bbox/mask conversion, regional delta isolation, overlap policies, LoRA filtering, desktop JSON import, region compilation, prompt emphasis, regional conditioning masks, character identity validation, face box/crop/blend behavior, and exact Lanczos output size. The repository's integration test also loads the package through ComfyUI 0.28, checks `/object_info`, and verifies the served browser extension byte-for-byte.
