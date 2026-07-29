# ComfyUI dependency inventory

Status: Phase 0 / Gate 1 candidate
K2Lab baseline: `add2a38`
Pinned shared core: `k2core@a82b0b32a891e19eac5c5f6e35f8a9bfb715f9dc`

## Scan scope and result

The tracked repository, configuration, scripts, tests, documentation, pinned `k2core`
source, worker environment bootstrap, workflow-like JSON, environment variables, model
paths, and the configured local ComfyUI checkout were inspected.

Search terms from the migration specification produced no production imports of
`nodes`, `folder_paths`, `CLIPLoader`, `VAELoader`, or the node-level `KSampler`.
There are no tracked ComfyUI workflow JSON files and no subprocess call to a ComfyUI HTTP
server. The implementation imports lower-level `comfy.*` modules dynamically inside
`k2core.worker.runtime`.

The `src/k2_region_lab/worker/runtime.py` file is only a compatibility re-export. Source
references below use the pinned `k2core` repository path and line at the inventoried
revision, not the ignored `.venv` installation path.

Classifications:

- **A** — replace with an upstream library.
- **B** — reimplement in K2Lab-owned `k2core`.
- **C** — vendor a minimal isolated component after license review.
- **D** — keep temporarily, isolated inside `ComfyUIBackend`.
- **E** — remove because unused.

## Direct runtime dependencies

| ID | Source and line | Dependency / purpose | Inputs → outputs | Ownership / product | Strategy | Phase / risk / test |
| --- | --- | --- | --- | --- | --- | --- |
| C01 | `k2core/worker/runtime.py:492`, `:1239`, `:1270` | `comfy.cli_args.args`; mutates global low-VRAM, reserve-VRAM, and CPU-VAE settings | K2Lab memory policy → ComfyUI global runtime flags | Generic lifecycle; desktop only today | **D**, then **B** device policy with no global state | Phase 8; **high**. Snapshot flags and compare load/offload/OOM behavior on CUDA and ROCm. |
| C02 | `k2core/worker/runtime.py:516-533` | `comfy.sd.load_diffusion_model`, `load_clip(CLIPType.KREA2)`, `VAE`; creates all model components | safetensors paths/options → ModelPatcher, CLIP wrapper, VAE wrapper | Generic Krea loading; desktop only | **D**; replace transformer/VAE with **B**, text encoder with **A** Transformers where verified | Phases 2–3; **critical**. Strict key/shape/count/dtype reports, reference embeddings, load/unload and VRAM tests. |
| C03 | `k2core/worker/runtime.py:528`, `:565`, `:1115-1126`, `:1259` | `comfy.utils.load_torch_file` and `state_dict_prefix_replace`; safe tensor/metadata loading and key cleanup | file path → state dict + metadata | Generic loader; desktop only | **D**, then **A** safetensors/PyTorch plus small **B** key-map utility | Phases 2, 4, 8; **medium**. Tensor hashes, metadata, prefix and malformed-file fixtures. |
| C04 | `k2core/worker/runtime.py:561-571` | `comfy.lora`, `comfy.lora_convert`; maps model keys, converts adapter formats, creates patch objects | LoRA state + active model → adapter patches and coverage | Generic parsing plus K2 key alignment; desktop only | **D**; **B** parser/target resolver using safetensors | Phase 4; **critical**. Complete/missing pairs, rank/alpha, strength/order, unsupported targets, exact delta comparison. |
| C05 | `k2core/worker/runtime.py:792-979` | `comfy.weight_adapter.WeightAdapterBase` and `BypassInjectionManager`; hosts token-selective projector and unfused routed LoRA deltas | adapter objects + K2 token masks → cloned patched model with forward hooks | K2Lab-specific behavior on ComfyUI hook API; desktop only | **D**, then **B** native module adapters/hooks; do not vendor before review | Phases 4 and 6; **critical**. Per-target delta checkpoints, gate leakage, enable/disable, global/regional coexistence. |
| C06 | `k2core/worker/runtime.py:1054-1072`, `:1202-1213`, `:1241-1292` | `comfy.model_management`; unload, cache clearing, free-memory boundary, device choice, VAE handoff and reserved VRAM | model/device/memory targets → residency changes and cleanup | Generic lifecycle with K2 policy; desktop only | **D**, then **B** device manager on PyTorch APIs | Phase 8; **critical**. Load/unload stress, OOM recovery, cancellation, persistent VRAM/RAM measurements. |
| C07 | `k2core/worker/runtime.py:1115-1160` | `comfy.utils.tiled_scale` and ComfyUI OOM classifier for neural post-upscale | image tensor + upscaler → tiled image tensor | Generic upscaling; desktop only | **D**; **A** verified tiling/upscaler utility or small **B** implementation | Phase 8/9; **medium**. Seam, shape, scale, OOM and memory-envelope fixtures. |
| C08 | `k2core/worker/runtime.py:1489-1538`, `:1820-1925`, `:2470-2491` | `comfy.samplers.KSampler` registries and `comfy.sample.fix_empty_latent_channels` / `prepare_noise` | names/model/shape/seed → validated choice, latent shape, seeded noise | Generic sampling; desktop only | **D**, then **B** exact required schedules/RNG behavior | Phase 3; **critical**. Sigma sequence, initial-noise tensor, repeated seeds, dimensions, invalid names. |
| C09 | `k2core/worker/runtime.py:1615`, `:2015`, `:2517` | `comfy.sample.sample`; complete denoising loop, CFG, denoise mask and callbacks | model/noise/steps/CFG/sampler/scheduler/conditioning/latent/mask → sampled latents | Generic loop plus installed K2 hooks; desktop only | **D**, then **B** loop with selected Diffusers utilities only when parity-proven | Phases 3, 6, 7; **critical**. Per-step latent/transformer/scheduler checkpoints, cancellation, masked edit retention. |
| C10 | `k2core/worker/runtime.py:1304-1759`, `:1761-2136`, `:2456-2569` | ComfyUI CLIP and VAE object methods (`tokenize`, scheduled encoding, encode/decode) retained from C02 | prompt/image/latent → conditioning, latent, RGB tensor | Generic component APIs; desktop only | **D**; **A/B** native tokenizer/encoder and VAE wrapper | Phases 2, 3, 7; **critical**. Token IDs/spans, embeddings, VAE latent scaling and decoded tolerance. |
| C11 | `k2core/worker/runtime.py:1578-1640`, `:1977-2039` | `ModelPatcher.model_options["transformer_options"]["optimized_attention_override"]`; injects K2 spatial attention into ComfyUI's Krea forward | bound region/token plan → per-attention-call bias/permissions | K2Lab-specific; desktop only | **D**, then **B** explicit native transformer control point | Phases 5–6; **critical**. Q/K/V, masks, call counts, overlaps, leakage and no-region baseline. |
| C12 | `k2core/worker/runtime.py:677-729`, `:849-1031` | `ModelPatcher` clone, attachments, patches, and injections; composes projector and LoRA without mutating FP8 base weights | base patcher + K2 adapters → disposable generation patcher | K2Lab-specific composition on ComfyUI API; desktop only | **D**, then **B** K2-owned loaded pipeline/adapter registry | Phases 4 and 6; **critical**. Base-weight hashes, deterministic order, stale-delta/reload tests. |

## Indirect filesystem, environment, and process dependencies

| ID | Source and line | Dependency / purpose | Inputs → outputs | Ownership / product | Strategy | Phase / risk / test |
| --- | --- | --- | --- | --- | --- | --- |
| I01 | `src/k2_region_lab/config.py:25-58`, `:166-177` | Discovers a Python interpreter below the configured ComfyUI root | ComfyUI root/config/env → worker executable | Desktop orchestration | **D**; native backend must use its own configured runtime | Gate 2 / Phase 9; **high**. Default still selects ComfyUI; explicit native does not require this path. |
| I02 | `src/k2_region_lab/desktop/worker_client.py:41-95` | Launches the ComfyUI interpreter and injects ComfyUI root into `PYTHONPATH` | settings + environment → isolated GPU subprocess | Desktop-only transport | **D**; backend-aware launcher or native worker entry point | Gate 2 / Phase 9; **high**. Environment-origin tests and real worker smoke tests. |
| I03 | `src/k2_region_lab/worker/bootstrap.py:14-55` | Borrows pinned `k2core` into a different environment while keeping that environment's numerical/ComfyUI packages | exact package directory → imported core in GPU process | Desktop bootstrap | **D**; retain for legacy, simplify native packaging later | Gate 2 / Phase 9; **medium**. Import-origin and version mismatch tests. |
| I04 | `src/k2_region_lab/desktop/main_window.py:5141-5174`; `src/k2_region_lab/worker/entrypoint.py:49-68` | Transports ComfyUI root/model directories and memory knobs in loose dictionaries | Qt settings → worker filesystem payload | Product configuration leaking into transport | **B** shared `PipelineConfig` and registry; preserve legacy fields in adapter | Gates 2–3; **high**. Schema round-trip and old-config compatibility tests. |
| I05 | `src/k2_region_lab/config.py:93-155`; `k2_region_lab.toml:4-18` | Defaults transformer, encoder, VAE, LoRA and upscaler directories under `~/ComfyUI/models` | config/env → absolute discovery paths | Generic registry with legacy defaults | **D**, then **B** arbitrary-path registry plus legacy scanner | Phase 1; **medium**. Absolute path, symlink, duplicate, missing, hash and legacy import tests. |
| I06 | `k2core/worker/runtime.py:240-251` | Checks Krea support through files inside the ComfyUI checkout | ComfyUI root → capability boolean | Compatibility probe | **D**; native capability derives from native component/version checks | Gates 2–4; **medium**. Old/new/missing Krea support probes. |
| I07 | `k2core/face_detail.py:16,76-79`; `README.md:62-70` | Defaults face detector to a model inside the FantasyPortrait custom node | ComfyUI root → ONNX model path | Generic detector wrapper with legacy discovery | **D** legacy import; **B** explicit model registry entry | Phase 1/7; **high** licensing/provenance. Explicit path and legacy discovery tests. |
| I08 | `src/k2_region_lab/desktop/main_window.py:1421,1430`; `k2core/sampling.py:4-65` | UI options copy the ordered ComfyUI sampler/scheduler registries | static names → UI choice then runtime validation | Product surface coupled to ComfyUI breadth | **D** until required combinations are approved; **B** capability-driven list | Gates 2/5; **high** semantic breadth. Characterize every advertised pair or narrow only with product approval. |
| I09 | `src/k2_region_lab/desktop/worker_client.py:64-76` | Sets allocator and ROCm attention environment defaults for the ComfyUI/Torch worker | inherited environment → allocator/attention backend | Runtime policy | **D**, then explicit **B** `DevicePlan`/attention setting | Phase 8; **high** performance/parity. CUDA and ROCm matrix with setting recorded in metadata. |
| I10 | `pyproject.toml:12`; `uv.lock:370-372`; compatibility re-exports throughout `src/k2_region_lab` | Runtime behavior is supplied by a separately pinned Git dependency, not tracked source in this repository | package pin → imported business logic/runtime | Shared desktop core; intended future RunPod core | Keep boundary, but implement in reviewed `k2core` branches and update pin deliberately | Every phase; **high** coordination. Test both upstream package and consuming repo on each pin update. |

## Requested symbol findings with no production dependency

| Search term | Finding | Classification |
| --- | --- | --- |
| `import nodes`, `from nodes` | No production hit. Test and third-party environment matches are unrelated. | **E** / no action |
| `folder_paths` | No tracked or pinned-core hit. | **E** / no action |
| `custom_nodes` | Only the default face-detector asset path and user documentation. No custom-node Python import. | Covered by I07 |
| `CLIPLoader`, `VAELoader`, node `KSampler` | No production node-class usage. Lower-level equivalents are C02/C08/C09. | **E** for node wrappers |
| ComfyUI workflow JSON | None tracked. User prompt JSON files are not ComfyUI workflows and were not modified. | **E** / no action |
| ComfyUI server/API subprocess | None. The worker imports ComfyUI in-process. | **E** / no action |
| RunPod container/setup | None in the repository. | Blocked pending scope; see architecture |

## Dependency and license observations

- The inspected ComfyUI checkout declares GPL-3.0. Direct vendoring would require a formal
  compatibility/distribution review and potentially reciprocal obligations. No ComfyUI
  code should be copied during implementation based solely on this inventory.
- K2Lab and the pinned `k2core` checkout contain no license file or package
  `License-Expression`. This must be resolved before evaluating compatibility or shipping
  vendored code.
- `ComfyUI-WanVideoWrapper` and its `fantasyportrait` directory contain Apache-2.0 license
  text. That does not by itself establish the provenance or redistribution terms of the
  `face_det.onnx` artifact.
- PyTorch, safetensors, Transformers, Diffusers, ONNX Runtime, and model licenses must be
  recorded at the exact versions/artifacts chosen in later phases.
- Phase 0 copied no third-party source and changed no dependencies, lockfiles, or notices.

## Inventory conclusion

All ComfyUI runtime dependencies can remain contained by a first
`ComfyUIBackend`. The highest-risk replacements are not filesystem discovery; they are
Krea component loading, prompt/token/embedding parity, scheduler/noise semantics,
ModelPatcher-based unfused LoRA composition, optimized-attention override behavior, and
memory/offload lifecycle. Those require golden intermediate checkpoints before native
implementation.
