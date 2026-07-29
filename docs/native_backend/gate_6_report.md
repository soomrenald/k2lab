# Gate 6 native ordinary-LoRA report

Date: 2026-07-29

Gate status: **PASS**

The product default remains `comfyui`; the native backend remains developer-only while
regional prompting and later phases are implemented.

## Pinned implementation

- K2Lab: `e7368237326e76770385dc573ae19b2a9688d7f2`
- k2core: `5ae1d9b29652ba08213aa469fb76045fdd7a5462`
- ComfyUI reference: `285a98944c397a4a81f15ac63d69fa3dbc0a27b9`

The versioned ordinary-LoRA fixture is
`tests/fixtures/parity/ordinary_lora/krea2_turbo_teapot_rank4_512.json`. Large PNG and
safetensors artifacts remain outside Git and are identified by SHA-256.

## Required behavior matrix

| Requirement | Status | Evidence |
| --- | --- | --- |
| No LoRA | PASS | Clean native golden and two-step base run |
| One LoRA | PASS | Rank-4 standard adapter and 256-target direct LoKr adapter |
| Multiple LoRAs | PASS | Ordered half-strength pair executed without target loss |
| Zero strength | PASS | Pixel-exact match to no-LoRA base; report status `disabled_zero_strength` |
| Negative strength | PASS | Negative delta changed output in the opposite algebraic direction |
| Missing keys | PASS | Incomplete A/B, down/up, and direct-LoKr pairs rejected |
| Incompatible LoRA | PASS | Unknown, bare-parameter, DoRA, mixed-format, decomposed LoKr, and shape-incompatible targets rejected before graph mutation |
| Repeated enable/disable | PASS | Base-after-adapter output was pixel-exact to initial base; per-generation wrappers unload with the transformer |
| ComfyUI parity | PASS | Eight-step standard-LoRA fixture passes all thresholds |

Every adapter is fully parsed and every model target and shape is validated before the
first wrapper is installed. There is no silent partial application or destructive merge.
Result and PNG metadata include adapter hash, tensor and target counts, target names,
format, rank, strength, storage bytes, status, and unmatched keys.

## Standard-LoRA parity

The ordinary-LoRA acceptance fixture uses a deterministic synthetic rank-4 A/B adapter
against `diffusion_model.blocks.0.attn.wq`. Its matrices are deliberately weak to measure
adapter semantics without turning the known scaled-FP8 graph difference into a chaotic
two-step image divergence.

| Measurement | Observed | Acceptance |
| --- | ---: | ---: |
| Size / mode | exact 512×512 RGB | exact |
| Cosine | 0.9998517 | at least 0.999 |
| Mean absolute error | 0.0071454 | at most 0.015 |
| RMSE | 0.0138494 | at most 0.04 |
| Maximum absolute error | 0.3960784 | at most 0.5 |
| PSNR | 37.1714 dB | at least 30 dB |

Human review found the native and Comfy outputs visually equivalent. Native generation
took 18.35 seconds; the current Comfy low-VRAM reference took 77.53 seconds on the same
local ROCm system.

## Existing direct-LoKr compatibility

The tracked implementation also supports the direct factor format used by existing
K2Lab adapters:

- `realism_engine_krea2_v3.1.safetensors` SHA-256
  `a6712629445a2e91a616568e82befa8c8c7518e891a0f7c9918138634b5b54a5`;
- all 768 tensors mapped to 256 executable targets;
- 1,562,320,896 bytes of factor storage reported;
- factorized forward math matched an explicit Kronecker matrix within `9.54e-7`;
- a two-step strength-0.1 native/Comfy comparison measured cosine `0.9991538`, MAE
  `0.0172480`, RMSE `0.0331193`, and PSNR `29.5984 dB`;
- human review found those outputs visually equivalent.

Direct LoKr is supported without materializing merged Kronecker matrices. Decomposed and
Tucker LoKr remain an explicit unsupported-format error rather than partial application.

## Automated tests

- k2core: 169 passed, 2 intentionally skipped, 6 subtests passed.
- K2Lab: 182 passed, 2 intentionally skipped, 6 subtests passed.

## Rollback

Revert this fixture/report commit and K2Lab pin commit `e736823`, then revert k2core LoKr
commit `5ae1d9b` and standard-LoRA commit `39bce9c`. The normal backend remains ComfyUI.
