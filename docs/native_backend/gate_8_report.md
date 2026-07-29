# Gate 8 native regional-LoRA report

Date: 2026-07-29

Gate status: **PASS WITH APPROVED DIFFERENCE**

The native backend remains developer-only and the product default remains `comfyui`.

## Pinned implementation

- k2core: `76dc7119d80218d7874448d0b181cedccf22b0dc`
- K2Lab before this report: `e9f86e4`
- ComfyUI reference: `285a98944c397a4a81f15ac63d69fa3dbc0a27b9`

The versioned fixture is
`tests/fixtures/parity/regional_lora/krea2_two_vases_regional_lora_512.json`.
Generated PNG and synthetic safetensors artifacts remain outside Git and are
content-addressed in the fixture.

## Native design

- Every active adapter is parsed and validated before graph mutation.
- Each regional adapter delta is multiplied by its compiled text/image token mask
  inside the transformer linear forward path.
- Multiple adapters on one target retain declaration order and sum their independently
  routed deltas.
- Global adapters use all text and image lanes and compose with regional adapters.
- Standard regional routing omits unsafe main-stream K/V and bare modulation targets.
- Character-identity routing preserves the current anchored Q/K/V behavior and trigger
  clause semantics.
- Main-stream, text-refiner, layerwise text, and projector stream shapes have explicit
  route-mask implementations.
- LoRA delta statistics can drive the same bounded next-step attention response used by
  the current backend.
- All state, masks, instrumentation records, and adapter wrappers are job-local and are
  released when the transformer unloads.

## Required combination matrix

| Case | Status | Evidence |
| --- | --- | --- |
| One regional Q-only LoRA | PASS WITH APPROVED DIFFERENCE | Eight-step native/Comfy image fixture |
| Multiple regional LoRAs | PASS | Independent left and right adapters executed together |
| Global plus regional LoRAs | PASS | Ordered global, left, right three-adapter fixture |
| One LoRA assigned multiple regions | PASS | Deterministic maximum-union route test |
| Standard explicit Q/K/V/output/MLP targets | PASS | Q/output/MLP applied; unsafe K/V explicitly reported skipped |
| Character-identity Q/K/V/output/MLP targets | PASS | All five targets applied in both backends |
| Q-only targeting | PASS | Synthetic rank-4 Q fixture |
| Token-delta tracking | PASS | Exact modified/total token observations |
| One-way attention restrictions | PASS | Shared Gate 7 partition controller and exact call counts |
| Step state and delta adaptation | PASS | Two-step scale changed from 1.0 to 0.906605 |
| Overlapping routes | PASS | Independent masks add in shared lanes; single-adapter assignments use max union |
| Instrumentation toggles | PASS | All nine toggles exercised; off by default |

k2core passed 182 tests, 2 intentional environment skips, and 12 subtests.

## Cross-backend parity and visual review

The representative eight-step case applies a weak Q-only adapter to the right vase. Both
backends compiled the same 173-token prompt, spans `10:73` and `73:136`, 312 routed image
tokens, and 224 main/16 text-refiner attention calls.

| Measurement | Observed |
| --- | ---: |
| Size / mode | exact 512×512 RGB |
| Cosine | 0.9982741 |
| Mean absolute error | 0.0134105 |
| RMSE | 0.0441076 |
| Maximum absolute error | 0.7176471 |
| PSNR | 27.1097 dB |
| Exact-pixel fraction | 0.5385933 |

Human review found equivalent vase structure, color, placement, scale, containment,
highlights, and clean background, with no unexplained artifact. The remaining numerical
difference is consistent with the approved scaled-FP8 execution-graph drift.

The two-step global-plus-left-plus-right case measured cosine `0.9930216`; its deliberately
short, dark trajectory remained visually equivalent. The character-identity five-target
case measured cosine `0.9972933` and preserved the same identity-routed highlight and
composition. Their broader tolerances are recorded explicitly rather than being presented
as exact pixel parity.

## Leakage and instrumentation review

The five-target character fixture produced records for Q, K, V, hidden-state
contribution, attention output, MLP output, residual input, token flags, and attention
masks. Every native record reported `maximum_outside_route = 0.0`; the current backend
reported `outside_gate_delta_rms = 0.0`.

The standard five-target fixture applied Q, output, and MLP targets while reporting K and
V as locality-skipped. Each applied target modified exactly the routed 63 text plus 312
image lanes per call and no lane outside the route.

Instrumentation-on and instrumentation-off runs decoded to identical pixels. Timings were
10.7388 and 10.7057 seconds respectively, about 0.31% observed overhead in this short run.
When instrumentation and adaptation are both disabled, no collector or tensor-statistics
work is installed.

## Regression, memory, and performance

- The no-LoRA native hash remains the Gate 5 golden `b04a747c…`.
- Explicitly disabling regional prompting with a global LoRA produced the exact Gate 6
  ordinary-LoRA pixel hash `002499bd…`.
- Native completed the representative generation in 16.34 seconds versus 82.66 seconds
  for the current Comfy low-VRAM path.
- Isolated maximum process RSS was 19,901,772 KiB native versus 32,261,024 KiB Comfy.
- The Comfy reference peaked at 11,779,470,336 allocated and 12,840,861,696 reserved GPU
  bytes. Native GPU allocator telemetry is intentionally deferred to the K2Lab-owned
  device manager in Gate 10; Gate 8 records process memory plus exact adapter storage.
- The Q-only adapter installed 98,304 executable bytes. The five-target adapter installed
  249,856 executable bytes.

## Known differences and risks

- Native and Comfy scaled-FP8 graphs remain visually equivalent but not pixel-exact.
- Character-identity routing intentionally retains K/V behavior that standard routing
  omits; this is compatibility behavior, not a newly broadened default.
- Native peak GPU allocator telemetry and wider-device validation belong to Gate 10.
- The deferred paid RunPod soak from Gate 5 remains unresolved and does not become less
  necessary because local Gate 8 passed.

## Rollback

Revert this K2Lab report/fixture/pin commit, then revert k2core commits `76dc711`,
`160a5a9`, and `e456b81`. `K2LAB_BACKEND=comfyui` remains the immediate runtime rollback,
and ComfyUI is still the default.
