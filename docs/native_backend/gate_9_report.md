# Gate 9 native image-editing report

Date: 2026-07-29

Gate status: **PASS**

The native backend remains developer-only and the product default remains `comfyui`.
The original automated geometry, determinism, and parity matrix still passes, but the
owner later rejected its two-step medium example because it did not visibly follow the
emerald-green-glass instruction. Numerical edit magnitude is not semantic prompt
adherence, so that example no longer counts as a representative feature pass.

## Pinned implementation

- k2core: `8f7ec5c53401643000cce1db12fb6be65d61d2ff`
- K2Lab before this report: `40cfb3d9d0ada45cd855a509b02d1016f60a9efc`
- ComfyUI reference: `285a98944c397a4a81f15ac63d69fa3dbc0a27b9`

The versioned fixture is
`tests/fixtures/parity/image_edit/krea2_two_vases_image_edit_512.json`.
Generated PNG artifacts remain outside Git and are content-addressed in the fixture.

## Native design

- The source is validated, EXIF-transposed, converted to RGB, and edge-padded to the
  Krea 16-pixel canvas contract without changing requested output dimensions.
- The upstream Qwen Image VAE posterior mode is normalized with the checkpoint latent
  mean and standard deviation.
- Partial strength uses the same expanded-and-truncated shifted-flow sigma schedule as
  the current Comfy sampler.
- Seeded noise is generated with an explicit CPU `torch.Generator`.
- Every masked Euler step restores the noisy source outside the latent mask, and the
  decoded candidate is composited over the original source with an exact bounded mask.
- Regional reference and edit clauses share the Gate 7 attention controller. Reference
  description retention changes only reference-region scales.
- Global and regional LoRAs use the Gate 8 non-destructive route stack.
- Strength zero bypasses text encoding, VAE, transformer, sampling, and decode, returning
  source pixels exactly in both backends.
- Source textual metadata is retained unless the request explicitly replaces the same
  K2Lab-owned key.

## Required Gate 9 matrix

| Case | Status | Evidence |
| --- | --- | --- |
| Strength 0 | PASS | Zero changed pixels; no model execution |
| Low strength 0.1 | PASS | 89,262 in-support pixels changed; exact exterior |
| Medium strength 0.5 | PASS | 92,995 in-support pixels changed; exact exterior |
| Full denoise 1.0 | PASS WITH APPROVED DIFFERENCE | Full local reconstruction; boundary stress documented |
| Same-size image | PASS | 512×512 remains 512×512 |
| Resized image | PASS | 320×320 whole-image edit and source metadata retention |
| Multiple aspect ratios | PASS | 333×509→336×512 and 509×333→512×336 internal alignment; requested sizes restored |
| Regional editing | PASS | Three semantic layers, exact token spans and 56/4 attention calls |
| Masked editing | PASS | Zero changed pixels outside every composite support |
| Regional LoRA | PASS | Q-only adapter applied regionally to 312 image tokens |
| Determinism | PASS | Repeated medium request is pixel-exact |
| Boundary review | PASS WITH APPROVED DIFFERENCE | Narrow-feather stress and default-feather output reviewed |
| Failure recovery | PASS | Invalid source classified before GPU work; fresh worker retry completed |

k2core passed 188 tests, 2 intentional environment skips, and 12 subtests.

## Cross-backend parity and visual review

The representative request is a two-step, strength-0.5 regional edit with two source
reference regions and one overlapping edit region. Both backends compiled the same
220-token prompt, token spans `0:63`, `63:126`, and `126:179`, a 32×32 image-token grid,
and 56 main/4 text-refiner attention calls.

| Measurement inside composite support | Observed |
| --- | ---: |
| Cosine | 0.9996564 |
| Mean absolute error | 0.0079054 |
| RMSE | 0.0174867 |
| Maximum absolute error | 0.6588235 |
| PSNR | 35.1458 dB |
| Pearson correlation | 0.9987771 |

Across the full image, cosine was `0.9998960` and 75.85% of channel values were exact.
Both outputs preserved every pixel outside the composite support. Visual review found
equivalent source preservation, vase structure, highlights, edit magnitude, and blend
boundary. Subsequent owner review correctly found that neither output visibly changed
the right vase from blue ceramic to emerald-green glass. The remaining numerical
difference is the already-approved scaled-FP8 graph drift, but parity with an
unsuccessful edit does not establish feature success.

## Corrective semantic-edit candidate

The original example combined only two denoising steps at strength 0.5 with a retained
reference clause that still described the target as a blue ceramic vase. A corrective
candidate uses eight steps, denoise 0.75, no competing blue-vase reference clause, a box
that covers the whole vase including its rim and base, regional prompt strength 2.0,
and this explicit instruction:

`replace the entire right blue ceramic vase, including its rim, neck, body, and base,
with vivid emerald-green translucent glass`

Both native and Comfy visibly turn the complete right vase emerald green while
preserving the left red vase and the protected exterior. In the full right-vase crop,
the blue-dominant pixel fraction changes from `0.6266` in the source to `0.0000` in both
outputs. Native-to-Comfy full-image comparison measures cosine `0.9998339`, MAE
`0.0034189`, RMSE `0.0129130`, and PSNR `37.7794 dB`. The owner approved this
corrective representative output on 2026-07-30.

## Strength, determinism, and geometry

Mean absolute change inside support increased from `3.314/255` at strength 0.1 to
`9.080/255` at 0.5 and `70.879/255` at 1.0. Strength zero was exactly unchanged. Two
separate medium-strength worker processes produced the same pixel SHA-256.

The unaligned portrait and landscape sources were edge-padded only for model execution,
then cropped back to exactly 333×509 and 509×333. Their custom source metadata and prior
project payloads survived. Every regional case had zero changed exterior pixels.

## Boundary review

The deliberate full-denoise/16-pixel-feather stress case visibly exposes the rectangular
semantic transition because sigma 1 reconstructs the target area independently from the
source. The normal 48-pixel composite feather softens that transition but cannot make a
fully independent local reconstruction semantically continuous in every prompt.

This is approved as a bounded, explainable control interaction: exterior pixels remain
exact, the current Comfy backend has the same mask/composite semantics, and users can
lower denoise, increase feather, enlarge the edit area, or select whole-image editing.
Low and medium representative edits showed no unexplained boundary artifact.

## Performance

Observed isolated-process completion for the representative two-step local request was
approximately 20.2 seconds native and 31.3 seconds Comfy, including process startup and
component graph construction. This is directional local evidence rather than a controlled
benchmark. Gate 10 owns full memory/device benchmarking.

## Known differences and risks

- Full-denoise local edits can expose the finite rectangular blend support, especially
  with deliberately low feather.
- Native and Comfy scaled-FP8 execution is visually equivalent but not pixel-exact.
- Native projector controls remain explicitly unsupported.
- Face refinement is a separate current workflow and is not claimed as native here.
- Native peak allocator telemetry and wider-device validation belong to Gate 10.
- The deferred paid RunPod soak remains unresolved.

## Rollback

Revert this K2Lab report/fixture/pin commit, then revert k2core commits `8f7ec5c`,
`6309542`, and `b198a20`. `K2LAB_BACKEND=comfyui` remains the immediate runtime rollback,
and ComfyUI is still the default.

## Recommended next step

Proceed to Gate 10 device and memory ownership while keeping native opt-in and preserving
the Comfy fallback.
