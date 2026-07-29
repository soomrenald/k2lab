# Gate 7 native regional-prompting report

Date: 2026-07-29

Gate status: **PASS WITH APPROVED DIFFERENCE**

The native backend remains developer-only and the product default remains `comfyui`.

## Pinned implementation

- k2core: `1c571a8d5415612cf1a41a7c848bcc6f76e75290`
- K2Lab before this report: `520a6ca`
- ComfyUI reference: `285a98944c397a4a81f15ac63d69fa3dbc0a27b9`

The versioned fixture is
`tests/fixtures/parity/regional_prompting/krea2_two_vases_512.json`. Generated images
and diagnostics remain outside Git and are content-addressed in the fixture.

## Native design

The native backend now uses the same K2Lab-owned regional compiler and exact chunked
softmax controller as the reference backend. It does not import ComfyUI or KJNodes.

- Pixel and explicit normalized boxes resolve to clipped pixel coordinates.
- Boxes rasterize to the Krea image-token grid using fractional area coverage.
- Project order and priority preserve front-to-back overlap semantics.
- Global and regional clauses compile into one conditioning sequence.
- Native Qwen prefix counts bind each clause to exact text-token spans.
- Main-stream and text-refiner attention use job-local regional ownership and soft fields.
- Image-to-image attention remains unmodified.
- Late-step relaxation matches the current backend.
- The controller and device cache are created per generation and cleared in `finally`.
- PNG and result metadata contain boxes, roles, token spans, grid size, settings, and
  attention-call counts.

## Required matrix

| Case | Status | Evidence |
| --- | --- | --- |
| One region | PASS | Compiler/mask matrix |
| Multiple non-overlapping regions | PASS | Unit matrix and two-vase end-to-end parity |
| Overlapping regions | PASS | Competition/order tests and 192-token overlap diagnostic |
| Edge-touching regions | PASS | Half-open adjacent masks have no cross-ownership |
| Full-canvas region | PASS | Full-canvas background matrix |
| Very small region | PASS | One-pixel fractional-coverage matrix |
| Invalid region | PASS | Outside-canvas and invalid-coordinate rejection |
| Differing aspect ratios | PASS | Landscape, portrait, and aligned-grid matrix |
| Regional plus global prompt | PASS | Exact shared compiled prompt in both backends |
| Current K2Lab parity | PASS WITH APPROVED DIFFERENCE | Exact semantic checkpoints and visual review below |

The core suite passed 176 tests, 2 intentional skips, and 12 subtests. Both Torch-only
attention partition tests separately passed in the configured ROCm worker environment.

## End-to-end parity

The synthetic fixture requests a red vase on the left and a blue vase on the right.
Native and Comfy produced the same 173-token compiled prompt, token spans `10:73` and
`73:136`, 32×32 image-token grid, 224 main-stream attention calls, and 16 text-refiner
attention calls.

| Measurement | Observed |
| --- | ---: |
| Size / mode | exact 512×512 RGB |
| Cosine | 0.9988357 |
| Mean absolute error | 0.0109999 |
| RMSE | 0.0363296 |
| Maximum absolute error | 0.6705882 |
| PSNR | 28.7948 dB |
| Exact-pixel fraction | 0.5426623 |

Both images contain two equally scaled glossy vases, red on the left and blue on the
right, inside their assigned regions on the white studio background. Human review found
their structure, color, placement, scale, and containment visually equivalent. The
remaining pixel difference is consistent with the already characterized scaled-FP8 graph
drift amplified by explicit attention bias.

Native completed in 16.92 seconds versus 77.64 seconds for the current Comfy low-VRAM
path on the same local ROCm system.

## Regression and diagnostics

An eight-step no-region native rerun produced the exact Gate 5 decoded-pixel SHA-256
`b04a747c3b92720d30e0ee592465dfb0a8b869d764e9c058ba0baa0f96e45b87`,
proving that an unused regional bridge does not change ordinary generation.

The reusable diagnostic renderer produces:

- full-resolution soft region-field preview;
- gridded latent/image-token mask preview;
- text-token ownership bar and region span summary;
- hard-mask overlap visualization, with overlaps highlighted separately.

The two-vase mask and token diagnostic hashes, plus a separate 192-token overlap
visualization hash, are recorded in the fixture.

## Rollback

Revert this report/fixture/pin commit, then revert k2core commits `1c571a8`, `ad3bcd7`,
and `60045d9`. The normal ComfyUI backend remains available and is still the default.
