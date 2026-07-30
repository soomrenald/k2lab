# Third-party notices

Last reviewed: 2026-07-29

This file records the third-party software boundaries used by the K2Lab desktop
application. It is an engineering inventory, not a legal opinion or a replacement for
the license texts shipped by each dependency. `pyproject.toml` and `uv.lock` are the
authoritative dependency and version records.

## Runtime dependencies

| Component | Upstream | License recorded by upstream/package metadata |
| --- | --- | --- |
| NumPy | <https://github.com/numpy/numpy> | BSD-3-Clause plus separately identified bundled components; see the installed distribution |
| Pillow | <https://github.com/python-pillow/Pillow> | MIT-CMU |
| Pydantic | <https://github.com/pydantic/pydantic> | MIT |
| PySide6 / Qt for Python | <https://code.qt.io/cgit/pyside/pyside-setup.git/> | LGPL-3.0-only OR GPL-2.0-only OR GPL-3.0-only |
| safetensors | <https://github.com/huggingface/safetensors> | Apache-2.0 |
| ONNX Runtime | <https://github.com/microsoft/onnxruntime> | MIT |

The optional native-model environment additionally installs:

| Component | Upstream | License recorded by upstream |
| --- | --- | --- |
| Diffusers | <https://github.com/huggingface/diffusers> | Apache-2.0 |
| PyTorch | <https://github.com/pytorch/pytorch> | BSD-3-Clause; its distribution includes additional third-party notices |
| Transformers | <https://github.com/huggingface/transformers> | Apache-2.0 |

The separately pinned `k2core` repository is maintained with K2Lab, but it currently
has no declared project license. That is a release blocker and must be resolved by the
copyright owner before either repository is distributed under a public license.

## ComfyUI compatibility boundary

This repository does not contain copied or vendored ComfyUI source. The production
compatibility backend connects to a separately installed
[ComfyUI](https://github.com/comfyanonymous/ComfyUI) checkout. ComfyUI declares
GPL-3.0; its license and any custom-node notices remain with that external installation.
No ComfyUI license is being applied to K2Lab by this inventory.

## Models and user-supplied assets

K2Lab does not grant rights to Krea, Qwen, LoRA, face-detector, or upscaler weights.
Those assets are configured by path and are not included in this source repository.
A SHA-256 registry entry proves file identity, not permission to use or redistribute
the file. A release that downloads or bundles any model must preserve that model's
license, model-card restrictions, attribution, and redistribution terms separately.

The existing FantasyPortrait face detector has unresolved source and redistribution
provenance and must not be bundled in a native release until that record is completed.

## Distribution checklist

Before publishing a binary, installer, or container:

1. generate a complete bill of materials from the final lockfile and image;
2. include every dependency's license and required notice files;
3. resolve the K2Lab and `k2core` first-party licenses;
4. record licenses for every bundled model and detector asset;
5. repeat review whenever a dependency, base image, or model hash changes.
