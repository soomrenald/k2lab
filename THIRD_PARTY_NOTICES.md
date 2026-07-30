# Third-party notices

Last reviewed: 2026-07-30

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

The K2Lab source and the separately pinned `k2core` source are licensed under
Apache-2.0. Their `LICENSE` files cover first-party software only and do not grant
rights to separately supplied models or user assets.

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

The Gate 12 release workload uses these exact externally hosted artifacts:

| Artifact | Reviewed source and identity | Recorded upstream terms |
| --- | --- | --- |
| Krea 2 Turbo FP8 transformer | [`Comfy-Org/Krea-2`](https://huggingface.co/Comfy-Org/Krea-2/tree/483928f7dcd0fe4ae7d8d96336540d6ac2a7a8e0), SHA-256 `eb4dd8c612cfd10f64f25b057e6e6bbcb5737c94a7372177e456dbf7579502f1` | [Krea 2 Community License Agreement v1](https://cdn.jsdelivr.net/gh/krea-ai/krea-2%40db3984fbc6e13b34c0064990fc2d95ac64d00058/assets/hf_samples/LICENSE.pdf), plus the incorporated [Acceptable Use Policy](https://www.krea.ai/krea-2-use-policy) |
| Qwen3-VL 4B FP8 text encoder | Same reviewed mirror revision, SHA-256 `54bd5144df0bbc25dd6ccadfcb826b521445a1b06ae5a42570bdd2974ca87094` | The [official Qwen3-VL-4B-Instruct repository](https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct) records Apache-2.0. The mirror does not separately document the FP8 conversion chain, so redistribution of that converted artifact still requires a confirmed derivation/notice record. |
| Qwen-Image VAE | Same reviewed mirror revision, SHA-256 `a70580f0213e67967ee9c95f05bb400e8fb08307e017a924bf3441223e023d1f` | The [official Qwen-Image repository and VAE](https://huggingface.co/Qwen/Qwen-Image/tree/main/vae) record Apache-2.0. |
| FantasyPortrait face detector | [`acvlab/FantasyPortrait/face_det.onnx`](https://huggingface.co/acvlab/FantasyPortrait/blob/main/face_det.onnx), upstream and installed SHA-256 `7ea8de1da304c1459a11f637798bb1140805365aeb3cf6637ca6d61909720aec` | The official FantasyPortrait repository records Apache-2.0 and identifies this exact file at commit `14df15c`. |

The FantasyPortrait detector's source and file identity are therefore resolved for the
currently tested hash. It remains externally supplied; if a future package bundles it,
that package must include the Apache-2.0 license and required notices.

The Krea agreement is a model-use boundary, not a notice-only dependency. The approved
project policy is recorded in `MODEL_USE_POLICY.md`: K2Lab is a noncommercial
open-source project; its releases do not bundle or redistribute weights or ship
prepopulated model caches; explicit RunPod downloads go directly from an authorized
upstream provider into the operator's private workspace; operators accept the
applicable upstream terms; and public/shared deployments require content filtering or
an equivalent review process appropriate to their use case. The tested desktop and
RunPod configurations are private, single-operator workspaces with operator review.

Because the project does not redistribute the reviewed converted Qwen FP8 encoder, its
incomplete conversion/notice chain is not a blocker for source-only distribution. It
remains a blocker for any future release that would bundle, mirror, or download that
artifact.

## Distribution checklist

Before publishing a binary, installer, or container:

1. generate a complete bill of materials from the final lockfile and image;
2. include every dependency's license and required notice files;
3. include the K2Lab and `k2core` Apache-2.0 licenses;
4. enforce `MODEL_USE_POLICY.md`; do not include model weights, and do not expose
   public/shared inference without appropriate filtering or equivalent review;
5. record licenses for every bundled model and detector asset;
6. repeat review whenever a dependency, base image, or model hash changes.
