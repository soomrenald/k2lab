from __future__ import annotations

import platform
import sys
from pathlib import Path
from typing import Any

from k2_region_lab.config import ModelDirectories
from k2_region_lab.model import ArtifactSet, discover_model_artifacts
from k2_region_lab.model.manifests import build_tensor_manifest


def probe_runtime(comfyui_root: Path) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "python": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "comfyui_root": str(comfyui_root),
        "comfyui_krea_support": all(
            path.is_file()
            for path in (
                comfyui_root / "comfy/ldm/krea2/model.py",
                comfyui_root / "comfy/text_encoders/krea2.py",
            )
        ),
    }
    try:
        import torch
    except ImportError as error:
        payload.update(
            {
                "torch_available": False,
                "accelerator_available": False,
                "error": str(error),
            }
        )
        return payload

    device_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
    devices = []
    for index in range(device_count):
        properties = torch.cuda.get_device_properties(index)
        devices.append(
            {
                "index": index,
                "name": torch.cuda.get_device_name(index),
                "total_memory": properties.total_memory,
                "major": properties.major,
                "minor": properties.minor,
            }
        )
    payload.update(
        {
            "torch_available": True,
            "torch_version": torch.__version__,
            "hip_version": torch.version.hip,
            "cuda_version": torch.version.cuda,
            "accelerator_available": bool(device_count),
            "device_count": device_count,
            "devices": devices,
            "bf16_supported": bool(device_count and torch.cuda.is_bf16_supported()),
            "float8_e4m3fn": hasattr(torch, "float8_e4m3fn"),
        }
    )
    return payload


def validate_model_artifacts(
    directories: ModelDirectories, manifest_directory: Path
) -> tuple[ArtifactSet, list[dict[str, Any]]]:
    artifacts = discover_model_artifacts(directories)
    results = [
        build_tensor_manifest(artifact, manifest_directory).to_payload()
        for artifact in artifacts.present()
    ]
    return artifacts, results


class ComfyBaselineRuntime:
    """Owns baseline Comfy model objects exclusively inside the GPU worker."""

    def __init__(self, comfyui_root: Path) -> None:
        self.comfyui_root = comfyui_root
        self.model = None
        self.clip = None
        self.vae = None

    def load(self, artifacts: ArtifactSet) -> dict[str, Any]:
        if not artifacts.complete:
            raise RuntimeError("all three model artifacts are required")
        capabilities = probe_runtime(self.comfyui_root)
        if not capabilities.get("accelerator_available"):
            raise RuntimeError("ROCm accelerator is unavailable to the worker process")

        root_text = str(self.comfyui_root)
        if root_text not in sys.path:
            sys.path.insert(0, root_text)
        import comfy.sd
        import comfy.utils

        self.model = comfy.sd.load_diffusion_model(
            str(artifacts.transformer.path),
            model_options={"fp8_optimizations": True},
        )
        self.clip = comfy.sd.load_clip(
            [str(artifacts.text_encoder.path)],
            embedding_directory=[],
            clip_type=comfy.sd.CLIPType.KREA2,
        )
        vae_state, metadata = comfy.utils.load_torch_file(
            str(artifacts.vae.path), return_metadata=True
        )
        self.vae = comfy.sd.VAE(sd=vae_state, metadata=metadata)
        self.vae.throw_exception_if_invalid()
        return {
            "transformer": type(self.model).__name__,
            "text_encoder": type(self.clip).__name__,
            "vae": type(self.vae).__name__,
        }
