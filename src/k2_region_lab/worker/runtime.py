from __future__ import annotations

import platform
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable

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
            "native_scaled_fp8": bool(
                torch.version.hip
                and tuple(int(part) for part in torch.version.hip.split(".")[:2]) >= (6, 5)
            ),
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

    def load(self, artifacts: ArtifactSet, *, reserve_vram_gb: float = 2.0) -> dict[str, Any]:
        if not artifacts.complete:
            raise RuntimeError("all three model artifacts are required")
        capabilities = probe_runtime(self.comfyui_root)
        if not capabilities.get("accelerator_available"):
            raise RuntimeError("ROCm accelerator is unavailable to the worker process")

        root_text = str(self.comfyui_root)
        if root_text not in sys.path:
            sys.path.insert(0, root_text)
        from comfy.cli_args import args

        args.lowvram = True
        args.reserve_vram = max(0.5, reserve_vram_gb)
        import comfy.sd
        import comfy.utils

        self.model = comfy.sd.load_diffusion_model(
            str(artifacts.transformer.path),
            model_options={"fp8_optimizations": bool(capabilities.get("native_scaled_fp8"))},
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
            "reserve_vram_gb": args.reserve_vram,
            "native_scaled_fp8": capabilities.get("native_scaled_fp8", False),
        }

    @property
    def loaded(self) -> bool:
        return all(component is not None for component in (self.model, self.clip, self.vae))

    def generate(
        self,
        *,
        prompt: str,
        width: int,
        height: int,
        steps: int,
        seed: int,
        output_directory: Path,
        progress: Callable[[int, int], None] | None = None,
    ) -> dict[str, Any]:
        if not self.loaded:
            raise RuntimeError("baseline components must be loaded before generation")
        if width <= 0 or height <= 0 or width % 16 or height % 16:
            raise ValueError("baseline dimensions must be positive multiples of 16")
        if not 1 <= steps <= 100:
            raise ValueError("steps must be between 1 and 100")

        import numpy as np
        import torch
        from PIL import Image, PngImagePlugin

        import comfy.model_management
        import comfy.sample

        positive = self.clip.encode_from_tokens_scheduled(self.clip.tokenize(prompt))
        negative = self.clip.encode_from_tokens_scheduled(self.clip.tokenize(""))
        latent = torch.zeros(
            [1, 4, height // 8, width // 8],
            device=comfy.model_management.intermediate_device(),
            dtype=comfy.model_management.intermediate_dtype(),
        )
        latent = comfy.sample.fix_empty_latent_channels(
            self.model, latent, downscale_ratio_spacial=8
        )
        noise = comfy.sample.prepare_noise(latent, seed)

        def callback(step: int, denoised, current, total: int) -> None:
            del denoised, current
            if progress is not None:
                progress(step + 1, total)

        samples = comfy.sample.sample(
            self.model,
            noise,
            steps,
            1.0,
            "euler",
            "simple",
            positive,
            negative,
            latent,
            denoise=1.0,
            callback=callback,
            disable_pbar=True,
            seed=seed,
        )
        images = self.vae.decode(samples)
        image_tensor = images[0]
        while image_tensor.ndim > 3 and image_tensor.shape[0] == 1:
            image_tensor = image_tensor[0]
        if image_tensor.ndim != 3 or image_tensor.shape[-1] != 3:
            raise RuntimeError(f"unexpected decoded image shape: {tuple(images.shape)}")
        array = (
            image_tensor.detach()
            .to(device="cpu", dtype=torch.float32)
            .clamp(0, 1)
            .numpy()
            * 255.0
        ).round().astype(np.uint8)

        output_directory.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        output_path = output_directory / f"baseline_{stamp}_seed-{seed}.png"
        metadata = PngImagePlugin.PngInfo()
        metadata.add_text("k2lab_mode", "krea2_turbo_baseline")
        metadata.add_text("prompt", prompt)
        metadata.add_text("seed", str(seed))
        metadata.add_text("steps", str(steps))
        metadata.add_text("size", f"{width}x{height}")
        Image.fromarray(array).save(output_path, pnginfo=metadata)
        return {
            "image_path": str(output_path),
            "width": width,
            "height": height,
            "steps": steps,
            "seed": seed,
            "sampler": "euler",
            "scheduler": "simple",
            "cfg": 1.0,
        }
