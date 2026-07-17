from __future__ import annotations

import glob
import gc
import json
import os
import platform
import sys
import traceback
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable

from k2_region_lab.config import ModelDirectories
from k2_region_lab.model import ArtifactSet, discover_model_artifacts
from k2_region_lab.model.manifests import build_tensor_manifest
from k2_region_lab.memory import (
    GIB,
    effective_minimum_system_ram_gb,
    effective_reserve_vram_gb,
    memory_policy,
)
from k2_region_lab.output import validate_filename_prefix
from k2_region_lab.regional_prompting import (
    RegionalPromptPlan,
    compile_regional_prompt_plan,
)
from k2_region_lab.regions import RegionDefinition


class CriticalGpuMemoryPressure(RuntimeError):
    """Raised only between denoising steps so recovery starts before a hard OOM."""


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

    try:
        accelerator_available = torch.cuda.is_available()
        device_count = torch.cuda.device_count() if accelerator_available else 0
    except Exception as error:
        payload.update(
            {
                "torch_available": True,
                "torch_version": torch.__version__,
                "hip_version": torch.version.hip,
                "cuda_version": torch.version.cuda,
                "accelerator_available": False,
                "device_count": 0,
                "devices": [],
                "error": f"{type(error).__name__}: {error}",
            }
        )
        return payload
    devices = []
    try:
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
    except Exception as error:
        payload["device_query_error"] = f"{type(error).__name__}: {error}"
    hip_parts = ()
    if torch.version.hip:
        try:
            hip_parts = tuple(int(part) for part in torch.version.hip.split(".")[:2])
        except ValueError:
            pass
    payload.update(
        {
            "torch_available": True,
            "torch_version": torch.__version__,
            "hip_version": torch.version.hip,
            "cuda_version": torch.version.cuda,
            "accelerator_available": bool(device_count and devices),
            "device_count": device_count,
            "devices": devices,
            "bf16_supported": bool(device_count and torch.cuda.is_bf16_supported()),
            "float8_e4m3fn": hasattr(torch, "float8_e4m3fn"),
            "native_scaled_fp8": hip_parts >= (6, 5),
        }
    )
    return payload


def diagnose_accelerator(comfyui_root: Path) -> dict[str, Any]:
    """Return copyable host/process evidence and targeted remediation hints."""

    payload = probe_runtime(comfyui_root)
    device_paths = [Path("/dev/kfd")]
    device_paths.extend(Path(path) for path in sorted(glob.glob("/dev/dri/renderD*")))
    payload.update(
        {
            "pid": os.getpid(),
            "uid": os.getuid() if hasattr(os, "getuid") else None,
            "gid": os.getgid() if hasattr(os, "getgid") else None,
            "groups": list(os.getgroups()) if hasattr(os, "getgroups") else [],
            "cwd": str(Path.cwd()),
            "device_paths": [
                {
                    "path": str(path),
                    "exists": path.exists(),
                    "readable": os.access(path, os.R_OK),
                    "writable": os.access(path, os.W_OK),
                }
                for path in device_paths
            ],
            "environment": {
                name: os.environ.get(name)
                for name in (
                    "VIRTUAL_ENV",
                    "PYTHONHOME",
                    "PYTHONPATH",
                    "LD_LIBRARY_PATH",
                    "ROCR_VISIBLE_DEVICES",
                    "HIP_VISIBLE_DEVICES",
                    "CUDA_VISIBLE_DEVICES",
                )
            },
        }
    )
    initialization_error = None
    if payload.get("torch_available") and not payload.get("accelerator_available"):
        try:
            import torch

            torch.cuda.init()
        except Exception as error:
            initialization_error = f"{type(error).__name__}: {error}"
    if initialization_error:
        payload["initialization_error"] = initialization_error

    recommendations: list[str] = []
    kfd = next(item for item in payload["device_paths"] if item["path"] == "/dev/kfd")
    if not payload.get("torch_available"):
        recommendations.append("Select the ComfyUI ROCm Python interpreter containing torch.")
    elif not payload.get("hip_version"):
        recommendations.append("The worker has a non-ROCm PyTorch build; install a ROCm build.")
    if not kfd["exists"]:
        recommendations.append(
            "The worker cannot see /dev/kfd; launch outside a sandbox/container "
            "or expose the AMD devices."
        )
    elif not kfd["readable"] or not kfd["writable"]:
        recommendations.append(
            "The worker lacks /dev/kfd access; verify the user belongs to the "
            "render and video groups."
        )
    if any(
        payload["environment"].get(name)
        for name in ("ROCR_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES")
    ):
        recommendations.append("Check accelerator visibility environment variables shown below.")
    if not recommendations and payload.get("accelerator_available"):
        recommendations.append("ROCm accelerator probe succeeded; model loading can proceed.")
    elif not recommendations:
        recommendations.append(
            "ROCm device files are visible; use the Torch initialization error "
            "below to inspect the runtime."
        )
    payload["recommendations"] = recommendations
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
        self.vae_path: Path | None = None
        self.memory_policy_key = "safe_16gb"
        self.reserve_vram_gb = 4.0
        self.warning_free_gb = 4.0
        self.critical_free_gb = 2.0
        self.minimum_system_ram_gb = 14.0
        self.cpu_vae = False
        self.oom_recovery = True

    def load(
        self,
        artifacts: ArtifactSet,
        *,
        memory_policy_key: str = "safe_16gb",
        reserve_vram_gb: float = 4.0,
        minimum_system_ram_gb: float = 14.0,
        cpu_vae: bool = False,
        oom_recovery: bool = True,
    ) -> dict[str, Any]:
        if not artifacts.complete:
            raise RuntimeError("all three model artifacts are required")
        capabilities = probe_runtime(self.comfyui_root)
        if not capabilities.get("accelerator_available"):
            raise RuntimeError("ROCm accelerator is unavailable to the worker process")

        root_text = str(self.comfyui_root)
        if root_text not in sys.path:
            sys.path.insert(0, root_text)
        from comfy.cli_args import args

        policy = memory_policy(memory_policy_key)
        self.memory_policy_key = policy.key
        self.reserve_vram_gb = effective_reserve_vram_gb(
            policy.key, reserve_vram_gb
        )
        self.warning_free_gb = max(self.reserve_vram_gb, policy.warning_free_gb)
        self.critical_free_gb = min(self.warning_free_gb, policy.critical_free_gb)
        self.minimum_system_ram_gb = effective_minimum_system_ram_gb(
            policy.key, minimum_system_ram_gb
        )
        self.cpu_vae = bool(cpu_vae)
        self.oom_recovery = bool(oom_recovery)
        import psutil

        if psutil.virtual_memory().available < self.minimum_system_ram_gb * GIB:
            raise MemoryError(
                "insufficient available system RAM for the selected offload policy: "
                f"requires at least {self.minimum_system_ram_gb:.1f} GiB"
            )
        args.lowvram = True
        args.reserve_vram = self.reserve_vram_gb
        args.cpu_vae = self.cpu_vae
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
        self.vae_path = artifacts.vae.path
        self.vae = comfy.sd.VAE(sd=vae_state, metadata=metadata)
        self.vae.throw_exception_if_invalid()
        return {
            "transformer": type(self.model).__name__,
            "text_encoder": type(self.clip).__name__,
            "vae": type(self.vae).__name__,
            "memory_policy": self.memory_policy_key,
            "reserve_vram_gb": self.reserve_vram_gb,
            "minimum_system_ram_gb": self.minimum_system_ram_gb,
            "cpu_vae": self.cpu_vae,
            "oom_recovery": self.oom_recovery,
            "native_scaled_fp8": capabilities.get("native_scaled_fp8", False),
            "memory": self.memory_snapshot("model loaded"),
        }

    @property
    def loaded(self) -> bool:
        return all(component is not None for component in (self.model, self.clip, self.vae))

    def memory_snapshot(self, stage: str) -> dict[str, Any]:
        import psutil
        import torch

        free_vram, total_vram = torch.cuda.mem_get_info(torch.cuda.current_device())
        ram = psutil.virtual_memory()
        return {
            "stage": stage,
            "gpu_free_bytes": free_vram,
            "gpu_total_bytes": total_vram,
            "gpu_allocated_bytes": torch.cuda.memory_allocated(),
            "gpu_reserved_bytes": torch.cuda.memory_reserved(),
            "ram_available_bytes": ram.available,
            "ram_total_bytes": ram.total,
            "warning_free_bytes": int(self.warning_free_gb * GIB),
            "critical_free_bytes": int(self.critical_free_gb * GIB),
            "minimum_ram_bytes": int(self.minimum_system_ram_gb * GIB),
            "memory_policy": self.memory_policy_key,
            "cpu_vae": self.cpu_vae,
        }

    def _ensure_memory(
        self,
        stage: str,
        event: Callable[[str, dict[str, Any]], None] | None,
    ) -> dict[str, Any]:
        import comfy.model_management

        snapshot = self.memory_snapshot(stage)
        if snapshot["ram_available_bytes"] < snapshot["minimum_ram_bytes"]:
            raise MemoryError(
                f"available system RAM is below the {self.minimum_system_ram_gb:.1f} GiB guard"
            )
        action = "observed"
        if snapshot["gpu_free_bytes"] < snapshot["warning_free_bytes"]:
            device = comfy.model_management.get_torch_device()
            comfy.model_management.free_memory(snapshot["warning_free_bytes"], device)
            comfy.model_management.soft_empty_cache()
            snapshot = self.memory_snapshot(stage)
            action = "offloaded_to_ram"
        snapshot["action"] = action
        if event is not None:
            event(f"Memory check: {stage}", {"memory": snapshot})
        return snapshot

    @staticmethod
    def _is_oom(error: BaseException) -> bool:
        if isinstance(error, CriticalGpuMemoryPressure):
            return True
        try:
            import torch

            if isinstance(error, torch.OutOfMemoryError):
                return True
        except (ImportError, AttributeError):
            pass
        return "out of memory" in str(error).casefold()

    def _switch_vae_to_cpu(self) -> None:
        if self.cpu_vae:
            return
        if self.vae_path is None:
            raise RuntimeError("VAE path is unavailable for CPU fallback")
        from comfy.cli_args import args

        import comfy.model_management
        import comfy.sd
        import comfy.utils

        old_vae = self.vae
        if old_vae is not None:
            try:
                comfy.model_management.unload_model_and_clones(
                    old_vae.patcher,
                    all_devices=True,
                )
            except (AttributeError, RuntimeError):
                pass
        self.vae = None
        del old_vae
        gc.collect()
        comfy.model_management.soft_empty_cache()
        args.cpu_vae = True
        vae_state, metadata = comfy.utils.load_torch_file(
            str(self.vae_path), return_metadata=True
        )
        self.vae = comfy.sd.VAE(sd=vae_state, metadata=metadata)
        self.vae.throw_exception_if_invalid()
        self.cpu_vae = True

    def _recover_from_oom(
        self,
        event: Callable[[str, dict[str, Any]], None] | None,
    ) -> None:
        from comfy.cli_args import args

        import comfy.model_management

        before = self.memory_snapshot("before OOM cleanup")
        if before["ram_available_bytes"] < before["minimum_ram_bytes"]:
            raise MemoryError(
                "GPU OOM recovery stopped because available system RAM is below "
                f"the {self.minimum_system_ram_gb:.1f} GiB guard"
            )
        device = comfy.model_management.get_torch_device()
        target_free = int(max(self.reserve_vram_gb, 5.0) * GIB)
        comfy.model_management.free_memory(target_free, device)
        gc.collect()
        comfy.model_management.soft_empty_cache(force=True)
        self.reserve_vram_gb = max(self.reserve_vram_gb, 5.0)
        self.warning_free_gb = max(self.warning_free_gb, self.reserve_vram_gb)
        args.reserve_vram = self.reserve_vram_gb
        comfy.model_management.EXTRA_RESERVED_VRAM = int(self.reserve_vram_gb * GIB)
        self._switch_vae_to_cpu()
        if event is not None:
            event(
                "OOM recovery prepared",
                {
                    "memory": self.memory_snapshot("OOM cleanup"),
                    "retry_reserve_vram_gb": self.reserve_vram_gb,
                    "cpu_vae": True,
                },
            )

    def generate(
        self,
        *,
        prompt: str,
        width: int,
        height: int,
        steps: int,
        seed: int,
        output_directory: Path,
        filename_prefix: str = "baseline",
        regions: tuple[RegionDefinition, ...] = (),
        regional_prompting: bool = True,
        regional_prompt_strength: float = 1.0,
        regional_feather_pixels: float = 32.0,
        progress: Callable[[int, int, dict[str, Any]], None] | None = None,
        event: Callable[[str, dict[str, Any]], None] | None = None,
    ) -> dict[str, Any]:
        if not self.loaded:
            raise RuntimeError("baseline components must be loaded before generation")
        if width <= 0 or height <= 0 or width % 16 or height % 16:
            raise ValueError("baseline dimensions must be positive multiples of 16")
        if not 1 <= steps <= 100:
            raise ValueError("steps must be between 1 and 100")
        filename_prefix = validate_filename_prefix(filename_prefix)
        regional_plan = (
            compile_regional_prompt_plan(
                width,
                height,
                prompt,
                regions,
                strength=regional_prompt_strength,
                falloff_pixels=regional_feather_pixels,
            )
            if regional_prompting and regions
            else None
        )

        oom_message: str | None = None
        try:
            return self._generate_once(
                prompt=prompt,
                width=width,
                height=height,
                steps=steps,
                seed=seed,
                output_directory=output_directory,
                filename_prefix=filename_prefix,
                regional_plan=regional_plan,
                progress=progress,
                event=event,
                oom_recovered=False,
            )
        except Exception as error:
            if not self.oom_recovery or not self._is_oom(error):
                raise
            oom_message = str(error)
            if error.__traceback__ is not None:
                traceback.clear_frames(error.__traceback__)
            error.__traceback__ = None
        gc.collect()
        if event is not None:
            event(
                "GPU OOM detected; preparing one safe retry",
                {
                    "error": oom_message,
                    "memory": self.memory_snapshot("OOM detected"),
                },
            )
        self._recover_from_oom(event)
        return self._generate_once(
            prompt=prompt,
            width=width,
            height=height,
            steps=steps,
            seed=seed,
            output_directory=output_directory,
            filename_prefix=filename_prefix,
            regional_plan=regional_plan,
            progress=progress,
            event=event,
            oom_recovered=True,
        )

    def _generate_once(
        self,
        *,
        prompt: str,
        width: int,
        height: int,
        steps: int,
        seed: int,
        output_directory: Path,
        filename_prefix: str,
        regional_plan: RegionalPromptPlan | None,
        progress: Callable[[int, int, dict[str, Any]], None] | None,
        event: Callable[[str, dict[str, Any]], None] | None,
        oom_recovered: bool,
    ) -> dict[str, Any]:
        import numpy as np
        import torch
        from PIL import Image, PngImagePlugin

        import comfy.sample

        self._ensure_memory("before text encoding", event)

        conditioned_prompt = (
            regional_plan.prompt
            if regional_plan is not None and regional_plan.regions
            else prompt
        )
        positive = self.clip.encode_from_tokens_scheduled(
            self.clip.tokenize(conditioned_prompt)
        )
        negative = self.clip.encode_from_tokens_scheduled(self.clip.tokenize(""))
        if regional_plan is not None and regional_plan.regions:
            if event is not None:
                event("Unified regional prompt prepared", regional_plan.summary())
        self._ensure_memory("before denoising", event)
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
            snapshot = self.memory_snapshot(f"denoising step {step + 1}/{total}")
            if progress is not None:
                progress(
                    step + 1,
                    total,
                    snapshot,
                )
            if snapshot["gpu_free_bytes"] < snapshot["critical_free_bytes"]:
                raise CriticalGpuMemoryPressure(
                    "critical GPU memory pressure after denoising step "
                    f"{step + 1}/{total}: "
                    f"{snapshot['gpu_free_bytes'] / GIB:.2f} GiB free"
                )

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
        self._ensure_memory("before VAE decode", event)
        images = self._decode_vae(samples)
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
        output_path = output_directory / f"{filename_prefix}_{stamp}_seed-{seed}.png"
        metadata = PngImagePlugin.PngInfo()
        metadata.add_text("k2lab_mode", "krea2_turbo_baseline")
        metadata.add_text("prompt", conditioned_prompt)
        metadata.add_text("global_prompt", prompt)
        metadata.add_text("seed", str(seed))
        metadata.add_text("steps", str(steps))
        metadata.add_text("size", f"{width}x{height}")
        metadata.add_text("filename_prefix", filename_prefix)
        regional_summary = (
            regional_plan.summary()
            if regional_plan is not None and regional_plan.regions
            else {"backend": "disabled", "region_count": 0}
        )
        metadata.add_text("regional_prompting", json.dumps(regional_summary))
        metadata.add_text("memory_policy", self.memory_policy_key)
        metadata.add_text("oom_recovered", str(oom_recovered).lower())
        metadata.add_text("cpu_vae", str(self.cpu_vae).lower())
        Image.fromarray(array).save(output_path, pnginfo=metadata)
        return {
            "image_path": str(output_path),
            "width": width,
            "height": height,
            "steps": steps,
            "seed": seed,
            "filename_prefix": filename_prefix,
            "regional_prompting": regional_summary,
            "sampler": "euler",
            "scheduler": "simple",
            "cfg": 1.0,
            "memory_policy": self.memory_policy_key,
            "reserve_vram_gb": self.reserve_vram_gb,
            "cpu_vae": self.cpu_vae,
            "oom_recovered": oom_recovered,
            "memory": self.memory_snapshot("generation complete"),
        }

    def _decode_vae(self, samples):
        import torch

        # ComfyUI's tiled fallback normalizes with in-place tensor operations.
        # PyTorch 2.10 requires those operations to remain inside inference mode
        # when the tiled accumulator was created as an inference tensor.
        with torch.inference_mode():
            return self.vae.decode(samples)
