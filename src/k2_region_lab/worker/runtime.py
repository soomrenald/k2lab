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
from k2_region_lab.image_edit import (
    composite_regional_edit,
    edge_pad_to_krea,
    edit_global_conditioning_prompt,
    load_source_image,
    regional_composite_mask,
    regional_edit_conditioning,
    regional_reference_emphases,
)
from k2_region_lab.face_detail import (
    BACKEND as FACE_DETAIL_BACKEND,
    FaceDetailSettings,
    OnnxNanoFaceDetector,
    assign_faces_to_regional_loras,
    composite_face_crop,
    discover_face_detector,
    expanded_square_crop,
)
from k2_region_lab.lora import (
    adapter_prefixes,
    align_krea_lora_state_dict,
    inspect_lora_header,
)
from k2_region_lab.model import ArtifactSet, discover_model_artifacts
from k2_region_lab.model.manifests import build_tensor_manifest
from k2_region_lab.memory import (
    GIB,
    effective_minimum_system_ram_gb,
    effective_reserve_vram_gb,
    memory_policy,
    oom_recovery_reserve_vram_gb,
)
from k2_region_lab.output import validate_filename_prefix
from k2_region_lab.projector import (
    DEFAULT_PROJECTOR_PRESET,
    PROJECTOR_VECTOR_COUNT,
    effective_projector_values,
    projector_token_delta_mask,
    projector_preset_values,
    validate_projector_values,
)
from k2_region_lab.regional_lora import (
    LoraDeltaRoute,
    character_identity_triggers,
    compile_lora_delta_routes,
    route_allows_adapter_target,
)
from k2_region_lab.regional_prompting import (
    BoundRegionalPromptPlan,
    PromptEmphasis,
    RegionalPromptPlan,
    character_identity_prompt,
    compile_regional_prompt_plan,
    krea_prompt_token_count,
)
from k2_region_lab.regions import RegionDefinition
from k2_region_lab.sampling import (
    DEFAULT_SAMPLER,
    DEFAULT_SCHEDULER,
    validate_sampler,
    validate_scheduler,
)
from k2_region_lab.spatial_attention import KreaSpatialAttentionOverride


class CriticalGpuMemoryPressure(RuntimeError):
    """Raised only between denoising steps so recovery starts before a hard OOM."""


class LoraDeltaStatistics:
    """Accumulate routed per-token delta magnitudes without synchronizing each layer."""

    def __init__(self, routes: tuple[LoraDeltaRoute, ...]) -> None:
        self.routes = {route.lora_id: route for route in routes}
        self.values: dict[str, dict[str, Any]] = {
            route.lora_id: {
                "text_energy": None,
                "text_count": 0,
                "image_energy": None,
                "image_count": 0,
                "step_text_energy": None,
                "step_text_count": 0,
                "step_image_energy": None,
                "step_image_count": 0,
                "delta_reference": None,
                "calls": 0,
            }
            for route in routes
        }

    @staticmethod
    def _add(previous, value):
        return value if previous is None else previous + value

    def observe(self, route: LoraDeltaRoute, token_norms, *, route_kind: str) -> None:
        state = self.values[route.lora_id]
        state["calls"] += 1
        batch = int(token_norms.shape[0])
        text_count = len(route.text_token_mask)
        enabled_text = sum(value > 0.0 for value in route.text_token_mask)
        if route_kind == "text_layerwise":
            text_norms = token_norms
            image_norms = None
            folded_batches = batch // text_count
            text_observations = folded_batches * enabled_text * int(token_norms.shape[1])
        elif route_kind == "text_projector":
            text_norms = token_norms
            image_norms = None
            text_observations = batch * enabled_text * int(token_norms.shape[2])
        elif route_kind == "text_refiner":
            text_norms = token_norms
            image_norms = None
            text_observations = batch * enabled_text
        else:
            text_norms = token_norms[:, :text_count]
            image_norms = token_norms[:, text_count:]
            text_observations = batch * enabled_text
        if enabled_text:
            text_energy = text_norms.square().sum()
            state["text_energy"] = self._add(state["text_energy"], text_energy)
            state["text_count"] += text_observations
            state["step_text_energy"] = self._add(
                state["step_text_energy"], text_energy
            )
            state["step_text_count"] += text_observations
        enabled_image = sum(value > 0.0 for value in route.image_token_mask)
        if image_norms is not None and enabled_image:
            image_energy = image_norms.square().sum()
            state["image_energy"] = self._add(state["image_energy"], image_energy)
            state["image_count"] += batch * enabled_image
            state["step_image_energy"] = self._add(
                state["step_image_energy"], image_energy
            )
            state["step_image_count"] += batch * enabled_image

    @staticmethod
    def _rms(energy, count: int) -> float:
        if energy is None or count == 0:
            return 0.0
        return float((energy / count).sqrt().item())

    def summary(self, lora_id: str) -> dict[str, Any]:
        state = self.values[lora_id]
        return {
            "observed_forward_calls": state["calls"],
            "text_delta_rms": self._rms(state["text_energy"], state["text_count"]),
            "image_delta_rms": self._rms(state["image_energy"], state["image_count"]),
            "outside_gate_delta_rms": 0.0,
        }

    def regional_attention_scales(self, gain: float) -> dict[str, float]:
        """Return a bounded next-step attention response for each regional route.

        A route's first measured step defines its own reference magnitude. Later
        steps compare their routed text/image delta RMS to that reference, so
        heterogeneous LoRA ranks and strengths do not compete on raw scale.
        """
        if not 0.0 <= gain <= 1.0:
            raise ValueError("LoRA delta adaptation gain must be between zero and one")
        region_values: dict[str, list[float]] = {}
        for lora_id, route in self.routes.items():
            if route.global_scope or not route.region_ids:
                continue
            state = self.values[lora_id]
            components = [
                self._rms(state["step_text_energy"], state["step_text_count"]),
                self._rms(state["step_image_energy"], state["step_image_count"]),
            ]
            components = [value for value in components if value > 0.0]
            if not components:
                continue
            observed = sum(components) / len(components)
            reference = state["delta_reference"]
            if reference is None:
                reference = observed
            ratio = observed / max(float(reference), 1e-12)
            scale = min(1.5, max(0.5, 1.0 + gain * (ratio - 1.0)))
            state["delta_reference"] = 0.85 * float(reference) + 0.15 * observed
            for region_id in route.region_ids:
                region_values.setdefault(region_id, []).append(scale)
        return {
            region_id: sum(scales) / len(scales)
            for region_id, scales in region_values.items()
        }

    def reset_step_measurements(self) -> None:
        for state in self.values.values():
            for prefix in ("text", "image"):
                state[f"step_{prefix}_energy"] = None
                state[f"step_{prefix}_count"] = 0


def accelerator_backend(hip_version: str | None, cuda_version: str | None) -> str:
    if hip_version:
        return "rocm"
    if cuda_version:
        return "cuda"
    return "unknown"


def native_scaled_fp8_supported(
    backend: str,
    runtime_version: str | None,
    devices: list[dict[str, Any]],
) -> bool:
    """Match the native FP8 paths exposed by current ComfyUI releases."""
    if backend == "rocm":
        try:
            version = tuple(int(part) for part in str(runtime_version).split(".")[:2])
        except ValueError:
            version = ()
        return version >= (6, 5)
    if backend == "cuda":
        return any(
            int(device.get("major", 0)) >= 9
            or (
                int(device.get("major", 0)) == 8
                and int(device.get("minor", 0)) >= 9
            )
            for device in devices
        )
    return False


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
        selected_device_index = torch.cuda.current_device() if device_count else None
    except Exception as error:
        backend = accelerator_backend(torch.version.hip, torch.version.cuda)
        payload.update(
            {
                "torch_available": True,
                "torch_version": torch.__version__,
                "hip_version": torch.version.hip,
                "cuda_version": torch.version.cuda,
                "accelerator_backend": backend,
                "accelerator_available": False,
                "device_count": 0,
                "selected_device_index": None,
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
    backend = accelerator_backend(torch.version.hip, torch.version.cuda)
    selected_devices = [
        device for device in devices if device["index"] == selected_device_index
    ]
    payload.update(
        {
            "torch_available": True,
            "torch_version": torch.__version__,
            "hip_version": torch.version.hip,
            "cuda_version": torch.version.cuda,
            "accelerator_backend": backend,
            "accelerator_available": bool(device_count and devices),
            "device_count": device_count,
            "selected_device_index": selected_device_index,
            "devices": devices,
            "bf16_supported": bool(device_count and torch.cuda.is_bf16_supported()),
            "float8_e4m3fn": hasattr(torch, "float8_e4m3fn"),
            "native_scaled_fp8": native_scaled_fp8_supported(
                backend,
                torch.version.hip if backend == "rocm" else torch.version.cuda,
                selected_devices,
            ),
        }
    )
    return payload


def diagnose_accelerator(comfyui_root: Path) -> dict[str, Any]:
    """Return copyable host/process evidence and targeted remediation hints."""

    payload = probe_runtime(comfyui_root)
    backend = str(payload.get("accelerator_backend", "unknown"))
    if backend == "cuda":
        device_paths = [Path("/dev/nvidiactl"), Path("/dev/dxg")]
        device_paths.extend(
            Path(path) for path in sorted(glob.glob("/dev/nvidia[0-9]*"))
        )
    else:
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
    if not payload.get("torch_available"):
        recommendations.append(
            "Select a ComfyUI CUDA or ROCm Python interpreter containing PyTorch."
        )
    elif backend == "unknown":
        recommendations.append(
            "The worker has a CPU-only PyTorch build; install a CUDA or ROCm build."
        )
    elif backend == "rocm" and not payload.get("accelerator_available"):
        kfd = next(item for item in payload["device_paths"] if item["path"] == "/dev/kfd")
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
    elif backend == "cuda" and not payload.get("accelerator_available"):
        control_paths = [
            item
            for item in payload["device_paths"]
            if item["path"] in {"/dev/nvidiactl", "/dev/dxg"}
        ]
        visible_control = next(
            (item for item in control_paths if item["exists"]), None
        )
        if visible_control is None:
            recommendations.append(
                "The worker cannot see NVIDIA or WSL GPU device files; install the "
                "NVIDIA driver or expose the NVIDIA devices to the container."
            )
        elif not visible_control["readable"] or not visible_control["writable"]:
            recommendations.append(
                f"The worker lacks access to {visible_control['path']}; verify device "
                "permissions."
            )
    if any(
        payload["environment"].get(name)
        for name in ("ROCR_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES")
    ):
        recommendations.append("Check accelerator visibility environment variables shown below.")
    if not recommendations and payload.get("accelerator_available"):
        recommendations.append(
            f"{backend.upper()} accelerator probe succeeded; model loading can proceed."
        )
    elif not recommendations:
        recommendations.append(
            "GPU device files are visible; use the Torch initialization error "
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

    def __init__(
        self, comfyui_root: Path, *, face_detector_path: Path | None = None
    ) -> None:
        self.comfyui_root = comfyui_root
        self.face_detector_path = face_detector_path
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
            raise RuntimeError("GPU accelerator is unavailable to the worker process")

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
            "accelerator_backend": capabilities.get("accelerator_backend", "unknown"),
            "memory": self.memory_snapshot("model loaded"),
        }

    @property
    def loaded(self) -> bool:
        return all(component is not None for component in (self.model, self.clip, self.vae))

    def _load_lora_patches(
        self, specification: dict[str, Any]
    ) -> tuple[dict, dict[str, str] | None, dict[str, Any]]:
        if not self.loaded:
            raise RuntimeError("load the Krea 2 baseline before validating LoRAs")
        path = Path(str(specification["path"])).expanduser().resolve()
        if path.suffix.casefold() != ".safetensors" or not path.is_file():
            raise ValueError(f"LoRA path is not a readable safetensors file: {path}")

        import comfy.lora
        import comfy.lora_convert
        import comfy.utils

        state, metadata = comfy.utils.load_torch_file(
            str(path), safe_load=True, return_metadata=True
        )
        key_map = comfy.lora.model_lora_keys_unet(self.model.model, {})
        aligned = align_krea_lora_state_dict(state, key_map)
        converted = comfy.lora_convert.convert_lora(aligned)
        patches = comfy.lora.load_lora(converted, key_map, log_missing=False)
        normalized_prefixes = adapter_prefixes(converted)
        unmatched_prefixes = [
            prefix for prefix in normalized_prefixes if prefix not in key_map
        ]
        header = inspect_lora_header(path)
        adapter_count = int(header["adapter_count"])
        report = {
            **header,
            "id": str(specification.get("id", path.stem)),
            "display_name": str(specification.get("name", path.stem)),
            "strength": float(specification.get("strength", 1.0)),
            "global": bool(specification.get("global", True)),
            "region_ids": list(specification.get("region_ids", [])),
            "routing_mode": str(specification.get("routing_mode", "standard")),
            "trigger_phrase": str(specification.get("trigger_phrase", "")),
            "matched_model_targets": len(patches),
            "unmatched_adapter_targets": max(0, adapter_count - len(patches)),
            "compatible": (
                bool(patches)
                and len(patches) == adapter_count
                and int(header["complete_adapter_pairs"]) == adapter_count
            ),
            "model_only": True,
            "unmatched_prefix_examples": unmatched_prefixes[:8],
            "model_mapping_examples": [
                key
                for key in key_map
                if "blocks.0" in key or "transformer_blocks.0" in key
            ][:12],
        }
        return patches, metadata, report

    def diagnose_loras(self, specifications: list[dict[str, Any]]) -> list[dict[str, Any]]:
        reports = []
        for specification in specifications:
            patches, _metadata, report = self._load_lora_patches(specification)
            report["status"] = "compatible" if report["compatible"] else "incompatible"
            reports.append(report)
            del patches
        gc.collect()
        return reports

    def _apply_routed_loras(
        self,
        specifications: list[dict[str, Any]],
        *,
        base_model,
        width: int,
        height: int,
        text_token_count: int,
        regional_plan: RegionalPromptPlan | None,
        bound_plan: BoundRegionalPromptPlan | None,
        event: Callable[[str, dict[str, Any]], None] | None,
    ):
        routes = compile_lora_delta_routes(
            specifications,
            width=width,
            height=height,
            text_token_count=text_token_count,
            regional_plan=regional_plan,
            bound_plan=bound_plan,
        )
        routes_by_id = {route.lora_id: route for route in routes}
        reports: list[dict[str, Any]] = []
        target_entries: dict[str, list[tuple[Any, LoraDeltaRoute | None]]] = {}
        skipped_targets: dict[str, list[str]] = {}
        metadata_items = []
        for specification in specifications:
            path = Path(str(specification["path"])).expanduser().resolve()
            lora_id = str(specification.get("id", path.stem))
            strength = float(specification.get("strength", 1.0))
            if strength == 0.0:
                reports.append(
                    {
                        **inspect_lora_header(path),
                        "id": lora_id,
                        "display_name": str(specification.get("name", path.stem)),
                        "strength": strength,
                        "global": bool(specification.get("global", True)),
                        "region_ids": list(specification.get("region_ids", [])),
                        "routing_mode": str(
                            specification.get("routing_mode", "standard")
                        ),
                        "trigger_phrase": str(
                            specification.get("trigger_phrase", "")
                        ),
                        "status": "disabled",
                        "compatible": None,
                        "model_only": True,
                    }
                )
                continue
            route = routes_by_id[lora_id]
            patches, metadata, report = self._load_lora_patches(specification)
            if not report["compatible"]:
                raise ValueError(
                    f"LoRA {report['display_name']} matched "
                    f"{report['matched_model_targets']}/{report['adapter_count']} "
                    "Krea 2 model targets"
                )
            for key, adapter in patches.items():
                if route_allows_adapter_target(route, str(key)):
                    target_entries.setdefault(key, []).append((adapter, route))
                else:
                    skipped_targets.setdefault(route.lora_id, []).append(str(key))
            if metadata:
                metadata_items.append({"id": lora_id, "metadata": metadata})
            report["status"] = "applied_global" if route.global_scope else "applied_regional"
            report["application_mode"] = (
                "unfused_token_delta_gate"
                if route.global_scope
                else "unfused_region_text_image_delta_gate_v3"
            )
            skipped = skipped_targets.get(route.lora_id, [])
            report["applied_model_targets"] = len(patches) - len(skipped)
            report["locality_skipped_targets"] = len(skipped)
            report["locality_skipped_target_examples"] = skipped[:8]
            report["route"] = route.summary()
            if report["applied_model_targets"] == 0:
                raise ValueError(
                    f"Regional LoRA {report['display_name']!r} has no targets that "
                    "can be routed locally; no LoRA was applied"
                )
            reports.append(report)

        projector_bypass = base_model.get_attachment("k2_projector_bypass_adapter")
        if projector_bypass is not None:
            target_entries.setdefault(projector_bypass["target"], []).insert(
                0, (projector_bypass["adapter"], None)
            )

        statistics = LoraDeltaStatistics(routes)
        if not target_entries:
            return base_model, reports, statistics
        generation_model, installed_targets = self._install_routed_lora_bypass(
            base_model, target_entries, statistics
        )
        expected_targets = len(target_entries)
        if installed_targets != expected_targets:
            raise ValueError(
                f"LoRA routing mapped {expected_targets} model targets but installed "
                f"only {installed_targets}"
            )
        if metadata_items:
            generation_model.set_attachments("lora_metadata", metadata_items)
        if event is not None:
            for report in reports:
                if report["status"] == "disabled":
                    continue
                route = report["route"]
                scope = "Global" if report["global"] else ", ".join(route["region_names"])
                event(
                    f"Applied LoRA {report['display_name']} to {scope} at {report['strength']:.2f}",
                    {"lora": report},
                )
        return generation_model, reports, statistics

    def _apply_global_projector_vector(
        self,
        *,
        enabled: bool,
        preset: str,
        values,
        multiplier: float,
        identity_protection: float = 1.0,
        bound_plan: BoundRegionalPromptPlan | None = None,
        event: Callable[[str, dict[str, Any]], None] | None = None,
    ):
        """Patch Krea's global 1×12 text-fusion projector before LoRA routing.

        The projector reduces the layerwise text-fusion axis for every token. Face
        identity prompt spans can retain some or all of the baseline layer mixture;
        every other token continues to receive the complete preset delta.
        """

        raw_values = (
            projector_preset_values(preset)
            if not values
            else validate_projector_values(values)
        )
        effective_values = effective_projector_values(raw_values, multiplier)
        summary = {
            "enabled": bool(enabled),
            "scope": "global",
            "preset": preset,
            "values": list(raw_values),
            "multiplier": float(multiplier),
            "effective_values": list(effective_values),
            "target": "diffusion_model.txtfusion.projector.weight",
            "identity_protection": float(identity_protection),
        }
        if not enabled or not any(effective_values):
            summary["status"] = "disabled" if not enabled else "zero_effect"
            return self.model, summary

        import torch

        target = summary["target"]
        target_weight = self.model.model.state_dict().get(target)
        expected_shape = (1, PROJECTOR_VECTOR_COUNT)
        if target_weight is None:
            raise RuntimeError(f"Krea projector target is missing: {target}")
        if tuple(target_weight.shape) != expected_shape:
            raise RuntimeError(
                f"unexpected Krea projector shape {tuple(target_weight.shape)}; "
                f"expected {expected_shape}"
            )
        delta = torch.tensor((effective_values,), dtype=torch.float32)
        protected = tuple(
            (identity.start, identity.end)
            for identity in (bound_plan.face_identities if bound_plan else ())
        )
        summary["protected_token_spans"] = [list(span) for span in protected]
        summary["protected_regions"] = [
            identity.region_id
            for identity in (bound_plan.face_identities if bound_plan else ())
        ]
        if protected and identity_protection > 0.0:
            import torch.nn.functional as functional

            import comfy.weight_adapter

            token_mask = projector_token_delta_mask(
                bound_plan.text_token_count,
                protected,
                identity_protection,
            )
            base_adapter_type = comfy.weight_adapter.WeightAdapterBase

            class TokenSelectiveProjectorDelta(base_adapter_type):
                name = "k2_token_selective_projector_delta"

                def __init__(self, weight, mask) -> None:
                    self.weights = (weight,)
                    self.loaded_keys = set()
                    self.mask = mask
                    self._weight_cache = {}
                    self._mask_cache = {}

                def h(self, x, base_out):
                    del base_out
                    if x.ndim != 4 or x.shape[1] != len(self.mask):
                        raise RuntimeError(
                            "Krea projector identity protection expected "
                            f"{len(self.mask)} text tokens, received {tuple(x.shape)}"
                        )
                    cache_key = (x.device, x.dtype)
                    weight = self._weight_cache.get(cache_key)
                    if weight is None:
                        weight = self.weights[0].to(device=x.device, dtype=x.dtype)
                        self._weight_cache[cache_key] = weight
                    mask = self._mask_cache.get(cache_key)
                    if mask is None:
                        mask = torch.tensor(
                            self.mask, device=x.device, dtype=x.dtype
                        ).view(1, -1, 1, 1)
                        self._mask_cache[cache_key] = mask
                    return functional.linear(x, weight) * mask

            patched_model = self.model.clone()
            patched_model.set_attachments(
                "k2_projector_bypass_adapter",
                {
                    "target": target,
                    "adapter": TokenSelectiveProjectorDelta(delta, token_mask),
                },
            )
            summary["status"] = "applied_token_selective_diff"
            summary["protected_token_count"] = sum(
                value < 1.0 for value in token_mask
            )
            patched_model.set_attachments("projector_settings", summary)
            if event is not None:
                event(
                    "Applied token-selective global projector vector "
                    f"({preset}) at {float(multiplier):.4f}×; protected "
                    f"{summary['protected_token_count']} face-identity token(s)",
                    {"projector": summary},
                )
            return patched_model, summary

        patched_model = self.model.clone()
        patched_keys = patched_model.add_patches({target: ("diff", (delta,))})
        if target not in patched_keys:
            raise RuntimeError("could not install the Krea projector vector patch")
        summary["status"] = "applied_global_diff"
        patched_model.set_attachments("projector_settings", summary)
        if event is not None:
            event(
                f"Applied global projector vector ({preset}) at {float(multiplier):.4f}×",
                {"projector": summary},
            )
        return patched_model, summary

    @staticmethod
    def _install_routed_lora_bypass(generation_model, target_entries, statistics):
        import torch

        import comfy.weight_adapter

        base_adapter_type = comfy.weight_adapter.WeightAdapterBase

        class RoutedCompositeAdapter(base_adapter_type):
            name = "k2_routed_composite"

            def __init__(self, entries, *, route_kind: str) -> None:
                self.entries = entries
                self.route_kind = route_kind
                self.weights = []
                self.loaded_keys = set()
                self._prepared: set[int] = set()
                self._mask_cache = {}

            def _prepare_adapter(self, adapter, route, x) -> None:
                adapter.multiplier = route.strength if route is not None else 1.0
                for name in (
                    "is_conv",
                    "conv_dim",
                    "kernel_size",
                    "in_channels",
                    "out_channels",
                    "kw_dict",
                ):
                    setattr(adapter, name, getattr(self, name))
                identity = id(adapter)
                if identity in self._prepared:
                    return
                weights = getattr(adapter, "weights", None)
                if isinstance(weights, (tuple, list)):
                    moved = []
                    for weight in weights:
                        if isinstance(weight, torch.Tensor):
                            dtype = x.dtype if weight.is_floating_point() else weight.dtype
                            moved.append(weight.to(device=x.device, dtype=dtype))
                        else:
                            moved.append(weight)
                    adapter.weights = type(weights)(moved)
                self._prepared.add(identity)

            def _mask(self, route, x):
                key = (
                    route.lora_id,
                    self.route_kind,
                    tuple(x.shape),
                    x.device,
                    x.dtype,
                )
                mask = self._mask_cache.get(key)
                if mask is None:
                    if route.global_scope:
                        mask = torch.ones((), device=x.device, dtype=x.dtype)
                    elif self.route_kind == "text_layerwise":
                        values = route.layerwise_text_batch_mask(int(x.shape[0]))
                        mask = torch.tensor(values, device=x.device, dtype=x.dtype).view(-1, 1, 1)
                    elif self.route_kind == "text_projector":
                        values = route.sequence_mask(int(x.shape[1]), text_fusion=True)
                        mask = torch.tensor(values, device=x.device, dtype=x.dtype).view(
                            1, -1, 1, 1
                        )
                    else:
                        text_fusion = self.route_kind == "text_refiner"
                        text_count = len(route.text_token_mask)
                        image_count = len(route.image_token_mask)
                        expected_counts = (
                            {text_count}
                            if text_fusion
                            else {image_count, text_count + image_count}
                        )
                        token_axes = [
                            axis
                            for axis, length in enumerate(x.shape[:-1])
                            if int(length) in expected_counts
                        ]
                        if len(token_axes) != 1:
                            raise ValueError(
                                f"LoRA route {route.display_name!r} could not identify "
                                f"one token axis in input shape {tuple(x.shape)}; "
                                f"expected one of {sorted(expected_counts)}"
                            )
                        token_axis = token_axes[0]
                        values = route.sequence_mask(
                            int(x.shape[token_axis]), text_fusion=text_fusion
                        )
                        mask_shape = [1] * x.ndim
                        mask_shape[token_axis] = len(values)
                        mask = torch.tensor(
                            values, device=x.device, dtype=x.dtype
                        ).view(*mask_shape)
                    self._mask_cache[key] = mask
                return mask

            def h(self, x, base_out):
                total = torch.zeros_like(base_out)
                for adapter, route in self.entries:
                    self._prepare_adapter(adapter, route, x)
                    delta = adapter.h(x, base_out)
                    if route is None:
                        total = total + delta
                        continue
                    applied = delta * self._mask(route, x)
                    token_norms = torch.linalg.vector_norm(
                        applied.detach(), dim=-1, dtype=torch.float32
                    )
                    statistics.observe(route, token_norms, route_kind=self.route_kind)
                    total = total + applied
                return total

        manager = comfy.weight_adapter.BypassInjectionManager()
        unsupported = []
        parameter_entries = {}
        for key, entries in target_entries.items():
            if not all(isinstance(adapter, base_adapter_type) for adapter, _route in entries):
                unsupported.append(key)
                continue
            lowered = str(key).casefold()
            if lowered.endswith(".last.modulation.lin") or lowered.endswith(
                ".last.modulation.lin.weight"
            ):
                if not all(route.global_scope for _adapter, route in entries):
                    unsupported.append(key)
                    continue
                parameter_entries[key] = entries
                continue
            manager.add_adapter(
                key,
                RoutedCompositeAdapter(
                    entries,
                    route_kind=(
                        "text_layerwise"
                        if ".txtfusion.layerwise_blocks." in str(key)
                        else "text_projector"
                        if ".txtfusion.projector." in str(key)
                        else "text_refiner"
                        if ".txtfusion." in str(key) or ".txtmlp." in str(key)
                        else "combined"
                    ),
                ),
                strength=1.0,
            )
        if unsupported:
            raise ValueError(
                "regional LoRA routing does not support non-adapter patches: "
                + ", ".join(map(str, unsupported[:4]))
            )
        patched_model = generation_model.clone()
        parameter_target_count = 0
        for key, entries in parameter_entries.items():
            installed = True
            for adapter, route in entries:
                patched_keys = patched_model.add_patches(
                    {key: adapter}, strength_patch=route.strength
                )
                if key not in patched_keys:
                    installed = False
                    break
            parameter_target_count += int(installed)
        injections = manager.create_injections(patched_model.model)
        if manager.get_hook_count():
            patched_model.set_injections("k2_routed_loras", injections)
        return patched_model, manager.get_hook_count() + parameter_target_count

    def _prepare_vae_handoff(
        self,
        generation_model,
        event: Callable[[str, dict[str, Any]], None] | None,
    ) -> None:
        """Offload denoising state before VAE decode enters inference mode.

        Comfy may otherwise decide to unload the quantized transformer from inside
        ``VAE.decode``. PyTorch forbids Comfy's quantized parameter reconstruction
        when that unload happens under inference mode.
        """

        self._release_generation_model(generation_model)
        if event is not None:
            event(
                "Transformer offloaded before VAE decode",
                {"memory": self.memory_snapshot("VAE handoff complete")},
            )

    @staticmethod
    def _release_generation_model(generation_model) -> None:
        import comfy.model_management

        comfy.model_management.unload_all_models()
        generation_model.remove_injections("k2_routed_loras")
        generation_model.remove_injections("k2_projector_delta")
        gc.collect()
        comfy.model_management.soft_empty_cache(force=True)

    def _release_gpu_for_post_upscale(
        self,
        event: Callable[[str, dict[str, Any]], None] | None,
    ) -> None:
        """Move every Krea component off the accelerator before upscaling."""

        import comfy.model_management

        comfy.model_management.unload_all_models()
        gc.collect()
        comfy.model_management.soft_empty_cache(force=True)
        if event is not None:
            event(
                "Krea GPU state released before post-upscale",
                {"memory": self.memory_snapshot("post-upscale handoff")},
            )

    def _post_upscale_image(
        self,
        image,
        *,
        scale: int,
        method: str,
        model_path: Path | None,
        event: Callable[[str, dict[str, Any]], None] | None,
    ):
        from PIL import Image

        if scale not in {2, 4}:
            raise ValueError("post-upscale scale must be 2 or 4")
        target_size = (image.width * scale, image.height * scale)
        if method == "lanczos":
            if event is not None:
                event(
                    f"CPU Lanczos post-upscale {scale}× started",
                    {"target_width": target_size[0], "target_height": target_size[1]},
                )
            result = image.resize(target_size, Image.Resampling.LANCZOS)
            return result, {
                "enabled": True,
                "backend": "pillow-lanczos",
                "scale": scale,
                "model": None,
            }
        if method != "model":
            raise ValueError(f"unsupported post-upscale method: {method!r}")
        if model_path is None or not model_path.expanduser().is_file():
            raise ValueError("the selected neural upscaler model is not readable")

        import numpy as np
        import torch
        from spandrel import ImageModelDescriptor, ModelLoader

        import comfy.model_management
        import comfy.utils

        resolved_model_path = model_path.expanduser().resolve()
        if event is not None:
            event(
                f"Neural post-upscale {scale}× started",
                {"model": str(resolved_model_path)},
            )
        state = comfy.utils.load_torch_file(str(resolved_model_path), safe_load=True)
        if "module.layers.0.residual_group.blocks.0.norm1.weight" in state:
            state = comfy.utils.state_dict_prefix_replace(state, {"module.": ""})
        upscale_model = ModelLoader().load_from_state_dict(state).eval()
        del state
        if not isinstance(upscale_model, ImageModelDescriptor):
            raise ValueError("upscaler must be a single-image ESRGAN-compatible model")

        device = comfy.model_management.get_torch_device()
        source = torch.from_numpy(np.asarray(image).copy()).to(dtype=torch.float32)
        source = source.div_(255.0).unsqueeze(0).movedim(-1, -3).to(device)
        tile = 512
        upscale_model.to(device)
        try:
            with torch.no_grad():
                while True:
                    try:
                        output = comfy.utils.tiled_scale(
                            source,
                            lambda tile_input: upscale_model(tile_input.float()),
                            tile_x=tile,
                            tile_y=tile,
                            overlap=32,
                            upscale_amount=upscale_model.scale,
                            output_device=torch.device("cpu"),
                        )
                        break
                    except Exception as error:
                        comfy.model_management.raise_non_oom(error)
                        tile //= 2
                        if tile < 128:
                            raise
        finally:
            upscale_model.to("cpu")
            del source
            gc.collect()
            comfy.model_management.soft_empty_cache(force=True)

        output = output.clamp_(0, 1).movedim(-3, -1)[0]
        array = (output.mul_(255.0).round_().to(torch.uint8).numpy())
        result = Image.fromarray(array)
        if result.size != target_size:
            result = result.resize(target_size, Image.Resampling.LANCZOS)
        return result, {
            "enabled": True,
            "backend": "spandrel-tiled",
            "scale": scale,
            "native_model_scale": float(upscale_model.scale),
            "tile_size": tile,
            "model": str(resolved_model_path),
        }

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
                "GPU OOM recovery could not start because available system RAM is below "
                f"the {self.minimum_system_ram_gb:.1f} GiB guard"
            )
        device = comfy.model_management.get_torch_device()
        total_vram_gb = before["gpu_total_bytes"] / GIB
        retry_reserve_gb = oom_recovery_reserve_vram_gb(
            self.reserve_vram_gb, total_vram_gb
        )
        target_free = int(retry_reserve_gb * GIB)
        comfy.model_management.free_memory(target_free, device)
        gc.collect()
        comfy.model_management.soft_empty_cache(force=True)
        self.reserve_vram_gb = retry_reserve_gb
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
        sampler: str = DEFAULT_SAMPLER,
        scheduler: str = DEFAULT_SCHEDULER,
        seed: int,
        output_directory: Path,
        filename_prefix: str = "baseline",
        regions: tuple[RegionDefinition, ...] = (),
        emphases: tuple[PromptEmphasis, ...] = (),
        regional_prompting: bool = True,
        regional_prompt_strength: float = 1.0,
        regional_outside_penalty: float = 1.0,
        regional_feather_pixels: float = 128.0,
        regional_subject_competition: bool = True,
        regional_subject_fill: bool = True,
        regional_late_step_scale: float = 0.35,
        regional_lora_delta_adaptation: bool = False,
        regional_lora_delta_adaptation_gain: float = 0.35,
        projector_enabled: bool = False,
        projector_preset: str = DEFAULT_PROJECTOR_PRESET,
        projector_values: tuple[float, ...] = (),
        projector_multiplier: float = 1.0,
        projector_identity_protection: float = 1.0,
        post_upscale: bool = False,
        upscale_scale: int = 2,
        upscale_method: str = "lanczos",
        upscale_model_path: Path | None = None,
        loras: list[dict[str, Any]] | None = None,
        project_json: dict[str, Any] | None = None,
        progress: Callable[[int, int, dict[str, Any]], None] | None = None,
        event: Callable[[str, dict[str, Any]], None] | None = None,
    ) -> dict[str, Any]:
        if not self.loaded:
            raise RuntimeError("baseline components must be loaded before generation")
        if width <= 0 or height <= 0 or width % 16 or height % 16:
            raise ValueError("baseline dimensions must be positive multiples of 16")
        if not 1 <= steps <= 100:
            raise ValueError("steps must be between 1 and 100")
        sampler = validate_sampler(sampler)
        scheduler = validate_scheduler(scheduler)
        if not 0.0 <= regional_lora_delta_adaptation_gain <= 1.0:
            raise ValueError("LoRA delta adaptation gain must be between zero and one")
        if not 0.0 <= projector_identity_protection <= 1.0:
            raise ValueError("projector identity protection must be between zero and one")
        if post_upscale and upscale_method == "model":
            if upscale_model_path is None or not upscale_model_path.expanduser().is_file():
                raise ValueError("select a readable neural upscaler model before generation")
        filename_prefix = validate_filename_prefix(filename_prefix)
        lora_specifications = list(loras or [])
        regional_plan = (
            compile_regional_prompt_plan(
                width,
                height,
                prompt,
                regions,
                strength=regional_prompt_strength,
                outside_penalty=regional_outside_penalty,
                falloff_pixels=regional_feather_pixels,
                subject_competition=regional_subject_competition,
                subject_fill=regional_subject_fill,
                late_step_scale=regional_late_step_scale,
                emphases=emphases,
                character_identity_triggers=character_identity_triggers(
                    lora_specifications
                ),
            )
            if regional_prompting and (regions or emphases)
            else None
        )

        oom_message: str | None = None
        try:
            return self._generate_once(
                prompt=prompt,
                width=width,
                height=height,
                steps=steps,
                sampler=sampler,
                scheduler=scheduler,
                seed=seed,
                output_directory=output_directory,
                filename_prefix=filename_prefix,
                regional_plan=regional_plan,
                regional_lora_delta_adaptation=regional_lora_delta_adaptation,
                regional_lora_delta_adaptation_gain=regional_lora_delta_adaptation_gain,
                projector_enabled=projector_enabled,
                projector_preset=projector_preset,
                projector_values=projector_values,
                projector_multiplier=projector_multiplier,
                projector_identity_protection=projector_identity_protection,
                post_upscale=post_upscale,
                upscale_scale=upscale_scale,
                upscale_method=upscale_method,
                upscale_model_path=upscale_model_path,
                loras=lora_specifications,
                project_json=project_json,
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
            sampler=sampler,
            scheduler=scheduler,
            seed=seed,
            output_directory=output_directory,
            filename_prefix=filename_prefix,
            regional_plan=regional_plan,
            regional_lora_delta_adaptation=regional_lora_delta_adaptation,
            regional_lora_delta_adaptation_gain=regional_lora_delta_adaptation_gain,
            projector_enabled=projector_enabled,
            projector_preset=projector_preset,
            projector_values=projector_values,
            projector_multiplier=projector_multiplier,
            projector_identity_protection=projector_identity_protection,
            post_upscale=post_upscale,
            upscale_scale=upscale_scale,
            upscale_method=upscale_method,
            upscale_model_path=upscale_model_path,
            loras=list(loras or []),
            project_json=project_json,
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
        sampler: str,
        scheduler: str,
        seed: int,
        output_directory: Path,
        filename_prefix: str,
        regional_plan: RegionalPromptPlan | None,
        regional_lora_delta_adaptation: bool,
        regional_lora_delta_adaptation_gain: float,
        projector_enabled: bool,
        projector_preset: str,
        projector_values: tuple[float, ...],
        projector_multiplier: float,
        projector_identity_protection: float,
        post_upscale: bool,
        upscale_scale: int,
        upscale_method: str,
        upscale_model_path: Path | None,
        loras: list[dict[str, Any]],
        project_json: dict[str, Any] | None,
        progress: Callable[[int, int, dict[str, Any]], None] | None,
        event: Callable[[str, dict[str, Any]], None] | None,
        oom_recovered: bool,
    ) -> dict[str, Any]:
        import numpy as np
        import torch
        from PIL import Image, PngImagePlugin

        import comfy.sample
        import comfy.samplers

        if sampler not in comfy.samplers.KSampler.SAMPLERS:
            raise ValueError(
                f"sampler {sampler!r} is unavailable in the installed ComfyUI runtime"
            )
        if scheduler not in comfy.samplers.KSampler.SCHEDULERS:
            raise ValueError(
                f"scheduler {scheduler!r} is unavailable in the installed ComfyUI runtime"
            )

        self._ensure_memory("before text encoding", event)

        conditioned_prompt = (
            regional_plan.prompt
            if regional_plan is not None
            and (regional_plan.regions or regional_plan.emphases)
            else prompt
        )
        positive = self.clip.encode_from_tokens_scheduled(
            self.clip.tokenize(conditioned_prompt)
        )
        negative = self.clip.encode_from_tokens_scheduled(self.clip.tokenize(""))
        if not positive:
            raise RuntimeError("Krea text encoder returned no positive conditioning")
        text_token_counts = {int(condition[0].shape[1]) for condition in positive}
        if len(text_token_counts) != 1:
            raise RuntimeError("Krea conditioning must use one text sequence length")
        conditioning_text_token_count = text_token_counts.pop()
        bound_regional_plan: BoundRegionalPromptPlan | None = None
        if regional_plan is not None and (
            regional_plan.regions or regional_plan.emphases
        ):
            bound_regional_plan = regional_plan.bind_tokens(
                lambda prefix: krea_prompt_token_count(self.clip.tokenize(prefix)),
                conditioning_text_token_count=conditioning_text_token_count,
            )
            if event is not None:
                event("Unified spatial prompt prepared", bound_regional_plan.summary())
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
        if loras:
            self._ensure_memory("before LoRA loading", event)
        generation_model, projector_summary = self._apply_global_projector_vector(
            enabled=projector_enabled,
            preset=projector_preset,
            values=projector_values,
            multiplier=projector_multiplier,
            identity_protection=projector_identity_protection,
            bound_plan=bound_regional_plan,
            event=event,
        )
        generation_model, lora_reports, lora_statistics = self._apply_routed_loras(
            loras,
            base_model=generation_model,
            width=width,
            height=height,
            text_token_count=conditioning_text_token_count,
            regional_plan=regional_plan,
            bound_plan=bound_regional_plan,
            event=event,
        )

        def callback(step: int, denoised, current, total: int) -> None:
            del denoised, current
            if attention_override is not None:
                attention_override.set_denoising_progress(step + 1, total)
                if regional_lora_delta_adaptation:
                    attention_override.set_lora_delta_scales(
                        lora_statistics.regional_attention_scales(
                            regional_lora_delta_adaptation_gain
                        )
                    )
                    lora_statistics.reset_step_measurements()
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

        attention_override = (
            KreaSpatialAttentionOverride(
                bound_regional_plan,
                lora_delta_adaptation=regional_lora_delta_adaptation,
                lora_delta_adaptation_gain=regional_lora_delta_adaptation_gain,
            )
            if bound_regional_plan is not None
            and (bound_regional_plan.spans or bound_regional_plan.emphases)
            else None
        )
        if regional_lora_delta_adaptation and attention_override is not None and event is not None:
            event(
                "LoRA delta-adaptive spatial guidance enabled",
                {"gain": regional_lora_delta_adaptation_gain},
            )
        transformer_options = generation_model.model_options.setdefault(
            "transformer_options", {}
        )
        missing = object()
        previous_override = transformer_options.get(
            "optimized_attention_override", missing
        )
        if attention_override is not None:
            if previous_override is not missing:
                raise RuntimeError(
                    "another optimized-attention override is already installed"
                )
            transformer_options["optimized_attention_override"] = attention_override
        try:
            samples = comfy.sample.sample(
                generation_model,
                noise,
                steps,
                1.0,
                sampler,
                scheduler,
                positive,
                negative,
                latent,
                denoise=1.0,
                callback=callback,
                disable_pbar=True,
                seed=seed,
            )
        finally:
            if attention_override is not None:
                attention_override.clear()
                if previous_override is missing:
                    transformer_options.pop("optimized_attention_override", None)
                else:
                    transformer_options["optimized_attention_override"] = previous_override
        if attention_override is not None:
            if attention_override.matched_calls == 0:
                raise RuntimeError(
                    "Krea main-stream attention was not reached by the spatial override"
                )
            if attention_override.text_refiner_calls == 0:
                raise RuntimeError(
                    "Krea text-refiner attention was not reached by the regional "
                    "text partition"
                )
            if event is not None:
                event(
                    "Unified spatial attention applied",
                    {
                        "attention_calls": attention_override.matched_calls,
                        "text_refiner_attention_calls": attention_override.text_refiner_calls,
                    },
                )
                if regional_lora_delta_adaptation:
                    event(
                        "LoRA delta-adaptive spatial guidance finalized",
                        attention_override.summary(),
                    )
        for report in lora_reports:
            if report.get("status") not in {"applied_global", "applied_regional"}:
                continue
            delta_summary = lora_statistics.summary(str(report["id"]))
            report["delta_statistics"] = delta_summary
            if event is not None:
                event(
                    f"LoRA delta measured for {report['display_name']}",
                    {"lora_id": report["id"], **delta_summary},
                )
        self._ensure_memory("before VAE decode", event)
        self._prepare_vae_handoff(generation_model, event)
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

        output_image = Image.fromarray(array)
        upscale_summary: dict[str, Any] = {
            "enabled": False,
            "backend": "disabled",
            "scale": 1,
            "model": None,
        }
        if post_upscale:
            self._release_gpu_for_post_upscale(event)
            output_image, upscale_summary = self._post_upscale_image(
                output_image,
                scale=upscale_scale,
                method=upscale_method,
                model_path=upscale_model_path,
                event=event,
            )
            if event is not None:
                event(
                    f"Post-upscale complete: {output_image.width}×{output_image.height}",
                    {"post_upscale": upscale_summary},
                )

        output_directory.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        output_path = output_directory / f"{filename_prefix}_{stamp}_seed-{seed}.png"
        metadata = PngImagePlugin.PngInfo()
        metadata.add_text("k2lab_mode", "krea2_turbo_baseline")
        metadata.add_text("prompt", conditioned_prompt)
        metadata.add_text("global_prompt", prompt)
        metadata.add_text("seed", str(seed))
        metadata.add_text("steps", str(steps))
        metadata.add_text("sampler", sampler)
        metadata.add_text("scheduler", scheduler)
        metadata.add_text("size", f"{output_image.width}x{output_image.height}")
        metadata.add_text("base_size", f"{width}x{height}")
        metadata.add_text("filename_prefix", filename_prefix)
        if project_json is not None:
            metadata.add_text(
                "k2lab_project",
                json.dumps(project_json, separators=(",", ":")),
            )
        regional_summary = self._regional_summary(
            regional_plan, bound_regional_plan, attention_override
        )
        metadata.add_text("regional_prompting", json.dumps(regional_summary))
        metadata.add_text("projector", json.dumps(projector_summary))
        metadata.add_text("post_upscale", json.dumps(upscale_summary))
        metadata.add_text("loras", json.dumps(lora_reports))
        metadata.add_text("memory_policy", self.memory_policy_key)
        metadata.add_text("oom_recovered", str(oom_recovered).lower())
        metadata.add_text("cpu_vae", str(self.cpu_vae).lower())
        output_image.save(output_path, pnginfo=metadata)
        return {
            "image_path": str(output_path),
            "width": output_image.width,
            "height": output_image.height,
            "base_width": width,
            "base_height": height,
            "steps": steps,
            "seed": seed,
            "filename_prefix": filename_prefix,
            "regional_prompting": regional_summary,
            "projector": projector_summary,
            "post_upscale": upscale_summary,
            "loras": lora_reports,
            "sampler": sampler,
            "scheduler": scheduler,
            "cfg": 1.0,
            "memory_policy": self.memory_policy_key,
            "reserve_vram_gb": self.reserve_vram_gb,
            "cpu_vae": self.cpu_vae,
            "oom_recovered": oom_recovered,
            "memory": self.memory_snapshot("generation complete"),
        }

    def edit_image(
        self,
        *,
        image_path: Path,
        prompt: str,
        regions: tuple[RegionDefinition, ...],
        loras: list[dict[str, Any]],
        reference_prompt: str = "",
        reference_regions: tuple[RegionDefinition, ...] = (),
        prompt_emphases: tuple[PromptEmphasis, ...] = (),
        seed: int = 0,
        steps: int = 8,
        sampler: str = DEFAULT_SAMPLER,
        scheduler: str = DEFAULT_SCHEDULER,
        denoise: float = 0.15,
        latent_feather_pixels: int = 64,
        composite_feather_pixels: int = 48,
        edit_entire_image: bool = False,
        preserve_identity: bool = True,
        reference_description_retention: float = 1.0,
        regional_prompt_strength: float = 1.0,
        regional_outside_penalty: float = 1.0,
        regional_feather_pixels: float = 128.0,
        regional_subject_competition: bool = True,
        regional_subject_fill: bool = True,
        regional_late_step_scale: float = 0.35,
        regional_lora_delta_adaptation: bool = False,
        regional_lora_delta_adaptation_gain: float = 0.35,
        projector_enabled: bool = False,
        projector_preset: str = DEFAULT_PROJECTOR_PRESET,
        projector_values: tuple[float, ...] | list[float] | None = None,
        projector_multiplier: float = 1.0,
        projector_identity_protection: float = 1.0,
        project_json: dict[str, Any] | None = None,
        output_directory: Path | None = None,
        progress: Callable[[int, int, dict[str, Any]], None] | None = None,
        event: Callable[[str, dict[str, Any]], None] | None = None,
    ) -> dict[str, Any]:
        """Run source-latent img2img with the baseline regional routing stack."""

        if not self.loaded:
            raise RuntimeError("baseline components must be loaded before image editing")
        if not 1 <= steps <= 100:
            raise ValueError("image-edit steps must be between 1 and 100")
        if not 0.0 < denoise <= 1.0:
            raise ValueError("image-edit denoise must be in (0, 1]")
        if not 0 <= latent_feather_pixels <= 256:
            raise ValueError("image-edit latent feather must be between 0 and 256 pixels")
        if not 0 <= composite_feather_pixels <= 256:
            raise ValueError("image-edit composite feather must be between 0 and 256 pixels")
        if not 0.0 <= reference_description_retention <= 1.0:
            raise ValueError("reference description retention must be between zero and one")
        sampler = validate_sampler(sampler)
        scheduler = validate_scheduler(scheduler)

        import numpy as np
        import torch
        from PIL import Image, PngImagePlugin

        import comfy.model_management
        import comfy.sample
        import comfy.samplers

        if sampler not in comfy.samplers.KSampler.SAMPLERS:
            raise ValueError(
                f"sampler {sampler!r} is unavailable in the installed ComfyUI runtime"
            )
        if scheduler not in comfy.samplers.KSampler.SCHEDULERS:
            raise ValueError(
                f"scheduler {scheduler!r} is unavailable in the installed ComfyUI runtime"
            )

        source_path = image_path.expanduser().resolve()
        source_image, source_metadata = load_source_image(source_path)
        padded_source, geometry = edge_pad_to_krea(source_image)
        target_regions = tuple(region for region in regions if region.enabled)
        filtered_loras = [
            lora
            for lora in loras
            if preserve_identity
            or not (
                str(lora.get("id", "")).startswith("reference:")
                and str(lora.get("routing_mode", "")) == "character_identity"
            )
        ]
        conditioning_regions = regional_edit_conditioning(
            reference_regions,
            target_regions,
            prompt,
            preserve_identity=preserve_identity,
        )
        active_edit_regions = tuple(
            region for region in conditioning_regions if region.spatial_role == "edit"
        )
        if not edit_entire_image and not target_regions:
            raise ValueError(
                "regional image editing requires an edit box or Edit entire image"
            )
        if not prompt.strip() and not active_edit_regions:
            raise ValueError(
                "a blank image-edit global prompt requires at least one active regional prompt"
            )

        conditioned_global_prompt = edit_global_conditioning_prompt(
            reference_prompt,
            prompt,
            edit_entire_image=edit_entire_image,
        )
        conditioned_emphases = regional_reference_emphases(prompt_emphases)

        regional_plan = (
            compile_regional_prompt_plan(
                source_image.width,
                source_image.height,
                conditioned_global_prompt,
                conditioning_regions,
                strength=regional_prompt_strength,
                outside_penalty=regional_outside_penalty,
                falloff_pixels=regional_feather_pixels,
                subject_competition=regional_subject_competition,
                subject_fill=regional_subject_fill,
                late_step_scale=regional_late_step_scale,
                emphases=conditioned_emphases,
                character_identity_triggers=character_identity_triggers(filtered_loras),
            )
            if conditioning_regions
            else None
        )
        conditioned_prompt = (
            regional_plan.prompt
            if regional_plan is not None and regional_plan.regions
            else conditioned_global_prompt
        )
        if not conditioned_prompt:
            raise ValueError("image editing requires prompt text")

        self._ensure_memory("before image-edit text encoding", event)
        positive = self.clip.encode_from_tokens_scheduled(
            self.clip.tokenize(conditioned_prompt)
        )
        negative = self.clip.encode_from_tokens_scheduled(self.clip.tokenize(""))
        if not positive:
            raise RuntimeError("Krea text encoder returned no image-edit conditioning")
        text_token_counts = {int(condition[0].shape[1]) for condition in positive}
        if len(text_token_counts) != 1:
            raise RuntimeError("Krea image-edit conditioning must use one text sequence length")
        text_token_count = text_token_counts.pop()
        bound_plan = None
        if regional_plan is not None and regional_plan.regions:
            bound_plan = regional_plan.bind_tokens(
                lambda prefix: krea_prompt_token_count(self.clip.tokenize(prefix)),
                conditioning_text_token_count=text_token_count,
            )
            if event is not None:
                event("Unified spatial edit prompt prepared", bound_plan.summary())

        pixels = torch.from_numpy(
            np.asarray(padded_source, dtype=np.float32).copy() / 255.0
        ).unsqueeze(0)
        self._ensure_memory("before image-edit VAE encode", event)
        latent = self._encode_vae(pixels)
        latent = comfy.sample.fix_empty_latent_channels(
            self.model, latent, downscale_ratio_spacial=8
        )
        noise = comfy.sample.prepare_noise(latent, seed)
        if edit_entire_image:
            denoise_mask = torch.ones(
                (1, latent.shape[-2], latent.shape[-1]),
                dtype=torch.float32,
                device="cpu",
            )
        else:
            pixel_mask = regional_composite_mask(
                padded_source.size,
                target_regions,
                latent_feather_pixels,
            )
            denoise_mask = torch.from_numpy(
                np.asarray(pixel_mask, dtype=np.float32).copy() / 255.0
            ).unsqueeze(0)
        generation_model, projector_summary = self._apply_global_projector_vector(
            enabled=projector_enabled,
            preset=projector_preset,
            values=projector_values,
            multiplier=projector_multiplier,
            identity_protection=projector_identity_protection,
            bound_plan=bound_plan,
            event=event,
        )
        generation_model, lora_reports, lora_statistics = self._apply_routed_loras(
            filtered_loras,
            base_model=generation_model,
            width=geometry.aligned_width,
            height=geometry.aligned_height,
            text_token_count=text_token_count,
            regional_plan=regional_plan,
            bound_plan=bound_plan,
            event=event,
        )
        attention_override = (
            KreaSpatialAttentionOverride(
                bound_plan,
                lora_delta_adaptation=regional_lora_delta_adaptation,
                lora_delta_adaptation_gain=regional_lora_delta_adaptation_gain,
            )
            if bound_plan is not None and bound_plan.spans
            else None
        )
        if attention_override is not None:
            reference_ids = {region.region_id for region in reference_regions}
            attention_override.region_scales.update(
                {
                    region_id: reference_description_retention
                    for region_id in reference_ids
                }
            )

        def callback(step: int, denoised, current, total: int) -> None:
            del denoised, current
            if attention_override is not None:
                attention_override.set_denoising_progress(step + 1, total)
                if regional_lora_delta_adaptation:
                    attention_override.set_lora_delta_scales(
                        lora_statistics.regional_attention_scales(
                            regional_lora_delta_adaptation_gain
                        )
                    )
                    attention_override.region_scales.update(
                        {
                            region.region_id: reference_description_retention
                            for region in reference_regions
                        }
                    )
                    lora_statistics.reset_step_measurements()
            snapshot = self.memory_snapshot(f"image-edit step {step + 1}/{total}")
            if progress is not None:
                progress(step + 1, total, snapshot)
            if snapshot["gpu_free_bytes"] < snapshot["critical_free_bytes"]:
                raise CriticalGpuMemoryPressure(
                    "critical GPU memory pressure after image-edit denoising step "
                    f"{step + 1}/{total}"
                )

        transformer_options = generation_model.model_options.setdefault(
            "transformer_options", {}
        )
        missing = object()
        previous_override = transformer_options.get("optimized_attention_override", missing)
        if attention_override is not None:
            if previous_override is not missing:
                raise RuntimeError("another optimized-attention override is already installed")
            transformer_options["optimized_attention_override"] = attention_override
        self._ensure_memory("before image-edit denoising", event)
        try:
            samples = comfy.sample.sample(
                generation_model,
                noise,
                steps,
                1.0,
                sampler,
                scheduler,
                positive,
                negative,
                latent,
                denoise=denoise,
                noise_mask=denoise_mask,
                callback=callback,
                disable_pbar=True,
                seed=seed,
            )
        finally:
            if attention_override is not None:
                attention_override.clear()
                if previous_override is missing:
                    transformer_options.pop("optimized_attention_override", None)
                else:
                    transformer_options["optimized_attention_override"] = previous_override
        if attention_override is not None:
            if attention_override.matched_calls == 0:
                raise RuntimeError(
                    "Krea main-stream attention was not reached by the edit spatial override"
                )
            if attention_override.text_refiner_calls == 0:
                raise RuntimeError(
                    "Krea text-refiner attention was not reached by the edit text partition"
                )
        for report in lora_reports:
            if report.get("status") in {"applied_global", "applied_regional"}:
                report["delta_statistics"] = lora_statistics.summary(str(report["id"]))

        self._ensure_memory("before image-edit VAE decode", event)
        self._prepare_vae_handoff(generation_model, event)
        images = self._decode_vae(samples)
        image_tensor = images[0]
        while image_tensor.ndim > 3 and image_tensor.shape[0] == 1:
            image_tensor = image_tensor[0]
        if image_tensor.ndim != 3 or image_tensor.shape[-1] != 3:
            raise RuntimeError(f"unexpected image-edit decoded shape: {tuple(images.shape)}")
        array = (
            image_tensor.detach()
            .to(device="cpu", dtype=torch.float32)
            .clamp(0, 1)
            .numpy()
            * 255.0
        ).round().astype(np.uint8)
        candidate = Image.fromarray(array).crop(
            (0, 0, source_image.width, source_image.height)
        )
        preserve_outside = not edit_entire_image
        effective_composite_feather = min(
            composite_feather_pixels, latent_feather_pixels
        )
        if preserve_outside:
            output_image, mask = composite_regional_edit(
                source_image,
                candidate,
                target_regions,
                effective_composite_feather,
            )
            changed_bounds = mask.getbbox()
        else:
            output_image = candidate
            changed_bounds = (0, 0, source_image.width, source_image.height)

        destination = (output_directory or source_path.parent).expanduser().resolve()
        destination.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        output_path = destination / f"{source_path.stem}_edited_{stamp}_seed-{seed}.png"
        regional_summary = self._regional_summary(
            regional_plan, bound_plan, attention_override
        )
        edit_summary = {
            "source_image": str(source_path),
            "original_size": [source_image.width, source_image.height],
            "aligned_size": [geometry.aligned_width, geometry.aligned_height],
            "seed": seed,
            "steps": steps,
            "sampler": sampler,
            "scheduler": scheduler,
            "denoise": denoise,
            "latent_feather_pixels": latent_feather_pixels,
            "preserve_outside_regions": preserve_outside,
            "composite_feather_pixels": composite_feather_pixels,
            "effective_composite_feather_pixels": effective_composite_feather,
            "edit_entire_image": edit_entire_image,
            "preserve_identity": preserve_identity,
            "reference_description_retention": reference_description_retention,
            "reference_global_conditioning_applied": False,
            "composite_bounds": list(changed_bounds) if changed_bounds else None,
            "regional_prompting": regional_summary,
            "projector": projector_summary,
            "loras": lora_reports,
        }
        metadata = PngImagePlugin.PngInfo()
        for key, value in source_metadata.items():
            if key not in {"k2lab_mode", "source_image", "image_edit", "k2lab_project"}:
                metadata.add_text(key, value)
        metadata.add_text("k2lab_mode", "krea2_regional_image_edit")
        metadata.add_text("source_image", str(source_path))
        metadata.add_text("prompt", conditioned_prompt)
        metadata.add_text("global_prompt", prompt)
        metadata.add_text("image_edit", json.dumps(edit_summary))
        metadata.add_text("regional_prompting", json.dumps(regional_summary))
        metadata.add_text("loras", json.dumps(lora_reports))
        if project_json is not None:
            metadata.add_text("k2lab_project", json.dumps(project_json, separators=(",", ":")))
        output_image.save(output_path, pnginfo=metadata)
        return {
            "image_path": str(output_path),
            "source_image": str(source_path),
            "width": output_image.width,
            "height": output_image.height,
            "seed": seed,
            "image_edit": edit_summary,
            "memory": self.memory_snapshot("image editing complete"),
        }

    def refine_faces(
        self,
        *,
        image_path: Path,
        regions: tuple[RegionDefinition, ...],
        loras: list[dict[str, Any]],
        seed: int = 0,
        steps: int = 8,
        denoise: float = 0.15,
        crop_size: int = 512,
        padding: float = 2.0,
        feather: float = 0.12,
        blend: float = 0.5,
        lora_scale: float = 0.5,
        detector_threshold: float = 0.15,
        detector_provider: str = "auto",
        selected_face_indices: tuple[int, ...] | None = None,
        manual_face_paths: tuple[tuple[tuple[float, float], ...], ...] = (),
        project_json: dict[str, Any] | None = None,
        output_directory: Path | None = None,
        event: Callable[[str, dict[str, Any]], None] | None = None,
    ) -> dict[str, Any]:
        if not self.loaded:
            raise RuntimeError("baseline components must be loaded before face refinement")
        source_path = image_path.expanduser().resolve()
        if source_path.suffix.casefold() != ".png" or not source_path.is_file():
            raise ValueError(f"face refinement requires a readable PNG: {source_path}")
        settings = FaceDetailSettings(
            enabled=True,
            steps=steps,
            denoise=denoise,
            crop_size=crop_size,
            padding=padding,
            feather=feather,
            blend=blend,
            lora_scale=lora_scale,
            detector_threshold=detector_threshold,
            detector_provider=detector_provider,
        )

        from PIL import Image, PngImagePlugin

        with Image.open(source_path) as source:
            source_metadata = {
                str(key): str(value)
                for key, value in source.info.items()
                if isinstance(value, (str, int, float, bool))
            }
            source_image = source.convert("RGB")
        refined_image, summary = self._run_face_detail_pass(
            source_image,
            settings=settings,
            regions=regions,
            loras=loras,
            seed=seed,
            selected_face_indices=selected_face_indices,
            manual_face_paths=manual_face_paths,
            event=event,
        )
        destination = (output_directory or source_path.parent).expanduser().resolve()
        destination.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        output_path = destination / (
            f"{source_path.stem}_face_refined_{stamp}_seed-{seed}.png"
        )
        metadata = PngImagePlugin.PngInfo()
        for key, value in source_metadata.items():
            if key in {"k2lab_mode", "source_image", "seed", "face_detail"} or (
                key == "k2lab_project" and project_json is not None
            ):
                continue
            metadata.add_text(key, value)
        metadata.add_text("k2lab_mode", "krea2_face_refinement")
        metadata.add_text("source_image", str(source_path))
        metadata.add_text("seed", str(seed))
        metadata.add_text("face_detail", json.dumps(summary))
        if project_json is not None:
            metadata.add_text(
                "k2lab_project",
                json.dumps(project_json, separators=(",", ":")),
            )
        refined_image.save(output_path, pnginfo=metadata)
        return {
            "image_path": str(output_path),
            "source_image": str(source_path),
            "width": refined_image.width,
            "height": refined_image.height,
            "seed": seed,
            "face_detail": summary,
            "memory": self.memory_snapshot("face refinement complete"),
        }

    def _run_face_detail_pass(
        self,
        image,
        *,
        settings: FaceDetailSettings,
        regions: tuple[RegionDefinition, ...],
        loras: list[dict[str, Any]],
        seed: int,
        selected_face_indices: tuple[int, ...] | None = None,
        manual_face_paths: tuple[tuple[tuple[float, float], ...], ...] = (),
        event: Callable[[str, dict[str, Any]], None] | None = None,
    ):
        from PIL import Image

        summary: dict[str, Any] = {
            "enabled": settings.enabled,
            "backend": FACE_DETAIL_BACKEND if settings.enabled else "disabled",
            "status": "disabled",
            "detection_count": 0,
            "selected_count": 0,
            "refined_count": 0,
            "settings": {
                "steps": settings.steps,
                "denoise": settings.denoise,
                "crop_size": settings.crop_size,
                "padding": settings.padding,
                "feather": settings.feather,
                "blend": settings.blend,
                "lora_scale": settings.lora_scale,
                "detector_threshold": settings.detector_threshold,
                "detector_provider": settings.detector_provider,
            },
            "faces": [],
            "detections": [],
        }
        if not settings.enabled:
            return image, summary

        if manual_face_paths:
            from k2_region_lab.face_detail import DetectedFace
            from k2_region_lab.regions import PixelBox

            detections_list = []
            for path in manual_face_paths:
                if len(path) < 3:
                    raise ValueError("a manual face lasso requires at least three points")
                points = tuple(
                    (
                        min(max(float(x), 0.0), float(image.width)),
                        min(max(float(y), 0.0), float(image.height)),
                    )
                    for x, y in path
                )
                x_values = [point[0] for point in points]
                y_values = [point[1] for point in points]
                box = PixelBox(
                    min(x_values), min(y_values), max(x_values), max(y_values)
                )
                if box.width < 2.0 or box.height < 2.0:
                    raise ValueError("a manual face lasso must enclose a visible area")
                detections_list.append(DetectedFace(box, 1.0, points))
            detections = tuple(detections_list)
            summary["detector"] = "manual_lasso"
            summary["detector_execution_provider"] = None
        else:
            detector_path = getattr(
                self, "face_detector_path", None
            ) or discover_face_detector(self.comfyui_root)
            if detector_path is None:
                raise RuntimeError(
                    "automatic face detailing is enabled, but the bundled NanoDet "
                    "face_det.onnx model was not found under the configured ComfyUI root"
                )
            summary["detector"] = str(detector_path)
            detector = OnnxNanoFaceDetector(
                detector_path,
                threshold=settings.detector_threshold,
                provider=settings.detector_provider,
            )
            detections = detector.detect(image)
            summary["detector_execution_provider"] = detector.execution_provider
        summary["detection_count"] = len(detections)
        summary["detections"] = [
            {
                "index": index,
                "box": [face.box.x0, face.box.y0, face.box.x1, face.box.y1],
                "score": face.score,
                "manual_lasso": [list(point) for point in face.mask_points],
            }
            for index, face in enumerate(detections)
        ]
        if selected_face_indices is None:
            selected_detections = detections
            selected_indices = tuple(range(len(detections)))
        else:
            requested = set(selected_face_indices)
            selected_indices = tuple(
                index for index in range(len(detections)) if index in requested
            )
            selected_detections = tuple(detections[index] for index in selected_indices)
        summary["selected_indices"] = list(selected_indices)
        summary["selected_count"] = len(selected_detections)
        targets = assign_faces_to_regional_loras(selected_detections, regions, loras)
        if event is not None:
            event(
                f"{'Manual lasso supplied' if manual_face_paths else 'Face detector found'} "
                f"{len(detections)} face(s); "
                f"{len(selected_detections)} selected; "
                f"{len(targets)} matched a regional LoRA",
                {
                    "face_detail": {
                        "detection_count": len(detections),
                        "selected_count": len(selected_detections),
                        "target_count": len(targets),
                    }
                },
            )
        if not detections:
            summary["status"] = "no_faces_detected"
            return image, summary
        if not selected_detections:
            summary["status"] = "no_faces_selected"
            return image, summary
        if not targets:
            summary["status"] = "no_regional_lora_faces"
            return image, summary

        result = image.convert("RGB")
        face_reports: list[dict[str, Any]] = []
        for index, target in enumerate(targets):
            crop_box = expanded_square_crop(
                target.face.box,
                image.width,
                image.height,
                settings.padding,
            )
            source_crop = result.crop(crop_box).resize(
                (settings.crop_size, settings.crop_size), Image.Resampling.LANCZOS
            )
            detail_seed = (seed + 104729 * (index + 1)) % 2_147_483_648
            identity_triggers = character_identity_triggers(target.loras).get(
                target.region_id, ()
            )
            detail_prompt = character_identity_prompt(
                target.prompt, identity_triggers
            )
            detail_loras = []
            strengths = []
            for specification in target.loras:
                original_strength = float(specification.get("strength", 1.0))
                requested_strength = original_strength * settings.lora_scale
                effective_strength = max(-4.0, min(4.0, requested_strength))
                detail_loras.append(
                    {
                        **specification,
                        "strength": effective_strength,
                        "global": True,
                        "region_ids": [],
                    }
                )
                strengths.append(
                    {
                        "id": str(specification.get("id", "LoRA")),
                        "name": str(specification.get("name", "LoRA")),
                        "source": original_strength,
                        "requested": requested_strength,
                        "effective": effective_strength,
                    }
                )
            if event is not None:
                event(
                    f"Face detail {index + 1}/{len(targets)} started for "
                    f"{target.region_name}",
                    {
                        "face_detail": {
                            "region_id": target.region_id,
                            "crop_box": list(crop_box),
                            "seed": detail_seed,
                        }
                    },
                )
            refined_crop, lora_reports = self._refine_face_crop(
                source_crop,
                prompt=detail_prompt,
                loras=detail_loras,
                settings=settings,
                seed=detail_seed,
                event=event,
            )
            result = composite_face_crop(
                result,
                refined_crop,
                crop_box,
                settings.feather,
                settings.blend,
                target.face.mask_points,
            )
            report = {
                "region_id": target.region_id,
                "region_name": target.region_name,
                "detected_box": [
                    target.face.box.x0,
                    target.face.box.y0,
                    target.face.box.x1,
                    target.face.box.y1,
                ],
                "detector_score": target.face.score,
                "manual_lasso": [list(point) for point in target.face.mask_points],
                "crop_box": list(crop_box),
                "seed": detail_seed,
                "prompt": detail_prompt,
                "strengths": strengths,
                "loras": lora_reports,
            }
            face_reports.append(report)
            if event is not None:
                event(
                    f"Face detail {index + 1}/{len(targets)} completed for "
                    f"{target.region_name}",
                    {"face_detail": report},
                )
        summary["status"] = "complete"
        summary["refined_count"] = len(face_reports)
        summary["faces"] = face_reports
        return result, summary

    def _refine_face_crop(
        self,
        crop,
        *,
        prompt: str,
        loras: list[dict[str, Any]],
        settings: FaceDetailSettings,
        seed: int,
        event: Callable[[str, dict[str, Any]], None] | None,
    ):
        import numpy as np
        import torch
        from PIL import Image

        import comfy.model_management
        import comfy.sample

        self._ensure_memory("before face-detail text encoding", event)
        positive = self.clip.encode_from_tokens_scheduled(self.clip.tokenize(prompt))
        negative = self.clip.encode_from_tokens_scheduled(self.clip.tokenize(""))
        if not positive:
            raise RuntimeError("Krea text encoder returned no face-detail conditioning")
        text_token_counts = {int(condition[0].shape[1]) for condition in positive}
        if len(text_token_counts) != 1:
            raise RuntimeError("Krea face-detail conditioning must use one text sequence length")
        text_token_count = text_token_counts.pop()

        pixels = torch.from_numpy(
            np.asarray(crop.convert("RGB"), dtype=np.float32).copy() / 255.0
        ).unsqueeze(0)
        self._ensure_memory("before face-detail VAE encode", event)
        latent = self._encode_vae(pixels)
        latent = comfy.sample.fix_empty_latent_channels(
            self.model, latent, downscale_ratio_spacial=8
        )
        noise = comfy.sample.prepare_noise(latent, seed)
        self._ensure_memory("before face-detail denoising", event)
        generation_model, lora_reports, _statistics = self._apply_routed_loras(
            loras,
            base_model=self.model,
            width=settings.crop_size,
            height=settings.crop_size,
            text_token_count=text_token_count,
            regional_plan=None,
            bound_plan=None,
            event=event,
        )

        def callback(step: int, denoised, current, total: int) -> None:
            del denoised, current
            snapshot = self.memory_snapshot(
                f"face-detail denoising step {step + 1}/{total}"
            )
            if snapshot["gpu_free_bytes"] < snapshot["critical_free_bytes"]:
                raise CriticalGpuMemoryPressure(
                    "critical GPU memory pressure after face-detail denoising step "
                    f"{step + 1}/{total}: "
                    f"{snapshot['gpu_free_bytes'] / GIB:.2f} GiB free"
                )

        try:
            samples = comfy.sample.sample(
                generation_model,
                noise,
                settings.steps,
                1.0,
                "euler",
                "simple",
                positive,
                negative,
                latent,
                denoise=settings.denoise,
                callback=callback,
                disable_pbar=True,
                seed=seed,
            )
        except Exception:
            self._release_generation_model(generation_model)
            raise
        self._ensure_memory("before face-detail VAE decode", event)
        self._prepare_vae_handoff(generation_model, event)
        images = self._decode_vae(samples)
        image_tensor = images[0]
        while image_tensor.ndim > 3 and image_tensor.shape[0] == 1:
            image_tensor = image_tensor[0]
        if image_tensor.ndim != 3 or image_tensor.shape[-1] != 3:
            raise RuntimeError(
                f"unexpected face-detail decoded image shape: {tuple(images.shape)}"
            )
        array = (
            image_tensor.detach()
            .to(device="cpu", dtype=torch.float32)
            .clamp(0, 1)
            .numpy()
            * 255.0
        ).round().astype(np.uint8)
        return Image.fromarray(array), lora_reports

    def _encode_vae(self, pixels):
        import torch

        # ComfyUI may load VAE parameters as inference tensors. Autograd cannot
        # save those tensors for a backward pass, and face refinement never needs
        # gradients, so keep regular and tiled VAE encoding under no-grad.
        with torch.no_grad():
            return self.vae.encode(pixels)

    def _decode_vae(self, samples):
        import torch

        # ComfyUI's tiled fallback normalizes with in-place tensor operations.
        # PyTorch 2.10 requires those operations to remain inside inference mode
        # when the tiled accumulator was created as an inference tensor.
        with torch.inference_mode():
            return self.vae.decode(samples)

    @staticmethod
    def _regional_summary(regional_plan, bound_plan, attention_override):
        if regional_plan is None or not (
            regional_plan.regions or regional_plan.emphases
        ):
            return {"backend": "disabled", "region_count": 0}
        summary = regional_plan.summary()
        if bound_plan is not None:
            summary["text_token_count"] = bound_plan.text_token_count
            token_spans = {
                span.region_id: [span.start, span.end] for span in bound_plan.spans
            }
            for region in summary["regions"]:
                region["text_token_span"] = token_spans[region["id"]]
            summary["emphases"] = [
                {
                    "scope_id": emphasis.scope_id,
                    "phrase": emphasis.phrase,
                    "strength": emphasis.strength,
                    "text_token_span": [emphasis.start, emphasis.end],
                }
                for emphasis in bound_plan.emphases
            ]
        if attention_override is not None:
            summary["attention_calls"] = attention_override.matched_calls
            summary["text_refiner_attention_calls"] = attention_override.text_refiner_calls
            summary["attention_implementation"] = "chunked-exact-softmax-v1"
            summary["attention_query_chunk_size"] = (
                attention_override.query_chunk_size
            )
            summary["lora_delta_adaptation"] = attention_override.summary()
            summary["text_partition"] = "subject_keys_private_to_region"
            summary["subject_box_exclusion"] = True
            summary["cross_modal_partition"] = "subject_text_private_to_box"
            summary["image_to_image_attention"] = "unmodified"
        return summary
