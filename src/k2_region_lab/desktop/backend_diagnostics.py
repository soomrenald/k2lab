from __future__ import annotations

from dataclasses import replace
from importlib.metadata import PackageNotFoundError, version
from types import SimpleNamespace
from typing import Any

from k2core.backends import BackendCapabilities, ComfyUIBackend, NativeK2Backend
from k2core.inference import BackendName
from k2core.model import RegisteredModel, load_model_registry


LAST_NATIVE_PARITY_STATUS = (
    "Gate 12 100-job soak PASS; overall release readiness blocked (2026-07-29)"
)


def _package_version(name: str) -> str:
    try:
        return version(name)
    except PackageNotFoundError:
        return "source checkout"


def _registered_model(settings) -> RegisteredModel | None:
    path = settings.model_registry_path
    if path is None:
        return None
    try:
        registry = load_model_registry(path)
    except (OSError, RuntimeError, TypeError, ValueError):
        return None
    supplied_name = settings.registered_model_name.strip().casefold()
    if supplied_name:
        return next(
            (
                model
                for model in registry.models
                if model.name.casefold() == supplied_name
            ),
            None,
        )
    return registry.models[0] if len(registry.models) == 1 else None


def selected_backend_capabilities(
    backend_name: BackendName,
    *,
    loaded: bool = False,
) -> BackendCapabilities:
    if backend_name is BackendName.NATIVE:
        return NativeK2Backend().capabilities()
    capabilities = ComfyUIBackend(SimpleNamespace(loaded=loaded)).capabilities()
    return replace(
        capabilities,
        modes=capabilities.modes
        | frozenset({"ordinary_lora", "post_upscale", "projector"}),
    )


def _device_plan(load_payload: dict[str, Any]) -> dict[str, Any]:
    value = load_payload.get("device_plan")
    return dict(value) if isinstance(value, dict) else {}


def backend_diagnostic_rows(
    *,
    backend_name: BackendName,
    settings,
    load_payload: dict[str, Any] | None = None,
    loaded_loras: tuple[str, ...] = (),
    memory_text: str = "Not measured",
) -> list[dict[str, str]]:
    """Build prompt-safe developer diagnostics without loading or hashing model tensors."""

    payload = dict(load_payload or {})
    plan = _device_plan(payload)
    model = _registered_model(settings)
    if model is None:
        hashes = "Unavailable — configure one validated model registry entry"
    else:
        hashes = "\n".join(
            (
                f"transformer {model.transformer.sha256}",
                f"text encoder {model.text_encoder.sha256}",
                f"VAE {model.vae.sha256}",
                (
                    f"tokenizer {model.tokenizer.sha256}"
                    if model.tokenizer is not None
                    else "tokenizer not registered"
                ),
            )
        )
    if plan:
        placement = ", ".join(
            (
                f"transformer={plan.get('transformer_device', 'unknown')}",
                f"text={plan.get('text_encoder_device', 'unknown')}",
                f"VAE={plan.get('vae_device', 'unknown')}",
            )
        )
        dtype = (
            f"compute={plan.get('compute_dtype', 'unknown')}, "
            f"weights={plan.get('weight_dtype', 'unknown')}"
        )
    elif backend_name is BackendName.NATIVE:
        placement = "Resolved after native model load"
        dtype = "Resolved after native model load"
    else:
        placement = "Owned by the selected ComfyUI runtime"
        dtype = "Owned by checkpoint and ComfyUI runtime"
    capabilities = selected_backend_capabilities(
        backend_name,
        loaded=bool(payload),
    )
    unsupported = (
        "pose control, projector, post-upscale, face refinement"
        if backend_name is BackendName.NATIVE
        else "None in the current production UI"
    )
    attention = (
        "K2 exact chunked softmax override"
        if backend_name is BackendName.NATIVE
        else "ComfyUI optimized-attention override"
    )
    return [
        {"label": "Selected backend", "value": backend_name.value},
        {"label": "Backend version", "value": f"k2core {_package_version('k2core')}"},
        {"label": "Capabilities", "value": ", ".join(sorted(capabilities.modes))},
        {"label": "Unsupported", "value": unsupported},
        {"label": "Model hashes", "value": hashes},
        {"label": "Dtype", "value": dtype},
        {"label": "Device placement", "value": placement},
        {"label": "Attention", "value": attention},
        {
            "label": "Loaded LoRAs",
            "value": ", ".join(loaded_loras) if loaded_loras else "None",
        },
        {"label": "Model memory", "value": memory_text},
        {
            "label": "Last parity status",
            "value": (
                LAST_NATIVE_PARITY_STATUS
                if backend_name is BackendName.NATIVE
                else "Reference production backend"
            ),
        },
    ]
