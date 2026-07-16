from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from k2_region_lab.model.artifacts import ArtifactKind, ModelArtifact, read_safetensors_header


@dataclass(frozen=True, slots=True)
class ManifestResult:
    kind: ArtifactKind
    source_path: Path
    manifest_path: Path
    tensor_count: int
    compatible: bool
    errors: tuple[str, ...]
    warnings: tuple[str, ...]

    def to_payload(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["kind"] = self.kind.value
        payload["source_path"] = str(self.source_path)
        payload["manifest_path"] = str(self.manifest_path)
        return payload


EXPECTED_SHAPES: dict[ArtifactKind, dict[str, list[int]]] = {
    ArtifactKind.TRANSFORMER: {
        "first.weight": [6144, 64],
        "blocks.0.attn.wq.weight": [6144, 6144],
        "blocks.0.attn.wk.weight": [1536, 6144],
        "blocks.0.attn.wv.weight": [1536, 6144],
        "blocks.27.attn.wq.weight": [6144, 6144],
        "txtfusion.projector.weight": [1, 12],
        "txtfusion.layerwise_blocks.0.prenorm.scale": [2560],
        "last.linear.weight": [64, 6144],
    },
    ArtifactKind.TEXT_ENCODER: {
        "model.embed_tokens.weight": [151936, 2560],
        "model.layers.0.self_attn.q_proj.weight": [4096, 2560],
        "model.layers.35.self_attn.q_proj.weight": [4096, 2560],
        "model.norm.weight": [2560],
    },
    ArtifactKind.VAE: {
        "encoder.conv1.weight": [96, 3, 3, 3, 3],
        "decoder.conv1.weight": [384, 16, 3, 3, 3],
        "conv1.weight": [32, 32, 1, 1, 1],
        "conv2.weight": [16, 16, 1, 1, 1],
    },
}


def _indexed_count(tensor_names: set[str], expression: str) -> int:
    matcher = re.compile(expression)
    return len({int(match.group(1)) for name in tensor_names if (match := matcher.match(name))})


def _validate(kind: ArtifactKind, tensors: dict[str, Any]) -> tuple[list[str], list[str]]:
    errors: list[str] = []
    warnings: list[str] = []
    for name, expected_shape in EXPECTED_SHAPES[kind].items():
        descriptor = tensors.get(name)
        if descriptor is None:
            errors.append(f"missing required tensor: {name}")
            continue
        observed = descriptor.get("shape") if isinstance(descriptor, dict) else None
        if observed != expected_shape:
            errors.append(f"shape mismatch for {name}: expected {expected_shape}, got {observed}")

    names = set(tensors)
    if kind == ArtifactKind.TRANSFORMER:
        count = _indexed_count(names, r"blocks\.(\d+)\.")
        if count != 28:
            errors.append(f"expected 28 Krea transformer blocks, found {count}")
        if not any(name.endswith(".weight_scale") for name in names):
            warnings.append("transformer has no scaled-FP8 weight_scale tensors")
    elif kind == ArtifactKind.TEXT_ENCODER:
        count = _indexed_count(names, r"model\.layers\.(\d+)\.")
        if count != 36:
            errors.append(f"expected 36 Qwen3-VL text layers, found {count}")
    return errors, warnings


def build_tensor_manifest(artifact: ModelArtifact, output_directory: Path) -> ManifestResult:
    header = read_safetensors_header(artifact.path)
    metadata = header.get("__metadata__", {})
    tensors = {name: value for name, value in header.items() if name != "__metadata__"}
    errors, warnings = _validate(artifact.kind, tensors)
    output_directory.mkdir(parents=True, exist_ok=True)
    output_path = output_directory / f"{artifact.kind.value}_tensor_manifest.json"
    document = {
        "schema_version": "k2lab-tensor-manifest/1",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "artifact_kind": artifact.kind.value,
        "source_path": str(artifact.path),
        "source_size_bytes": artifact.size_bytes,
        "metadata": metadata,
        "compatible": not errors,
        "errors": errors,
        "warnings": warnings,
        "tensor_count": len(tensors),
        "tensors": [
            {
                "name": name,
                "dtype": descriptor.get("dtype"),
                "shape": descriptor.get("shape"),
                "data_offsets": descriptor.get("data_offsets"),
            }
            for name, descriptor in tensors.items()
            if isinstance(descriptor, dict)
        ],
    }
    output_path.write_text(json.dumps(document, indent=2), encoding="utf-8")
    return ManifestResult(
        kind=artifact.kind,
        source_path=artifact.path,
        manifest_path=output_path,
        tensor_count=len(tensors),
        compatible=not errors,
        errors=tuple(errors),
        warnings=tuple(warnings),
    )
