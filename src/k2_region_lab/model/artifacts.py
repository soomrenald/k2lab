from __future__ import annotations

import json
import struct
from collections import Counter
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any, Iterable

from k2_region_lab.config import ModelDirectories


MAX_HEADER_BYTES = 128 * 1024 * 1024


class ArtifactKind(StrEnum):
    TRANSFORMER = "transformer"
    TEXT_ENCODER = "text_encoder"
    VAE = "vae"


@dataclass(frozen=True, slots=True)
class SafetensorsSummary:
    tensor_count: int
    dtypes: tuple[tuple[str, int], ...]
    metadata_keys: tuple[str, ...]
    format_name: str | None
    quantized: bool


@dataclass(frozen=True, slots=True)
class ModelArtifact:
    kind: ArtifactKind
    path: Path
    size_bytes: int
    summary: SafetensorsSummary


@dataclass(frozen=True, slots=True)
class ArtifactSet:
    transformer: ModelArtifact | None
    text_encoder: ModelArtifact | None
    vae: ModelArtifact | None

    @property
    def complete(self) -> bool:
        return all((self.transformer, self.text_encoder, self.vae))

    def present(self) -> tuple[ModelArtifact, ...]:
        return tuple(
            artifact
            for artifact in (self.transformer, self.text_encoder, self.vae)
            if artifact is not None
        )


def read_safetensors_header(path: Path) -> dict[str, Any]:
    with path.open("rb") as handle:
        length_bytes = handle.read(8)
        if len(length_bytes) != 8:
            raise ValueError(f"{path} is too small to be a safetensors file")
        header_length = struct.unpack("<Q", length_bytes)[0]
        if header_length <= 1 or header_length > MAX_HEADER_BYTES:
            raise ValueError(f"{path} has an invalid safetensors header length: {header_length}")
        encoded = handle.read(header_length)
        if len(encoded) != header_length:
            raise ValueError(f"{path} has a truncated safetensors header")
    header = json.loads(encoded)
    if not isinstance(header, dict):
        raise ValueError(f"{path} safetensors header is not an object")
    return header


def read_safetensors_summary(path: Path) -> SafetensorsSummary:
    """Read metadata without mapping or loading any tensor payload."""

    header = read_safetensors_header(path)
    metadata = header.get("__metadata__", {})
    if not isinstance(metadata, dict):
        metadata = {}
    tensors = {key: value for key, value in header.items() if key != "__metadata__"}
    dtypes = Counter(
        value.get("dtype", "UNKNOWN")
        for value in tensors.values()
        if isinstance(value, dict)
    )
    quantization_text = str(metadata.get("_quantization_metadata", ""))
    quantized = bool(quantization_text) or any(dtype.startswith("F8") for dtype in dtypes)
    return SafetensorsSummary(
        tensor_count=len(tensors),
        dtypes=tuple(sorted(dtypes.items())),
        metadata_keys=tuple(sorted(metadata)),
        format_name=str(metadata["format"]) if "format" in metadata else None,
        quantized=quantized,
    )


def _candidate_files(directory: Path) -> tuple[Path, ...]:
    if not directory.is_dir():
        return ()
    return tuple(sorted(path for path in directory.glob("*.safetensors") if path.is_file()))


def _score_candidate(path: Path, required: Iterable[str], preferred: Iterable[str]) -> int:
    name = path.name.lower()
    required_terms = tuple(required)
    if not all(term in name for term in required_terms):
        return -1
    return 100 * len(required_terms) + sum(10 for term in preferred if term in name)


def _select(
    directory: Path,
    kind: ArtifactKind,
    required: tuple[str, ...],
    preferred: tuple[str, ...],
) -> ModelArtifact | None:
    scored = [
        (_score_candidate(path, required, preferred), path)
        for path in _candidate_files(directory)
    ]
    matches = [(score, path) for score, path in scored if score >= 0]
    if not matches:
        return None
    _, selected = max(matches, key=lambda item: (item[0], item[1].name))
    return ModelArtifact(
        kind=kind,
        path=selected,
        size_bytes=selected.stat().st_size,
        summary=read_safetensors_summary(selected),
    )


def discover_model_artifacts(directories: ModelDirectories) -> ArtifactSet:
    return ArtifactSet(
        transformer=_select(
            directories.diffusion_models,
            ArtifactKind.TRANSFORMER,
            required=("krea",),
            preferred=("krea2", "turbo", "fp8", "scaled"),
        ),
        text_encoder=_select(
            directories.text_encoders,
            ArtifactKind.TEXT_ENCODER,
            required=("qwen",),
            preferred=("qwen3vl", "4b", "fp8", "scaled"),
        ),
        vae=_select(
            directories.vae,
            ArtifactKind.VAE,
            required=("vae",),
            preferred=("qwen", "image"),
        ),
    )
