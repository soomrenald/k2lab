from __future__ import annotations

from math import isfinite
from typing import Final


PROJECTOR_VECTOR_COUNT: Final = 12
CUSTOM_PROJECTOR_PRESET: Final = "custom"
DEFAULT_PROJECTOR_PRESET: Final = "filter_bypass2"

# These are the 12-column txtfusion.projector deltas from the supplied reference
# table. They are independent from regional LoRA gates; an optional text-token mask
# can protect explicit face identity prompt spans from some or all of the delta.
PROJECTOR_PRESETS: Final[dict[str, tuple[float, ...]]] = {
    "filter_bypass2": (
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        -0.5117,
        -0.8906,
        0.0,
        0.0,
    ),
    "filter_bypass3": (
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        -0.5117,
        -0.8906,
        -0.6094,
        0.0,
    ),
    "skc3vo": (
        -5.4400,
        -16.1100,
        -37.1100,
        -50.3900,
        -70.7000,
        -39.4500,
        -39.8400,
        -143.7511,
        -51.1700,
        -89.0600,
        -60.9400,
        -11.2800,
    ),
    "z0jglf": (
        -13.6000,
        -40.2750,
        -92.7750,
        -159.7500,
        -176.7500,
        -98.6250,
        -99.6000,
        -359.3778,
        -127.9250,
        -222.6500,
        -152.3500,
        -28.2000,
    ),
}

PROJECTOR_PRESET_LABELS: Final[dict[str, str]] = {
    "filter_bypass2": "FilterBypass2",
    "filter_bypass3": "FilterBypass3",
    "skc3vo": "skc3vo",
    "z0jglf": "z0jglf",
    CUSTOM_PROJECTOR_PRESET: "Custom values",
}


def projector_preset_values(preset: str) -> tuple[float, ...]:
    try:
        return PROJECTOR_PRESETS[preset]
    except KeyError as error:
        raise ValueError(f"unknown projector preset: {preset!r}") from error


def validate_projector_values(values) -> tuple[float, ...]:
    vector = tuple(float(value) for value in values)
    if len(vector) != PROJECTOR_VECTOR_COUNT:
        raise ValueError(
            f"projector vector must contain {PROJECTOR_VECTOR_COUNT} values"
        )
    if not all(isfinite(value) for value in vector):
        raise ValueError("projector values must be finite")
    return vector


def effective_projector_values(values, multiplier: float) -> tuple[float, ...]:
    vector = validate_projector_values(values)
    scale = float(multiplier)
    if not isfinite(scale):
        raise ValueError("projector multiplier must be finite")
    return tuple(value * scale for value in vector)


def projector_token_delta_mask(
    text_token_count: int,
    protected_spans: tuple[tuple[int, int], ...],
    protection: float,
) -> tuple[float, ...]:
    """Scale the projector delta while preserving selected identity tokens.

    A value of one applies the complete projector preset delta. A fully protected
    token receives zero projector delta and therefore uses the baseline layer mix.
    """

    if text_token_count <= 0:
        raise ValueError("projector token mask requires a positive token count")
    amount = float(protection)
    if not 0.0 <= amount <= 1.0:
        raise ValueError("projector identity protection must be between zero and one")
    mask = [1.0] * text_token_count
    for start, end in protected_spans:
        if start < 0 or end <= start or end > text_token_count:
            raise ValueError("projector protected token span is outside the text sequence")
        for index in range(start, end):
            mask[index] = min(mask[index], 1.0 - amount)
    return tuple(mask)
