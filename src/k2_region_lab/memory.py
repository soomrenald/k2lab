from __future__ import annotations

from dataclasses import dataclass


GIB = 1024**3


@dataclass(frozen=True, slots=True)
class MemoryPolicy:
    key: str
    label: str
    reserve_vram_gb: float
    warning_free_gb: float
    critical_free_gb: float
    minimum_system_ram_gb: float
    cpu_vae: bool = False
    oom_recovery: bool = True


MEMORY_POLICIES = (
    MemoryPolicy(
        "low_8gb",
        "Low VRAM (8 GB)",
        1.0,
        1.0,
        0.5,
        24.0,
        cpu_vae=True,
    ),
    MemoryPolicy(
        "safe_12gb",
        "Safe 12 GB",
        2.0,
        2.0,
        1.0,
        18.0,
        cpu_vae=True,
    ),
    MemoryPolicy("performance", "Performance", 2.0, 2.0, 1.0, 12.0),
    MemoryPolicy("balanced", "Balanced", 3.0, 3.0, 1.5, 12.0),
    MemoryPolicy("safe_16gb", "Safe 16 GB", 4.0, 4.0, 2.0, 14.0),
    MemoryPolicy("large_24gb", "Large VRAM (24+ GB)", 3.0, 3.0, 1.5, 12.0),
    MemoryPolicy("custom", "Custom / any GPU", 0.5, 0.5, 0.25, 4.0),
    MemoryPolicy(
        "emergency",
        "Emergency",
        5.5,
        5.0,
        2.5,
        16.0,
        cpu_vae=True,
        oom_recovery=False,
    ),
)


def memory_policy(key: str) -> MemoryPolicy:
    for policy in MEMORY_POLICIES:
        if policy.key == key:
            return policy
    raise ValueError(f"unknown memory policy: {key}")


def effective_reserve_vram_gb(key: str, requested_gb: float) -> float:
    """Apply the selected policy's non-negotiable VRAM reserve floor."""
    return max(0.5, float(requested_gb), memory_policy(key).reserve_vram_gb)


def effective_minimum_system_ram_gb(key: str, requested_gb: float) -> float:
    """Apply the selected policy's non-negotiable system-RAM floor."""
    return max(4.0, float(requested_gb), memory_policy(key).minimum_system_ram_gb)


def oom_recovery_reserve_vram_gb(current_gb: float, total_gb: float) -> float:
    """Increase the reserve proportionally without consuming a small GPU.

    The original 16 GiB setup moves from a 4 GiB reserve to 5 GiB. Smaller and
    larger devices receive a bounded increase scaled to their actual capacity.
    """
    current = max(0.5, float(current_gb))
    total = max(1.0, float(total_gb))
    increase = max(0.5, min(1.5, total / 16.0))
    capacity_limit = max(0.5, total * 0.4)
    return current + min(increase, max(0.0, capacity_limit - current))
