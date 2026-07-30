from __future__ import annotations

import json
import platform
import zipfile
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from uuid import uuid4

from k2core.inference import BackendName


SCHEMA_VERSION = "k2lab-issue-report/1"
SAFE_DIAGNOSTIC_LABELS = frozenset(
    {
        "Selected backend",
        "Backend version",
        "Capabilities",
        "Unsupported",
        "Model hashes",
        "Dtype",
        "Device placement",
        "Attention",
        "Model memory",
        "Last parity status",
    }
)


def _package_version(name: str) -> str:
    try:
        return version(name)
    except PackageNotFoundError:
        return "source checkout"


def create_issue_report_bundle(
    *,
    data_directory: Path,
    backend_name: BackendName,
    diagnostics: list[dict[str, str]],
) -> Path:
    """Write a bounded, prompt-safe support bundle and return its absolute path."""

    created_at = datetime.now(UTC)
    report = {
        "schema_version": SCHEMA_VERSION,
        "created_at": created_at.isoformat(),
        "backend": backend_name.value,
        "versions": {
            "k2lab": _package_version("k2-region-lab"),
            "k2core": _package_version("k2core"),
            "python": platform.python_version(),
            "qt": _package_version("PySide6"),
        },
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
        },
        "diagnostics": [
            {"label": str(row["label"]), "value": str(row["value"])}
            for row in diagnostics
            if str(row.get("label")) in SAFE_DIAGNOSTIC_LABELS
        ],
        "privacy": {
            "full_prompts_included": False,
            "environment_variables_included": False,
            "debug_logs_included": False,
            "user_file_paths_included": False,
            "lora_names_included": False,
            "note": (
                "Attach logs or reproduction projects separately only after reviewing "
                "them for private prompts and paths."
            ),
        },
    }
    encoded = (json.dumps(report, indent=2, sort_keys=True) + "\n").encode("utf-8")
    readme = (
        "K2 Region Lab prompt-safe issue report\n\n"
        "This bundle intentionally excludes prompts, environment variables, logs, "
        "user paths, images, projects, and LoRA names. Review issue-report.json "
        "before sharing it.\n"
    )
    output_directory = data_directory.expanduser().resolve() / "issue-reports"
    output_directory.mkdir(parents=True, exist_ok=True)
    suffix = uuid4().hex[:8]
    path = output_directory / (
        f"k2lab-issue-report-{created_at:%Y%m%dT%H%M%SZ}-{suffix}.zip"
    )
    with zipfile.ZipFile(
        path,
        mode="x",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=9,
    ) as archive:
        archive.writestr("issue-report.json", encoded)
        archive.writestr("README.txt", readme.encode("utf-8"))
    return path
