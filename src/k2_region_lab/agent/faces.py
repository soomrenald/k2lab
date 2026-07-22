from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from typing import Awaitable, Callable

from k2_region_lab.agent.domain import (
    DetectedFaceRecord,
    FaceDetectionRequest,
    FaceDetectionResult,
    FileKind,
)
from k2_region_lab.agent.jobs import JobError
from k2_region_lab.agent.storage import WorkspaceLayout
from k2_region_lab.agent.transfers import TransferManager


FaceDetectionRunner = Callable[[list[str], dict[str, str]], Awaitable[tuple[int, bytes, bytes]]]


class FaceDetectionService:
    def __init__(
        self,
        layout: WorkspaceLayout,
        transfers: TransferManager,
        *,
        worker_python: Path,
        comfyui_root: Path,
        runner: FaceDetectionRunner | None = None,
    ) -> None:
        self._layout = layout
        self._transfers = transfers
        self._worker_python = worker_python
        self._comfyui_root = comfyui_root
        self._runner = runner or self._run_subprocess
        self._requires_worker_binary = runner is None

    async def detect(self, request: FaceDetectionRequest) -> FaceDetectionResult:
        record, image_path = await self._transfers.resolve_file(request.input_file_id)
        if record.kind not in {FileKind.INPUTS, FileKind.OUTPUTS}:
            raise JobError("input_file_invalid", "Face detection requires an input image.")
        if self._requires_worker_binary and (
            not self._worker_python.is_file() or not os.access(self._worker_python, os.X_OK)
        ):
            raise JobError(
                "worker_unavailable",
                "The configured GPU worker runtime is unavailable.",
                503,
            )
        command = [
            str(self._worker_python),
            "-m",
            "k2_region_lab.worker.detect_faces",
            "--image",
            str(image_path),
            "--comfyui-root",
            str(self._comfyui_root),
            "--threshold",
            str(request.threshold),
            "--provider",
            request.provider,
        ]
        detector_path: Path | None = None
        if request.face_detector_file_id:
            _record, detector_path = await self._transfers.resolve_file(
                request.face_detector_file_id, required_kind=FileKind.FACE_DETECTION
            )
        else:
            detector_files = sorted(
                path
                for path in self._layout.destination(FileKind.FACE_DETECTION.value).iterdir()
                if path.is_file() and not path.is_symlink()
            )
            detector_path = detector_files[0] if detector_files else None
        if detector_path is not None:
            command.extend(("--detector-path", str(detector_path)))
        code, stdout, _stderr = await self._runner(command, self._worker_environment())
        if code != 0:
            raise JobError(
                "face_detection_failed",
                "Face detection failed in the isolated worker.",
                500,
            )
        try:
            payload = json.loads(stdout.decode("utf-8"))
            faces = [
                DetectedFaceRecord(
                    index=index,
                    box=item["box"],
                    score=item["score"],
                )
                for index, item in enumerate(payload["faces"])
            ]
            return FaceDetectionResult(
                width=payload["width"],
                height=payload["height"],
                execution_provider=str(payload["execution_provider"]),
                faces=faces,
            )
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
            raise JobError(
                "face_detection_response_invalid",
                "Face detection returned an invalid response.",
                500,
            ) from error

    @staticmethod
    async def _run_subprocess(
        command: list[str], environment: dict[str, str]
    ) -> tuple[int, bytes, bytes]:
        process = await asyncio.create_subprocess_exec(
            *command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=environment,
        )
        try:
            stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=120)
        except TimeoutError as error:
            process.kill()
            await process.wait()
            raise JobError("face_detection_timeout", "Face detection timed out.", 504) from error
        return process.returncode or 0, stdout[:4_194_304], stderr[:65_536]

    @staticmethod
    def _worker_environment() -> dict[str, str]:
        allowed_names = {
            "HOME",
            "LANG",
            "LC_ALL",
            "LD_LIBRARY_PATH",
            "PATH",
            "PYTHONPATH",
            "TMPDIR",
            "VIRTUAL_ENV",
        }
        allowed_prefixes = ("CUDA_", "HIP_", "NVIDIA_", "ROCM_")
        return {
            key: value
            for key, value in os.environ.items()
            if key in allowed_names or key.startswith(allowed_prefixes)
        }
