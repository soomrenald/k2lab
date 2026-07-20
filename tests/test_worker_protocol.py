from __future__ import annotations

import json
import os
import struct
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from k2_region_lab.worker.protocol import CommandKind


def descriptor(shape: list[int], dtype: str = "BF16") -> dict:
    return {"dtype": dtype, "shape": shape, "data_offsets": [0, 2]}


def write_header(path: Path, tensors: dict) -> None:
    encoded = json.dumps(tensors, separators=(",", ":")).encode("utf-8")
    path.write_bytes(struct.pack("<Q", len(encoded)) + encoded)


def write_compatible_artifacts(root: Path) -> tuple[Path, Path, Path]:
    diffusion = root / "diffusion_models"
    text = root / "text_encoders"
    vae = root / "vae"
    diffusion.mkdir()
    text.mkdir()
    vae.mkdir()

    transformer = {
        "first.weight": descriptor([6144, 64]),
        "blocks.0.attn.wq.weight": descriptor([6144, 6144], "F8_E4M3"),
        "blocks.0.attn.wk.weight": descriptor([1536, 6144], "F8_E4M3"),
        "blocks.0.attn.wv.weight": descriptor([1536, 6144], "F8_E4M3"),
        "blocks.27.attn.wq.weight": descriptor([6144, 6144], "F8_E4M3"),
        "txtfusion.projector.weight": descriptor([1, 12]),
        "txtfusion.layerwise_blocks.0.prenorm.scale": descriptor([2560]),
        "last.linear.weight": descriptor([64, 6144]),
        "blocks.0.attn.wq.weight_scale": descriptor([], "F32"),
    }
    for index in range(1, 27):
        transformer[f"blocks.{index}.placeholder"] = descriptor([1])
    write_header(diffusion / "krea2_turbo_fp8_scaled.safetensors", transformer)

    text_tensors = {
        "model.embed_tokens.weight": descriptor([151936, 2560]),
        "model.layers.0.self_attn.q_proj.weight": descriptor([4096, 2560], "F8_E4M3"),
        "model.layers.35.self_attn.q_proj.weight": descriptor([4096, 2560], "F8_E4M3"),
        "model.norm.weight": descriptor([2560]),
    }
    for index in range(1, 35):
        text_tensors[f"model.layers.{index}.placeholder"] = descriptor([1])
    write_header(text / "qwen3vl_4b_fp8_scaled.safetensors", text_tensors)

    write_header(
        vae / "qwen_image_vae.safetensors",
        {
            "encoder.conv1.weight": descriptor([96, 3, 3, 3, 3]),
            "decoder.conv1.weight": descriptor([384, 16, 3, 3, 3]),
            "conv1.weight": descriptor([32, 32, 1, 1, 1]),
            "conv2.weight": descriptor([16, 16, 1, 1, 1]),
        },
    )
    return diffusion, text, vae


class WorkerProtocolTests(unittest.TestCase):
    def test_image_edit_has_a_dedicated_worker_command(self) -> None:
        self.assertEqual(CommandKind.EDIT_IMAGE.value, "edit_image")

    def test_external_worker_probes_validates_and_stops(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            diffusion, text, vae = write_compatible_artifacts(root)
            manifests = root / "manifests"
            payload = {
                "comfyui_root": str(root),
                "diffusion_models": str(diffusion),
                "text_encoders": str(text),
                "vae": str(vae),
                "manifest_directory": str(manifests),
            }
            commands = [
                {"command_id": "probe", "kind": "probe", "payload": payload},
                {
                    "command_id": "diagnose",
                    "kind": "diagnose_accelerator",
                    "payload": payload,
                },
                {"command_id": "validate", "kind": "validate_models", "payload": payload},
                {"command_id": "stop", "kind": "shutdown", "payload": payload},
            ]
            environment = os.environ.copy()
            project_root = Path(__file__).resolve().parents[1]
            environment["PYTHONPATH"] = str(project_root / "src")
            process = subprocess.run(
                [sys.executable, "-m", "k2_region_lab.worker.entrypoint"],
                input="".join(json.dumps(command) + "\n" for command in commands),
                text=True,
                capture_output=True,
                env=environment,
                cwd=project_root,
                timeout=15,
                check=False,
            )
            self.assertEqual(process.returncode, 0, process.stderr)
            events = [json.loads(line) for line in process.stdout.splitlines()]
            validation = next(
                event
                for event in events
                if event["command_id"] == "validate" and event["state"] == "ready"
            )
            self.assertTrue(validation["payload"]["complete"])
            self.assertTrue(all(item["compatible"] for item in validation["payload"]["manifests"]))
            self.assertEqual(len(tuple(manifests.glob("*_tensor_manifest.json"))), 3)
            diagnostic = next(
                event
                for event in events
                if event["command_id"] == "diagnose"
                and event["message"] == "Accelerator diagnostics complete"
            )
            self.assertIn("python_executable", diagnostic["payload"])
            self.assertIn("device_paths", diagnostic["payload"])
            self.assertTrue(diagnostic["payload"]["recommendations"])


if __name__ == "__main__":
    unittest.main()
