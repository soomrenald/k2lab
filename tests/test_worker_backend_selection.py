from __future__ import annotations

import io
import json
import unittest
from contextlib import redirect_stdout
from unittest.mock import patch

from k2_region_lab.worker import entrypoint


class Runtime:
    instances = []

    def __init__(self, comfyui_root, *, face_detector_path=None) -> None:
        self.comfyui_root = comfyui_root
        self.face_detector_path = face_detector_path
        self.loaded = False
        self.load_request = None
        self.generate_request = None
        self.instances.append(self)

    def load(self, artifacts, **request):
        self.loaded = True
        self.load_request = (artifacts, request)
        return {
            "transformer": "ModelPatcher",
            "text_encoder": "CLIP",
            "vae": "VAE",
        }

    def generate(self, **request):
        self.generate_request = request
        request["progress"](1, 1, {"gpu_free_bytes": 10})
        return {
            "image_path": "/tmp/reference.png",
            "width": request["width"],
            "height": request["height"],
            "seed": request["seed"],
        }


def command(command_id: str, kind: str, payload: dict) -> str:
    return json.dumps({"command_id": command_id, "kind": kind, "payload": payload})


class WorkerBackendSelectionTests(unittest.TestCase):
    def setUp(self) -> None:
        Runtime.instances.clear()

    def _run(self, environment: dict[str, str]) -> tuple[int, list[dict]]:
        load_payload = {
            "comfyui_root": "/tmp/ComfyUI",
            "diffusion_models": "/tmp/models/diffusion_models",
            "text_encoders": "/tmp/models/text_encoders",
            "vae": "/tmp/models/vae",
            "manifest_directory": "/tmp/manifests",
        }
        generation_payload = {
            **load_payload,
            "prompt": "synthetic contract fixture",
            "width": 1024,
            "height": 1024,
            "steps": 8,
            "sampler": "euler",
            "scheduler": "simple",
            "seed": 31,
            "output_directory": "/tmp",
            "filename_prefix": "fixture",
            "regions": [],
            "prompt_emphases": [],
            "loras": [],
        }
        encoded = "\n".join(
            (
                command("load-1", "load_model", load_payload),
                command("generate-1", "generate_baseline", generation_payload),
            )
        )
        output = io.StringIO()
        artifact_set = object()
        with (
            patch.dict("os.environ", environment, clear=True),
            patch.object(entrypoint, "configure_debug_logging"),
            patch.object(entrypoint, "ComfyBaselineRuntime", Runtime),
            patch.object(entrypoint, "discover_model_artifacts", return_value=artifact_set),
            patch("sys.stdin", io.StringIO(encoded)),
            redirect_stdout(output),
        ):
            result = entrypoint.main()
        return result, [json.loads(line) for line in output.getvalue().splitlines() if line]

    def test_unset_selector_routes_generation_through_comfyui_backend(self) -> None:
        result, events = self._run({})
        self.assertEqual(result, 0)
        runtime = Runtime.instances[0]
        self.assertEqual(runtime.generate_request["prompt"], "synthetic contract fixture")
        self.assertEqual(runtime.generate_request["sampler"], "euler")
        self.assertEqual(runtime.generate_request["scheduler"], "simple")
        completed = next(item for item in events if item["message"] == "Generation complete")
        self.assertEqual(
            completed["payload"],
            {
                "image_path": "/tmp/reference.png",
                "width": 1024,
                "height": 1024,
                "seed": 31,
            },
        )

    def test_explicit_comfyui_is_the_rollback_path(self) -> None:
        result, events = self._run({"K2LAB_BACKEND": "comfyui"})
        self.assertEqual(result, 0)
        self.assertTrue(any(item["message"] == "Generation complete" for item in events))

    def test_native_selector_fails_explicitly_without_loading_or_fallback(self) -> None:
        result, events = self._run({"K2LAB_BACKEND": "native"})
        self.assertEqual(result, 1)
        self.assertFalse(Runtime.instances[0].loaded)
        failure = next(item for item in events if item["state"] == "error")
        self.assertEqual(
            failure["payload"]["error"]["category"],
            "UnsupportedFeatureError",
        )
        self.assertEqual(failure["payload"]["error"]["backend_name"], "native")

    def test_invalid_selector_returns_structured_configuration_error(self) -> None:
        result, events = self._run({"K2LAB_BACKEND": "automatic"})
        self.assertEqual(result, 1)
        self.assertFalse(Runtime.instances)
        failure = next(item for item in events if item["state"] == "error")
        self.assertEqual(
            failure["payload"]["error"]["category"],
            "ConfigurationError",
        )


if __name__ == "__main__":
    unittest.main()
