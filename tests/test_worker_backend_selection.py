from __future__ import annotations

import io
import json
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from k2_region_lab.worker import entrypoint
from k2core.inference import GenerationRequest, LoadedPipeline, ProgressEvent


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
        self.assertFalse(Runtime.instances)
        failure = next(item for item in events if item["state"] == "error")
        self.assertEqual(
            failure["payload"]["error"]["category"],
            "ConfigurationError",
        )
        self.assertEqual(failure["payload"]["error"]["backend_name"], "native")

    def test_native_selector_loads_the_explicit_registered_model_without_comfy_runtime(
        self,
    ) -> None:
        class NativeBackend:
            instances = []

            def __init__(self) -> None:
                self.pipeline = None
                self.load_config = None
                self.instances.append(self)

            def load(self, config):
                self.load_config = config
                self.pipeline = SimpleNamespace(loaded=True)
                return LoadedPipeline(
                    backend_id="native",
                    metadata={"strict_loading": config.strict_loading},
                )

        registered = SimpleNamespace(name="fixture")
        load_payload = {
            "comfyui_root": "/tmp/ComfyUI",
            "diffusion_models": "/tmp/models/diffusion_models",
            "text_encoders": "/tmp/models/text_encoders",
            "vae": "/tmp/models/vae",
            "model_registry": "/tmp/models.toml",
            "registered_model": "fixture",
            "cpu_offload": True,
            "vae_tiling": True,
        }
        output = io.StringIO()
        with (
            patch.dict("os.environ", {"K2LAB_BACKEND": "native"}, clear=True),
            patch.object(entrypoint, "configure_debug_logging"),
            patch.object(entrypoint, "ComfyBaselineRuntime", Runtime),
            patch.object(entrypoint, "NativeK2Backend", NativeBackend),
            patch.object(
                entrypoint,
                "discover_model_artifacts",
                return_value=SimpleNamespace(complete=True),
            ),
            patch.object(
                entrypoint,
                "load_model_registry",
                return_value=SimpleNamespace(models=(registered,)),
            ),
            patch(
                "sys.stdin",
                io.StringIO(command("load-native", "load_model", load_payload)),
            ),
            redirect_stdout(output),
        ):
            result = entrypoint.main()

        events = [json.loads(line) for line in output.getvalue().splitlines() if line]
        self.assertEqual(result, 0)
        self.assertFalse(Runtime.instances)
        load_config = NativeBackend.instances[0].load_config
        self.assertIs(load_config.registered_model, registered)
        self.assertTrue(load_config.device_policy.cpu_offload)
        self.assertTrue(load_config.device_policy.vae_tiling)
        self.assertTrue(
            any(
                event["state"] == "ready"
                and event["payload"]["strict_loading"] is True
                for event in events
            )
        )

    def test_gate11_fixture_uses_shared_generation_schema_at_desktop_entrypoint(
        self,
    ) -> None:
        fixture_path = (
            Path(__file__).parent
            / "fixtures"
            / "gate11"
            / "native_clean_generation.json"
        )
        generation_fixture = json.loads(fixture_path.read_text(encoding="utf-8"))

        class NativeBackend:
            instances = []

            def __init__(self) -> None:
                self.pipeline = None
                self.generate_request = None
                self.instances.append(self)

            def load(self, config):
                self.pipeline = SimpleNamespace(loaded=True)
                return LoadedPipeline(
                    backend_id="native",
                    metadata={"model_name": config.registered_model.name},
                )

            def generate(self, request, **_callbacks):
                self.generate_request = request
                return SimpleNamespace(
                    to_payload=lambda: {
                        "image_path": (
                            "/workspace/outputs/gate11-native-clean.png"
                        ),
                        "width": request.width,
                        "height": request.height,
                        "seed": request.seed,
                    }
                )

        registered = SimpleNamespace(name="gate11-krea2")
        model_payload = {
            "comfyui_root": "/tmp/ComfyUI",
            "diffusion_models": "/tmp/models/diffusion_models",
            "text_encoders": "/tmp/models/text_encoders",
            "vae": "/tmp/models/vae",
            "model_registry": "/tmp/models.toml",
            "registered_model": "gate11-krea2",
        }
        encoded = "\n".join(
            (
                command("gate11-load", "load_model", model_payload),
                command(
                    "gate11-generation",
                    "generate_baseline",
                    {**model_payload, **generation_fixture},
                ),
            )
        )
        output = io.StringIO()
        with (
            patch.dict("os.environ", {"K2LAB_BACKEND": "native"}, clear=True),
            patch.object(entrypoint, "configure_debug_logging"),
            patch.object(entrypoint, "NativeK2Backend", NativeBackend),
            patch.object(
                entrypoint,
                "discover_model_artifacts",
                return_value=SimpleNamespace(complete=True),
            ),
            patch.object(
                entrypoint,
                "load_model_registry",
                return_value=SimpleNamespace(models=(registered,)),
            ),
            patch("sys.stdin", io.StringIO(encoded)),
            redirect_stdout(output),
        ):
            result = entrypoint.main()

        self.assertEqual(result, 0)
        request = NativeBackend.instances[0].generate_request
        self.assertIsInstance(request, GenerationRequest)
        self.assertEqual(request.correlation_id, "gate11-generation")
        self.assertEqual(request.prompt, generation_fixture["prompt"])
        self.assertEqual(request.seed, generation_fixture["seed"])
        self.assertEqual(request.sampler, generation_fixture["sampler"])
        self.assertEqual(request.scheduler, generation_fixture["scheduler"])
        self.assertEqual(request.regions[0].region_id, "teapot")
        self.assertEqual(request.prompt_emphases[0].phrase, "ceramic")
        events = [
            json.loads(line)
            for line in output.getvalue().splitlines()
            if line
        ]
        completed = next(
            event
            for event in events
            if event["command_id"] == "gate11-generation"
            and event["state"] == "ready"
        )
        self.assertEqual(completed["payload"]["seed"], 424242)

    def test_invalid_selector_returns_structured_configuration_error(self) -> None:
        result, events = self._run({"K2LAB_BACKEND": "automatic"})
        self.assertEqual(result, 1)
        self.assertFalse(Runtime.instances)
        failure = next(item for item in events if item["state"] == "error")
        self.assertEqual(
            failure["payload"]["error"]["category"],
            "ConfigurationError",
        )

    def test_native_phase_progress_is_not_labeled_as_denoising(self) -> None:
        output = io.StringIO()
        with redirect_stdout(output):
            entrypoint.emit_generation_progress(
                "native-progress",
                ProgressEvent(
                    correlation_id="native-progress",
                    phase="text_encoding",
                    fraction=0.0,
                ),
            )
            entrypoint.emit_generation_progress(
                "native-progress",
                ProgressEvent(
                    correlation_id="native-progress",
                    phase="vae_decode",
                    fraction=1.0,
                ),
            )
            entrypoint.emit_generation_progress(
                "native-progress",
                ProgressEvent(
                    correlation_id="native-progress",
                    phase="diffusion",
                    step=3,
                    total_steps=8,
                    fraction=3 / 8,
                    detail={
                        "sigma": 0.8,
                        "memory": {
                            "gpu_free_bytes": 12,
                            "gpu_total_bytes": 24,
                        },
                    },
                ),
            )

        events = [json.loads(line) for line in output.getvalue().splitlines()]
        self.assertEqual(
            [event["message"] for event in events],
            [
                "Prompt encoding started",
                "VAE decode complete",
                "Denoising step 3/8",
            ],
        )
        self.assertEqual(events[0]["payload"]["phase"], "text_encoding")
        self.assertEqual(events[2]["payload"]["memory"]["sigma"], 0.8)
        self.assertEqual(events[2]["payload"]["memory"]["gpu_free_bytes"], 12)


if __name__ == "__main__":
    unittest.main()
