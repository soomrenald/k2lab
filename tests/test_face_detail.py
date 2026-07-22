from __future__ import annotations

import tempfile
import unittest
import sys
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
from PIL import Image, PngImagePlugin

from k2_region_lab.face_detail import (
    DEFAULT_DETECTOR_RELATIVE_PATH,
    DetectedFace,
    FaceDetailSettings,
    OnnxNanoFaceDetector,
    assign_faces_to_regional_loras,
    composite_face_crop,
    expanded_square_crop,
)
from k2_region_lab.regions import PixelBox, RegionDefinition
from k2_region_lab.worker.runtime import ComfyBaselineRuntime


class FaceDetailGeometryTests(unittest.TestCase):
    def test_faces_match_only_regional_loras_and_are_assigned_once(self) -> None:
        regions = (
            RegionDefinition(
                "left", "Left character", PixelBox(0, 100, 120, 200), "person one", priority=2
            ),
            RegionDefinition(
                "wide", "Wide character", PixelBox(0, 100, 220, 200), "person two", priority=1
            ),
        )
        faces = (
            DetectedFace(PixelBox(30, 30, 70, 80), 0.95),
            DetectedFace(PixelBox(150, 40, 190, 90), 0.9),
        )
        loras = (
            {"id": "global", "global": True, "strength": 1.0},
            {"id": "left-lora", "global": False, "region_ids": ["left"]},
            {"id": "wide-lora", "global": False, "region_ids": ["wide"]},
        )

        targets = assign_faces_to_regional_loras(faces, regions, loras)

        self.assertEqual([target.region_id for target in targets], ["left", "wide"])
        self.assertEqual(targets[0].face, faces[0])
        self.assertEqual(targets[1].face, faces[1])
        self.assertEqual([item["id"] for item in targets[0].loras], ["left-lora"])

    def test_square_crop_stays_inside_image(self) -> None:
        crop = expanded_square_crop(PixelBox(0, 10, 40, 70), 200, 100, 2.0)

        self.assertEqual(crop, (0, 0, 100, 100))

    def test_feathered_composite_preserves_pixels_outside_crop(self) -> None:
        original = Image.new("RGB", (100, 100), "black")
        refined = Image.new("RGB", (40, 40), "white")

        result = composite_face_crop(original, refined, (30, 30, 70, 70), 0.2, 1.0)

        self.assertEqual(result.getpixel((10, 10)), (0, 0, 0))
        self.assertEqual(result.getpixel((50, 50)), (255, 255, 255))
        self.assertEqual(result.getpixel((30, 30)), (0, 0, 0))

    def test_detector_preprocessing_has_model_shape(self) -> None:
        detector = OnnxNanoFaceDetector(Path("/unused/model.onnx"))

        tensor, scale, offset_x, offset_y = detector._preprocess(
            Image.new("RGB", (640, 480), "white")
        )

        self.assertEqual(tensor.shape, (1, 3, 272, 160))
        self.assertEqual(tensor.dtype, np.float32)
        self.assertGreater(scale, 0)
        self.assertGreaterEqual(offset_x, 0)
        self.assertGreaterEqual(offset_y, 0)

    def test_auto_detector_provider_prefers_cuda_when_available(self) -> None:
        session = Mock()
        session.get_providers.return_value = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        session.get_inputs.return_value = [Mock(shape=[1, 3, 272, 160])]
        session.get_outputs.return_value = [Mock() for _index in range(6)]
        runtime = Mock()
        runtime.get_available_providers.return_value = [
            "CUDAExecutionProvider",
            "CPUExecutionProvider",
        ]
        runtime.InferenceSession.return_value = session
        detector = OnnxNanoFaceDetector(Path("/unused/model.onnx"), provider="auto")

        with patch.dict(sys.modules, {"onnxruntime": runtime}):
            detector._load_session()

        self.assertEqual(
            runtime.InferenceSession.call_args.kwargs["providers"],
            ["CUDAExecutionProvider", "CPUExecutionProvider"],
        )
        self.assertEqual(detector.execution_provider, "CUDAExecutionProvider")

    def test_cpu_detector_provider_never_requests_cuda(self) -> None:
        session = Mock()
        session.get_providers.return_value = ["CPUExecutionProvider"]
        session.get_inputs.return_value = [Mock(shape=[1, 3, 272, 160])]
        session.get_outputs.return_value = [Mock() for _index in range(6)]
        runtime = Mock()
        runtime.get_available_providers.return_value = [
            "CUDAExecutionProvider",
            "CPUExecutionProvider",
        ]
        runtime.InferenceSession.return_value = session
        detector = OnnxNanoFaceDetector(Path("/unused/model.onnx"), provider="cpu")

        with patch.dict(sys.modules, {"onnxruntime": runtime}):
            detector._load_session()

        self.assertEqual(
            runtime.InferenceSession.call_args.kwargs["providers"],
            ["CPUExecutionProvider"],
        )
        self.assertEqual(detector.execution_provider, "CPUExecutionProvider")

    def test_auto_detector_provider_falls_back_when_cuda_session_fails(self) -> None:
        session = Mock()
        session.get_providers.return_value = ["CPUExecutionProvider"]
        session.get_inputs.return_value = [Mock(shape=[1, 3, 272, 160])]
        session.get_outputs.return_value = [Mock() for _index in range(6)]
        runtime = Mock()
        runtime.get_available_providers.return_value = [
            "CUDAExecutionProvider",
            "CPUExecutionProvider",
        ]
        runtime.InferenceSession.side_effect = [RuntimeError("broken CUDA"), session]
        detector = OnnxNanoFaceDetector(Path("/unused/model.onnx"), provider="auto")

        with patch.dict(sys.modules, {"onnxruntime": runtime}):
            detector._load_session()

        self.assertEqual(runtime.InferenceSession.call_count, 2)
        self.assertEqual(
            runtime.InferenceSession.call_args.kwargs["providers"],
            ["CPUExecutionProvider"],
        )
        self.assertEqual(detector.execution_provider, "CPUExecutionProvider")


class FaceDetailRuntimeTests(unittest.TestCase):
    def test_runtime_refines_matched_crop_with_only_its_regional_lora(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            detector_path = root / DEFAULT_DETECTOR_RELATIVE_PATH
            detector_path.parent.mkdir(parents=True)
            detector_path.write_bytes(b"x" * 2048)
            runtime = ComfyBaselineRuntime.__new__(ComfyBaselineRuntime)
            runtime.comfyui_root = root
            runtime._refine_face_crop = Mock(
                return_value=(Image.new("RGB", (256, 256), "red"), [{"status": "applied"}])
            )
            region = RegionDefinition(
                "subject",
                "Subject",
                PixelBox(0, 0, 128, 128),
                "standing beside a window",
                face_identity_prompt="personface, a specific person with an oval face",
            )
            loras = [
                {
                    "id": "character",
                    "name": "Character",
                    "path": "/models/character.safetensors",
                    "strength": 1.5,
                    "global": False,
                    "region_ids": ["subject"],
                },
                {
                    "id": "style",
                    "name": "Style",
                    "path": "/models/style.safetensors",
                    "strength": 1.0,
                    "global": True,
                    "region_ids": [],
                },
            ]

            fake_detector = Mock()
            fake_detector.detect.return_value = (
                DetectedFace(PixelBox(40, 40, 80, 80), 0.92),
            )
            with patch(
                "k2_region_lab.worker.runtime.OnnxNanoFaceDetector",
                return_value=fake_detector,
            ):
                result, summary = runtime._run_face_detail_pass(
                    Image.new("RGB", (128, 128), "black"),
                    settings=FaceDetailSettings(
                        enabled=True,
                        crop_size=256,
                        feather=0.0,
                        blend=1.0,
                        lora_scale=2.0,
                    ),
                    regions=(region,),
                    loras=loras,
                    seed=10,
                    event=None,
                )

        self.assertEqual(summary["status"], "complete")
        self.assertEqual(summary["refined_count"], 1)
        self.assertEqual(result.getpixel((60, 60)), (255, 0, 0))
        self.assertEqual(result.getpixel((120, 120)), (0, 0, 0))
        detail_loras = runtime._refine_face_crop.call_args.kwargs["loras"]
        self.assertEqual(len(detail_loras), 1)
        self.assertEqual(detail_loras[0]["id"], "character")
        self.assertTrue(detail_loras[0]["global"])
        self.assertEqual(detail_loras[0]["region_ids"], [])
        self.assertEqual(detail_loras[0]["strength"], 3.0)
        detail_prompt = runtime._refine_face_crop.call_args.kwargs["prompt"]
        self.assertIn("personface, a specific person with an oval face", detail_prompt)
        self.assertNotIn("standing beside a window", detail_prompt)

    def test_standalone_refinement_saves_a_tagged_png(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "first_pass.png"
            source_metadata = PngImagePlugin.PngInfo()
            source_metadata.add_text("k2lab_project", '{"source":true}')
            source_metadata.add_text("global_prompt", "source prompt")
            Image.new("RGB", (64, 64), "blue").save(source, pnginfo=source_metadata)
            runtime = ComfyBaselineRuntime.__new__(ComfyBaselineRuntime)
            runtime.model = object()
            runtime.clip = object()
            runtime.vae = object()
            runtime._run_face_detail_pass = Mock(
                return_value=(Image.new("RGB", (64, 64), "green"), {"status": "complete"})
            )
            runtime.memory_snapshot = Mock(return_value={"stage": "complete"})

            report = runtime.refine_faces(
                image_path=source,
                regions=(),
                loras=[],
                seed=42,
                selected_face_indices=(1,),
                project_json={"schema": "k2-region-lab-project", "version": 14},
            )
            output_path = Path(report["image_path"])
            with Image.open(output_path) as refined:
                refined_metadata = dict(refined.info)

        self.assertIn("face_refined", output_path.name)
        self.assertEqual(report["source_image"], str(source.resolve()))
        runtime._run_face_detail_pass.assert_called_once()
        self.assertEqual(
            runtime._run_face_detail_pass.call_args.kwargs["selected_face_indices"],
            (1,),
        )
        self.assertEqual(refined_metadata["k2lab_mode"], "krea2_face_refinement")
        self.assertEqual(refined_metadata["global_prompt"], "source prompt")
        self.assertEqual(
            refined_metadata["k2lab_project"],
            '{"schema":"k2-region-lab-project","version":14}',
        )

    def test_selected_face_indices_filter_detected_targets(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            detector_path = root / DEFAULT_DETECTOR_RELATIVE_PATH
            detector_path.parent.mkdir(parents=True)
            detector_path.write_bytes(b"x" * 2048)
            runtime = ComfyBaselineRuntime.__new__(ComfyBaselineRuntime)
            runtime.comfyui_root = root
            runtime._refine_face_crop = Mock(
                return_value=(Image.new("RGB", (256, 256), "red"), [])
            )
            regions = (
                RegionDefinition("left", "Left", PixelBox(0, 0, 128, 256), "left face"),
                RegionDefinition("right", "Right", PixelBox(128, 0, 256, 256), "right face"),
            )
            loras = [
                {"id": "left", "global": False, "strength": 1.0, "region_ids": ["left"]},
                {
                    "id": "right",
                    "global": False,
                    "strength": 1.0,
                    "region_ids": ["right"],
                },
            ]
            fake_detector = Mock()
            fake_detector.detect.return_value = (
                DetectedFace(PixelBox(30, 40, 70, 80), 0.8),
                DetectedFace(PixelBox(180, 40, 220, 80), 0.9),
            )
            with patch(
                "k2_region_lab.worker.runtime.OnnxNanoFaceDetector",
                return_value=fake_detector,
            ):
                _result, summary = runtime._run_face_detail_pass(
                    Image.new("RGB", (256, 256), "black"),
                    settings=FaceDetailSettings(enabled=True, crop_size=256),
                    regions=regions,
                    loras=loras,
                    seed=0,
                    selected_face_indices=(1,),
                )

        self.assertEqual(summary["detection_count"], 2)
        self.assertEqual(summary["selected_indices"], [1])
        self.assertEqual(summary["selected_count"], 1)
        self.assertEqual(summary["refined_count"], 1)
        self.assertEqual(summary["faces"][0]["region_id"], "right")


if __name__ == "__main__":
    unittest.main()
