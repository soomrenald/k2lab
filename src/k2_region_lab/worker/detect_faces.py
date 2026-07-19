from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image

from k2_region_lab.face_detail import OnnxNanoFaceDetector, discover_face_detector


def main() -> int:
    parser = argparse.ArgumentParser(description="Detect faces for K2 Region Lab")
    parser.add_argument("--image", required=True, type=Path)
    parser.add_argument("--comfyui-root", required=True, type=Path)
    parser.add_argument("--detector-path", type=Path)
    parser.add_argument("--threshold", required=True, type=float)
    parser.add_argument(
        "--provider", choices=("auto", "cpu", "cuda"), default="auto"
    )
    arguments = parser.parse_args()

    detector_path = arguments.detector_path or discover_face_detector(
        arguments.comfyui_root
    )
    if detector_path is None:
        raise RuntimeError("face_det.onnx was not found under the configured ComfyUI root")
    with Image.open(arguments.image.expanduser().resolve()) as source:
        image = source.convert("RGB")
    detector = OnnxNanoFaceDetector(
        detector_path,
        threshold=arguments.threshold,
        provider=arguments.provider,
    )
    faces = detector.detect(image)
    print(
        json.dumps(
            {
                "width": image.width,
                "height": image.height,
                "execution_provider": detector.execution_provider,
                "faces": [
                    {
                        "box": [face.box.x0, face.box.y0, face.box.x1, face.box.y1],
                        "score": face.score,
                    }
                    for face in faces
                ],
            },
            separators=(",", ":"),
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
