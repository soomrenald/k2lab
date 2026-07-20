from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from PIL import Image

from k2_region_lab.regions import PixelBox, RegionDefinition


BACKEND = "krea-face-crop-detail-v1"
DETECTOR_PROVIDERS = ("auto", "cpu", "cuda")
DEFAULT_DETECTOR_RELATIVE_PATH = Path(
    "custom_nodes/ComfyUI-WanVideoWrapper/fantasyportrait/models/face_det.onnx"
)


@dataclass(frozen=True, slots=True)
class FaceDetailSettings:
    enabled: bool = False
    steps: int = 8
    denoise: float = 0.15
    crop_size: int = 512
    padding: float = 2.0
    feather: float = 0.12
    blend: float = 0.5
    lora_scale: float = 0.5
    detector_threshold: float = 0.15
    detector_provider: str = "auto"

    def __post_init__(self) -> None:
        if not 1 <= self.steps <= 100:
            raise ValueError("face-detail steps must be between 1 and 100")
        if not 0.0 < self.denoise <= 1.0:
            raise ValueError("face-detail denoise must be in (0, 1]")
        if self.crop_size not in {256, 512, 768, 1024}:
            raise ValueError("face-detail crop size must be 256, 512, 768, or 1024")
        if not 1.0 <= self.padding <= 4.0:
            raise ValueError("face-detail padding must be between 1 and 4")
        if not 0.0 <= self.feather <= 0.5:
            raise ValueError("face-detail feather must be between zero and 0.5")
        if not 0.0 <= self.blend <= 1.0:
            raise ValueError("face-detail blend must be between zero and one")
        if not 0.0 <= self.lora_scale <= 4.0:
            raise ValueError("face-detail LoRA scale must be between zero and four")
        if not 0.0 < self.detector_threshold < 1.0:
            raise ValueError("face detector threshold must be in (0, 1)")
        if self.detector_provider not in DETECTOR_PROVIDERS:
            raise ValueError(
                f"unsupported face detector provider: {self.detector_provider!r}"
            )


@dataclass(frozen=True, slots=True)
class DetectedFace:
    box: PixelBox
    score: float
    mask_points: tuple[tuple[float, float], ...] = ()

    @property
    def center(self) -> tuple[float, float]:
        return ((self.box.x0 + self.box.x1) / 2.0, (self.box.y0 + self.box.y1) / 2.0)


@dataclass(frozen=True, slots=True)
class FaceDetailTarget:
    region_id: str
    region_name: str
    prompt: str
    face: DetectedFace
    loras: tuple[dict[str, Any], ...]


def discover_face_detector(comfyui_root: Path) -> Path | None:
    candidate = comfyui_root.expanduser().resolve() / DEFAULT_DETECTOR_RELATIVE_PATH
    return candidate if candidate.is_file() and candidate.stat().st_size > 1024 else None


def face_detail_prompt(prompt: str) -> str:
    compact = " ".join(prompt.strip().split())
    if not compact:
        return "detailed natural face, preserve the same person and expression"
    first_sentence = compact.split(".", 1)[0].rstrip("!? ")
    return (
        f"close-up facial detail of {first_sentence}. Preserve the same person, "
        "head angle, expression, lighting, and photographic style."
    )


def face_identity_detail_prompt(identity_prompt: str) -> str:
    compact = " ".join(identity_prompt.strip().split()).rstrip(".!? ")
    if not compact:
        return "detailed natural face, preserve the same person and expression"
    return (
        f"close-up facial detail of {compact}. Generate the facial identity from "
        "this description and the assigned character LoRA. Preserve the source "
        "head angle, expression, lighting, and photographic style."
    )


def _contains(box: PixelBox, point: tuple[float, float]) -> bool:
    x, y = point
    return box.x0 <= x <= box.x1 and box.y0 <= y <= box.y1


def _intersection_ratio(inner: PixelBox, outer: PixelBox) -> float:
    width = max(0.0, min(inner.x1, outer.x1) - max(inner.x0, outer.x0))
    height = max(0.0, min(inner.y1, outer.y1) - max(inner.y0, outer.y0))
    return width * height / max(inner.width * inner.height, 1e-12)


def _distance_to_box(box: PixelBox, point: tuple[float, float]) -> float:
    x, y = point
    dx = max(box.x0 - x, 0.0, x - box.x1)
    dy = max(box.y0 - y, 0.0, y - box.y1)
    return float(np.hypot(dx, dy))


def assign_faces_to_regional_loras(
    detections: Iterable[DetectedFace],
    regions: Iterable[RegionDefinition],
    loras: Iterable[dict[str, Any]],
) -> tuple[FaceDetailTarget, ...]:
    faces = tuple(detections)
    loras_by_region: dict[str, list[dict[str, Any]]] = {}
    for specification in loras:
        if bool(specification.get("global", True)):
            continue
        if float(specification.get("strength", 1.0)) == 0.0:
            continue
        for region_id in map(str, specification.get("region_ids", ())):
            loras_by_region.setdefault(region_id, []).append(dict(specification))

    unused = set(range(len(faces)))
    targets: list[FaceDetailTarget] = []
    active_regions = sorted(
        (
            region
            for region in regions
            if region.enabled
            and (region.prompt.strip() or region.face_identity_prompt.strip())
            and region.region_id in loras_by_region
            and region.spatial_role != "background"
        ),
        key=lambda region: -region.priority,
    )
    for region in active_regions:
        candidates = []
        for index in unused:
            face = faces[index]
            overlap = _intersection_ratio(face.box, region.box)
            distance = _distance_to_box(region.box, face.center)
            nearby = distance <= max(face.box.width, face.box.height) * 1.25
            if not _contains(region.box, face.center) and overlap < 0.2 and not nearby:
                continue
            candidates.append(
                (
                    overlap,
                    -distance,
                    face.score,
                    face.box.width * face.box.height,
                    index,
                )
            )
        if not candidates:
            continue
        _overlap, _distance, _score, _area, selected = max(candidates)
        unused.remove(selected)
        targets.append(
            FaceDetailTarget(
                region_id=region.region_id,
                region_name=region.name,
                prompt=(
                    face_identity_detail_prompt(region.face_identity_prompt)
                    if region.face_identity_prompt.strip()
                    else face_detail_prompt(region.prompt)
                ),
                face=faces[selected],
                loras=tuple(loras_by_region[region.region_id]),
            )
        )
    return tuple(targets)


def expanded_square_crop(
    face: PixelBox,
    image_width: int,
    image_height: int,
    padding: float,
) -> tuple[int, int, int, int]:
    side = min(max(face.width, face.height) * padding, image_width, image_height)
    center_x = (face.x0 + face.x1) / 2.0
    center_y = (face.y0 + face.y1) / 2.0
    x0 = min(max(center_x - side / 2.0, 0.0), image_width - side)
    y0 = min(max(center_y - side / 2.0, 0.0), image_height - side)
    x1 = x0 + side
    y1 = y0 + side
    return (
        max(0, int(round(x0))),
        max(0, int(round(y0))),
        min(image_width, int(round(x1))),
        min(image_height, int(round(y1))),
    )


def composite_face_crop(
    image: Image.Image,
    refined_crop: Image.Image,
    crop_box: tuple[int, int, int, int],
    feather: float,
    blend: float,
    mask_points: tuple[tuple[float, float], ...] = (),
) -> Image.Image:
    x0, y0, x1, y1 = crop_box
    width, height = x1 - x0, y1 - y0
    if width <= 0 or height <= 0:
        raise ValueError("face-detail crop must have positive dimensions")
    refined = refined_crop.convert("RGB").resize((width, height), Image.Resampling.LANCZOS)
    original = image.convert("RGB")
    if not 0.0 <= blend <= 1.0:
        raise ValueError("face-detail blend must be between zero and one")
    if blend <= 0.0:
        return original
    original_crop = original.crop(crop_box)
    if mask_points:
        from PIL import ImageDraw, ImageFilter

        center_x = sum(point[0] for point in mask_points) / len(mask_points)
        center_y = sum(point[1] for point in mask_points) / len(mask_points)
        expanded = [
            (
                center_x + (point_x - center_x) * 1.12 - x0,
                center_y + (point_y - center_y) * 1.12 - y0,
            )
            for point_x, point_y in mask_points
        ]
        mask = Image.new("L", (width, height), 0)
        ImageDraw.Draw(mask).polygon(expanded, fill=255)
        if feather > 0.0:
            mask = mask.filter(
                ImageFilter.GaussianBlur(
                    radius=max(1.0, min(width, height) * feather * 0.5)
                )
            )
        if blend < 1.0:
            mask = mask.point(lambda value: round(value * blend))
        blended = Image.composite(refined, original_crop, mask)
        result = original.copy()
        result.paste(blended, (x0, y0))
        return result
    if feather <= 0.0:
        result = original.copy()
        result.paste(Image.blend(original_crop, refined, blend), (x0, y0))
        return result

    feather_pixels = max(1.0, min(width, height) * feather)
    yy, xx = np.mgrid[0:height, 0:width]
    edge_distance = np.minimum.reduce((xx, yy, width - 1 - xx, height - 1 - yy))
    alpha = np.clip(edge_distance / feather_pixels, 0.0, 1.0)
    alpha = alpha * alpha * (3.0 - 2.0 * alpha) * blend
    mask = Image.fromarray(np.round(alpha * 255.0).astype(np.uint8), mode="L")
    blended = Image.composite(refined, original_crop, mask)
    result = original.copy()
    result.paste(blended, (x0, y0))
    return result


class OnnxNanoFaceDetector:
    """Small CPU face detector backed by the bundled FantasyPortrait NanoDet model."""

    input_width = 160
    input_height = 272
    strides = (8, 16, 32)
    reg_max = 7

    def __init__(
        self,
        model_path: Path,
        *,
        threshold: float = 0.4,
        provider: str = "auto",
    ) -> None:
        self.model_path = model_path.expanduser().resolve()
        self.threshold = float(threshold)
        if provider not in DETECTOR_PROVIDERS:
            raise ValueError(f"unsupported face detector provider: {provider!r}")
        self.provider = provider
        self.execution_provider: str | None = None
        self._session = None

    def _load_session(self):
        if self._session is not None:
            return self._session
        try:
            import onnxruntime
        except Exception as error:
            raise RuntimeError(
                "face detection requires a working CPU onnxruntime installation in "
                "the configured worker environment; the installed package could not "
                f"be imported: {error}"
            ) from error
        available = tuple(onnxruntime.get_available_providers())
        if self.provider == "cuda":
            if "CUDAExecutionProvider" not in available:
                raise RuntimeError(
                    "CUDA face detection was selected, but this ONNX Runtime "
                    f"installation exposes only: {', '.join(available) or 'no providers'}"
                )
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        elif self.provider == "cpu":
            providers = ["CPUExecutionProvider"]
        elif "CUDAExecutionProvider" in available:
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        else:
            providers = ["CPUExecutionProvider"]
        try:
            self._session = onnxruntime.InferenceSession(
                str(self.model_path), providers=providers
            )
        except Exception as error:
            if self.provider == "auto" and providers[0] == "CUDAExecutionProvider":
                try:
                    self._session = onnxruntime.InferenceSession(
                        str(self.model_path), providers=["CPUExecutionProvider"]
                    )
                except Exception as cpu_error:
                    raise RuntimeError(
                        f"could not load the face detector {self.model_path} with "
                        f"CUDA ({error}) or CPU ({cpu_error})"
                    ) from cpu_error
            else:
                raise RuntimeError(
                    f"could not load the face detector {self.model_path}: {error}"
                ) from error
        active = tuple(self._session.get_providers())
        self.execution_provider = active[0] if active else providers[0]
        if self.provider == "cuda" and self.execution_provider != "CUDAExecutionProvider":
            raise RuntimeError(
                "CUDA face detection was selected, but ONNX Runtime activated "
                f"{self.execution_provider} instead"
            )
        inputs = self._session.get_inputs()
        outputs = self._session.get_outputs()
        input_shape = tuple(inputs[0].shape) if len(inputs) == 1 else ()
        expected_shape = (1, 3, self.input_height, self.input_width)
        if len(inputs) != 1 or input_shape != expected_shape or len(outputs) != 6:
            self._session = None
            raise RuntimeError(
                f"incompatible face detector model: {self.model_path}; expected one "
                f"rank-4 input shaped {expected_shape} and 6 outputs, found "
                f"input shape {input_shape or 'unknown'} and {len(outputs)} outputs. "
                "Select ComfyUI-WanVideoWrapper/fantasyportrait/models/face_det.onnx "
                "or reset the detector setting to Auto."
            )
        return self._session

    @staticmethod
    def _softmax(values: np.ndarray) -> np.ndarray:
        shifted = values - values.max(axis=1, keepdims=True)
        exponent = np.exp(shifted)
        return exponent / exponent.sum(axis=1, keepdims=True)

    @staticmethod
    def _nms(boxes: np.ndarray, scores: np.ndarray, threshold: float = 0.3) -> list[int]:
        if not len(boxes):
            return []
        x0, y0, x1, y1 = boxes.T
        areas = np.maximum(0.0, x1 - x0) * np.maximum(0.0, y1 - y0)
        order = scores.argsort()[::-1]
        keep: list[int] = []
        while order.size:
            current = int(order[0])
            keep.append(current)
            if order.size == 1:
                break
            rest = order[1:]
            ix0 = np.maximum(x0[current], x0[rest])
            iy0 = np.maximum(y0[current], y0[rest])
            ix1 = np.minimum(x1[current], x1[rest])
            iy1 = np.minimum(y1[current], y1[rest])
            intersection = np.maximum(0.0, ix1 - ix0) * np.maximum(0.0, iy1 - iy0)
            union = areas[current] + areas[rest] - intersection
            iou = intersection / np.maximum(union, 1e-12)
            order = rest[iou <= threshold]
        return keep

    def _preprocess(self, image: Image.Image) -> tuple[np.ndarray, float, float, float]:
        rgb = image.convert("RGB")
        scale = min(self.input_width / rgb.width, self.input_height / rgb.height)
        resized_width = max(1, int(round(rgb.width * scale)))
        resized_height = max(1, int(round(rgb.height * scale)))
        resized = rgb.resize((resized_width, resized_height), Image.Resampling.BILINEAR)
        offset_x = (self.input_width - resized_width) / 2.0
        offset_y = (self.input_height - resized_height) / 2.0
        canvas = Image.new("RGB", (self.input_width, self.input_height))
        canvas.paste(resized, (int(round(offset_x)), int(round(offset_y))))
        # The bundled NanoDet weights use OpenCV's BGR channel convention.
        array = np.asarray(canvas, dtype=np.float32)[:, :, ::-1]
        mean = np.asarray((103.53, 116.28, 123.675), dtype=np.float32)
        std = np.asarray((57.375, 57.12, 58.395), dtype=np.float32)
        array = (array - mean) / std
        return array.transpose(2, 0, 1)[None], scale, offset_x, offset_y

    def detect(self, image: Image.Image) -> tuple[DetectedFace, ...]:
        session = self._load_session()
        input_tensor, scale, offset_x, offset_y = self._preprocess(image)
        outputs = session.run(None, {session.get_inputs()[0].name: input_tensor})
        scores_by_stride = [np.squeeze(item) for item in outputs[:3]]
        boxes_by_stride = [np.squeeze(item) for item in outputs[3:6]]
        decoded_boxes: list[np.ndarray] = []
        decoded_scores: list[np.ndarray] = []
        for stride, raw_scores, distribution in zip(
            self.strides, scores_by_stride, boxes_by_stride, strict=True
        ):
            scores = raw_scores.reshape(-1)
            feature_height = int(np.ceil(self.input_height / stride))
            feature_width = int(np.ceil(self.input_width / stride))
            columns, rows = np.meshgrid(
                np.arange(feature_width, dtype=np.float32),
                np.arange(feature_height, dtype=np.float32),
            )
            centers = np.stack(
                (columns.ravel() * stride, rows.ravel() * stride), axis=1
            )
            distances = distribution.reshape((-1, self.reg_max + 1))
            distances = self._softmax(distances)
            distances *= np.arange(self.reg_max + 1, dtype=np.float32)[None]
            distances = distances.sum(axis=1).reshape((-1, 4)) * stride
            count = min(len(scores), len(centers), len(distances))
            scores, centers, distances = scores[:count], centers[:count], distances[:count]
            selected = scores >= self.threshold
            if not np.any(selected):
                continue
            centers, distances, scores = centers[selected], distances[selected], scores[selected]
            boxes = np.column_stack(
                (
                    centers[:, 0] - distances[:, 0],
                    centers[:, 1] - distances[:, 1],
                    centers[:, 0] + distances[:, 2],
                    centers[:, 1] + distances[:, 3],
                )
            )
            decoded_boxes.append(boxes)
            decoded_scores.append(scores)
        if not decoded_boxes:
            return ()
        boxes = np.concatenate(decoded_boxes)
        scores = np.concatenate(decoded_scores)
        boxes[:, (0, 2)] = (boxes[:, (0, 2)] - offset_x) / scale
        boxes[:, (1, 3)] = (boxes[:, (1, 3)] - offset_y) / scale
        boxes[:, (0, 2)] = boxes[:, (0, 2)].clip(0, image.width)
        boxes[:, (1, 3)] = boxes[:, (1, 3)].clip(0, image.height)
        faces = []
        for index in self._nms(boxes, scores):
            x0, y0, x1, y1 = map(float, boxes[index])
            if x1 - x0 < 8.0 or y1 - y0 < 8.0:
                continue
            faces.append(
                DetectedFace(PixelBox(x0, y0, x1, y1), float(scores[index]))
            )
        return tuple(
            sorted(faces, key=lambda face: (face.center[0], face.center[1], -face.score))
        )
