from __future__ import annotations

from typing import Any

from k2_region_lab.regional_prompting import BoundRegionalPromptPlan


def spatial_pair_bias(
    image_token_field: tuple[float, ...],
    strength: float,
    *,
    outside_penalty_ratio: float = 0.25,
    outside_penalty: float | None = None,
) -> tuple[float, ...]:
    """Convert a soft spatial field into additive attention-logit values."""
    if not 0.0 <= outside_penalty_ratio <= 1.0:
        raise ValueError("outside penalty ratio must be between zero and one")
    penalty = (
        strength * outside_penalty_ratio
        if outside_penalty is None
        else outside_penalty
    )
    if not 0.0 <= penalty <= 10.0:
        raise ValueError("outside penalty must be between zero and ten")
    return tuple((strength + penalty) * weight - penalty for weight in image_token_field)


class KreaSpatialAttentionOverride:
    """Inject a chunked text/image score bias into Krea's main stream only.

    ROCm cannot use its efficient SDPA kernels with a dense additive mask on the
    target GPU. Query chunking computes the exact biased softmax without ever
    materializing the complete per-head score matrix.
    """

    def __init__(
        self,
        plan: BoundRegionalPromptPlan,
        *,
        outside_penalty_ratio: float = 0.25,
        query_chunk_size: int = 256,
    ) -> None:
        self.plan = plan
        self.outside_penalty_ratio = outside_penalty_ratio
        if query_chunk_size <= 0:
            raise ValueError("attention query chunk size must be positive")
        self.query_chunk_size = query_chunk_size
        self.expected_sequence_length = (
            plan.text_token_count + plan.image_token_count
        )
        self.matched_calls = 0
        self.step_scale = 1.0
        self._cache: dict[tuple[str, int | None, str], Any] = {}

    def __call__(self, original, *args, **kwargs):
        q = args[0]
        k = args[1]
        if (
            q.shape[-2] != self.expected_sequence_length
            or k.shape[-2] != self.expected_sequence_length
        ):
            return original(*args, **kwargs)

        if kwargs.get("mask") is not None:
            raise RuntimeError(
                "Krea chunked spatial attention requires an unmasked main stream"
            )
        if not kwargs.get("skip_reshape", False) or q.ndim != 4:
            raise RuntimeError(
                "Krea chunked spatial attention expected head-shaped query tensors"
            )

        v = args[2]
        original_head_dim = q.shape[-1]
        scale = float(kwargs.get("scale", original_head_dim**-0.5))
        output = self._chunked_attention(q, k, v, scale)
        self.matched_calls += 1
        if kwargs.get("skip_output_reshape", False):
            return output
        return output.transpose(1, 2).reshape(output.shape[0], output.shape[2], -1)

    def _chunked_attention(self, q, k, v, scale: float):
        import torch

        output = torch.empty(
            (q.shape[0], q.shape[1], q.shape[2], v.shape[-1]),
            dtype=v.dtype,
            device=v.device,
        )
        key_transposed = k.transpose(-2, -1)
        pair_fields = self._pair_fields(q)
        for start in range(0, q.shape[-2], self.query_chunk_size):
            end = min(q.shape[-2], start + self.query_chunk_size)
            scores = torch.matmul(q[:, :, start:end], key_transposed) * scale
            scores = scores.float()
            self._add_spatial_bias(scores, start, end, pair_fields)
            probabilities = torch.softmax(scores, dim=-1).to(v.dtype)
            output[:, :, start:end] = torch.matmul(probabilities, v)
            del scores, probabilities
        return output

    def _pair_fields(self, reference):
        import torch

        device = reference.device
        key = (device.type, device.index, str(reference.dtype))
        cached = self._cache.get(key)
        if cached is not None:
            return cached

        fields = tuple(
            torch.tensor(
                spatial_pair_bias(
                    span.image_token_field,
                    self.plan.strength,
                    outside_penalty_ratio=self.outside_penalty_ratio,
                    outside_penalty=self.plan.outside_penalty
                    * (1.0 if span.spatial_role == "subject" else 0.25),
                ),
                dtype=torch.float32,
                device=device,
            )
            for span in self.plan.spans
        )
        self._cache[key] = fields
        return fields

    def _add_spatial_bias(self, scores, start, end, pair_fields) -> None:
        text_count = self.plan.text_token_count
        for span, pair in zip(self.plan.spans, pair_fields, strict=True):
            text_start = max(start, span.start)
            text_end = min(end, span.end)
            if text_start < text_end:
                scores[
                    :, :, text_start - start : text_end - start, text_count:
                ].add_(pair.reshape(1, 1, 1, -1), alpha=self.step_scale)

            image_start = max(start, text_count)
            image_end = end
            if image_start < image_end:
                image_pair = pair[
                    image_start - text_count : image_end - text_count
                ]
                scores[
                    :, :, image_start - start : image_end - start, span.start : span.end
                ].add_(image_pair.reshape(1, 1, -1, 1), alpha=self.step_scale)

    def set_denoising_progress(self, completed_steps: int, total_steps: int) -> None:
        """Keep placement strong early, then relax it for late detail refinement."""
        if total_steps <= 0:
            raise ValueError("total denoising steps must be positive")
        progress = min(1.0, max(0.0, completed_steps / total_steps))
        relaxation_start = 0.55
        if progress <= relaxation_start:
            self.step_scale = 1.0
            return
        fraction = (progress - relaxation_start) / (1.0 - relaxation_start)
        self.step_scale = 1.0 + fraction * (self.plan.late_step_scale - 1.0)

    def clear(self) -> None:
        self._cache.clear()
