from __future__ import annotations

import unittest

from PIL import Image

from k2_region_lab.worker.runtime import ComfyBaselineRuntime


class PostUpscaleTests(unittest.TestCase):
    def test_lanczos_upscale_runs_on_final_rgb_image(self) -> None:
        runtime = ComfyBaselineRuntime.__new__(ComfyBaselineRuntime)
        source = Image.new("RGB", (48, 32), "#7f4020")
        events: list[str] = []

        output, summary = runtime._post_upscale_image(
            source,
            scale=4,
            method="lanczos",
            model_path=None,
            event=lambda message, _payload: events.append(message),
        )

        self.assertEqual(output.size, (192, 128))
        self.assertEqual(summary["backend"], "pillow-lanczos")
        self.assertEqual(summary["scale"], 4)
        self.assertIn("CPU Lanczos post-upscale 4× started", events)

    def test_invalid_upscale_method_is_rejected(self) -> None:
        runtime = ComfyBaselineRuntime.__new__(ComfyBaselineRuntime)
        with self.assertRaisesRegex(ValueError, "unsupported post-upscale method"):
            runtime._post_upscale_image(
                Image.new("RGB", (16, 16)),
                scale=2,
                method="unknown",
                model_path=None,
                event=None,
            )


if __name__ == "__main__":
    unittest.main()
