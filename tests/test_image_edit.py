from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image, PngImagePlugin

from k2_region_lab.image_edit import (
    ImageEditState,
    composite_regional_edit,
    edge_pad_to_krea,
    edit_global_conditioning_prompt,
    load_source_image,
    regional_composite_mask,
    regional_edit_conditioning,
    regional_reference_emphases,
)
from k2_region_lab.project import (
    ProjectState,
    SavedLora,
    load_associated_image_project,
    project_document,
    project_state,
)
from k2_region_lab.regions import PixelBox, RegionDefinition
from k2_region_lab.regional_prompting import GLOBAL_EMPHASIS_SCOPE, PromptEmphasis


class ImageEditGeometryTests(unittest.TestCase):
    def test_source_loading_applies_supported_format_and_size_rules(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source.webp"
            Image.new("RGB", (300, 260), "navy").save(source)

            image, metadata = load_source_image(source)

        self.assertEqual(image.size, (300, 260))
        self.assertIsInstance(metadata, dict)

    def test_edge_padding_preserves_source_and_repeats_boundary_pixels(self) -> None:
        image = Image.new("RGB", (257, 259), "black")
        image.putpixel((256, 258), (10, 20, 30))

        padded, geometry = edge_pad_to_krea(image)

        self.assertEqual(padded.size, (272, 272))
        self.assertEqual(padded.crop((0, 0, 257, 259)).tobytes(), image.tobytes())
        self.assertEqual(padded.getpixel((271, 271)), (10, 20, 30))
        self.assertEqual((geometry.requested_width, geometry.requested_height), (257, 259))

    def test_union_mask_has_bounded_feather_and_no_overlap_accumulation(self) -> None:
        regions = (
            RegionDefinition("one", "One", PixelBox(20, 20, 50, 50), "red object"),
            RegionDefinition("two", "Two", PixelBox(40, 40, 70, 70), "blue object"),
        )

        mask = regional_composite_mask((100, 100), regions, 4)

        self.assertEqual(mask.getpixel((15, 30)), 0)
        self.assertGreater(mask.getpixel((18, 30)), 0)
        self.assertEqual(mask.getpixel((30, 30)), 255)
        self.assertEqual(mask.getpixel((45, 45)), 255)
        self.assertEqual(mask.getpixel((80, 80)), 0)

    def test_blank_global_composite_preserves_pixels_outside_mask_support(self) -> None:
        source = Image.new("RGB", (100, 100), "black")
        candidate = Image.new("RGB", (100, 100), "white")
        regions = (
            RegionDefinition("edit", "Edit", PixelBox(30, 30, 70, 70), "make white"),
        )

        result, _mask = composite_regional_edit(source, candidate, regions, 5)

        self.assertEqual(result.getpixel((10, 10)), (0, 0, 0))
        self.assertEqual(result.getpixel((50, 50)), (255, 255, 255))
        self.assertNotEqual(result.getpixel((27, 50)), (0, 0, 0))

    def test_reference_and_edit_regions_compile_as_separate_semantic_layers(self) -> None:
        reference = RegionDefinition(
            "person",
            "Person",
            PixelBox(10, 10, 80, 95),
            "a woman in a blue coat",
            face_identity_prompt="the same face",
            spatial_role="subject",
        )
        target = RegionDefinition(
            "sleeve-edit",
            "Sleeve edit",
            PixelBox(15, 45, 45, 80),
            "embroidered gold cuffs",
        )

        combined = regional_edit_conditioning(
            (reference,), (target,), "change the clothing"
        )

        self.assertEqual(combined[0], reference)
        self.assertEqual(combined[1].spatial_role, "edit")
        self.assertEqual(
            combined[1].prompt,
            "change the clothing. embroidered gold cuffs",
        )
        self.assertEqual(combined[1].box, target.box)

    def test_edit_global_conditioning_excludes_conflicting_source_scene(self) -> None:
        source_prompt = "two people reclining on a bed"
        edit_prompt = "remove the people and continue the bedding"

        self.assertEqual(
            edit_global_conditioning_prompt(
                source_prompt,
                edit_prompt,
                edit_entire_image=False,
            ),
            "",
        )
        self.assertEqual(
            edit_global_conditioning_prompt(
                source_prompt,
                edit_prompt,
                edit_entire_image=True,
            ),
            edit_prompt,
        )

    def test_reference_emphases_drop_only_the_omitted_global_scope(self) -> None:
        emphases = (
            PromptEmphasis(GLOBAL_EMPHASIS_SCOPE, "two people"),
            PromptEmphasis("person", "blue coat"),
        )

        retained = regional_reference_emphases(emphases)

        self.assertEqual(retained, (emphases[1],))


class ImageEditProjectTests(unittest.TestCase):
    def test_project_round_trip_preserves_independent_edit_state_and_lora_scope(self) -> None:
        edit_region = RegionDefinition(
            "edit-one",
            "Replace sign",
            PixelBox(20, 30, 180, 150),
            "a painted wooden sign",
        )
        state = ProjectState(
            canvas_width=512,
            canvas_height=512,
            image_edit=ImageEditState(
                source_image=Path("/images/source.jpg"),
                associated_project=Path("/images/source.k2lab.json"),
                width=300,
                height=260,
                reference_global_prompt="portrait in a studio",
                reference_regions=(
                    RegionDefinition(
                        "reference-person",
                        "Person",
                        PixelBox(0, 0, 200, 250),
                        "a specific person",
                    ),
                ),
                global_prompt="",
                denoise=0.42,
                composite_feather_pixels=24,
                regions=(edit_region,),
            ),
            loras=(
                SavedLora(
                    Path("/models/style.safetensors"),
                    edit_enabled=True,
                    edit_global_scope=False,
                    edit_region_ids=("edit-one",),
                    reference_enabled=True,
                    reference_global_scope=False,
                    reference_region_ids=("reference-person",),
                ),
            ),
        )

        restored = project_state(project_document(state))

        self.assertEqual(restored.image_edit.source_image, Path("/images/source.jpg"))
        self.assertEqual(
            restored.image_edit.associated_project,
            Path("/images/source.k2lab.json"),
        )
        self.assertEqual(restored.image_edit.reference_global_prompt, "portrait in a studio")
        self.assertEqual(
            restored.image_edit.reference_regions[0].region_id,
            "reference-person",
        )
        self.assertEqual(restored.image_edit.denoise, 0.42)
        self.assertEqual(restored.image_edit.regions, (edit_region,))
        self.assertTrue(restored.loras[0].edit_enabled)
        self.assertEqual(restored.loras[0].edit_region_ids, ("edit-one",))
        self.assertEqual(
            restored.loras[0].reference_region_ids,
            ("reference-person",),
        )

    def test_version_16_project_defaults_to_empty_edit_setup(self) -> None:
        document = project_document(ProjectState(512, 512))
        document["version"] = 16
        document.pop("image_edit")

        restored = project_state(document)

        self.assertEqual(restored.image_edit, ImageEditState())

    def test_associated_project_prefers_embedded_metadata_then_exact_sidecar(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image_path = root / "source.png"
            embedded = project_document(ProjectState(512, 512, seed=71))
            metadata = PngImagePlugin.PngInfo()
            metadata.add_text("k2lab_project", json.dumps(embedded))
            Image.new("RGB", (512, 512), "black").save(image_path, pnginfo=metadata)
            sidecar = image_path.with_suffix(".k2lab.json")
            sidecar.write_text(
                json.dumps(project_document(ProjectState(512, 512, seed=19))),
                encoding="utf-8",
            )

            associated = load_associated_image_project(image_path)

            self.assertIsNotNone(associated)
            state, provenance = associated
            self.assertEqual(state.seed, 71)
            self.assertEqual(provenance, image_path.resolve())

            plain_path = root / "plain.jpg"
            Image.new("RGB", (512, 512), "white").save(plain_path)
            plain_sidecar = plain_path.with_suffix(".k2lab.json")
            plain_sidecar.write_text(
                json.dumps(project_document(ProjectState(512, 512, seed=23))),
                encoding="utf-8",
            )
            state, provenance = load_associated_image_project(plain_path)
            self.assertEqual(state.seed, 23)
            self.assertEqual(provenance, plain_sidecar)


if __name__ == "__main__":
    unittest.main()
