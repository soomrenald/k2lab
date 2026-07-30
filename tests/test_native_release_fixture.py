from __future__ import annotations

import json
import unittest
from pathlib import Path


ROOT = Path(__file__).parent.parent
FIXTURE = (
    Path(__file__).parent
    / "fixtures"
    / "parity"
    / "device"
    / "gate12_a40_100_job_soak.json"
)
CLEAN_IMAGE_FIXTURE = (
    Path(__file__).parent
    / "fixtures"
    / "parity"
    / "integration"
    / "gate12_clean_native_image.json"
)
BACKEND_VRAM_FIXTURE = (
    Path(__file__).parent
    / "fixtures"
    / "parity"
    / "device"
    / "gate12_a40_backend_vram.json"
)
ROLLBACK_FIXTURE = (
    Path(__file__).parent
    / "fixtures"
    / "parity"
    / "integration"
    / "gate12_rollback_readiness.json"
)
CLEAN_DESKTOP_FIXTURE = (
    Path(__file__).parent
    / "fixtures"
    / "parity"
    / "integration"
    / "gate12_clean_desktop_acceptance.json"
)


class NativeReleaseFixtureTests(unittest.TestCase):
    def setUp(self) -> None:
        self.fixture = json.loads(FIXTURE.read_text(encoding="utf-8"))

    def test_soak_uses_pinned_source_models_and_release_workload(self) -> None:
        self.assertEqual(
            self.fixture["schema_version"],
            "k2lab-gate12-release-evidence/1",
        )
        implementation = self.fixture["implementation"]
        self.assertIn(
            implementation["k2core_commit"],
            (ROOT / "pyproject.toml").read_text(encoding="utf-8"),
        )
        self.assertEqual(len(implementation["runpod_commit"]), 40)
        self.assertEqual(self.fixture["hardware"]["gpu"], "NVIDIA A40")
        for digest in self.fixture["model_hashes"].values():
            self.assertEqual(len(digest), 64)

        scope = self.fixture["test_scope"]
        self.assertEqual(scope["generation_jobs"], 100)
        self.assertEqual((scope["width"], scope["height"], scope["steps"]), (512, 512, 8))
        self.assertEqual((scope["sampler"], scope["scheduler"]), ("euler", "simple"))

    def test_soak_passes_determinism_growth_and_cleanup_contracts(self) -> None:
        results = self.fixture["results"]
        self.assertEqual(results["status"], "passed")
        self.assertEqual(results["completed_jobs"], 100)
        self.assertTrue(results["pixel_exact"])
        self.assertEqual(results["unique_pixel_hashes"], 1)
        self.assertEqual(len(results["pixel_sha256"]), 64)

        memory = results["memory"]
        for summary in (
            memory["nvidia_used_mib"],
            memory["worker_rss_bytes"],
        ):
            self.assertTrue(summary["passed"])
            self.assertLessEqual(summary["growth"], summary["tolerance"])
        self.assertTrue(results["cleanup"]["passed"])
        self.assertEqual(results["cleanup"]["initial_gpu_used_mib"], 0)
        self.assertEqual(results["cleanup"]["terminal_gpu_used_mib"], 0)

    def test_cost_is_recomputed_and_release_blockers_remain_explicit(self) -> None:
        execution = self.fixture["execution"]
        recomputed_cost = (
            execution["session_seconds"]
            / 3600
            * execution["published_a40_rate_usd_per_hour"]
        )
        self.assertAlmostEqual(
            recomputed_cost,
            execution["estimated_session_cost_usd"],
        )
        self.assertLess(execution["estimated_session_cost_usd"], 0.13)
        self.assertEqual(len(self.fixture["raw_evidence"]["state_sha256"]), 64)

        readiness = self.fixture["release_readiness"]
        self.assertFalse(readiness["native_backend_default_changed"])
        self.assertTrue(readiness["comfyui_fallback_retained"])
        self.assertTrue(readiness["experimental_selector_present"])
        self.assertTrue(readiness["prompt_safe_issue_report_present"])
        self.assertFalse(readiness["clean_native_image_built_and_booted"])
        self.assertTrue(readiness["clean_native_image_built_and_booted_locally"])
        self.assertFalse(
            readiness["clean_native_image_built_and_booted_on_runpod_gpu"]
        )
        self.assertTrue(
            readiness["identical_comfyui_a40_peak_memory_baseline_complete"]
        )
        self.assertTrue(readiness["peak_memory_ratio_threshold_passed"])
        self.assertFalse(readiness["first_party_and_model_license_review_complete"])
        self.assertFalse(readiness["representative_output_human_approval_complete"])

    def test_clean_native_image_uses_pinned_source_and_immutable_base(self) -> None:
        evidence = json.loads(CLEAN_IMAGE_FIXTURE.read_text(encoding="utf-8"))
        self.assertEqual(
            evidence["schema_version"],
            "k2lab-clean-native-image-evidence/1",
        )
        source = evidence["source"]
        self.assertEqual(len(source["runpod_commit"]), 40)
        self.assertEqual(source["k2core_commit"], self.fixture["implementation"]["k2core_commit"])
        self.assertRegex(source["base_image"], r"@sha256:[0-9a-f]{64}$")

        image = evidence["image"]
        self.assertRegex(image["image_id"], r"^sha256:[0-9a-f]{64}$")
        for key in ("manifest_list_sha256", "manifest_sha256", "config_sha256"):
            self.assertRegex(image[key], r"^[0-9a-f]{64}$")
        self.assertGreater(image["size_bytes"], 0)
        self.assertFalse(image["published"])

    def test_clean_native_image_smoke_passes_without_overstating_release(self) -> None:
        evidence = json.loads(CLEAN_IMAGE_FIXTURE.read_text(encoding="utf-8"))
        runtime = evidence["runtime"]
        self.assertEqual(runtime["inference_backend"], "native")
        self.assertEqual(runtime["worker_python"], "/opt/k2lab-venv/bin/python")
        self.assertEqual(runtime["fastapi"], "0.139.2")

        validation = evidence["validation"]
        for check in (
            "build_passed",
            "comfyui_tree_absent",
            "native_import_smoke_passed",
            "pip_check_passed",
            "agent_boot_passed",
            "runtime_pip_setuptools_wheel_absent",
            "development_node_modules_absent",
            "ruff_passed",
        ):
            self.assertTrue(validation[check])
        self.assertEqual(validation["authenticated_health_status"], "ready")
        self.assertTrue(validation["models_and_worker_expected_false_without_weights"])

        boundaries = evidence["release_boundaries"]
        security = evidence["security"]
        self.assertFalse(security["initial_scan"]["passed"])
        self.assertEqual(security["initial_scan"]["high"], 4)
        self.assertTrue(security["trivy"]["passed"])
        self.assertEqual(security["trivy"]["high"], 0)
        self.assertEqual(security["trivy"]["critical"], 0)
        self.assertEqual(security["trivy"]["secrets"], 0)
        self.assertTrue(security["sbom"]["emitted"])
        self.assertEqual(security["sbom"]["format"], "SPDX-2.3 JSON")
        for report in (security["trivy"], security["sbom"]):
            self.assertRegex(report["report_sha256"], r"^[0-9a-f]{64}$")
            self.assertGreater(report["report_bytes"], 0)

        self.assertTrue(boundaries["vulnerability_scan_complete"])
        self.assertTrue(boundaries["sbom_emitted"])
        self.assertFalse(boundaries["published_candidate_booted_on_runpod_gpu"])
        self.assertFalse(boundaries["clean_desktop_install_tested"])
        self.assertFalse(boundaries["rollback_drill_complete"])
        self.assertFalse(boundaries["release_approved"])

    def test_paired_a40_vram_evidence_passes_release_threshold_and_cleanup(self) -> None:
        evidence = json.loads(BACKEND_VRAM_FIXTURE.read_text(encoding="utf-8"))
        self.assertEqual(
            evidence["schema_version"],
            "k2lab-gate12-backend-vram-evidence/1",
        )
        self.assertEqual(evidence["hardware"]["gpu"], "NVIDIA A40")
        self.assertEqual(
            evidence["workload"]["fixture_sha256"],
            self.fixture["test_scope"]["fixture_sha256"],
        )
        self.assertEqual(evidence["model_hashes"], self.fixture["model_hashes"])

        comfy = evidence["backends"]["comfyui"]
        native = evidence["backends"]["native"]
        comparison = evidence["comparison"]
        recomputed_ratio = (
            native["lifecycle_peak_used_mib"] / comfy["lifecycle_peak_used_mib"]
        )
        self.assertAlmostEqual(
            recomputed_ratio,
            comparison["native_to_comfyui_peak_ratio"],
        )
        self.assertLessEqual(recomputed_ratio, comparison["limit"])
        self.assertTrue(comparison["passed"])
        self.assertTrue(evidence["execution"]["cleanup_passed"])
        self.assertEqual(evidence["execution"]["initial_gpu_used_mib"], 0)
        self.assertEqual(evidence["execution"]["terminal_gpu_used_mib"], 0)
        self.assertRegex(
            evidence["source"]["full_state_sha256"],
            r"^[0-9a-f]{64}$",
        )

    def test_rollback_drill_passes_without_claiming_release_approval(self) -> None:
        evidence = json.loads(ROLLBACK_FIXTURE.read_text(encoding="utf-8"))
        self.assertEqual(
            evidence["schema_version"],
            "k2lab-gate12-rollback-readiness/1",
        )
        self.assertEqual(evidence["desktop"]["unset_backend_selects"], "comfyui")
        self.assertTrue(evidence["desktop"]["session_fallback_present"])
        self.assertTrue(evidence["runpod"]["preserved_manifest_resolves"])
        self.assertRegex(
            evidence["runpod"]["preserved_image"],
            r"@sha256:[0-9a-f]{64}$",
        )
        self.assertTrue(evidence["native_candidate"]["published"])
        self.assertTrue(evidence["native_candidate"]["signed"])
        self.assertRegex(
            evidence["native_candidate"]["image"],
            r"@sha256:[0-9a-f]{64}$",
        )
        acceptance = evidence["acceptance"]
        self.assertRegex(acceptance["pod_id_suffix"], r"^[a-z0-9]{6}$")
        self.assertEqual(acceptance["native_generation_backend"], "native")
        self.assertEqual(acceptance["rollback_generation_backend"], "comfyui")
        self.assertTrue(acceptance["same_pod_image_swap"])
        self.assertTrue(acceptance["pod_deleted"])
        self.assertTrue(acceptance["pod_volume_deleted"])
        serialized = json.dumps(evidence).lower()
        for forbidden in ("api_key", "agent_secret", "session_token", '"prompt"'):
            self.assertNotIn(forbidden, serialized)

        result = evidence["result"]
        self.assertTrue(result["prerequisites_passed"])
        self.assertTrue(result["image_swap_to_native_completed"])
        self.assertTrue(result["image_swap_back_to_comfyui_completed"])
        self.assertTrue(result["rollback_drill_complete"])
        self.assertEqual(result["status"], "passed")
        self.assertFalse(result["release_approved"])

    def test_clean_desktop_acceptance_is_prompt_safe_and_non_persistent(self) -> None:
        evidence = json.loads(CLEAN_DESKTOP_FIXTURE.read_text(encoding="utf-8"))
        self.assertEqual(
            evidence["schema_version"],
            "k2lab-gate12-clean-desktop-acceptance/1",
        )
        self.assertRegex(evidence["source"]["wheel_sha256"], r"^[0-9a-f]{64}$")
        self.assertTrue(evidence["environment"]["fresh_virtual_environment"])
        self.assertFalse(evidence["environment"]["repository_venv_used"])
        self.assertFalse(evidence["environment"]["comfyui_python_package_visible"])
        self.assertTrue(evidence["environment"]["model_directories_empty"])

        checks = evidence["checks"]
        for check in (
            "installed_qml_loaded",
            "experimental_selector_present",
            "unset_backend_selected_comfyui",
            "native_session_selection_passed",
            "native_capability_gating_passed",
            "one_click_comfyui_fallback_passed",
        ):
            self.assertTrue(checks[check])
        self.assertFalse(checks["backend_environment_persisted"])

        report = evidence["issue_report"]
        self.assertRegex(report["sha256"], r"^[0-9a-f]{64}$")
        for field in (
            "private_prompt_included",
            "environment_variables_included",
            "debug_logs_included",
            "user_file_paths_included",
            "lora_names_included",
        ):
            self.assertFalse(report[field])
        self.assertTrue(
            evidence["result"]["clean_desktop_selector_fallback_report_complete"]
        )
        self.assertFalse(evidence["result"]["release_approved"])


if __name__ == "__main__":
    unittest.main()
