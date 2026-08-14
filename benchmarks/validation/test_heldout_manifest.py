from __future__ import annotations

from pathlib import Path

import pytest
from benchmarks.tooling.errors import HarborSuiteError
from benchmarks.tooling.heldout_manifest import validate_manifest
from benchmarks.validation.heldout_fixtures import _manifest, _write


def test_valid_manifest_freezes_c1_c2_and_budget_ladder(tmp_path: Path) -> None:
    manifest = validate_manifest(_write(tmp_path, _manifest()))

    assert manifest["conditions"][0] == {
        "id": "C1",
        "role": "PRIMARY_CONTROL",
        "jacobian_enabled": False,
    }
    assert manifest["experiment"]["stages"]["pilot"]["repetitions"] == 3


def test_manifest_rejects_unknown_stage_task(tmp_path: Path) -> None:
    value = _manifest()
    value["experiment"]["stages"]["pilot"]["task_ids"][0] = "unknown"

    with pytest.raises(HarborSuiteError, match="unknown task ids"):
        validate_manifest(_write(tmp_path, value))


def test_manifest_rejects_control_with_jacobian_image(tmp_path: Path) -> None:
    value = _manifest()
    value["conditions"][0]["image"] = "registry.invalid/c1@sha256:" + "1" * 64

    with pytest.raises(
        HarborSuiteError, match="strict configuration validation failed"
    ):
        validate_manifest(_write(tmp_path, value))


def test_manifest_rejects_non_digest_pinned_treatment_image(tmp_path: Path) -> None:
    value = _manifest()
    value["conditions"][1]["image"] = "registry.invalid/c2:latest"

    with pytest.raises(HarborSuiteError, match="held-out manifest is invalid"):
        validate_manifest(_write(tmp_path, value))


def test_manifest_rejects_v2_schema_version(tmp_path: Path) -> None:
    value = _manifest()
    value["schema_version"] = "2"

    with pytest.raises(
        HarborSuiteError, match="strict configuration validation failed"
    ):
        validate_manifest(_write(tmp_path, value))


def test_manifest_rejects_extra_top_level_field(tmp_path: Path) -> None:
    value = _manifest()
    value["legacy_provenance"] = "should be rejected"

    with pytest.raises(
        HarborSuiteError, match="strict configuration validation failed"
    ):
        validate_manifest(_write(tmp_path, value))


def test_manifest_rejects_extra_nested_field(tmp_path: Path) -> None:
    value = _manifest()
    value["snapshot_lock"]["extra_field"] = "rejected"

    with pytest.raises(
        HarborSuiteError, match="strict configuration validation failed"
    ):
        validate_manifest(_write(tmp_path, value))


def test_manifest_rejects_string_coercion_of_integer(tmp_path: Path) -> None:
    value = _manifest()
    value["experiment"]["max_tokens"] = "100000"

    with pytest.raises(
        HarborSuiteError, match="strict configuration validation failed"
    ):
        validate_manifest(_write(tmp_path, value))


def test_manifest_rejects_bool_coercion_of_integer(tmp_path: Path) -> None:
    value = _manifest()
    value["dataset"]["minimum_independent_families"] = True

    with pytest.raises(
        HarborSuiteError, match="strict configuration validation failed"
    ):
        validate_manifest(_write(tmp_path, value))


def test_manifest_rejects_wrong_root_type(tmp_path: Path) -> None:
    path = tmp_path / "manifest.json"
    path.write_text("[]", encoding="utf-8")

    with pytest.raises(
        HarborSuiteError, match="strict configuration validation failed"
    ):
        validate_manifest(path)


def test_manifest_rejects_missing_required_field(tmp_path: Path) -> None:
    value = _manifest()
    del value["snapshot_lock"]

    with pytest.raises(
        HarborSuiteError, match="strict configuration validation failed"
    ):
        validate_manifest(_write(tmp_path, value))


def test_manifest_rejects_wrong_condition_id(tmp_path: Path) -> None:
    value = _manifest()
    value["conditions"][0]["id"] = "C3"

    with pytest.raises(
        HarborSuiteError, match="strict configuration validation failed"
    ):
        validate_manifest(_write(tmp_path, value))


def test_manifest_rejects_wrong_agent_name(tmp_path: Path) -> None:
    value = _manifest()
    value["experiment"]["agent"]["name"] = "claude"

    with pytest.raises(
        HarborSuiteError, match="strict configuration validation failed"
    ):
        validate_manifest(_write(tmp_path, value))
