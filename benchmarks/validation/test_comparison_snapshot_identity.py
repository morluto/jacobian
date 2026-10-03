"""Pairing must retain the exact benchmark lock, not a subset of its fields."""

import json
from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest
from benchmarks.tooling.benchmark_snapshots import lock_digest_of
from benchmarks.tooling.observation_comparison import compare_evidence, render_markdown
from benchmarks.validation.observation_results_support import _evidence
from jsonschema import Draft202012Validator

ROOT = Path(__file__).parents[2]


def validate_report(report: dict[str, Any]) -> None:
    schema = json.loads(
        (ROOT / "benchmarks/schemas/comparison-report.schema.json").read_text()
    )
    Draft202012Validator(schema).validate(report)


@pytest.mark.parametrize("conditions", [("control", "treatment"), ("C1", "C2")])
def test_equal_snapshots_remain_pairable_and_report_the_identity(
    conditions: tuple[str, str],
) -> None:
    control, treatment = (_evidence(condition, [1.0]) for condition in conditions)
    report = compare_evidence(control, treatment)
    assert report["status"] == "VALID"
    assert report["snapshot_id"] == control["snapshot_id"]
    assert report["schema_version"] == "3"
    assert control["snapshot_id"] in render_markdown(report)
    validate_report(report)


@pytest.mark.parametrize("conditions", [("control", "treatment"), ("C1", "C2")])
def test_different_snapshots_cannot_report_a_valid_pair(
    conditions: tuple[str, str],
) -> None:
    control, treatment = (_evidence(condition, [1.0]) for condition in conditions)
    treatment["snapshot_id"] = "sha256:" + "b" * 64
    report = compare_evidence(control, treatment)
    assert report["status"] == "INVALID"
    assert "fixed invariant differs: snapshot_id" in report["validation_failures"]
    assert report["snapshot_id"] is None
    assert "no agreed snapshot" in render_markdown(report)
    validate_report(report)


def test_lock_drift_outside_fixed_invariants_is_not_hidden() -> None:
    lock_path = next(
        (ROOT / "benchmarks/snapshots/agent-workflow-v1").glob("*.lock.json")
    )
    original = json.loads(lock_path.read_text())
    changed = deepcopy(original)
    changed["source"]["registry_digest"] = "sha256:" + "0" * 64
    assert original["source"]["registry_digest"] != changed["source"]["registry_digest"]
    control, treatment = _evidence("control", [1.0]), _evidence("treatment", [1.0])
    control["snapshot_id"] = lock_digest_of(original)
    treatment["snapshot_id"] = lock_digest_of(changed)
    assert control["source_sha"] == treatment["source_sha"]
    assert control["fixed_invariants"] == treatment["fixed_invariants"]
    report = compare_evidence(control, treatment)
    assert report["status"] == "INVALID"
    assert report["snapshot_id"] is None
    validate_report(report)


def test_contradictory_harbor_versions_cannot_pair_under_the_same_snapshot() -> None:
    control, treatment = _evidence("control", [1.0]), _evidence("treatment", [1.0])
    treatment["harbor_version"] = "99.0.0"
    report = compare_evidence(control, treatment)
    assert report["status"] == "INVALID"
    assert "fixed invariant differs: harbor_version" in report["validation_failures"]
    assert report["snapshot_id"] == control["snapshot_id"]
    validate_report(report)
