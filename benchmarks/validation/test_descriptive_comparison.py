"""Repeated benchmark observations do not manufacture inferential evidence."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

import pytest
from benchmarks.tooling.heldout_observations import _family_binding
from benchmarks.tooling.observation_comparison import compare_evidence
from benchmarks.validation.observation_results_support import _DIGEST, _evidence, _trial


def _arms(
    rows: list[tuple[str, str, list[float]]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    arms = []
    for condition in ("control", "treatment"):
        value = _evidence(condition, [])
        value["schema_version"] = "5"
        value["task_family_binding"] = {
            "manifest_digest": _DIGEST,
            "tasks": [
                {"task": task, "family": family, "digest": _DIGEST}
                for task, family, _ in rows
            ],
        }
        value["fixed_invariants"]["tasks"] = [
            {"task": task, "digest": _DIGEST} for task, _, _ in rows
        ]
        for task, _, deltas in rows:
            for repetition, delta in enumerate(deltas):
                reward = (
                    float(delta < 0) if condition == "control" else float(delta > 0)
                )
                trial = _trial(repetition, reward)
                trial["task"] = task
                value["trials"].append(trial)
        arms.append(value)
    return arms[0], arms[1]


@pytest.mark.parametrize("repetitions", [1, 10, 25])
def test_repetition_count_never_promotes_one_task(repetitions: int) -> None:
    report = compare_evidence(*_arms([("a", "one", [1.0] * repetitions)]))
    metric = report["metrics"]["correctness"]
    assert report["status"] == "VALID"
    assert (report["task_count"], report["family_count"]) == (1, 1)
    assert report["repetitions_per_task"] == {"a": repetitions}
    assert metric["task_average"]["paired_delta"] == 1
    assert metric["interpretation"] == "descriptive-only"
    assert metric["bootstrap_95_interval"] is metric["mcnemar_exact_p"] is None


def test_unbalanced_repetitions_keep_pair_and_task_weights_explicit() -> None:
    report = compare_evidence(*_arms([("a", "one", [1.0] * 9), ("b", "two", [-1.0])]))
    metric = report["metrics"]["correctness"]
    assert metric["paired_delta"] == pytest.approx(0.8)
    assert metric["weighting"] == "equal-observed-pairs"
    assert metric["task_average"]["paired_delta"] == 0
    assert metric["task_average"]["weighting"] == "equal-observed-task-means"


def test_unbalanced_families_do_not_replace_task_weighting() -> None:
    report = compare_evidence(
        *_arms([("a", "one", [1.0]), ("b", "one", [1.0]), ("c", "two", [-1.0])])
    )
    metric = report["metrics"]["correctness"]
    assert metric["task_average"]["paired_delta"] == pytest.approx(1 / 3)
    assert [family["paired_delta"] for family in metric["families"]] == [1, -1]
    assert [family["task_count"] for family in metric["families"]] == [2, 1]
    assert report["family_count"] == 2


def test_duplicate_or_permuted_repetitions_preserve_task_estimates() -> None:
    rows = [("a", "one", [1.0, -1.0, 1.0]), ("b", "two", [-1.0, 1.0, 0.0])]
    original = compare_evidence(*_arms(rows))["metrics"]["correctness"]
    for transformed in (
        [(task, family, deltas * 5) for task, family, deltas in rows],
        [(task, family, list(reversed(deltas))) for task, family, deltas in rows],
    ):
        metric = compare_evidence(*_arms(transformed))["metrics"]["correctness"]
        assert metric["task_average"] == original["task_average"]
        assert metric["families"] == original["families"]
        assert metric["bootstrap_95_interval"] is metric["mcnemar_exact_p"] is None


def test_optional_metric_missingness_remains_visible() -> None:
    control, treatment = _arms([("a", "one", [1.0, 1.0]), ("b", "one", [-1.0])])
    treatment["trials"][0]["cost_usd"] = None
    treatment["trials"][2]["cost_usd"] = None
    metric = compare_evidence(control, treatment)["metrics"]["cost_usd"]
    assert metric["pair_count"] == 1
    assert metric["missing_pair_count"] == 2
    assert metric["task_average"]["observed_task_count"] == 1
    assert metric["task_average"]["missing_task_count"] == 1
    assert metric["tasks"][1]["paired_delta"] is None
    assert metric["families"][0]["missing_task_count"] == 1


def test_missing_pair_still_invalid_even_when_task_means_exist() -> None:
    control, treatment = _arms([("a", "one", [1.0, 1.0])])
    treatment["trials"].pop()
    report = compare_evidence(control, treatment)
    assert report["status"] == "INVALID"
    assert (
        "control/treatment trials do not pair exactly" in report["validation_failures"]
    )


@pytest.mark.parametrize(
    "mutation", ["family", "manifest", "digest", "missing", "duplicate", "trial-digest"]
)
def test_conflicting_or_incomplete_family_binding_is_invalid(mutation: str) -> None:
    control, treatment = _arms([("a", "one", [1.0]), ("b", "two", [1.0])])
    binding = treatment["task_family_binding"]
    if mutation == "family":
        binding["tasks"][0]["family"] = "other"
    elif mutation == "manifest":
        binding["manifest_digest"] = "sha256:" + "b" * 64
    elif mutation == "digest":
        binding["tasks"][0]["digest"] = "sha256:" + "b" * 64
    elif mutation == "missing":
        binding["tasks"].pop()
    elif mutation == "duplicate":
        binding["tasks"].append(deepcopy(binding["tasks"][0]))
    else:
        treatment["trials"][0]["task_digest"] = "sha256:" + "b" * 64
    assert compare_evidence(control, treatment)["status"] == "INVALID"


def test_legacy_evidence_has_unknown_families_and_no_inference() -> None:
    report = compare_evidence(
        _evidence("control", [0.0] * 25), _evidence("treatment", [1.0] * 25)
    )
    assert report["status"] == "VALID"
    assert report["family_count"] is None
    metric = report["metrics"]["correctness"]
    assert metric["families"] is None
    assert metric["interpretation"] == "descriptive-only"
    assert metric["bootstrap_95_interval"] is metric["mcnemar_exact_p"] is None


def test_many_tasks_in_one_family_never_claim_independence() -> None:
    report = compare_evidence(*_arms([(f"task{i}", "one", [1.0]) for i in range(25)]))
    assert report["task_count"] == 25
    assert report["family_count"] == 1
    assert report["metrics"]["correctness"]["interpretation"] == "descriptive-only"


@pytest.mark.parametrize("defect", [None, "manifest", "task", "repetition", "digest"])
def test_family_projection_requires_frozen_selected_stage(defect: str | None) -> None:
    manifest = {
        "tasks": [
            {"id": "a", "family": "one", "digest": _DIGEST},
            {"id": "unselected", "family": "two", "digest": _DIGEST},
        ],
        "experiment": {"stages": {"pilot": {"task_ids": ["a"], "repetitions": 1}}},
    }
    plan = {"manifest_digest": _DIGEST, "stage": "pilot"}
    trial = {"task": "a", "repetition": 0, "task_digest": _DIGEST}
    digest = _DIGEST
    if defect == "manifest":
        digest = "sha256:" + "b" * 64
    elif defect == "task":
        trial["task"] = "unselected"
    elif defect == "repetition":
        trial["repetition"] = 1
    elif defect == "digest":
        trial["task_digest"] = "sha256:" + "b" * 64
    binding, failures = _family_binding(manifest, digest, plan, plan, [trial])
    if defect is None:
        assert failures == []
        assert binding == {
            "manifest_digest": _DIGEST,
            "tasks": [{"task": "a", "digest": _DIGEST, "family": "one"}],
        }
    else:
        assert binding is None
        assert failures


def test_one_missing_arm_binding_is_not_silently_imputed() -> None:
    control, treatment = _arms([("a", "one", [1.0])])
    treatment["task_family_binding"] = None
    report = compare_evidence(control, treatment)
    assert report["status"] == "INVALID"
    assert report["family_count"] is None


def test_family_row_order_is_not_semantic() -> None:
    control, treatment = _arms([("a", "one", [1.0]), ("b", "two", [1.0])])
    treatment["task_family_binding"]["tasks"].reverse()
    assert compare_evidence(control, treatment)["status"] == "VALID"


def test_schema_migration_and_null_inference_contract() -> None:
    import json
    from pathlib import Path

    from jsonschema import Draft202012Validator

    schemas = Path(__file__).parents[1] / "schemas"
    evidence_schema = Draft202012Validator(
        json.loads((schemas / "observation-evidence.schema.json").read_text())
    )
    report_schema = Draft202012Validator(
        json.loads((schemas / "comparison-report.schema.json").read_text())
    )
    control, treatment = _arms([("a", "one", [1.0])])
    assert evidence_schema.is_valid(control)
    report = compare_evidence(control, treatment)
    assert report_schema.is_valid(report)
    report["metrics"]["correctness"]["mcnemar_exact_p"] = 1.0
    assert not report_schema.is_valid(report)
    control.pop("task_family_binding")
    assert not evidence_schema.is_valid(control)
    control["schema_version"] = "4"
    assert evidence_schema.is_valid(control)


@pytest.mark.parametrize("location", ["runtime_snapshot", "fixed_invariants"])
def test_matching_arms_cannot_contradict_their_runtime_manifest(location: str) -> None:
    control, treatment = _arms([("a", "one", [1.0])])
    for arm in (control, treatment):
        runtime = (
            arm[location]
            if location == "runtime_snapshot"
            else arm[location]["runtime"]
        )
        runtime["manifest_digest"] = "sha256:" + "b" * 64
    report = compare_evidence(control, treatment)
    assert report["status"] == "INVALID"
    assert report["family_count"] is None


@pytest.mark.parametrize("digest", [_DIGEST, "sha256:" + "b" * 64])
def test_duplicate_frozen_task_rows_cannot_be_overwritten(digest: str) -> None:
    control, treatment = _arms([("a", "one", [1.0])])
    for arm in (control, treatment):
        arm["fixed_invariants"]["tasks"].insert(0, {"task": "a", "digest": digest})
    assert compare_evidence(control, treatment)["status"] == "INVALID"
