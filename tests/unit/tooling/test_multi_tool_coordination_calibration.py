from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import pytest
from benchmarks.tooling.multi_tool_coordination_calibration import (
    _EXTENSION_TASK_ORDER,
    _FROZEN_TASK_ORDER,
    _TASK_ORDER,
    CoordinationCalibrationExtensionSpec,
    CoordinationCalibrationSpec,
    FrozenCoordinationEvaluationSpec,
    calibration_selection,
    load_frozen_evaluation,
    load_spec,
    main,
)
from pydantic import ValidationError

ROOT = Path(__file__).parents[3]
SPEC = ROOT / "benchmarks/config/multi-tool-coordination-pr2-calibration.json"
EXTENSION_SPEC = (
    ROOT / "benchmarks/config/multi-tool-coordination-pr2-calibration-extension.json"
)
FROZEN_SPEC = (
    ROOT / "benchmarks/config/multi-tool-coordination-pr3-frozen-comparison.json"
)


def test_pr2_calibration_freezes_the_cross_dataset_matrix() -> None:
    spec = load_spec(SPEC)

    assert tuple(task.task_id for task in spec.tasks) == _TASK_ORDER
    assert spec.repetitions_per_task * len(spec.tasks) == 24
    assert Counter(task.dataset for task in spec.tasks) == {
        "multi-tool-coordination-v1": 4,
        "symbolic-coordination-v1": 8,
    }
    probes = {probe for task in spec.tasks for probe in task.coordination_probe}
    assert "one-direction evidence audit" in probes
    assert "timeout record interpretation" in probes
    assert "equivalent sparse encoding" in probes
    assert spec.wrong_answer_retries == 0
    assert spec.tool_call_reward == 0
    assert spec.selection.target_minimum_tasks == 4
    assert not spec.causal_claim_authorized


def test_pr2_task_contracts_are_present_and_digest_bound() -> None:
    spec = load_spec(SPEC)

    for task in spec.tasks:
        path = ROOT / "benchmarks/datasets" / task.dataset / task.task_id
        assert path.is_dir()
        assert all(
            (path / relative).is_file()
            for relative in (
                "instruction.md",
                "environment/input.json",
                "environment/submission_schema.json",
                "tests/verifier.py",
                "tests/verifier_support.py",
            )
        )
        assert len(task.harbor_task_digest) == 64


def test_pr2_extension_freezes_six_new_candidates_without_changing_runtime() -> None:
    initial = load_spec(SPEC)
    extension = load_spec(EXTENSION_SPEC)

    assert isinstance(extension, CoordinationCalibrationExtensionSpec)
    assert tuple(task.task_id for task in extension.tasks) == _EXTENSION_TASK_ORDER
    assert not set(_EXTENSION_TASK_ORDER) & set(_TASK_ORDER)
    assert extension.parent_selected_task_ids == (
        "symbolic-coordination-semantic-equivalence-01",
    )
    assert extension.model == initial.model
    assert extension.agent_instructions == initial.agent_instructions
    assert extension.timeout_seconds_per_rollout == initial.timeout_seconds_per_rollout
    assert extension.repetitions_per_task * len(extension.tasks) == 12
    assert extension.selection.extension_policy == "no-further-extension"


def test_pr2_extension_task_contracts_are_digest_bound() -> None:
    extension = load_spec(EXTENSION_SPEC)

    for task in extension.tasks:
        path = ROOT / "benchmarks/datasets" / task.dataset / task.task_id
        assert path.is_dir()
        assert len(task.harbor_task_digest) == 64


def test_pr2_freezes_only_mixed_tasks_for_the_pr3_comparison() -> None:
    frozen = load_frozen_evaluation(FROZEN_SPEC)

    assert isinstance(frozen, FrozenCoordinationEvaluationSpec)
    assert frozen.selected_task_ids == _FROZEN_TASK_ORDER
    assert tuple(task.task_id for task in frozen.tasks) == _FROZEN_TASK_ORDER
    assert frozen.comparison.conditions == ("baseline", "treatment")
    assert frozen.comparison.repetitions_per_task_per_condition == 5
    assert frozen.comparison.tool_call_reward == 0
    assert frozen.target_minimum_met is False
    assert frozen.extension_exhausted is True
    assert frozen.causal_claim_authorized is False


def test_frozen_comparison_rejects_substituted_tasks_and_calibration() -> None:
    value = json.loads(FROZEN_SPEC.read_text())
    value["tasks"][0], value["tasks"][1] = value["tasks"][1], value["tasks"][0]
    with pytest.raises(ValidationError, match="frozen PR3 order"):
        FrozenCoordinationEvaluationSpec.model_validate(value)

    value = json.loads(FROZEN_SPEC.read_text())
    value["calibrations"].reverse()
    with pytest.raises(ValidationError, match="calibration references"):
        FrozenCoordinationEvaluationSpec.model_validate(value)

    value = json.loads(FROZEN_SPEC.read_text())
    value["calibrations"][0]["manifest_sha256"] = "sha256:" + "0" * 64
    with pytest.raises(ValidationError, match="calibration references"):
        FrozenCoordinationEvaluationSpec.model_validate(value)


def _record(task_id: str, acceptance: str, repetition: int) -> dict[str, object]:
    return {
        "task_id": task_id,
        "repetition": repetition,
        "terminal": {"acceptance": acceptance},
    }


def test_selection_uses_only_predeclared_mixed_outcomes_in_task_order() -> None:
    spec = load_spec(SPEC)
    records = []
    for index, task in enumerate(spec.tasks):
        if index < 5:
            labels = ("ACCEPTED", "REJECTED")
        elif index == 5:
            labels = ("ACCEPTED", "ACCEPTED")
        elif index == 6:
            labels = ("REJECTED", "REJECTED")
        elif index == 7:
            labels = ("ACCEPTED", "INCONCLUSIVE")
        else:
            labels = ("INCONCLUSIVE", "INCONCLUSIVE")
        records.extend(
            _record(task.task_id, label, repetition)
            for repetition, label in enumerate(labels, start=1)
        )

    result = calibration_selection(spec, records)

    assert result["eligible_task_ids"] == list(_TASK_ORDER[:5])
    assert result["selected_task_ids"] == list(_TASK_ORDER[:5])
    assert result["target_minimum_met"] is True
    assert result["extension_required"] is False
    assert result["extension_is_automatic"] is False
    rows = {row["task_id"]: row for row in result["tasks"]}
    assert rows[_TASK_ORDER[0]]["acceptance_rate"] == 0.5
    assert rows[_TASK_ORDER[0]]["wilson_95"] == pytest.approx(
        (0.094531205734, 0.905468794266)
    )
    assert rows[_TASK_ORDER[7]]["labelled_rollouts"] == 1
    assert rows[_TASK_ORDER[7]]["inconclusive"] == 1


def test_extension_selection_combines_parent_and_new_mixed_tasks() -> None:
    extension = load_spec(EXTENSION_SPEC)
    records = []
    for index, task in enumerate(extension.tasks):
        labels = ("ACCEPTED", "REJECTED") if index < 3 else ("ACCEPTED", "ACCEPTED")
        records.extend(
            _record(task.task_id, label, repetition)
            for repetition, label in enumerate(labels, start=1)
        )

    result = calibration_selection(extension, records)

    assert result["initial_selected_task_ids"] == [
        "symbolic-coordination-semantic-equivalence-01"
    ]
    assert result["new_selected_task_ids"] == list(_EXTENSION_TASK_ORDER[:3])
    assert result["selected_task_ids"] == [
        "symbolic-coordination-semantic-equivalence-01",
        *_EXTENSION_TASK_ORDER[:3],
    ]
    assert result["target_minimum_met"] is True
    assert result["extension_required"] is False
    assert result["extension_exhausted"] is True


def test_extension_never_authorizes_a_second_extension() -> None:
    extension = load_spec(EXTENSION_SPEC)
    records = [
        _record(task.task_id, "ACCEPTED", repetition)
        for task in extension.tasks
        for repetition in range(1, extension.repetitions_per_task + 1)
    ]

    result = calibration_selection(extension, records)

    assert result["selected_task_ids"] == [
        "symbolic-coordination-semantic-equivalence-01"
    ]
    assert result["target_minimum_met"] is False
    assert result["extension_required"] is False
    assert result["extension_exhausted"] is True


def test_substituted_task_order_fails_closed() -> None:
    value = json.loads(SPEC.read_text())
    value["tasks"][0], value["tasks"][1] = value["tasks"][1], value["tasks"][0]

    with pytest.raises(ValidationError, match="frozen PR2 calibration order"):
        load_spec_from_value(value)


def test_extension_parent_binding_and_task_order_fail_closed() -> None:
    value = json.loads(EXTENSION_SPEC.read_text())
    value["parent_manifest_sha256"] = "sha256:" + "0" * 64
    with pytest.raises(ValidationError):
        CoordinationCalibrationExtensionSpec.model_validate(value)

    value = json.loads(EXTENSION_SPEC.read_text())
    value["tasks"][0], value["tasks"][1] = value["tasks"][1], value["tasks"][0]
    with pytest.raises(ValidationError, match="frozen PR2 extension order"):
        CoordinationCalibrationExtensionSpec.model_validate(value)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("duplicate", "duplicate task repetition"),
        ("substituted", "unexpected task ID"),
        ("invalid-terminal", "invalid terminal acceptance"),
    ],
)
def test_selection_rejects_stale_or_substituted_records(
    mutation: str, message: str
) -> None:
    spec = load_spec(SPEC)
    records = [
        _record(task.task_id, "INCONCLUSIVE", repetition)
        for task in spec.tasks
        for repetition in range(1, spec.repetitions_per_task + 1)
    ]
    if mutation == "duplicate":
        records[-1] = records[0]
    elif mutation == "substituted":
        records[-1]["task_id"] = "not-a-frozen-task"
    else:
        records[-1]["terminal"] = {"acceptance": "TIMEOUT"}

    with pytest.raises(ValueError, match=message):
        calibration_selection(spec, records)


def test_selection_requires_the_complete_frozen_matrix() -> None:
    spec = load_spec(SPEC)
    records = [
        _record(task.task_id, "INCONCLUSIVE", repetition)
        for task in spec.tasks
        for repetition in range(1, spec.repetitions_per_task + 1)
    ]

    with pytest.raises(ValueError, match="exactly 24 rollout records"):
        calibration_selection(spec, records[:-1])


def test_extension_selection_requires_all_twelve_rollouts() -> None:
    extension = load_spec(EXTENSION_SPEC)
    records = [
        _record(task.task_id, "INCONCLUSIVE", repetition)
        for task in extension.tasks
        for repetition in range(1, extension.repetitions_per_task + 1)
    ]

    with pytest.raises(ValueError, match="exactly 12 rollout records"):
        calibration_selection(extension, records[:-1])


def load_spec_from_value(value: object) -> CoordinationCalibrationSpec:
    return CoordinationCalibrationSpec.model_validate(value)


def test_execution_requires_explicit_operator_opt_in() -> None:
    with pytest.raises(SystemExit, match="model execution is opt-in"):
        main(["run", "--spec", str(SPEC)])
