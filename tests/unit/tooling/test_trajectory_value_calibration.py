from __future__ import annotations

import json
from pathlib import Path

import pytest
from benchmarks.tooling.trajectory_value_calibration import (
    TrajectoryValueCalibrationSpec,
    _task_contract,
    load_spec,
    summarize,
)
from pydantic import ValidationError

ROOT = Path(__file__).resolve().parents[3]
SPEC = ROOT / "benchmarks/config/trajectory-value-calibration-v1.json"
EXTENSION_SPEC = (
    ROOT / "benchmarks/config/trajectory-value-calibration-extension-v1.json"
)


def _record(dataset: str, task: str, acceptance: str) -> dict[str, object]:
    return {
        "dataset_id": dataset,
        "task_id": task,
        "terminal": {"acceptance": acceptance},
    }


def test_preregistered_spec_is_closed_and_binds_real_harbor_tasks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "benchmarks.tooling.trajectory_value_calibration._harbor_task_digest",
        lambda _path: "sha256:" + "0" * 64,
    )
    spec = load_spec(SPEC)
    assert len(spec.candidates) == 8
    assert spec.repetitions_per_candidate == 2
    assert spec.retries_for_wrong_answers == 0
    contracts = [_task_contract(candidate) for candidate in spec.candidates]
    assert all(contract.harbor_digest.startswith("sha256:") for contract in contracts)
    assert all(
        "verifier.py" in contract.verifier_file_digests for contract in contracts
    )


def test_unknown_spec_fields_fail_closed() -> None:
    value = json.loads(SPEC.read_text(encoding="utf-8"))
    value["post_label_tuning"] = True
    with pytest.raises(ValidationError):
        TrajectoryValueCalibrationSpec.model_validate(value)


def test_extension_is_a_separate_preregistered_candidate_batch() -> None:
    initial = load_spec(SPEC)
    extension = load_spec(EXTENSION_SPEC)
    assert extension.calibration_id != initial.calibration_id
    assert len(extension.candidates) == 8
    assert not {
        (candidate.dataset_id, candidate.task_id) for candidate in initial.candidates
    }.intersection(
        (candidate.dataset_id, candidate.task_id) for candidate in extension.candidates
    )


def test_candidate_coverage_is_required() -> None:
    value = json.loads(SPEC.read_text(encoding="utf-8"))
    value["candidates"] = value["candidates"][:4]
    with pytest.raises(ValidationError, match="every declared trap"):
        TrajectoryValueCalibrationSpec.model_validate(value)


def test_selection_uses_only_labelled_terminal_outcomes_in_candidate_order() -> None:
    spec = load_spec(SPEC)
    records: list[dict[str, object]] = []
    expected: list[str] = []
    for index, candidate in enumerate(spec.candidates):
        if index < 5:
            outcomes = ("ACCEPTED", "REJECTED")
            if index < 4:
                expected.append(candidate.task_id)
        elif index == 5:
            outcomes = ("ACCEPTED", "ACCEPTED")
        elif index == 6:
            outcomes = ("REJECTED", "REJECTED")
        else:
            outcomes = ("ACCEPTED", "INCONCLUSIVE")
        records.extend(
            _record(candidate.dataset_id, candidate.task_id, outcome)
            for outcome in outcomes
        )
    summary = summarize(spec, records)
    assert [row["task_id"] for row in summary["selected_tasks"]] == expected
    assert summary["accepted"] == 8
    assert summary["rejected"] == 7
    assert summary["inconclusive"] == 1


def test_schema_matches_checked_in_contract() -> None:
    checked_in = json.loads(
        (
            ROOT
            / "docs/reference/evaluations/schemas/trajectory-value-calibration-v1.schema.json"
        ).read_text(encoding="utf-8")
    )
    assert checked_in == TrajectoryValueCalibrationSpec.model_json_schema()
