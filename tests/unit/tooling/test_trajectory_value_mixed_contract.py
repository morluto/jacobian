from __future__ import annotations

import json
from pathlib import Path
from typing import cast

import pytest
from benchmarks.tooling.trajectory_value_mixed_contract import (
    TrajectoryValueMixedStudyContract,
    load_frozen_study,
    validate_current_harbor_tasks,
    validate_frozen_study,
)
from pydantic import ValidationError

ROOT = Path(__file__).resolve().parents[3]
SPEC = ROOT / "benchmarks/config/trajectory-value-mixed-study-v1.json"
SCHEMA = (
    ROOT
    / "docs/reference/evaluations/schemas/trajectory-value-mixed-study-v1.schema.json"
)


def _value() -> dict[str, object]:
    return cast(dict[str, object], json.loads(SPEC.read_text(encoding="utf-8")))


def test_frozen_contract_recomputes_calibration_selected_population() -> None:
    validated = load_frozen_study(SPEC)
    assert [task.task_id for task in validated.contract.tasks] == [
        "graph-artifact-composition",
        "apollonius-gap-repair",
        "rp2-homology-lattice",
    ]
    assert validated.contract.repetitions_per_task == 8
    assert len(validated.contract.tasks) * validated.contract.repetitions_per_task == 24
    assert validated.contract.h3_warning_rule.startswith("any-strictly-negative")
    assert set(validated.task_contracts) == {
        ("mathematical-benchmarks-v1", "graph-artifact-composition"),
        ("mathematical-benchmarks-v1", "apollonius-gap-repair"),
        ("mathematical-benchmarks-v1", "rp2-homology-lattice"),
    }


def test_unknown_fields_fail_closed() -> None:
    value = _value()
    value["post_label_task_tuning"] = True
    with pytest.raises(ValidationError):
        TrajectoryValueMixedStudyContract.model_validate(value)


def test_estimator_order_is_preregistered() -> None:
    value = _value()
    estimators = value["estimator_comparison"]
    assert isinstance(estimators, list)
    estimators[0], estimators[1] = estimators[1], estimators[0]
    with pytest.raises(ValidationError, match="preregistered order"):
        TrajectoryValueMixedStudyContract.model_validate(value)


def test_task_substitution_fails_against_calibration_evidence() -> None:
    value = _value()
    tasks = value["tasks"]
    assert isinstance(tasks, list)
    tasks[0]["task_id"] = "polynomial-normalization"
    tasks[0]["task_group"] = "polynomial-normalization"
    contract = TrajectoryValueMixedStudyContract.model_validate(value)
    with pytest.raises(ValueError, match="differs from calibration selection"):
        validate_frozen_study(contract)


def test_calibration_evidence_digest_drift_fails_closed() -> None:
    value = _value()
    sources = value["calibration_sources"]
    assert isinstance(sources, list)
    sources[0]["summary_digest"] = "sha256:" + "0" * 64
    contract = TrajectoryValueMixedStudyContract.model_validate(value)
    with pytest.raises(ValueError, match="summary digest drift"):
        validate_frozen_study(contract)


def test_current_harbor_task_drift_fails_before_execution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    validated = load_frozen_study(SPEC, verify_current_tasks=False)

    class DriftedTask:
        def as_record(self) -> dict[str, object]:
            return {"drifted": True}

    monkeypatch.setattr(
        "benchmarks.tooling.trajectory_value_mixed_contract._task_contract",
        lambda _candidate: DriftedTask(),
    )
    with pytest.raises(ValueError, match="frozen Harbor task drift"):
        validate_current_harbor_tasks(validated)


def test_schema_matches_checked_in_contract() -> None:
    assert json.loads(SCHEMA.read_text(encoding="utf-8")) == (
        TrajectoryValueMixedStudyContract.model_json_schema()
    )
