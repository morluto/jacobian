"""Validate the frozen mixed-difficulty trajectory-value study contract.

The contract binds completed calibration evidence and recomputes task selection
from the preregistered rule.  It contains no main-study labels and cannot use a
post-label task substitution to change the frozen study population.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Self

from pydantic import ConfigDict, Field, model_validator

from benchmarks.tooling.trajectory_value_calibration import (
    CalibrationCandidate,
    CalibrationSelectionRule,
    TrajectoryValueCalibrationSpec,
    _task_contract,
    summarize,
)
from benchmarks.tooling.trajectory_value_calibration import (
    load_spec as load_calibration_spec,
)
from benchmarks.tooling.trajectory_value_study import StudyModel
from benchmarks.tooling.trajectory_value_study_verifier import (
    file_digest,
    object_digest,
)
from jacobian.contracts.results import ContractModel

_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_SPEC = _ROOT / "benchmarks/config/trajectory-value-mixed-study-v1.json"
_IDENTIFIER = r"^[a-z0-9][a-z0-9._-]{0,127}$"
_DIGEST = r"^sha256:[0-9a-f]{64}$"
_RELATIVE_JSON = r"^(?:[a-zA-Z0-9._-]+/)+[a-zA-Z0-9._-]+\.json$"
_CALIBRATION_TAGS = Literal[
    "candidate-checker-repair",
    "bounded-search-unknown",
    "scope-assurance",
    "one-sided-evidence",
    "artifact-binding",
    "capability-routing",
]
_ESTIMATORS = (
    "GROUP_ROLLOUT",
    "NUMCA_NUMERICAL",
    "REASONING_TEXT",
    "JACOBIAN_TYPED_EXACT",
    "ABSTRACT_VALUE_STATE",
    "ABSTRACT_VALUE_STATE_TEXT",
)


class CalibrationEvidenceSource(ContractModel):
    calibration_id: str = Field(pattern=_IDENTIFIER)
    manifest_path: str = Field(pattern=_RELATIVE_JSON)
    manifest_digest: str = Field(pattern=_DIGEST)
    summary_path: str = Field(pattern=_RELATIVE_JSON)
    summary_digest: str = Field(pattern=_DIGEST)


class FrozenMixedTask(ContractModel):
    dataset_id: str = Field(pattern=_IDENTIFIER)
    task_id: str = Field(pattern=_IDENTIFIER)
    task_group: str = Field(pattern=_IDENTIFIER)
    task_family: str = Field(min_length=1, max_length=128)
    calibration_id: str = Field(pattern=_IDENTIFIER)
    calibration_tags: tuple[_CALIBRATION_TAGS, ...] = Field(min_length=1)
    accepted: int = Field(ge=1, strict=True)
    rejected: int = Field(ge=1, strict=True)
    labelled: int = Field(ge=2, strict=True)
    success_rate_millionths: int = Field(ge=200_000, le=800_000, strict=True)
    calibration_result_digest: str = Field(pattern=_DIGEST)
    task_contract_digest: str = Field(pattern=_DIGEST)

    @model_validator(mode="after")
    def bind_rate_to_counts(self) -> Self:
        if self.accepted + self.rejected != self.labelled:
            raise ValueError("calibration counts must add to labelled rollouts")
        rate = round(self.accepted * 1_000_000 / self.labelled)
        if rate != self.success_rate_millionths:
            raise ValueError("success rate must be derived from calibration counts")
        if self.task_group != self.task_id:
            raise ValueError("v1 freezes one task per task group")
        return self


class FrozenSelectionPolicy(ContractModel):
    calibration_rule: CalibrationSelectionRule = CalibrationSelectionRule()
    source_combination: Literal["source-order-then-candidate-order"] = (
        "source-order-then-candidate-order"
    )
    maximum_main_tasks: Literal[4] = 4
    minimum_main_tasks: Literal[1, 2] = 1


class TrajectoryValueMixedStudyContract(ContractModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        json_schema_extra={
            "$id": "https://jacobian.invalid/docs/reference/evaluations/schemas/trajectory-value-mixed-study-v1.schema.json"
        },
    )

    schema_version: Literal["1"] = "1"
    study_id: str = Field(pattern=_IDENTIFIER)
    model: StudyModel
    calibration_sources: tuple[CalibrationEvidenceSource, ...] = Field(
        min_length=1, max_length=4
    )
    selection_policy: FrozenSelectionPolicy = FrozenSelectionPolicy()
    tasks: tuple[FrozenMixedTask, ...] = Field(min_length=1, max_length=4)
    repetitions_per_task: Literal[8] = 8
    timeout_seconds: Literal[420] = 420
    sandbox: Literal["workspace-write"] = "workspace-write"
    reasoning_log_mode: Literal["REQUIRED"] = "REQUIRED"
    tool_mode: Literal["direct"] = "direct"
    web_search: Literal["disabled"] = "disabled"
    agent_instructions: str = Field(min_length=1, max_length=8192)
    estimator_comparison: tuple[
        Literal[
            "GROUP_ROLLOUT",
            "NUMCA_NUMERICAL",
            "REASONING_TEXT",
            "JACOBIAN_TYPED_EXACT",
            "ABSTRACT_VALUE_STATE",
            "ABSTRACT_VALUE_STATE_TEXT",
        ],
        ...,
    ] = Field(min_length=6, max_length=6)
    h3_warning_rule: Literal[
        "any-strictly-negative-preterminal-selected-state-value-delta"
    ] = "any-strictly-negative-preterminal-selected-state-value-delta"
    terminal_reward: Literal["clean-room-verifier-acceptance-only"] = (
        "clean-room-verifier-acceptance-only"
    )
    retries_for_wrong_answers: Literal[0] = 0
    training_performed: Literal[False] = False
    scorer_intervention: Literal[False] = False
    exact_resume_supported: Literal[False] = False
    intermediate_value_surrogate: Literal[
        "leave-one-trajectory-out-success-frequency-among-compatible-cross-rollout-states"
    ] = "leave-one-trajectory-out-success-frequency-among-compatible-cross-rollout-states"
    label_status_at_freeze: Literal["main-labels-not-collected"] = (
        "main-labels-not-collected"
    )

    @model_validator(mode="after")
    def require_unique_frozen_population(self) -> Self:
        if self.estimator_comparison != _ESTIMATORS:
            raise ValueError("the six estimators must remain in preregistered order")
        source_ids = [source.calibration_id for source in self.calibration_sources]
        if len(set(source_ids)) != len(source_ids):
            raise ValueError("calibration sources must be unique")
        identities = [(task.dataset_id, task.task_id) for task in self.tasks]
        if len(set(identities)) != len(identities):
            raise ValueError("frozen tasks must be unique")
        if not self.selection_policy.minimum_main_tasks <= len(self.tasks):
            raise ValueError("the mixed study requires at least one task group")
        return self


@dataclass(frozen=True, slots=True)
class ValidatedFrozenStudy:
    contract: TrajectoryValueMixedStudyContract
    task_contracts: Mapping[tuple[str, str], Mapping[str, Any]]


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected one JSON object: {path}")
    return value


def _repo_file(relative: str) -> Path:
    candidate = _ROOT / relative
    if candidate.is_symlink() or not candidate.is_file():
        raise ValueError(f"frozen evidence must be a regular file: {relative}")
    resolved = candidate.resolve(strict=True)
    try:
        resolved.relative_to(_ROOT.resolve(strict=True))
    except ValueError as exc:
        raise ValueError("frozen evidence must remain inside the repository") from exc
    return resolved


def _verify_artifacts(
    manifest_path: Path, artifacts: Mapping[str, object]
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    manifest_root = manifest_path.parent.resolve(strict=True)
    for relative, expected_digest in sorted(artifacts.items()):
        if not isinstance(expected_digest, str):
            raise ValueError("malformed calibration artifact binding")
        artifact = manifest_path.parent / relative
        if artifact.is_symlink() or not artifact.is_file():
            raise ValueError(f"calibration artifact is not regular: {relative}")
        try:
            artifact.resolve(strict=True).relative_to(manifest_root)
        except ValueError as exc:
            raise ValueError("calibration artifact escapes its evidence root") from exc
        if file_digest(artifact) != expected_digest:
            raise ValueError(f"calibration artifact digest drift: {relative}")
        if relative.startswith("runs/") and relative.endswith("/run.json"):
            records.append(_read_json(artifact))
    return records


def _bound_calibration_spec(
    manifest: Mapping[str, Any],
) -> TrajectoryValueCalibrationSpec:
    raw_spec = manifest.get("spec")
    if not isinstance(raw_spec, dict) or not isinstance(raw_spec.get("path"), str):
        raise ValueError("calibration manifest has no spec binding")
    calibration_spec = load_calibration_spec(_repo_file(raw_spec["path"]))
    if object_digest(calibration_spec.model_dump(mode="json")) != raw_spec.get(
        "digest"
    ):
        raise ValueError("calibration spec digest drift")
    return calibration_spec


def _bound_task_contracts(
    manifest: Mapping[str, Any],
) -> dict[tuple[str, str], Mapping[str, Any]]:
    task_contracts: dict[tuple[str, str], Mapping[str, Any]] = {}
    raw_contracts = manifest.get("task_contracts")
    if not isinstance(raw_contracts, list):
        raise ValueError("calibration manifest has no task contracts")
    for raw in raw_contracts:
        if not isinstance(raw, dict):
            raise ValueError("malformed calibration task contract")
        key = (str(raw.get("dataset_id")), str(raw.get("task_id")))
        if key in task_contracts:
            raise ValueError("duplicate calibration task contract")
        task_contracts[key] = raw
    return task_contracts


def _source_selection(
    source: CalibrationEvidenceSource,
) -> tuple[
    list[dict[str, Any]],
    dict[tuple[str, str], Mapping[str, Any]],
    TrajectoryValueCalibrationSpec,
]:
    manifest_path = _repo_file(source.manifest_path)
    summary_path = _repo_file(source.summary_path)
    if file_digest(manifest_path) != source.manifest_digest:
        raise ValueError(f"calibration manifest digest drift: {source.calibration_id}")
    if file_digest(summary_path) != source.summary_digest:
        raise ValueError(f"calibration summary digest drift: {source.calibration_id}")
    manifest = _read_json(manifest_path)
    summary = _read_json(summary_path)
    if manifest.get("calibration_id") != source.calibration_id:
        raise ValueError("calibration manifest identity mismatch")
    if summary.get("calibration_id") != source.calibration_id:
        raise ValueError("calibration summary identity mismatch")
    if manifest.get("source_tree_clean_at_start") is not True:
        raise ValueError("calibration source tree was not clean")
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, dict):
        raise ValueError("calibration manifest has no artifact bindings")
    records = _verify_artifacts(manifest_path, artifacts)
    summary_relative = summary_path.relative_to(manifest_path.parent).as_posix()
    if artifacts.get(summary_relative) != source.summary_digest:
        raise ValueError("calibration manifest does not bind its summary")
    calibration_spec = _bound_calibration_spec(manifest)
    recomputed = summarize(calibration_spec, records)
    if recomputed != summary:
        raise ValueError("calibration summary differs from raw terminal records")
    selected = recomputed["selected_tasks"]
    return selected, _bound_task_contracts(manifest), calibration_spec


def _frozen_row(
    *,
    source: CalibrationEvidenceSource,
    row: Mapping[str, Any],
    task_contract: Mapping[str, Any],
) -> dict[str, Any]:
    success_rate = row.get("success_rate")
    if not isinstance(success_rate, (int, float)):
        raise ValueError("selected calibration row has no success rate")
    return {
        "dataset_id": row.get("dataset_id"),
        "task_id": row.get("task_id"),
        "task_group": row.get("task_id"),
        "task_family": row.get("task_family"),
        "calibration_id": source.calibration_id,
        "calibration_tags": row.get("calibration_tags"),
        "accepted": row.get("accepted"),
        "rejected": row.get("rejected"),
        "labelled": row.get("labelled"),
        "success_rate_millionths": round(success_rate * 1_000_000),
        "calibration_result_digest": object_digest(dict(row)),
        "task_contract_digest": object_digest(dict(task_contract)),
    }


def validate_frozen_study(
    contract: TrajectoryValueMixedStudyContract,
) -> ValidatedFrozenStudy:
    """Recompute the frozen task population from digest-bound calibrations."""

    derived: list[dict[str, Any]] = []
    selected_contracts: dict[tuple[str, str], Mapping[str, Any]] = {}
    for source in contract.calibration_sources:
        rows, task_contracts, calibration_spec = _source_selection(source)
        if (
            calibration_spec.selection_rule
            != contract.selection_policy.calibration_rule
        ):
            raise ValueError("calibration selection rule differs from frozen policy")
        if (
            calibration_spec.model != contract.model
            or calibration_spec.agent_instructions != contract.agent_instructions
            or calibration_spec.timeout_seconds != contract.timeout_seconds
            or calibration_spec.sandbox != contract.sandbox
            or calibration_spec.reasoning_log_mode != contract.reasoning_log_mode
            or calibration_spec.tool_mode != contract.tool_mode
            or calibration_spec.web_search != contract.web_search
            or calibration_spec.terminal_reward != contract.terminal_reward
            or calibration_spec.retries_for_wrong_answers
            != contract.retries_for_wrong_answers
            or calibration_spec.training_performed != contract.training_performed
            or calibration_spec.scorer_intervention != contract.scorer_intervention
        ):
            raise ValueError("main execution protocol differs from calibration")
        for row in rows:
            if not isinstance(row, dict) or row.get("selection_eligible") is not True:
                raise ValueError("calibration selected a non-eligible task")
            key = (str(row.get("dataset_id")), str(row.get("task_id")))
            task_contract = task_contracts.get(key)
            if task_contract is None:
                raise ValueError("selected task lacks a bound task contract")
            derived.append(
                _frozen_row(source=source, row=row, task_contract=task_contract)
            )
            selected_contracts[key] = task_contract
    maximum = contract.selection_policy.maximum_main_tasks
    expected = tuple(FrozenMixedTask.model_validate(row) for row in derived[:maximum])
    if contract.tasks != expected:
        raise ValueError("frozen task population differs from calibration selection")
    return ValidatedFrozenStudy(contract=contract, task_contracts=selected_contracts)


def validate_current_harbor_tasks(
    validated: ValidatedFrozenStudy,
) -> ValidatedFrozenStudy:
    """Fail before execution when a selected Harbor contract has drifted."""

    for task in validated.contract.tasks:
        actual = _task_contract(
            CalibrationCandidate(
                dataset_id=task.dataset_id,
                task_id=task.task_id,
                task_family=task.task_family,
                calibration_tags=task.calibration_tags,
            )
        ).as_record()
        if object_digest(actual) != task.task_contract_digest:
            raise ValueError(f"frozen Harbor task drift: {task.task_id}")
    return validated


def load_frozen_study(
    path: Path, *, verify_current_tasks: bool = True
) -> ValidatedFrozenStudy:
    contract = TrajectoryValueMixedStudyContract.model_validate_json(
        path.read_text(encoding="utf-8")
    )
    validated = validate_frozen_study(contract)
    if verify_current_tasks:
        return validate_current_harbor_tasks(validated)
    return validated


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, default=_DEFAULT_SPEC)
    parser.add_argument("--schema-output", type=Path)
    parser.add_argument("--print-derived-tasks", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    if args.schema_output is not None:
        _write_json(
            args.schema_output,
            TrajectoryValueMixedStudyContract.model_json_schema(),
        )
        return
    validated = load_frozen_study(args.spec)
    if args.print_derived_tasks:
        print(
            json.dumps(
                [task.model_dump(mode="json") for task in validated.contract.tasks],
                indent=2,
                sort_keys=True,
            )
        )


if __name__ == "__main__":
    main()


__all__ = [
    "CalibrationEvidenceSource",
    "FrozenMixedTask",
    "FrozenSelectionPolicy",
    "TrajectoryValueMixedStudyContract",
    "ValidatedFrozenStudy",
    "load_frozen_study",
    "validate_current_harbor_tasks",
    "validate_frozen_study",
]
