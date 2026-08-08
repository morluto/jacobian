"""Run the frozen PR2 multi-dataset coordination calibration."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Literal, Self

from pydantic import ConfigDict, Field, model_validator

from benchmarks.tooling.command_runner import git_head_sha
from benchmarks.tooling.harbor_suite import (
    get_suite,
    select_task_refs,
    task_digest,
    verifier_bundle_checksum,
)
from benchmarks.tooling.multi_tool_coordination_study import (
    StudyModel,
    StudyTask,
    _artifact_manifest,
    _codex_version,
    _digest,
    _infrastructure_failure_record,
    _model_record,
    _now,
    _object_digest,
    _parent_revision,
    _repository_is_clean,
    _run_one,
    _tmux_session,
    _write_json,
)
from jacobian.contracts.results import ContractModel

_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_SPEC = _ROOT / "benchmarks/config/multi-tool-coordination-pr2-calibration.json"
_DEFAULT_OUTPUT = _ROOT / "benchmarks/results/multi-tool-coordination-pr2-calibration"
_PUBLIC_FILES = (
    "instruction.md",
    "environment/input.json",
    "environment/submission_schema.json",
)
_TASK_ORDER = (
    "coordination-graph-set-distance-01",
    "coordination-cycle-lattice-01",
    "coordination-rational-slice-01",
    "coordination-directed-proportionality-01",
    "symbolic-coordination-valid-inverse-05",
    "symbolic-coordination-near-miss-01",
    "symbolic-coordination-one-direction-01",
    "symbolic-coordination-keller-only-01",
    "symbolic-coordination-collision-found-01",
    "symbolic-coordination-grid-exhausted-01",
    "symbolic-coordination-search-timeout-01",
    "symbolic-coordination-semantic-equivalence-01",
)
_EXTENSION_TASK_ORDER = (
    "symbolic-coordination-valid-inverse-04",
    "symbolic-coordination-near-miss-04",
    "symbolic-coordination-one-direction-03",
    "symbolic-coordination-semantic-equivalence-02",
    "symbolic-coordination-semantic-equivalence-03",
    "symbolic-coordination-semantic-equivalence-04",
)
_FROZEN_TASK_ORDER = (
    "symbolic-coordination-semantic-equivalence-01",
    "symbolic-coordination-semantic-equivalence-04",
)
_FROZEN_TASK_DIGESTS = (
    "799301c8118daf0c9328aafaace67e904eba0679bffd06ea44ff3c120137df4e",
    "9a99f6990bf3ac9b0fc5f81ad208ebb0b7f88917ab7f6d2018ae7d2cd324905c",
)
_FROZEN_CALIBRATION_REFS = (
    (
        "multi-tool-coordination-pr2-calibration",
        "sha256:b65f5a32438e44dbc58af7f1323f3d38d86a703df41a1ab8daa37044010944bb",
        "sha256:ce69b7bc01c25a5e187c735c8b8c7234197e6273e086e7462869b283e21ba362",
    ),
    (
        "multi-tool-coordination-pr2-calibration-extension",
        "sha256:ac1ac4b541c7771fa5ee2a8aeb302bad1b159f3fffe23e9bc90206cb833bd49a",
        "sha256:6d1f233d05a88b8cb9e46fefc53e437b2c1c85c0977a8d4e349700af3314a518",
    ),
)


class CalibrationTask(StudyTask):
    dataset: Literal["multi-tool-coordination-v1", "symbolic-coordination-v1"]


class CalibrationSelection(ContractModel):
    required_labelled_rollouts: Literal[2] = 2
    eligible_accepted_count: Literal[1] = 1
    eligible_rejected_count: Literal[1] = 1
    ordering: Literal["declared-task-order"] = "declared-task-order"
    maximum_selected_tasks: Literal[6] = 6
    target_minimum_tasks: Literal[4] = 4
    uncertainty: Literal["wilson-score-95"] = "wilson-score-95"
    extension_policy: Literal[
        "one-separately-preregistered-extension-of-at-most-six-new-candidates"
    ]


class CalibrationExtensionSelection(ContractModel):
    required_labelled_rollouts: Literal[2] = 2
    eligible_accepted_count: Literal[1] = 1
    eligible_rejected_count: Literal[1] = 1
    ordering: Literal["declared-task-order"] = "declared-task-order"
    maximum_selected_tasks: Literal[6] = 6
    target_minimum_tasks: Literal[4] = 4
    uncertainty: Literal["wilson-score-95"] = "wilson-score-95"
    extension_policy: Literal["no-further-extension"]


class CoordinationCalibrationSpec(ContractModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["1"] = "1"
    study_id: Literal["multi-tool-coordination-pr2-calibration"]
    evidence_class: Literal["public-host-local-difficulty-calibration"]
    causal_claim_authorized: Literal[False] = False
    harbor_execution_claimed: Literal[False] = False
    source_base_revision: Literal["098abff900e67f8e80c83df84fcc121358862750"]
    model: StudyModel
    repetitions_per_task: Literal[2] = 2
    timeout_seconds_per_rollout: Literal[600] = 600
    sandbox: Literal["workspace-write"] = "workspace-write"
    reasoning_log_mode: Literal["REQUIRED"] = "REQUIRED"
    web_search: Literal["disabled"] = "disabled"
    wrong_answer_retries: Literal[0] = 0
    terminal_reward: Literal["task-owned-clean-room-verifier-only"]
    tool_call_reward: Literal[0] = 0
    tasks: tuple[CalibrationTask, ...] = Field(min_length=12, max_length=12)
    selection: CalibrationSelection
    stop_rules: tuple[str, ...] = Field(min_length=4, max_length=8)
    agent_instructions: str = Field(min_length=1, max_length=4096)

    @model_validator(mode="after")
    def require_frozen_matrix(self) -> Self:
        if tuple(task.task_id for task in self.tasks) != _TASK_ORDER:
            raise ValueError("tasks must match the frozen PR2 calibration order")
        if len({task.task_id for task in self.tasks}) != len(self.tasks):
            raise ValueError("task IDs must be unique")
        if len({task.harbor_task_digest for task in self.tasks}) != len(self.tasks):
            raise ValueError("task digests must be unique")
        datasets = Counter(task.dataset for task in self.tasks)
        if datasets != {
            "multi-tool-coordination-v1": 4,
            "symbolic-coordination-v1": 8,
        }:
            raise ValueError("calibration must retain the frozen 4+8 dataset split")
        return self


class CoordinationCalibrationExtensionSpec(ContractModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["1"] = "1"
    study_id: Literal["multi-tool-coordination-pr2-calibration-extension"]
    evidence_class: Literal["public-host-local-difficulty-calibration-extension"]
    causal_claim_authorized: Literal[False] = False
    harbor_execution_claimed: Literal[False] = False
    source_base_revision: Literal["a33d6cd66f431cf1d904fcc719f24383869d4bd4"]
    parent_study_id: Literal["multi-tool-coordination-pr2-calibration"]
    parent_manifest_sha256: Literal[
        "sha256:b65f5a32438e44dbc58af7f1323f3d38d86a703df41a1ab8daa37044010944bb"
    ]
    parent_summary_sha256: Literal[
        "sha256:ce69b7bc01c25a5e187c735c8b8c7234197e6273e086e7462869b283e21ba362"
    ]
    parent_selected_task_ids: tuple[str, ...] = Field(min_length=1, max_length=1)
    extension_number: Literal[1] = 1
    model: StudyModel
    repetitions_per_task: Literal[2] = 2
    timeout_seconds_per_rollout: Literal[600] = 600
    sandbox: Literal["workspace-write"] = "workspace-write"
    reasoning_log_mode: Literal["REQUIRED"] = "REQUIRED"
    web_search: Literal["disabled"] = "disabled"
    wrong_answer_retries: Literal[0] = 0
    terminal_reward: Literal["task-owned-clean-room-verifier-only"]
    tool_call_reward: Literal[0] = 0
    tasks: tuple[CalibrationTask, ...] = Field(min_length=6, max_length=6)
    selection: CalibrationExtensionSelection
    selection_rationale: tuple[str, ...] = Field(min_length=3, max_length=6)
    stop_rules: tuple[str, ...] = Field(min_length=4, max_length=8)
    agent_instructions: str = Field(min_length=1, max_length=4096)

    @model_validator(mode="after")
    def require_frozen_extension(self) -> Self:
        if tuple(task.task_id for task in self.tasks) != _EXTENSION_TASK_ORDER:
            raise ValueError("tasks must match the frozen PR2 extension order")
        if len({task.harbor_task_digest for task in self.tasks}) != len(self.tasks):
            raise ValueError("task digests must be unique")
        if any(task.dataset != "symbolic-coordination-v1" for task in self.tasks):
            raise ValueError("extension tasks must use symbolic-coordination-v1")
        if set(_EXTENSION_TASK_ORDER) & set(_TASK_ORDER):
            raise ValueError("extension tasks must be new candidates")
        if self.parent_selected_task_ids != (
            "symbolic-coordination-semantic-equivalence-01",
        ):
            raise ValueError("parent selected tasks must match the frozen calibration")
        return self


CalibrationSpec = CoordinationCalibrationSpec | CoordinationCalibrationExtensionSpec


class FrozenCalibrationRef(ContractModel):
    study_id: Literal[
        "multi-tool-coordination-pr2-calibration",
        "multi-tool-coordination-pr2-calibration-extension",
    ]
    manifest_sha256: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    summary_sha256: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")


class FrozenComparisonDesign(ContractModel):
    conditions: tuple[Literal["baseline", "treatment"], ...] = Field(
        min_length=2, max_length=2
    )
    order: Literal["complete-baseline-before-product-change"]
    repetitions_per_task_per_condition: Literal[5] = 5
    timeout_seconds_per_rollout: Literal[600] = 600
    reasoning_log_mode: Literal["REQUIRED"] = "REQUIRED"
    web_search: Literal["disabled"] = "disabled"
    wrong_answer_retries: Literal[0] = 0
    terminal_reward: Literal["task-owned-clean-room-verifier-only"]
    tool_call_reward: Literal[0] = 0


class FrozenCoordinationEvaluationSpec(ContractModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["1"] = "1"
    study_id: Literal["multi-tool-coordination-pr3-frozen-comparison"]
    evidence_class: Literal["public-host-local-exploratory-controlled-comparison"]
    causal_claim_authorized: Literal[False] = False
    harbor_execution_claimed: Literal[False] = False
    model: StudyModel
    calibrations: tuple[FrozenCalibrationRef, ...] = Field(min_length=2, max_length=2)
    selection_rule: Literal[
        "exactly-one-accepted-and-one-rejected-across-two-calibration-rollouts"
    ]
    selected_task_ids: tuple[str, ...] = Field(min_length=2, max_length=2)
    tasks: tuple[CalibrationTask, ...] = Field(min_length=2, max_length=2)
    target_minimum_tasks: Literal[4] = 4
    target_minimum_met: Literal[False] = False
    extension_exhausted: Literal[True] = True
    intended_use: Literal["small-exploratory-paired-before-after-comparison"]
    comparison: FrozenComparisonDesign
    strategic_metrics: tuple[str, ...] = Field(min_length=3, max_length=8)
    stop_rules: tuple[str, ...] = Field(min_length=4, max_length=8)
    limitations: tuple[str, ...] = Field(min_length=2, max_length=6)
    agent_instructions: str = Field(min_length=1, max_length=4096)

    @model_validator(mode="after")
    def require_frozen_comparison(self) -> Self:
        if self.selected_task_ids != _FROZEN_TASK_ORDER:
            raise ValueError("selected tasks must match the frozen PR3 order")
        if tuple(task.task_id for task in self.tasks) != _FROZEN_TASK_ORDER:
            raise ValueError("tasks must match the frozen PR3 order")
        if (
            tuple(task.harbor_task_digest for task in self.tasks)
            != _FROZEN_TASK_DIGESTS
        ):
            raise ValueError("task digests must match the frozen PR3 bindings")
        calibration_refs = tuple(
            (ref.study_id, ref.manifest_sha256, ref.summary_sha256)
            for ref in self.calibrations
        )
        if calibration_refs != _FROZEN_CALIBRATION_REFS:
            raise ValueError("calibration references must match the frozen bindings")
        if self.comparison.conditions != ("baseline", "treatment"):
            raise ValueError("comparison conditions must be baseline then treatment")
        return self


def load_spec(path: Path) -> CalibrationSpec:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("calibration specification must be a JSON object")
    if value.get("study_id") == "multi-tool-coordination-pr2-calibration":
        return CoordinationCalibrationSpec.model_validate(value)
    if value.get("study_id") == "multi-tool-coordination-pr2-calibration-extension":
        return CoordinationCalibrationExtensionSpec.model_validate(value)
    raise ValueError("unsupported coordination calibration study_id")


def load_frozen_evaluation(path: Path) -> FrozenCoordinationEvaluationSpec:
    return FrozenCoordinationEvaluationSpec.model_validate_json(
        path.read_text(encoding="utf-8")
    )


def _task_records(
    spec: CalibrationSpec,
) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    for task in spec.tasks:
        refs = select_task_refs(get_suite(task.dataset), (task.task_id,))
        if len(refs) != 1:
            raise RuntimeError(f"task selection is ambiguous: {task.task_id}")
        ref = refs[0]
        observed = task_digest(ref.path)
        if observed != task.harbor_task_digest:
            raise RuntimeError(f"task digest drift: {task.task_id}")
        public = {relative: _digest(ref.path / relative) for relative in _PUBLIC_FILES}
        records[task.task_id] = {
            "path": ref.path,
            "dataset": task.dataset,
            "harbor_task_digest": observed,
            "public_files": public,
            "public_bundle_digest": _object_digest(public),
            "verifier_bundle_digest": "sha256:"
            + verifier_bundle_checksum(ref.path / "tests"),
        }
    return records


def _wilson_interval(successes: int, total: int) -> tuple[float, float] | None:
    if total == 0:
        return None
    z = 1.959963984540054
    rate = successes / total
    denominator = 1 + z * z / total
    center = (rate + z * z / (2 * total)) / denominator
    radius = (
        z
        * math.sqrt(rate * (1 - rate) / total + z * z / (4 * total * total))
        / denominator
    )
    return round(center - radius, 12), round(center + radius, 12)


def _validated_labels(
    spec: CalibrationSpec, records: Sequence[Mapping[str, object]]
) -> dict[str, list[str]]:
    by_task: dict[str, list[str]] = {task.task_id: [] for task in spec.tasks}
    expected_count = len(spec.tasks) * spec.repetitions_per_task
    if len(records) != expected_count:
        raise ValueError(
            f"calibration requires exactly {expected_count} rollout records"
        )
    seen_slots: set[tuple[str, int]] = set()
    for record in records:
        task_id = record.get("task_id")
        repetition = record.get("repetition")
        if not isinstance(task_id, str) or task_id not in by_task:
            raise ValueError("calibration record has an unexpected task ID")
        if not isinstance(repetition, int) or isinstance(repetition, bool):
            raise ValueError("calibration record has an invalid repetition")
        if not 1 <= repetition <= spec.repetitions_per_task:
            raise ValueError(
                "calibration record repetition is outside the frozen range"
            )
        slot = (task_id, repetition)
        if slot in seen_slots:
            raise ValueError("calibration contains a duplicate task repetition")
        seen_slots.add(slot)
        terminal = record.get("terminal")
        acceptance = (
            terminal.get("acceptance") if isinstance(terminal, Mapping) else None
        )
        if acceptance not in {"ACCEPTED", "REJECTED", "INCONCLUSIVE"}:
            raise ValueError("calibration record has an invalid terminal acceptance")
        if acceptance != "INCONCLUSIVE":
            by_task[task_id].append(str(acceptance))
    return by_task


def calibration_selection(
    spec: CalibrationSpec, records: Sequence[Mapping[str, object]]
) -> dict[str, object]:
    by_task = _validated_labels(spec, records)

    rows = []
    eligible = []
    for task in spec.tasks:
        labels = by_task[task.task_id]
        accepted = labels.count("ACCEPTED")
        rejected = labels.count("REJECTED")
        is_eligible = (
            len(labels) == spec.selection.required_labelled_rollouts
            and accepted == spec.selection.eligible_accepted_count
            and rejected == spec.selection.eligible_rejected_count
        )
        if is_eligible:
            eligible.append(task.task_id)
        rows.append(
            {
                "task_id": task.task_id,
                "dataset": task.dataset,
                "labelled_rollouts": len(labels),
                "accepted": accepted,
                "rejected": rejected,
                "inconclusive": spec.repetitions_per_task - len(labels),
                "acceptance_rate": (
                    round(accepted / len(labels), 12) if labels else None
                ),
                "wilson_95": _wilson_interval(accepted, len(labels)),
                "eligible": is_eligible,
            }
        )
    initial = (
        list(spec.parent_selected_task_ids)
        if isinstance(spec, CoordinationCalibrationExtensionSpec)
        else []
    )
    selected = (initial + eligible)[: spec.selection.maximum_selected_tasks]
    result: dict[str, object] = {
        "schema_version": "1",
        "rule": spec.selection.model_dump(mode="json"),
        "tasks": rows,
        "eligible_task_ids": eligible,
        "selected_task_ids": selected,
        "target_minimum_met": len(selected) >= spec.selection.target_minimum_tasks,
        "extension_required": (
            not isinstance(spec, CoordinationCalibrationExtensionSpec)
            and len(selected) < spec.selection.target_minimum_tasks
        ),
        "extension_is_automatic": False,
    }
    if isinstance(spec, CoordinationCalibrationExtensionSpec):
        result.update(
            {
                "extension_number": spec.extension_number,
                "initial_selected_task_ids": initial,
                "new_selected_task_ids": selected[len(initial) :],
                "extension_exhausted": True,
            }
        )
    return result


def run_calibration(spec_path: Path, output: Path) -> None:
    spec = load_spec(spec_path)
    if output.exists():
        raise RuntimeError(
            "output directory already exists; runs are never overwritten"
        )
    if not _repository_is_clean():
        raise RuntimeError("source tree must be clean at model-execution start")
    session = _tmux_session()
    source_revision = git_head_sha(_ROOT)
    if _parent_revision() != spec.source_base_revision:
        raise RuntimeError(
            "the calibration commit is not based directly on its frozen base revision"
        )
    output.mkdir(parents=True)
    started_at = _now()
    if _codex_version(_ROOT) != spec.model.codex_cli_version:
        raise RuntimeError("Codex CLI version does not match the preregistration")
    model_record, model_cache_digest = _model_record(spec)
    task_records = _task_records(spec)
    records: list[dict[str, Any]] = []
    for task in spec.tasks:
        for repetition in range(1, spec.repetitions_per_task + 1):
            try:
                record = _run_one(
                    spec=spec,
                    task=task,
                    task_record=task_records[task.task_id],
                    repetition=repetition,
                    output=output,
                )
            except Exception as exc:
                run_path = output / "runs" / f"{task.task_id}-r{repetition:02d}"
                existing = run_path / "run.json"
                record = (
                    json.loads(existing.read_text(encoding="utf-8"))
                    if existing.is_file()
                    else _infrastructure_failure_record(
                        task=task,
                        repetition=repetition,
                        output=output,
                        error=exc,
                    )
                )
            records.append(record)
    outcomes = Counter(record["terminal"]["acceptance"] for record in records)
    selection = calibration_selection(spec, records)
    summary = {
        "schema_version": "1",
        "study_id": spec.study_id,
        "evidence_class": spec.evidence_class,
        "run_count": len(records),
        "task_count": len(spec.tasks),
        "outcomes": dict(sorted(outcomes.items())),
        "reasoning_protocol": dict(
            sorted(Counter(record["reasoning_protocol"] for record in records).items())
        ),
        "extraction_error_count": sum(
            record["extraction_error"] is not None for record in records
        ),
        "selection": selection,
        "causal_claim_authorized": False,
        "harbor_execution_claimed": False,
    }
    if isinstance(spec, CoordinationCalibrationExtensionSpec):
        summary["extension_of"] = {
            "study_id": spec.parent_study_id,
            "manifest_sha256": spec.parent_manifest_sha256,
            "summary_sha256": spec.parent_summary_sha256,
        }
    _write_json(output / "summary.json", summary)
    manifest = {
        "schema_version": "1",
        "study_id": spec.study_id,
        "evidence_class": spec.evidence_class,
        "source_revision": source_revision,
        "source_tree_clean_at_start": True,
        "tmux_session": session,
        "started_at": started_at,
        "ended_at": _now(),
        "spec": {
            "path": spec_path.relative_to(_ROOT).as_posix(),
            "digest": _digest(spec_path),
        },
        "codex": {
            "version": spec.model.codex_cli_version,
            "model": spec.model.model_id,
            "reasoning_effort": spec.model.reasoning_effort,
            "model_catalog_record": model_record,
            "model_cache_digest": model_cache_digest,
            "api_key_forwarded": False,
        },
        "task_contracts": {
            task_id: {key: value for key, value in record.items() if key != "path"}
            for task_id, record in task_records.items()
        },
        "selection": selection,
        "outcomes": dict(sorted(outcomes.items())),
        "artifacts": _artifact_manifest(output),
    }
    if isinstance(spec, CoordinationCalibrationExtensionSpec):
        manifest["extension_of"] = {
            "study_id": spec.parent_study_id,
            "manifest_sha256": spec.parent_manifest_sha256,
            "summary_sha256": spec.parent_summary_sha256,
        }
    _write_json(output / "manifest.json", manifest)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    run = subparsers.add_parser("run")
    run.add_argument("--spec", type=Path, default=_DEFAULT_SPEC)
    run.add_argument("--output", type=Path, default=_DEFAULT_OUTPUT)
    run.add_argument("--execute", action="store_true")
    validate = subparsers.add_parser("validate")
    validate.add_argument("--spec", type=Path, default=_DEFAULT_SPEC)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "validate":
        spec = load_spec(args.spec)
        _task_records(spec)
        return 0
    if not args.execute:
        raise SystemExit("model execution is opt-in; pass --execute inside named tmux")
    run_calibration(args.spec.resolve(), args.output.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
