"""Run the preregistered mixed-difficulty Codex study and test H1--H3.

The runner exposes only each frozen Harbor task's public bundle, records
observable Codex and Jacobian evidence, and invokes the task-owned clean-room
verifier after Codex exits.  Analysis is retrospective and observation-only:
it cannot alter prompts, tools, retries, runtime behavior, terminal reward, or
mathematical assurance.
"""

from __future__ import annotations

import argparse
import asyncio
import gzip
import hashlib
import itertools
import json
import re
import shutil
import statistics
import tempfile
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Literal, Self

from pydantic import ConfigDict, Field, model_validator

from benchmarks.tooling.codex_visibility import inspect_surface
from benchmarks.tooling.command_runner import (
    ToolCommandStatus,
    git_head_sha,
    operator_environment,
    run_operator_command,
)
from benchmarks.tooling.trajectory_value_calibration import (
    CalibrationCandidate,
    HarborTaskContract,
    _copy_workspace,
    _regular_files,
    _task_contract,
)
from benchmarks.tooling.trajectory_value_mixed_contract import (
    FrozenMixedTask,
    TrajectoryValueMixedStudyContract,
    ValidatedFrozenStudy,
    load_frozen_study,
)
from benchmarks.tooling.trajectory_value_study import (
    _codex_version,
    _mcp_server,
    _model_cache,
    _read_reasoning_resource,
    _reasoning_run_ids,
    _repository_is_clean,
)
from benchmarks.tooling.trajectory_value_study_verifier import (
    file_digest,
    object_digest,
)
from benchmarks.validation._verifier_child import (
    VerifierExecutionError,
    run_verifier_in_child,
)
from jacobian.contracts.results import ContractModel
from jacobian.eval.telemetry import parse_agent_transcript
from jacobian.eval.trajectory_state import (
    CleanRoomTerminalEvidence,
    StateBoundary,
    TerminalAcceptance,
    TrajectoryExtraction,
    extract_codex_trajectory,
)
from jacobian.eval.trajectory_value import (
    LabelledTrajectory,
    TrajectoryValueCorpus,
    ValueEstimatorConfig,
)
from jacobian.eval.trajectory_value_abstraction import (
    EstimatorEvaluationV2,
    EstimatorKindV2,
    OfflineValueComparisonV2,
    SemanticStateValueEstimate,
    evaluate_semantic_trajectories,
)

_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_SPEC = _ROOT / "benchmarks/config/trajectory-value-hypothesis-study-v1.json"
_IDENTIFIER = r"^[a-z0-9][a-z0-9._-]{0,127}$"
_DIGEST = r"^sha256:[0-9a-f]{64}$"
_RELATIVE_JSON = r"^(?:[a-zA-Z0-9._-]+/)+[a-zA-Z0-9._-]+\.json$"
_PUBLIC_FILES = {
    "instruction.md": Path("instruction.md"),
    "input.json": Path("environment/input.json"),
    "submission_schema.json": Path("environment/submission_schema.json"),
}
_CODEX_ENVIRONMENT = (
    "HOME",
    "PATH",
    "CODEX_HOME",
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "ALL_PROXY",
    "NO_PROXY",
    "http_proxy",
    "https_proxy",
    "all_proxy",
    "no_proxy",
)
_TYPED_ESTIMATORS = (
    EstimatorKindV2.JACOBIAN_TYPED_EXACT,
    EstimatorKindV2.ABSTRACT_VALUE_STATE,
    EstimatorKindV2.ABSTRACT_VALUE_STATE_TEXT,
)
_BASELINES = (
    EstimatorKindV2.GROUP_ROLLOUT,
    EstimatorKindV2.NUMCA_NUMERICAL,
)


class FrozenMixedStudyReference(ContractModel):
    path: str = Field(pattern=_RELATIVE_JSON)
    file_digest: str = Field(pattern=_DIGEST)
    study_id: str = Field(pattern=_IDENTIFIER)


class H1Plan(ContractModel):
    question: Literal[
        "typed/value-state estimation outperforms group and Numca baselines on mixed terminal outcomes"
    ]
    metrics: tuple[Literal["brier_score", "mean_absolute_error"], ...] = (
        "brier_score",
        "mean_absolute_error",
    )
    baselines: tuple[Literal["GROUP_ROLLOUT", "NUMCA_NUMERICAL"], ...] = (
        "GROUP_ROLLOUT",
        "NUMCA_NUMERICAL",
    )
    typed_estimators: tuple[
        Literal[
            "JACOBIAN_TYPED_EXACT",
            "ABSTRACT_VALUE_STATE",
            "ABSTRACT_VALUE_STATE_TEXT",
        ],
        ...,
    ] = (
        "JACOBIAN_TYPED_EXACT",
        "ABSTRACT_VALUE_STATE",
        "ABSTRACT_VALUE_STATE_TEXT",
    )
    directional_support_rule: Literal[
        "strictly-lower-brier-and-mae-than-each-baseline"
    ] = "strictly-lower-brier-and-mae-than-each-baseline"
    uncertainty: Literal["per-estimate-wilson-score-95-and-reported-mean-width"] = (
        "per-estimate-wilson-score-95-and-reported-mean-width"
    )
    metric_weighting: Literal["equal-per-trajectory"] = "equal-per-trajectory"


class H2Plan(ContractModel):
    question: Literal[
        "reasoning text adds predictive information beyond abstract typed mathematical state"
    ]
    pair_scope: Literal[
        "different-trajectories-same-task-group-same-abstract-state"
    ] = "different-trajectories-same-task-group-same-abstract-state"
    require_different_reasoning_digest: Literal[True] = True
    require_mixed_terminal_outcome: Literal[True] = True
    branch_separation: Literal[
        "different-reasoning-text-and-abstract-plus-text-cluster-ids"
    ] = "different-reasoning-text-and-abstract-plus-text-cluster-ids"
    directional_support_rule: Literal[
        "at-least-one-mixed-outcome-pair-separated-and-hybrid-strictly-lower-brier-and-mae-than-abstract"
    ] = "at-least-one-mixed-outcome-pair-separated-and-hybrid-strictly-lower-brier-and-mae-than-abstract"


class H3Plan(ContractModel):
    question: Literal[
        "a negative estimated-value change predicts eventual failure before terminal verification"
    ]
    warning_rule: Literal[
        "any-strictly-negative-preterminal-selected-state-value-delta"
    ] = "any-strictly-negative-preterminal-selected-state-value-delta"
    threshold_tuned_on_main_labels: Literal[False] = False
    lead_time_unit: Literal["later-selected-preterminal-observations"] = (
        "later-selected-preterminal-observations"
    )
    false_alarm_denominator: Literal["accepted-labelled-trajectories"] = (
        "accepted-labelled-trajectories"
    )
    directional_support_rule: Literal[
        "positive-recall-precision-above-failure-prevalence-and-positive-mean-true-positive-lead"
    ] = "positive-recall-precision-above-failure-prevalence-and-positive-mean-true-positive-lead"


class TrajectoryValueHypothesisStudySpec(ContractModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        json_schema_extra={
            "$id": "https://jacobian.invalid/docs/reference/evaluations/schemas/trajectory-value-hypothesis-study-v1.schema.json"
        },
    )

    schema_version: Literal["1"] = "1"
    analysis_id: str = Field(pattern=_IDENTIFIER)
    mixed_study: FrozenMixedStudyReference
    evaluator_config: ValueEstimatorConfig = ValueEstimatorConfig()
    h1: H1Plan
    h2: H2Plan
    h3: H3Plan
    representative_selection: Literal[
        "lexicographically-first-accepted-and-rejected-labelled-trajectory-per-task-group"
    ] = "lexicographically-first-accepted-and-rejected-labelled-trajectory-per-task-group"
    terminal_reward: Literal["clean-room-verifier-acceptance-only"] = (
        "clean-room-verifier-acceptance-only"
    )
    scorer_mode: Literal["retrospective-observation-only"] = (
        "retrospective-observation-only"
    )
    retries_for_wrong_answers: Literal[0] = 0
    learned_components: Literal[False] = False
    training_performed: Literal[False] = False
    scorer_intervention: Literal[False] = False
    causal_claim_authorized: Literal[False] = False

    @model_validator(mode="after")
    def freeze_hypothesis_order_and_metrics(self) -> Self:
        if self.h1.metrics != ("brier_score", "mean_absolute_error"):
            raise ValueError("H1 metrics must remain in preregistered order")
        if self.h1.baselines != tuple(kind.value for kind in _BASELINES):
            raise ValueError("H1 baselines must remain in preregistered order")
        if self.h1.typed_estimators != tuple(kind.value for kind in _TYPED_ESTIMATORS):
            raise ValueError("H1 typed estimators must remain in preregistered order")
        return self


def _repo_file(relative: str) -> Path:
    candidate = _ROOT / relative
    if candidate.is_symlink() or not candidate.is_file():
        raise ValueError(f"preregistered file must be regular: {relative}")
    resolved = candidate.resolve(strict=True)
    try:
        resolved.relative_to(_ROOT.resolve(strict=True))
    except ValueError as exc:
        raise ValueError(
            "preregistered file must remain inside the repository"
        ) from exc
    return resolved


def _historical_mixed_path(mixed_path: Path, expected_digest: str) -> Path:
    if mixed_path.is_symlink() or not mixed_path.is_file():
        raise ValueError("historical mixed-study snapshot must be a regular file")
    mixed_path = mixed_path.resolve(strict=True)
    try:
        mixed_path.relative_to(_ROOT.resolve(strict=True))
    except ValueError as exc:
        raise ValueError(
            "historical mixed-study snapshot must remain inside the repository"
        ) from exc
    if file_digest(mixed_path) != expected_digest:
        raise ValueError("historical mixed-study snapshot digest drift")
    return mixed_path


def _historical_manifest(study_root: Path) -> dict[str, Any]:
    manifest_path = study_root / "manifest.json"
    if manifest_path.is_symlink() or not manifest_path.is_file():
        raise ValueError("historical study manifest must be a regular file")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise ValueError("historical study manifest must be one JSON object")
    if manifest.get("source_tree_clean_at_start") is not True:
        raise ValueError("historical study did not start from a clean tree")
    return manifest


def _validate_historical_preregistration(
    manifest: Mapping[str, Any],
    spec_path: Path,
    spec: TrajectoryValueHypothesisStudySpec,
) -> None:
    preregistration = manifest.get("preregistration")
    if not isinstance(preregistration, dict):
        raise ValueError("historical study has no preregistration binding")
    try:
        spec_relative = (
            spec_path.resolve(strict=True)
            .relative_to(_ROOT.resolve(strict=True))
            .as_posix()
        )
    except ValueError as exc:
        raise ValueError("preregistration must remain inside the repository") from exc
    if (
        preregistration.get("path") != spec_relative
        or preregistration.get("file_digest") != file_digest(spec_path)
        or preregistration.get("object_digest")
        != object_digest(spec.model_dump(mode="json"))
        or preregistration.get("labels_available_at_revision") is not False
    ):
        raise ValueError("historical preregistration binding drift")


def _historical_mixed_contract(
    manifest: Mapping[str, Any],
    study_root: Path,
    mixed_path: Path,
    spec: TrajectoryValueHypothesisStudySpec,
) -> TrajectoryValueMixedStudyContract:
    mixed = TrajectoryValueMixedStudyContract.model_validate_json(
        mixed_path.read_text(encoding="utf-8")
    )
    mixed_binding = manifest.get("mixed_study")
    if not isinstance(mixed_binding, dict):
        raise ValueError("historical study has no mixed-study binding")
    if (
        mixed_binding.get("path") != spec.mixed_study.path
        or mixed_binding.get("file_digest") != spec.mixed_study.file_digest
        or mixed_binding.get("object_digest")
        != object_digest(mixed.model_dump(mode="json"))
    ):
        raise ValueError("historical mixed-study binding drift")

    artifacts = manifest.get("artifacts")
    mixed_relative = mixed_path.relative_to(study_root).as_posix()
    if (
        not isinstance(artifacts, dict)
        or artifacts.get(mixed_relative) != spec.mixed_study.file_digest
    ):
        raise ValueError("historical manifest does not bind the mixed-study snapshot")
    return mixed


def _historical_task_contracts(
    manifest: Mapping[str, Any], mixed: TrajectoryValueMixedStudyContract
) -> dict[tuple[str, str], Mapping[str, Any]]:
    raw_contracts = manifest.get("task_contracts")
    if not isinstance(raw_contracts, list):
        raise ValueError("historical study has no task contracts")
    task_contracts: dict[tuple[str, str], Mapping[str, Any]] = {}
    for raw in raw_contracts:
        if not isinstance(raw, dict):
            raise ValueError("malformed historical task contract")
        key = (str(raw.get("dataset_id")), str(raw.get("task_id")))
        if key in task_contracts:
            raise ValueError("duplicate historical task contract")
        task_contracts[key] = raw
    expected_keys = {(task.dataset_id, task.task_id) for task in mixed.tasks}
    if set(task_contracts) != expected_keys:
        raise ValueError("historical task-contract population drift")
    for task in mixed.tasks:
        raw = task_contracts[(task.dataset_id, task.task_id)]
        if object_digest(dict(raw)) != task.task_contract_digest:
            raise ValueError(f"historical task contract drift: {task.task_id}")
    return task_contracts


def _historical_frozen_study(
    *,
    spec_path: Path,
    spec: TrajectoryValueHypothesisStudySpec,
    mixed_path: Path,
) -> ValidatedFrozenStudy:
    """Replay one completed study without authorizing a new execution."""

    mixed_path = _historical_mixed_path(mixed_path, spec.mixed_study.file_digest)
    study_root = mixed_path.parent.parent
    manifest = _historical_manifest(study_root)
    _validate_historical_preregistration(manifest, spec_path, spec)
    mixed = _historical_mixed_contract(manifest, study_root, mixed_path, spec)
    task_contracts = _historical_task_contracts(manifest, mixed)
    return ValidatedFrozenStudy(contract=mixed, task_contracts=task_contracts)


def load_hypothesis_spec(
    path: Path,
    *,
    verify_current_tasks: bool = True,
    historical_mixed_path: Path | None = None,
) -> tuple[TrajectoryValueHypothesisStudySpec, ValidatedFrozenStudy]:
    """Load and cross-bind the label-free H1--H3 preregistration."""

    path = path.resolve(strict=True)
    spec = TrajectoryValueHypothesisStudySpec.model_validate_json(
        path.read_text(encoding="utf-8")
    )
    if historical_mixed_path is None:
        mixed_path = _repo_file(spec.mixed_study.path)
        if file_digest(mixed_path) != spec.mixed_study.file_digest:
            raise ValueError("frozen mixed-study file digest drift")
        validated = load_frozen_study(
            mixed_path, verify_current_tasks=verify_current_tasks
        )
    else:
        if verify_current_tasks:
            raise ValueError(
                "historical mixed-study snapshots cannot authorize new execution"
            )
        validated = _historical_frozen_study(
            spec_path=path,
            spec=spec,
            mixed_path=historical_mixed_path,
        )
    mixed = validated.contract
    if mixed.study_id != spec.mixed_study.study_id:
        raise ValueError("mixed-study identity mismatch")
    if mixed.terminal_reward != spec.terminal_reward:
        raise ValueError("terminal reward differs from the frozen mixed study")
    if mixed.retries_for_wrong_answers != spec.retries_for_wrong_answers:
        raise ValueError("retry policy differs from the frozen mixed study")
    if mixed.h3_warning_rule != spec.h3.warning_rule:
        raise ValueError("H3 warning rule differs from the frozen mixed study")
    if mixed.training_performed or mixed.scorer_intervention:
        raise ValueError("frozen mixed study authorizes forbidden intervention")
    if mixed.estimator_comparison != tuple(kind.value for kind in EstimatorKindV2):
        raise ValueError("frozen mixed study estimator order drift")
    return spec, validated


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _prepare_workspace(
    workspace: Path,
    mixed: TrajectoryValueMixedStudyContract,
    task: FrozenMixedTask,
    contract: HarborTaskContract,
) -> str:
    workspace.mkdir()
    for name, relative in _PUBLIC_FILES.items():
        shutil.copyfile(contract.path / relative, workspace / name)
    (workspace / "evidence").mkdir()
    prompt = mixed.agent_instructions.strip() + "\n"
    (workspace / "prompt.txt").write_text(prompt, encoding="utf-8")
    visible = {
        name: file_digest(workspace / name) for name in contract.public_file_digests
    }
    if visible != dict(contract.public_file_digests):
        raise RuntimeError(f"public bundle drift during preparation: {task.task_id}")
    return prompt


def _codex_arguments(
    *,
    workspace: Path,
    mixed: TrajectoryValueMixedStudyContract,
    mcp_url: str,
    prompt: str,
) -> tuple[str, ...]:
    return (
        "-a",
        "never",
        "exec",
        "--ephemeral",
        "--skip-git-repo-check",
        "--ignore-user-config",
        "--ignore-rules",
        "-C",
        str(workspace),
        "-s",
        mixed.sandbox,
        "--json",
        "-m",
        mixed.model.model_id,
        "-c",
        f"model_reasoning_effort={json.dumps(mixed.model.reasoning_effort)}",
        "-c",
        f"mcp_servers.jacobian.url={json.dumps(mcp_url)}",
        prompt,
    )


def _public_binding(
    workspace: Path, contract: HarborTaskContract
) -> tuple[bool, dict[str, str | None]]:
    observed: dict[str, str | None] = {}
    for name in contract.public_file_digests:
        path = workspace / name
        observed[name] = (
            file_digest(path) if path.is_file() and not path.is_symlink() else None
        )
    return observed == dict(contract.public_file_digests), observed


def _submission_digest(workspace: Path) -> str | None:
    submission = workspace / "submission.json"
    if submission.is_symlink() or not submission.is_file():
        return None
    if submission.stat().st_size > 8 * 1024 * 1024:
        return None
    return file_digest(submission)


def _verify_terminal(
    *,
    task: FrozenMixedTask,
    contract: HarborTaskContract,
    workspace: Path,
    run_dir: Path,
    source_binding_digest: str,
    command_status: ToolCommandStatus,
    exit_code: int | None,
) -> tuple[dict[str, Any], CleanRoomTerminalEvidence]:
    checker_digest = object_digest(
        {
            "task_harbor_digest": contract.harbor_digest,
            "verifier_file_digests": dict(contract.verifier_file_digests),
        }
    )
    public_valid, public_observed = _public_binding(workspace, contract)
    before_digest = _submission_digest(workspace)
    base: dict[str, Any] = {
        "clean_room": True,
        "task_id": task.task_id,
        "task_harbor_digest": contract.harbor_digest,
        "verifier_digest": checker_digest,
        "verifier_file_digests": dict(contract.verifier_file_digests),
        "public_file_digests_expected": dict(contract.public_file_digests),
        "public_file_digests_observed": public_observed,
        "input_binding_valid": public_valid,
        "submission_digest_before_verifier": before_digest,
    }
    if command_status is ToolCommandStatus.TIMED_OUT:
        status: Literal["COMPLETED", "TIMEOUT", "CANCELLED", "ERROR"] = "TIMEOUT"
        outcome = {
            **base,
            "verifier_execution_status": "NOT_RUN",
            "acceptance": "INCONCLUSIVE",
            "reason": "MODEL_TIMEOUT",
            "artifact_binding_valid": False,
            "reward": None,
        }
    elif command_status is not ToolCommandStatus.EXITED or exit_code != 0:
        status = "ERROR"
        outcome = {
            **base,
            "verifier_execution_status": "NOT_RUN",
            "acceptance": "INCONCLUSIVE",
            "reason": "MODEL_ERROR",
            "artifact_binding_valid": False,
            "reward": None,
        }
    elif not public_valid:
        status = "ERROR"
        outcome = {
            **base,
            "verifier_execution_status": "NOT_RUN",
            "acceptance": "INCONCLUSIVE",
            "reason": "PUBLIC_INPUT_DRIFT",
            "artifact_binding_valid": False,
            "reward": None,
        }
    elif before_digest is None:
        status = "ERROR"
        outcome = {
            **base,
            "verifier_execution_status": "NOT_RUN",
            "acceptance": "INCONCLUSIVE",
            "reason": "MISSING_OR_UNBOUND_SUBMISSION",
            "artifact_binding_valid": False,
            "reward": None,
        }
    else:
        logs = run_dir / "verification" / "logs"
        logs.mkdir(parents=True)
        try:
            reward = run_verifier_in_child(
                task=contract.path,
                app=workspace,
                logs=logs,
                timeout_seconds=120.0,
            )
        except (VerifierExecutionError, ValueError) as exc:
            status = "ERROR"
            outcome = {
                **base,
                "verifier_execution_status": "ERROR",
                "acceptance": "INCONCLUSIVE",
                "reason": type(exc).__name__,
                "artifact_binding_valid": False,
                "reward": None,
            }
        else:
            after_digest = _submission_digest(workspace)
            artifact_valid = after_digest == before_digest
            status = "COMPLETED" if artifact_valid else "ERROR"
            accepted = reward.get("reward") == 1.0
            evidence_reward = reward.get("evidence_validity")
            outcome = {
                **base,
                "verifier_execution_status": (
                    "COMPLETED" if artifact_valid else "ERROR"
                ),
                "acceptance": (
                    "ACCEPTED"
                    if artifact_valid and accepted
                    else "REJECTED"
                    if artifact_valid
                    else "INCONCLUSIVE"
                ),
                "reason": (
                    "TERMINAL_CLEAN_ROOM_REWARD"
                    if artifact_valid
                    else "SUBMISSION_MUTATED_DURING_VERIFICATION"
                ),
                "artifact_binding_valid": artifact_valid,
                "submission_digest_after_verifier": after_digest,
                "submission_evidence_valid": (
                    evidence_reward == 1.0
                    if isinstance(evidence_reward, (int, float))
                    else None
                ),
                "false_certification": reward.get("false_certification"),
                "reward": reward,
            }
    acceptance = str(outcome["acceptance"]) if status == "COMPLETED" else "INCONCLUSIVE"
    terminal = CleanRoomTerminalEvidence(
        verifier_digest=checker_digest,
        source_binding_digest=source_binding_digest,
        clean_room=True,
        verifier_execution_status=status,
        acceptance=TerminalAcceptance(acceptance),
        input_binding_valid=(
            bool(outcome["input_binding_valid"]) if status == "COMPLETED" else None
        ),
        artifact_binding_valid=(
            bool(outcome["artifact_binding_valid"]) if status == "COMPLETED" else None
        ),
    )
    return outcome, terminal


def _run_one(
    *,
    mixed: TrajectoryValueMixedStudyContract,
    task: FrozenMixedTask,
    contract: HarborTaskContract,
    repetition: int,
    output: Path,
    source_revision: str,
) -> dict[str, Any]:
    trajectory_id = f"{task.task_id}-main-r{repetition:02d}"
    run_dir = output / "runs" / trajectory_id
    run_dir.mkdir(parents=True)
    with tempfile.TemporaryDirectory(prefix=f"jacobian-{trajectory_id}-") as raw:
        isolated = Path(raw)
        workspace = isolated / "workspace"
        state_dir = isolated / "state"
        prompt = _prepare_workspace(workspace, mixed, task, contract)
        with _mcp_server(
            state_dir=state_dir,
            run_dir=run_dir,
            tenant_id=trajectory_id,
        ) as mcp_url:
            surface = asyncio.run(inspect_surface(mcp_url, 30))
            _write_json(run_dir / "surface.json", surface)
            command = run_operator_command(
                "codex",
                _codex_arguments(
                    workspace=workspace,
                    mixed=mixed,
                    mcp_url=mcp_url,
                    prompt=prompt,
                ),
                cwd=workspace,
                timeout_seconds=mixed.timeout_seconds,
                stdout_limit_bytes=32 * 1024 * 1024,
                stderr_limit_bytes=4 * 1024 * 1024,
                environment=operator_environment(include=_CODEX_ENVIRONMENT),
            )
            transcript = run_dir / "codex.jsonl"
            transcript.write_bytes(command.stdout)
            (run_dir / "codex.stderr").write_bytes(command.stderr)
            run_ids = _reasoning_run_ids(transcript)
            reasoning_text = ""
            if len(run_ids) == 1:
                try:
                    reasoning_text = asyncio.run(
                        _read_reasoning_resource(mcp_url, run_ids[0])
                    )
                except Exception:
                    reasoning_text = ""
            (run_dir / "reasoning-log.jsonl").write_text(
                reasoning_text, encoding="utf-8"
            )
        verifier, terminal = _verify_terminal(
            task=task,
            contract=contract,
            workspace=workspace,
            run_dir=run_dir,
            source_binding_digest=file_digest(transcript),
            command_status=command.status,
            exit_code=command.exit_code,
        )
        _write_json(run_dir / "verifier.json", verifier)
        terminal_for_analysis, exclusion_reason = _publish_workspace_and_extract(
            transcript=transcript,
            workspace=workspace,
            run_dir=run_dir,
            task=task,
            source_revision=source_revision,
            original_terminal=terminal,
            original_verifier=verifier,
        )
        telemetry = parse_agent_transcript(transcript)
        record = {
            "schema_version": "1",
            "trajectory_id": trajectory_id,
            "dataset_id": task.dataset_id,
            "task_id": task.task_id,
            "task_group": task.task_group,
            "task_family": task.task_family,
            "repetition": repetition,
            "source_revision": source_revision,
            "task_contract": contract.as_record(),
            "prompt_digest": file_digest(workspace / "prompt.txt"),
            "command": {
                "status": command.status.value,
                "exit_code": command.exit_code,
                "diagnostic": command.diagnostic,
                "stdout_exceeded": command.stdout_exceeded,
                "stderr_exceeded": command.stderr_exceeded,
            },
            "reasoning_run_ids": list(run_ids),
            "reasoning_protocol": telemetry.get("reasoning_protocol"),
            "usage": telemetry.get("usage"),
            "terminal": terminal_for_analysis.model_dump(mode="json"),
            "clean_room_verifier": verifier,
            "surface_digest": surface["surface_digest"],
            "catalog_digest": surface["catalog"]["catalog_digest"],
            "policy_digest": surface["catalog"]["policy_digest"],
        }
        if exclusion_reason is not None:
            record.update(
                {
                    "original_clean_room_verifier": verifier,
                    "exclusion_reason": exclusion_reason,
                    "rerun_performed": False,
                }
            )
        _write_json(run_dir / "run.json", record)
        return record


def _publish_workspace_and_extract(
    *,
    transcript: Path,
    workspace: Path,
    run_dir: Path,
    task: FrozenMixedTask,
    source_revision: str,
    original_terminal: CleanRoomTerminalEvidence,
    original_verifier: Mapping[str, Any],
) -> tuple[CleanRoomTerminalEvidence, str | None]:
    """Publish raw terminal artifacts before fallible retrospective extraction."""

    _copy_workspace(workspace, run_dir / "workspace")
    try:
        extraction = extract_codex_trajectory(
            transcript,
            task_family=task.task_family,
            terminal_evidence=original_terminal,
        )
    except Exception as exc:
        failure = {
            "schema_version": "1",
            "trajectory_id": run_dir.name,
            "source_revision": source_revision,
            "disposition": "INCONCLUSIVE",
            "rerun_performed": False,
            "reason": "RUNNER_EXTRACTION_FAILURE",
            "extraction_failure": {
                "type": type(exc).__name__,
                "message": str(exc),
            },
            "original_clean_room_verifier": dict(original_verifier),
            "missing_artifacts": ["extraction"],
            "preserved_artifacts": ["workspace", "raw submission"],
        }
        _write_json(run_dir / "infrastructure-failure.json", failure)
        return (
            CleanRoomTerminalEvidence(
                verifier_digest=original_terminal.verifier_digest,
                source_binding_digest=original_terminal.source_binding_digest,
                clean_room=True,
                verifier_execution_status="ERROR",
                acceptance=TerminalAcceptance.INCONCLUSIVE,
            ),
            "runner extraction failed after raw submission publication",
        )
    _write_json(run_dir / "extraction.json", extraction.model_dump(mode="json"))
    return original_terminal, None


def _corpus(
    spec: TrajectoryValueHypothesisStudySpec,
    output: Path,
    records: Sequence[Mapping[str, Any]],
) -> tuple[TrajectoryValueCorpus, list[dict[str, str]]]:
    labelled: list[LabelledTrajectory] = []
    exclusions: list[dict[str, str]] = []
    for record in records:
        trajectory_id = str(record["trajectory_id"])
        extraction_path = output / "runs" / trajectory_id / "extraction.json"
        if not extraction_path.is_file() or extraction_path.is_symlink():
            exclusions.append(
                {
                    "trajectory_id": trajectory_id,
                    "reason": str(
                        record.get(
                            "exclusion_reason",
                            "missing exact trajectory extraction",
                        )
                    ),
                }
            )
            continue
        extraction = TrajectoryExtraction.model_validate(
            json.loads(extraction_path.read_text(encoding="utf-8"))
        )
        evidence = extraction.terminal_evidence
        reason: str | None = None
        if evidence is None or evidence.acceptance not in {
            TerminalAcceptance.ACCEPTED,
            TerminalAcceptance.REJECTED,
        }:
            reason = "terminal verifier outcome is inconclusive"
        elif not (evidence.input_binding_valid and evidence.artifact_binding_valid):
            reason = "terminal label is not bound to exact input and submission"
        elif not any(
            state.boundary is StateBoundary.PLAN for state in extraction.states
        ):
            reason = "no successful PLAN boundary"
        if reason is not None:
            exclusions.append({"trajectory_id": trajectory_id, "reason": reason})
            continue
        labelled.append(
            LabelledTrajectory(
                trajectory_id=trajectory_id,
                task_group=str(record["task_group"]),
                extraction=extraction,
            )
        )
    return (
        TrajectoryValueCorpus(
            corpus_id=spec.analysis_id,
            evaluator_config=spec.evaluator_config,
            trajectories=tuple(labelled),
        ),
        exclusions,
    )


def _bind_historical_soft_state_digests(extraction: dict[str, Any]) -> None:
    states = extraction.get("states")
    if not isinstance(states, list):
        raise ValueError("historical corpus states are malformed")
    for state in states:
        if not isinstance(state, dict):
            raise ValueError("historical corpus state is malformed")
        expected = object_digest(state.get("soft_state"))
        existing = state.get("soft_state_digest")
        if existing is not None and existing != expected:
            raise ValueError("historical soft-state binding drift")
        state["soft_state_digest"] = expected


def load_historical_corpus(path: Path) -> TrajectoryValueCorpus:
    """Replay a v1 corpus recorded before mandatory derived-state bindings."""

    path = path.resolve(strict=True)
    if path.is_symlink() or not path.is_file():
        raise ValueError("historical corpus must be a regular file")
    manifest_path = path.parent / "manifest.json"
    if manifest_path.is_symlink() or not manifest_path.is_file():
        raise ValueError("historical corpus manifest must be a regular file")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    artifacts = manifest.get("artifacts") if isinstance(manifest, dict) else None
    if not isinstance(artifacts, dict) or artifacts.get(path.name) != file_digest(path):
        raise ValueError("historical manifest does not bind the corpus")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(
        payload.get("trajectories"), list
    ):
        raise ValueError("historical corpus is malformed")
    for raw in payload["trajectories"]:
        if not isinstance(raw, dict) or not isinstance(raw.get("extraction"), dict):
            raise ValueError("historical corpus extraction is malformed")
        extraction = raw["extraction"]
        source_digest = extraction.get("source_digest")
        terminal = extraction.get("terminal_evidence")
        if not isinstance(source_digest, str) or not isinstance(terminal, dict):
            raise ValueError("historical corpus terminal binding is malformed")
        existing = terminal.get("source_binding_digest")
        if existing is not None and existing != source_digest:
            raise ValueError("historical terminal source binding drift")
        terminal["source_binding_digest"] = source_digest
        _bind_historical_soft_state_digests(extraction)
    return TrajectoryValueCorpus.model_validate(payload)


def _evaluation(comparison: OfflineValueComparisonV2, kind: EstimatorKindV2) -> Any:
    return next(item for item in comparison.evaluations if item.estimator is kind)


def _estimate_maps(
    comparison: OfflineValueComparisonV2,
) -> dict[EstimatorKindV2, dict[str, SemanticStateValueEstimate]]:
    return {
        evaluation.estimator: {
            estimate.observation_id: estimate for estimate in evaluation.estimates
        }
        for evaluation in comparison.evaluations
    }


def _h1(comparison: OfflineValueComparisonV2) -> dict[str, Any]:
    metrics = {
        evaluation.estimator: evaluation.metrics
        for evaluation in comparison.evaluations
    }
    results: list[dict[str, Any]] = []
    supported: list[str] = []
    for estimator in _TYPED_ESTIMATORS:
        estimate_metrics = metrics[estimator]
        deltas = []
        estimator_supported = True
        for baseline in _BASELINES:
            baseline_metrics = metrics[baseline]
            brier_delta = round(
                estimate_metrics.brier_score - baseline_metrics.brier_score, 12
            )
            mae_delta = round(
                estimate_metrics.mean_absolute_error
                - baseline_metrics.mean_absolute_error,
                12,
            )
            deltas.append(
                {
                    "baseline": baseline.value,
                    "brier_delta": brier_delta,
                    "mae_delta": mae_delta,
                    "strictly_better_on_both": brier_delta < 0 and mae_delta < 0,
                }
            )
            estimator_supported &= brier_delta < 0 and mae_delta < 0
        if estimator_supported:
            supported.append(estimator.value)
        results.append(
            {
                "estimator": estimator.value,
                "directional_support": estimator_supported,
                "deltas": deltas,
            }
        )
    return {
        "question": "H1",
        "decision_rule": "strictly-lower-brier-and-mae-than-each-baseline",
        "metrics": {
            kind.value: metrics[kind].model_dump(mode="json")
            for kind in EstimatorKindV2
        },
        "typed_estimator_comparisons": results,
        "supported_estimators": supported,
        "h1_directionally_supported": bool(supported),
    }


def _h2(comparison: OfflineValueComparisonV2) -> dict[str, Any]:
    estimates = _estimate_maps(comparison)
    abstract = estimates[EstimatorKindV2.ABSTRACT_VALUE_STATE]
    hybrid = estimates[EstimatorKindV2.ABSTRACT_VALUE_STATE_TEXT]
    text = estimates[EstimatorKindV2.REASONING_TEXT]
    pairs: list[dict[str, Any]] = []
    ordered = sorted(abstract.values(), key=lambda item: item.observation_id)
    for left, right in itertools.combinations(ordered, 2):
        if (
            left.trajectory_id == right.trajectory_id
            or left.task_group != right.task_group
            or left.abstract_value_state_digest != right.abstract_value_state_digest
            or left.reasoning_text_digest == right.reasoning_text_digest
            or left.eventual_terminal_reward == right.eventual_terminal_reward
        ):
            continue
        pairs.append(
            {
                "left": left.observation_id,
                "right": right.observation_id,
                "task_group": left.task_group,
                "abstract_value_state_digest": left.abstract_value_state_digest,
                "left_reward": left.eventual_terminal_reward,
                "right_reward": right.eventual_terminal_reward,
                "reasoning_text_separated": (
                    text[left.observation_id].cluster_id
                    != text[right.observation_id].cluster_id
                ),
                "hybrid_separated": (
                    hybrid[left.observation_id].cluster_id
                    != hybrid[right.observation_id].cluster_id
                ),
            }
        )
    abstract_metrics = _evaluation(
        comparison, EstimatorKindV2.ABSTRACT_VALUE_STATE
    ).metrics
    hybrid_metrics = _evaluation(
        comparison, EstimatorKindV2.ABSTRACT_VALUE_STATE_TEXT
    ).metrics
    brier_delta = round(hybrid_metrics.brier_score - abstract_metrics.brier_score, 12)
    mae_delta = round(
        hybrid_metrics.mean_absolute_error - abstract_metrics.mean_absolute_error, 12
    )
    separated = sum(pair["hybrid_separated"] for pair in pairs)
    supported = bool(pairs) and separated > 0 and brier_delta < 0 and mae_delta < 0
    return {
        "question": "H2",
        "pair_rule": (
            "different trajectories; same task group and exact abstract-state digest; "
            "different reasoning digest; different terminal reward"
        ),
        "mixed_outcome_compatible_pair_count": len(pairs),
        "reasoning_text_separated_pair_count": sum(
            pair["reasoning_text_separated"] for pair in pairs
        ),
        "hybrid_separated_pair_count": separated,
        "hybrid_brier_delta_vs_abstract": brier_delta,
        "hybrid_mae_delta_vs_abstract": mae_delta,
        "h2_directionally_supported": supported,
        "pairs": pairs,
    }


def _ratio(numerator: int, denominator: int) -> float | None:
    return round(numerator / denominator, 12) if denominator else None


def _first_negative_warning(
    states: Sequence[SemanticStateValueEstimate],
) -> tuple[int, SemanticStateValueEstimate, float] | None:
    for index in range(1, len(states)):
        delta = round(
            states[index].estimated_value - states[index - 1].estimated_value, 12
        )
        if delta < 0:
            return index, states[index], delta
    return None


def _confusion_key(predicted_failure: bool, failed: bool) -> str:
    return {
        (True, True): "true_positive",
        (True, False): "false_positive",
        (False, True): "false_negative",
        (False, False): "true_negative",
    }[(predicted_failure, failed)]


def _optional_mean(values: Sequence[int]) -> float | None:
    return round(statistics.fmean(values), 12) if values else None


def _h3_evaluation(evaluation: EstimatorEvaluationV2) -> dict[str, Any]:
    by_trajectory: dict[str, list[SemanticStateValueEstimate]] = defaultdict(list)
    for estimate in evaluation.estimates:
        by_trajectory[estimate.trajectory_id].append(estimate)
    confusion = {
        "true_positive": 0,
        "false_positive": 0,
        "true_negative": 0,
        "false_negative": 0,
    }
    warnings: list[dict[str, Any]] = []
    accepted_count = failed_count = 0
    true_positive_leads: list[int] = []
    all_leads: list[int] = []
    for trajectory_id, raw_states in sorted(by_trajectory.items()):
        states = sorted(raw_states, key=lambda item: item.state_index)
        failed = states[0].eventual_terminal_reward == 0
        failed_count += failed
        accepted_count += not failed
        first = _first_negative_warning(states)
        confusion[_confusion_key(first is not None, failed)] += 1
        if first is None:
            continue
        index, state, delta = first
        lead = len(states) - index - 1
        all_leads.append(lead)
        if failed:
            true_positive_leads.append(lead)
        warnings.append(
            {
                "trajectory_id": trajectory_id,
                "first_warning_observation_id": state.observation_id,
                "first_value_delta": delta,
                "later_selected_preterminal_observations": lead,
                "eventual_terminal_reward": state.eventual_terminal_reward,
            }
        )
    true_positive = confusion["true_positive"]
    false_positive = confusion["false_positive"]
    false_negative = confusion["false_negative"]
    precision = _ratio(true_positive, true_positive + false_positive)
    recall = _ratio(true_positive, true_positive + false_negative)
    prevalence = _ratio(failed_count, failed_count + accepted_count)
    precision_lift = (
        round(precision - prevalence, 12)
        if precision is not None and prevalence is not None
        else None
    )
    mean_tp_lead = _optional_mean(true_positive_leads)
    supported = bool(
        recall is not None
        and recall > 0
        and precision_lift is not None
        and precision_lift > 0
        and mean_tp_lead is not None
        and mean_tp_lead > 0
    )
    return {
        "rule": "any-strictly-negative-preterminal-selected-state-value-delta",
        **confusion,
        "precision": precision,
        "recall": recall,
        "failure_prevalence": prevalence,
        "precision_lift_over_failure_prevalence": precision_lift,
        "false_alarm_count": false_positive,
        "false_alarm_rate_among_accepted": _ratio(false_positive, accepted_count),
        "mean_true_positive_lead_observations": mean_tp_lead,
        "median_true_positive_lead_observations": (
            statistics.median(true_positive_leads) if true_positive_leads else None
        ),
        "mean_all_warning_lead_observations": _optional_mean(all_leads),
        "directional_support": supported,
        "warnings": warnings,
    }


def _h3(comparison: OfflineValueComparisonV2) -> dict[str, Any]:
    results = {
        evaluation.estimator.value: _h3_evaluation(evaluation)
        for evaluation in comparison.evaluations
    }
    return {
        "question": "H3",
        "lead_time_unit": "later-selected-preterminal-observations",
        "threshold_tuned_on_main_labels": False,
        "estimators": results,
        "supported_estimators": [
            kind for kind, result in results.items() if result["directional_support"]
        ],
        "h3_directionally_supported": any(
            result["directional_support"] for result in results.values()
        ),
    }


def _representatives(corpus: TrajectoryValueCorpus) -> list[dict[str, Any]]:
    by_group: dict[str, list[LabelledTrajectory]] = defaultdict(list)
    for trajectory in corpus.trajectories:
        by_group[trajectory.task_group].append(trajectory)
    selected: list[dict[str, Any]] = []
    for task_group, trajectories in sorted(by_group.items()):
        for acceptance in (
            TerminalAcceptance.ACCEPTED,
            TerminalAcceptance.REJECTED,
        ):
            candidates = sorted(
                (
                    trajectory
                    for trajectory in trajectories
                    if trajectory.extraction.terminal_evidence is not None
                    and trajectory.extraction.terminal_evidence.acceptance is acceptance
                ),
                key=lambda item: item.trajectory_id,
            )
            if not candidates:
                continue
            trajectory = candidates[0]
            selected.append(
                {
                    "task_group": task_group,
                    "trajectory_id": trajectory.trajectory_id,
                    "acceptance": acceptance.value,
                    "state_count": len(trajectory.extraction.states),
                    "eligible_milestone_count": sum(
                        state.milestone_eligible
                        for state in trajectory.extraction.states
                    ),
                    "reasoning_protocol_final_state": (
                        trajectory.extraction.states[
                            -1
                        ].hard_state.reasoning_protocol_state.value
                    ),
                    "source_digest": trajectory.extraction.source_digest,
                }
            )
    return selected


def analyze_comparison(
    spec: TrajectoryValueHypothesisStudySpec,
    comparison: OfflineValueComparisonV2,
    *,
    run_count: int,
    exclusions: Sequence[Mapping[str, str]] = (),
) -> dict[str, Any]:
    """Apply only the preregistered H1--H3 analysis to a frozen comparison."""

    if comparison.corpus_id != spec.analysis_id:
        raise ValueError("comparison corpus differs from preregistered analysis")
    if comparison.evaluator_config != spec.evaluator_config:
        raise ValueError("comparison evaluator differs from preregistered analysis")
    corpus = comparison.source_corpus
    accepted = sum(
        trajectory.extraction.terminal_evidence is not None
        and trajectory.extraction.terminal_evidence.acceptance
        is TerminalAcceptance.ACCEPTED
        for trajectory in corpus.trajectories
    )
    rejected = len(corpus.trajectories) - accepted
    return {
        "schema_version": "1",
        "analysis_id": spec.analysis_id,
        "evidence_class": "public-host-local-codex-predictive-validity-pilot",
        "run_count": run_count,
        "labelled_trajectory_count": len(corpus.trajectories),
        "accepted_count": accepted,
        "rejected_count": rejected,
        "mixed_terminal_outcomes": accepted > 0 and rejected > 0,
        "excluded": [dict(item) for item in exclusions],
        "h1": _h1(comparison),
        "h2": _h2(comparison),
        "h3": _h3(comparison),
        "representative_trajectories": _representatives(corpus),
        "terminal_reward": spec.terminal_reward,
        "scorer_mode": spec.scorer_mode,
        "learned_components": False,
        "training_performed": False,
        "scorer_intervention": False,
        "causal_claim_authorized": False,
    }


def analyze_study(
    spec: TrajectoryValueHypothesisStudySpec,
    output: Path,
    records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    corpus, exclusions = _corpus(spec, output, records)
    comparison = evaluate_semantic_trajectories(corpus)
    _write_json(output / "corpus.json", corpus.model_dump(mode="json"))
    _write_json(output / "comparison.json", comparison.model_dump(mode="json"))
    summary = analyze_comparison(
        spec, comparison, run_count=len(records), exclusions=exclusions
    )
    _write_json(output / "analysis.json", summary)
    return summary


def _package_comparison(output: Path, *, source_revision: str) -> dict[str, Any]:
    source = output / "comparison.json"
    destination = output / "comparison.json.gz"
    if source.is_symlink() or destination.is_symlink():
        raise RuntimeError("comparison evidence must not be a symlink")
    if source.is_file() == destination.is_file():
        raise RuntimeError("exactly one comparison representation is required")
    if source.is_file():
        raw = source.read_bytes()
        OfflineValueComparisonV2.model_validate_json(raw)
        compressed = gzip.compress(raw, compresslevel=9, mtime=0)
        temporary = output / ".comparison.json.gz.tmp"
        temporary.write_bytes(compressed)
        temporary.replace(destination)
        source.unlink()
    else:
        compressed = destination.read_bytes()
        try:
            raw = gzip.decompress(compressed)
        except (EOFError, gzip.BadGzipFile) as exc:
            raise RuntimeError("compressed comparison evidence is invalid") from exc
        OfflineValueComparisonV2.model_validate_json(raw)
    metadata = {
        "path": destination.relative_to(output).as_posix(),
        "encoding": "gzip",
        "compression_level": 9,
        "mtime": 0,
        "uncompressed_bytes": len(raw),
        "uncompressed_file_digest": "sha256:" + hashlib.sha256(raw).hexdigest(),
        "compressed_file_digest": file_digest(destination),
        "source_revision": source_revision,
        "packager_digest": file_digest(Path(__file__).resolve(strict=True)),
    }
    return metadata


def _artifact_manifest(output: Path) -> dict[str, str]:
    return {
        path.relative_to(output).as_posix(): file_digest(path)
        for path in _regular_files(output)
        if path.name != "manifest.json"
    }


def package_study(output: Path) -> dict[str, Any]:
    """Deterministically package a completed study without changing its analysis."""

    output = output.resolve(strict=True)
    manifest_path = output / "manifest.json"
    if manifest_path.is_symlink() or not manifest_path.is_file():
        raise RuntimeError("packaging requires a completed study manifest")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict) or not isinstance(
        manifest.get("artifacts"), dict
    ):
        raise RuntimeError("study manifest is malformed")
    if _artifact_manifest(output) != manifest["artifacts"]:
        raise RuntimeError("study artifacts do not match the immutable manifest")
    revision = git_head_sha(_ROOT)
    if revision is None:
        raise RuntimeError("unable to bind packaging source revision")
    packaging = _package_comparison(output, source_revision=revision)
    existing = manifest.get("packaging")
    if existing is not None and existing != {"comparison": packaging}:
        raise RuntimeError("study packaging provenance drift")
    manifest["packaging"] = {"comparison": packaging}
    manifest["artifacts"] = _artifact_manifest(output)
    _write_json(manifest_path, manifest)
    return packaging


def _bind_historical_terminal_sources(output: Path) -> None:
    for run_dir in sorted((output / "runs").iterdir()):
        if not run_dir.is_dir() or run_dir.is_symlink():
            raise RuntimeError("historical run root contains a non-directory entry")
        transcript = run_dir / "codex.jsonl"
        if transcript.is_symlink() or not transcript.is_file():
            raise RuntimeError(f"historical transcript is missing: {run_dir.name}")
        source_digest = file_digest(transcript)
        run_path = run_dir / "run.json"
        run_record = json.loads(run_path.read_text(encoding="utf-8"))
        terminal = run_record.get("terminal") if isinstance(run_record, dict) else None
        if not isinstance(terminal, dict):
            raise RuntimeError(f"historical terminal is malformed: {run_dir.name}")
        existing = terminal.get("source_binding_digest")
        if existing is not None and existing != source_digest:
            raise RuntimeError(f"historical terminal source drift: {run_dir.name}")
        terminal["source_binding_digest"] = source_digest
        _write_json(run_path, run_record)

        extraction_path = run_dir / "extraction.json"
        if extraction_path.is_file() and not extraction_path.is_symlink():
            extraction = json.loads(extraction_path.read_text(encoding="utf-8"))
            raw_terminal = (
                extraction.get("terminal_evidence")
                if isinstance(extraction, dict)
                else None
            )
            if not isinstance(raw_terminal, dict):
                raise RuntimeError(
                    f"historical extraction terminal is malformed: {run_dir.name}"
                )
            if extraction.get("source_digest") != source_digest:
                raise RuntimeError(
                    f"historical extraction source drift: {run_dir.name}"
                )
            existing = raw_terminal.get("source_binding_digest")
            if existing is not None and existing != source_digest:
                raise RuntimeError(
                    f"historical extraction terminal source drift: {run_dir.name}"
                )
            raw_terminal["source_binding_digest"] = source_digest
            validated = TrajectoryExtraction.model_validate(extraction)
            _write_json(extraction_path, validated.model_dump(mode="json"))


def reanalyze_historical_study(
    *, spec_path: Path, historical_mixed_path: Path, output: Path
) -> dict[str, Any]:
    """Migrate exact source bindings and rerun analysis without model execution."""

    if not _repository_is_clean():
        raise RuntimeError("historical reanalysis requires a clean source tree")
    output = output.resolve(strict=True)
    manifest_path = output / "manifest.json"
    if manifest_path.is_symlink() or not manifest_path.is_file():
        raise RuntimeError("historical reanalysis requires a completed manifest")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict) or not isinstance(
        manifest.get("artifacts"), dict
    ):
        raise RuntimeError("historical study manifest is malformed")
    if manifest.get("posthoc_reanalysis") is not None:
        raise RuntimeError("historical study was already reanalyzed")
    if _artifact_manifest(output) != manifest["artifacts"]:
        raise RuntimeError("historical study artifacts do not match the manifest")
    spec, _validated = load_hypothesis_spec(
        spec_path,
        verify_current_tasks=False,
        historical_mixed_path=historical_mixed_path,
    )
    revision = git_head_sha(_ROOT)
    if revision is None:
        raise RuntimeError("unable to bind historical reanalysis source revision")
    previous = {
        "analysis_file_digest": file_digest(output / "analysis.json"),
        "comparison_packaging": manifest.get("packaging", {}).get("comparison"),
        "corpus_file_digest": file_digest(output / "corpus.json"),
    }

    _bind_historical_terminal_sources(output)
    records = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted((output / "runs").glob("*/run.json"))
    ]
    comparison_gzip = output / "comparison.json.gz"
    if comparison_gzip.is_symlink() or not comparison_gzip.is_file():
        raise RuntimeError("historical comparison package is missing")
    comparison_gzip.unlink()
    summary = analyze_study(spec, output, records)
    packaging = _package_comparison(output, source_revision=revision)
    manifest["packaging"] = {"comparison": packaging}
    manifest["posthoc_reanalysis"] = {
        "source_revision": revision,
        "reason": "replay-after-source-binding-and-estimator-validation-fixes",
        "model_rerun": False,
        "terminal_verifier_outcomes_changed": False,
        "source_binding_migration": "terminal-source-binding-equals-exact-codex-jsonl-digest",
        "previous": previous,
        "runner_and_analysis_digest": file_digest(Path(__file__).resolve(strict=True)),
        "mixed_contract_digest": file_digest(
            _ROOT / "benchmarks/tooling/trajectory_value_mixed_contract.py"
        ),
        "exact_evaluator_digest": file_digest(
            _ROOT / "src/jacobian/eval/trajectory_value.py"
        ),
        "semantic_evaluator_digest": file_digest(
            _ROOT / "src/jacobian/eval/trajectory_value_abstraction.py"
        ),
    }
    manifest["artifacts"] = _artifact_manifest(output)
    _write_json(manifest_path, manifest)
    return summary


def _local_auth_status() -> str:
    result = run_operator_command(
        "codex",
        ("login", "status"),
        cwd=_ROOT,
        timeout_seconds=30,
        environment=operator_environment(include=("HOME", "PATH", "CODEX_HOME")),
    )
    stdout = result.stdout.decode("utf-8", errors="replace").strip()
    stderr = result.stderr.decode("utf-8", errors="replace").strip()
    status = "\n".join(part for part in (stdout, stderr) if part)
    if (
        result.status is not ToolCommandStatus.EXITED
        or result.exit_code != 0
        or "Logged in using ChatGPT" not in status
    ):
        raise RuntimeError("the preregistered study requires local ChatGPT login")
    return status


def _git_file_digest(revision: str, relative: str) -> str:
    result = run_operator_command(
        "git",
        ("show", f"{revision}:{relative}"),
        cwd=_ROOT,
        timeout_seconds=30,
    )
    if result.status is not ToolCommandStatus.EXITED or result.exit_code != 0:
        raise RuntimeError(f"unable to bind prior source file: {relative}")
    return "sha256:" + hashlib.sha256(result.stdout).hexdigest()


def _resume_tree_is_clean(output: Path) -> bool:
    result = run_operator_command(
        "git", ("status", "--porcelain"), cwd=_ROOT, timeout_seconds=30
    )
    if result.status is not ToolCommandStatus.EXITED or result.exit_code != 0:
        return False
    relative = output.relative_to(_ROOT).as_posix()
    lines = result.stdout.decode("utf-8", errors="replace").splitlines()
    return bool(lines) and all(line == f"?? {relative}/" for line in lines)


def _validate_prior_revision(revision: str) -> None:
    if re.fullmatch(r"[0-9a-f]{40}", revision) is None:
        raise RuntimeError("prior source revision must be a full commit SHA")
    result = run_operator_command(
        "git",
        ("cat-file", "-e", f"{revision}^{{commit}}"),
        cwd=_ROOT,
        timeout_seconds=30,
    )
    if result.status is not ToolCommandStatus.EXITED or result.exit_code != 0:
        raise RuntimeError("prior source revision is unavailable")


def _slot_id(task: FrozenMixedTask, repetition: int) -> str:
    return f"{task.task_id}-main-r{repetition:02d}"


def _read_existing_record(
    path: Path, task: FrozenMixedTask, repetition: int
) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"malformed completed rollout record: {path}")
    expected = _slot_id(task, repetition)
    if (
        value.get("trajectory_id") != expected
        or value.get("dataset_id") != task.dataset_id
        or value.get("task_id") != task.task_id
        or value.get("task_group") != task.task_group
        or value.get("repetition") != repetition
    ):
        raise RuntimeError(f"completed rollout identity drift: {expected}")
    return value


def _recover_interrupted_record(
    *,
    task: FrozenMixedTask,
    repetition: int,
    run_dir: Path,
    prior_revision: str,
) -> dict[str, Any]:
    trajectory_id = _slot_id(task, repetition)
    required = (
        "codex.jsonl",
        "codex.stderr",
        "reasoning-log.jsonl",
        "surface.json",
        "verifier.json",
    )
    if any(
        (run_dir / name).is_symlink() or not (run_dir / name).is_file()
        for name in required
    ):
        raise RuntimeError(
            f"interrupted rollout evidence is incomplete: {trajectory_id}"
        )
    transcript = run_dir / "codex.jsonl"
    verifier = json.loads((run_dir / "verifier.json").read_text(encoding="utf-8"))
    if not isinstance(verifier, dict):
        raise RuntimeError("interrupted verifier record is malformed")
    original_terminal = CleanRoomTerminalEvidence(
        verifier_digest=str(verifier["verifier_digest"]),
        source_binding_digest=file_digest(transcript),
        clean_room=True,
        verifier_execution_status="COMPLETED",
        acceptance=TerminalAcceptance(str(verifier["acceptance"])),
        input_binding_valid=bool(verifier["input_binding_valid"]),
        artifact_binding_valid=bool(verifier["artifact_binding_valid"]),
    )
    extraction_failure: dict[str, str]
    try:
        extract_codex_trajectory(
            transcript,
            task_family=task.task_family,
            terminal_evidence=original_terminal,
        )
    except Exception as exc:
        extraction_failure = {
            "type": type(exc).__name__,
            "message": str(exc),
        }
    else:
        extraction_failure = {
            "type": "InterruptedWorkspaceLoss",
            "message": "raw submission workspace was lost before replay publication",
        }
    terminal = CleanRoomTerminalEvidence(
        verifier_digest=original_terminal.verifier_digest,
        source_binding_digest=original_terminal.source_binding_digest,
        clean_room=True,
        verifier_execution_status="ERROR",
        acceptance=TerminalAcceptance.INCONCLUSIVE,
    )
    telemetry = parse_agent_transcript(transcript)
    surface = json.loads((run_dir / "surface.json").read_text(encoding="utf-8"))
    failure = {
        "schema_version": "1",
        "trajectory_id": trajectory_id,
        "source_revision": prior_revision,
        "disposition": "INCONCLUSIVE",
        "rerun_performed": False,
        "reason": "RUNNER_EXTRACTION_FAILURE_AND_RAW_WORKSPACE_LOSS",
        "extraction_failure": extraction_failure,
        "original_clean_room_verifier": verifier,
        "missing_artifacts": ["workspace", "raw submission", "extraction"],
    }
    _write_json(run_dir / "infrastructure-failure.json", failure)
    record = {
        "schema_version": "1",
        "trajectory_id": trajectory_id,
        "dataset_id": task.dataset_id,
        "task_id": task.task_id,
        "task_group": task.task_group,
        "task_family": task.task_family,
        "repetition": repetition,
        "source_revision": prior_revision,
        "command": {
            "status": "UNKNOWN_AFTER_RUNNER_FAILURE",
            "exit_code": None,
            "diagnostic": "verifier execution proves Codex returned, but command status was not published",
        },
        "reasoning_run_ids": list(_reasoning_run_ids(transcript)),
        "reasoning_protocol": telemetry.get("reasoning_protocol"),
        "usage": telemetry.get("usage"),
        "terminal": terminal.model_dump(mode="json"),
        "original_clean_room_verifier": verifier,
        "surface_digest": surface["surface_digest"],
        "catalog_digest": surface["catalog"]["catalog_digest"],
        "policy_digest": surface["catalog"]["policy_digest"],
        "exclusion_reason": "runner extraction failed and raw submission was unavailable for replay",
        "rerun_performed": False,
    }
    _write_json(run_dir / "run.json", record)
    return record


def _execute_or_resume_slots(
    *,
    mixed: TrajectoryValueMixedStudyContract,
    contracts: Mapping[tuple[str, str], HarborTaskContract],
    output: Path,
    revision: str,
    prior_revision: str | None,
    interrupted_revision: str | None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    records: list[dict[str, Any]] = []
    prior_ids_by_revision: dict[str, list[str]] = {}
    current_ids: list[str] = []
    expected_ids = {
        _slot_id(task, repetition)
        for task in mixed.tasks
        for repetition in range(1, mixed.repetitions_per_task + 1)
    }
    actual_dirs = (
        {path.name for path in (output / "runs").iterdir() if path.is_dir()}
        if (output / "runs").is_dir()
        else set()
    )
    if not actual_dirs <= expected_ids:
        raise RuntimeError("resume output contains an unexpected rollout directory")
    for task in mixed.tasks:
        for repetition in range(1, mixed.repetitions_per_task + 1):
            trajectory_id = _slot_id(task, repetition)
            run_dir = output / "runs" / trajectory_id
            record_path = run_dir / "run.json"
            if record_path.is_file() and not record_path.is_symlink():
                if prior_revision is None:
                    raise RuntimeError("fresh execution found a completed rollout")
                record = _read_existing_record(record_path, task, repetition)
                source_revision = str(record.get("source_revision", prior_revision))
                _validate_prior_revision(source_revision)
                records.append(record)
                prior_ids_by_revision.setdefault(source_revision, []).append(
                    trajectory_id
                )
                continue
            if run_dir.exists():
                if prior_revision is None:
                    raise RuntimeError("fresh execution found an interrupted rollout")
                source_revision = interrupted_revision or prior_revision
                records.append(
                    _recover_interrupted_record(
                        task=task,
                        repetition=repetition,
                        run_dir=run_dir,
                        prior_revision=source_revision,
                    )
                )
                prior_ids_by_revision.setdefault(source_revision, []).append(
                    trajectory_id
                )
                continue
            records.append(
                _run_one(
                    mixed=mixed,
                    task=task,
                    contract=contracts[(task.dataset_id, task.task_id)],
                    repetition=repetition,
                    output=output,
                    source_revision=revision,
                )
            )
            current_ids.append(trajectory_id)
    attempts: list[dict[str, Any]] = []
    for source_revision, trajectory_ids in prior_ids_by_revision.items():
        attempts.append(
            {
                "source_revision": source_revision,
                "runner_and_analysis_digest": _git_file_digest(
                    source_revision,
                    "benchmarks/tooling/trajectory_value_hypothesis_study.py",
                ),
                "trajectory_ids": trajectory_ids,
                "completed_normally": False,
            }
        )
    attempts.append(
        {
            "source_revision": revision,
            "runner_and_analysis_digest": file_digest(
                Path(__file__).resolve(strict=True)
            ),
            "trajectory_ids": current_ids,
            "completed_normally": True,
        }
    )
    return records, attempts


def _validate_execution_location(
    *,
    output: Path,
    resume_after_interruption: bool,
    prior_source_revision: str | None,
    interrupted_source_revision: str | None,
) -> None:
    if resume_after_interruption:
        if not output.is_dir() or (output / "manifest.json").exists():
            raise RuntimeError("resume requires one incomplete unmanifested study")
        if prior_source_revision is None:
            raise RuntimeError("resume requires the prior full source revision")
        _validate_prior_revision(prior_source_revision)
        if interrupted_source_revision is not None:
            _validate_prior_revision(interrupted_source_revision)
        return
    if output.exists():
        raise RuntimeError(f"output directory already exists: {output}")
    if prior_source_revision is not None or interrupted_source_revision is not None:
        raise RuntimeError("source revisions are valid only for resume")


def run_study(
    spec_path: Path,
    output: Path,
    *,
    execute: bool,
    resume_after_interruption: bool = False,
    prior_source_revision: str | None = None,
    interrupted_source_revision: str | None = None,
) -> dict[str, Any]:
    """Execute exactly one clean, immutable 24-rollout mixed study."""

    if not execute:
        raise RuntimeError("refusing external Codex execution without --execute")
    spec_path = spec_path.resolve(strict=True)
    output = output.resolve()
    _validate_execution_location(
        output=output,
        resume_after_interruption=resume_after_interruption,
        prior_source_revision=prior_source_revision,
        interrupted_source_revision=interrupted_source_revision,
    )
    spec, validated = load_hypothesis_spec(spec_path)
    mixed = validated.contract
    if _codex_version(_ROOT) != mixed.model.codex_cli_version:
        raise RuntimeError("Codex CLI version differs from the frozen study")
    model_record, model_cache_digest = _model_cache(mixed)  # type: ignore[arg-type]
    login_status = _local_auth_status()
    tree_clean = (
        _resume_tree_is_clean(output)
        if resume_after_interruption
        else _repository_is_clean()
    )
    if not tree_clean:
        raise RuntimeError("study execution requires a clean preregistration tree")
    revision = git_head_sha(_ROOT)
    if revision is None:
        raise RuntimeError("unable to bind the source revision")
    contracts = {
        (task.dataset_id, task.task_id): _task_contract(
            CalibrationCandidate(
                dataset_id=task.dataset_id,
                task_id=task.task_id,
                task_family=task.task_family,
                calibration_tags=task.calibration_tags,
            )
        )
        for task in mixed.tasks
    }
    output.mkdir(parents=True, exist_ok=resume_after_interruption)
    records, execution_attempts = _execute_or_resume_slots(
        mixed=mixed,
        contracts=contracts,
        output=output,
        revision=revision,
        prior_revision=prior_source_revision,
        interrupted_revision=interrupted_source_revision,
    )
    summary = analyze_study(spec, output, records)
    packaging = _package_comparison(output, source_revision=revision)
    manifest = {
        "schema_version": "1",
        "analysis_id": spec.analysis_id,
        "evidence_class": "public-host-local-codex-predictive-validity-pilot",
        "source_revision": prior_source_revision or revision,
        "source_tree_clean_at_start": True,
        "execution_attempts": execution_attempts,
        "preregistration": {
            "path": spec_path.relative_to(_ROOT).as_posix(),
            "file_digest": file_digest(spec_path),
            "object_digest": object_digest(spec.model_dump(mode="json")),
            "source_revision": prior_source_revision or revision,
            "labels_available_at_revision": False,
        },
        "mixed_study": {
            "path": spec.mixed_study.path,
            "file_digest": spec.mixed_study.file_digest,
            "object_digest": object_digest(mixed.model_dump(mode="json")),
            "agent_instructions_digest": object_digest(mixed.agent_instructions),
        },
        "codex": {
            "version": mixed.model.codex_cli_version,
            "model": mixed.model.model_id,
            "reasoning_effort": mixed.model.reasoning_effort,
            "model_catalog_record": model_record,
            "model_cache_digest": model_cache_digest,
            "authentication": login_status,
            "api_key_environment_forwarded": False,
            "sandbox": mixed.sandbox,
            "ephemeral": True,
            "user_config_loaded": False,
            "sampling_seed_available": False,
        },
        "jacobian": {
            "reasoning_log_mode": mixed.reasoning_log_mode,
            "one_ephemeral_state_directory_per_rollout": True,
            "surface_recorded_per_rollout": True,
        },
        "task_contracts": [
            contracts[(task.dataset_id, task.task_id)].as_record()
            for task in mixed.tasks
        ],
        "versions": {
            "runner_and_analysis_digest": file_digest(
                Path(__file__).resolve(strict=True)
            ),
            "mixed_contract_digest": file_digest(
                _ROOT / "benchmarks/tooling/trajectory_value_mixed_contract.py"
            ),
            "extractor_digest": file_digest(
                _ROOT / "src/jacobian/eval/trajectory_state.py"
            ),
            "exact_evaluator_digest": file_digest(
                _ROOT / "src/jacobian/eval/trajectory_value.py"
            ),
            "semantic_evaluator_digest": file_digest(
                _ROOT / "src/jacobian/eval/trajectory_value_abstraction.py"
            ),
            "state_schema_version": "1",
            "semantic_state_schema_version": "1",
            "evaluator_schema_version": "2",
            "analysis_schema_version": "1",
        },
        "budgets": {
            "tasks": len(mixed.tasks),
            "repetitions_per_task": mixed.repetitions_per_task,
            "rollouts": len(records),
            "timeout_seconds_per_rollout": mixed.timeout_seconds,
            "retries_for_wrong_answers": 0,
        },
        "outcomes": {
            "accepted": summary["accepted_count"],
            "rejected": summary["rejected_count"],
            "excluded": len(summary["excluded"]),
        },
        "packaging": {"comparison": packaging},
        "artifacts": _artifact_manifest(output),
        "terminal_reward": spec.terminal_reward,
        "learned_components": False,
        "training_performed": False,
        "scorer_intervention": False,
        "causal_claim_authorized": False,
    }
    _write_json(output / "manifest.json", manifest)
    return summary


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    run = subparsers.add_parser("run")
    run.add_argument("--spec", type=Path, default=_DEFAULT_SPEC)
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--execute", action="store_true")
    run.add_argument("--resume-after-interruption", action="store_true")
    run.add_argument("--prior-source-revision")
    run.add_argument("--interrupted-source-revision")
    package = subparsers.add_parser("package")
    package.add_argument("--output", type=Path, required=True)
    reanalyze = subparsers.add_parser("reanalyze-historical")
    reanalyze.add_argument("--spec", type=Path, default=_DEFAULT_SPEC)
    reanalyze.add_argument("--historical-mixed-study", type=Path, required=True)
    reanalyze.add_argument("--output", type=Path, required=True)
    schema = subparsers.add_parser("schema")
    schema.add_argument("--output", type=Path)
    validate = subparsers.add_parser("validate")
    validate.add_argument("--spec", type=Path, default=_DEFAULT_SPEC)
    validate.add_argument("--skip-current-tasks", action="store_true")
    validate.add_argument("--historical-mixed-study", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    if args.command == "schema":
        value = TrajectoryValueHypothesisStudySpec.model_json_schema()
        if args.output is None:
            print(json.dumps(value, indent=2, sort_keys=True))
        else:
            _write_json(args.output, value)
        return
    if args.command == "validate":
        spec, validated = load_hypothesis_spec(
            args.spec,
            verify_current_tasks=not args.skip_current_tasks,
            historical_mixed_path=args.historical_mixed_study,
        )
        print(
            json.dumps(
                {
                    "analysis_id": spec.analysis_id,
                    "tasks": [task.task_id for task in validated.contract.tasks],
                    "rollouts": len(validated.contract.tasks)
                    * validated.contract.repetitions_per_task,
                },
                sort_keys=True,
            )
        )
        return
    if args.command == "package":
        print(json.dumps(package_study(args.output), indent=2, sort_keys=True))
        return
    if args.command == "reanalyze-historical":
        summary = reanalyze_historical_study(
            spec_path=args.spec,
            historical_mixed_path=args.historical_mixed_study,
            output=args.output,
        )
        print(json.dumps(summary, indent=2, sort_keys=True))
        return
    summary = run_study(
        args.spec,
        args.output,
        execute=args.execute,
        resume_after_interruption=args.resume_after_interruption,
        prior_source_revision=args.prior_source_revision,
        interrupted_source_revision=args.interrupted_source_revision,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()


__all__ = [
    "H1Plan",
    "H2Plan",
    "H3Plan",
    "TrajectoryValueHypothesisStudySpec",
    "analyze_comparison",
    "analyze_study",
    "load_historical_corpus",
    "load_hypothesis_spec",
    "package_study",
    "reanalyze_historical_study",
    "run_study",
]
