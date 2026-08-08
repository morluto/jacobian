"""Calibrate mixed-difficulty Harbor tasks with real, isolated Codex rollouts.

The runner exposes only each task's public files, attaches an observation-only
Jacobian MCP server, and invokes the task-owned clean-room verifier after Codex
exits.  It never retries a mathematically rejected answer and never turns an
incomplete model or verifier execution into a failure label.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import shutil
import tempfile
import tomllib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Self

from pydantic import ConfigDict, Field, model_validator

from benchmarks.tooling.codex_visibility import inspect_surface
from benchmarks.tooling.command_runner import (
    ToolCommandRequest,
    ToolCommandStatus,
    ToolResolver,
    git_head_sha,
    operator_environment,
    run_operator_command,
    run_tool_command,
)
from benchmarks.tooling.trajectory_value_study import (
    StudyModel,
    _codex_version,
    _mcp_server,
    _model_cache,
    _reasoning_run_ids,
    _repository_is_clean,
    _required_reasoning_log,
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

_ROOT = Path(__file__).resolve().parents[2]
_DATASETS = _ROOT / "benchmarks" / "datasets"
_DEFAULT_SPEC = _ROOT / "benchmarks/config/trajectory-value-calibration-v1.json"
_IDENTIFIER = r"^[a-z0-9][a-z0-9._-]{0,127}$"
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
_PUBLIC_FILES = {
    "instruction.md": Path("instruction.md"),
    "input.json": Path("environment/input.json"),
    "submission_schema.json": Path("environment/submission_schema.json"),
}
_HARBOR_VERSION = "0.20.0"
_DIAGNOSTIC_LIMIT = 1024


class CalibrationCandidate(ContractModel):
    dataset_id: str = Field(pattern=_IDENTIFIER)
    task_id: str = Field(pattern=_IDENTIFIER)
    task_family: str = Field(min_length=1, max_length=128)
    calibration_tags: tuple[
        Literal[
            "candidate-checker-repair",
            "bounded-search-unknown",
            "scope-assurance",
            "one-sided-evidence",
            "artifact-binding",
            "capability-routing",
        ],
        ...,
    ] = Field(min_length=1)


class CalibrationSelectionRule(ContractModel):
    minimum_labelled_rollouts: Literal[2] = 2
    minimum_success_rate_millionths: Literal[200000] = 200000
    maximum_success_rate_millionths: Literal[800000] = 800000
    maximum_selected_tasks: Literal[4] = 4
    ordering: Literal["candidate-order"] = "candidate-order"
    uncertainty: Literal["wilson-95"] = "wilson-95"


class TrajectoryValueCalibrationSpec(ContractModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        json_schema_extra={
            "$id": "https://jacobian.invalid/docs/reference/evaluations/schemas/trajectory-value-calibration-v1.schema.json"
        },
    )

    schema_version: Literal["1"] = "1"
    calibration_id: str = Field(pattern=_IDENTIFIER)
    model: StudyModel
    repetitions_per_candidate: Literal[2] = 2
    timeout_seconds: int = Field(ge=60, le=900, strict=True)
    sandbox: Literal["workspace-write"] = "workspace-write"
    reasoning_log_mode: Literal["REQUIRED"] = "REQUIRED"
    tool_mode: Literal["direct"] = "direct"
    web_search: Literal["disabled"] = "disabled"
    agent_instructions: str = Field(min_length=1, max_length=8192)
    selection_rule: CalibrationSelectionRule = CalibrationSelectionRule()
    candidates: tuple[CalibrationCandidate, ...] = Field(min_length=4, max_length=16)
    terminal_reward: Literal["clean-room-verifier-acceptance-only"] = (
        "clean-room-verifier-acceptance-only"
    )
    retries_for_wrong_answers: Literal[0] = 0
    training_performed: Literal[False] = False
    scorer_intervention: Literal[False] = False

    @model_validator(mode="after")
    def require_unique_candidates_and_coverage(self) -> Self:
        identities = [
            (candidate.dataset_id, candidate.task_id) for candidate in self.candidates
        ]
        if len(set(identities)) != len(identities):
            raise ValueError("calibration candidates must be unique")
        covered = {
            tag for candidate in self.candidates for tag in candidate.calibration_tags
        }
        required = {
            "candidate-checker-repair",
            "bounded-search-unknown",
            "scope-assurance",
            "one-sided-evidence",
            "artifact-binding",
            "capability-routing",
        }
        if covered != required:
            raise ValueError("calibration candidates must cover every declared trap")
        return self


@dataclass(frozen=True, slots=True)
class HarborTaskContract:
    dataset_id: str
    task_id: str
    path: Path
    harbor_digest: str
    public_file_digests: Mapping[str, str]
    verifier_file_digests: Mapping[str, str]

    def as_record(self) -> dict[str, Any]:
        return {
            "dataset_id": self.dataset_id,
            "task_id": self.task_id,
            "harbor_digest": self.harbor_digest,
            "public_file_digests": dict(self.public_file_digests),
            "verifier_file_digests": dict(self.verifier_file_digests),
        }


def load_spec(path: Path) -> TrajectoryValueCalibrationSpec:
    """Load a closed preregistered calibration contract."""

    return TrajectoryValueCalibrationSpec.model_validate_json(
        path.read_text(encoding="utf-8")
    )


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _regular_files(root: Path) -> tuple[Path, ...]:
    files: list[Path] = []
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise RuntimeError(f"symlink is forbidden in evaluation evidence: {path}")
        if path.is_file():
            files.append(path)
        elif not path.is_dir():
            raise RuntimeError(
                f"special file is forbidden in evaluation evidence: {path}"
            )
    return tuple(files)


def _harbor_task_digest(task: Path) -> str:
    script = (
        "import sys; from pathlib import Path; "
        "from benchmarks.tooling.harbor_digest import task_digest; "
        "print(task_digest(Path(sys.argv[1])))"
    )
    uvx = ToolResolver().resolve("uvx")
    if uvx is None:
        raise RuntimeError("uvx is required for the pinned Harbor task digest")
    result = run_tool_command(
        ToolCommandRequest(
            executable=uvx,
            arguments=(
                "--from",
                f"harbor=={_HARBOR_VERSION}",
                "--with",
                "tomli-w==1.2.0",
                "--with",
                "jsonschema",
                "python",
                "-c",
                script,
                str(task),
            ),
            environment=operator_environment(),
            cwd=str(_ROOT),
            timeout_seconds=180.0,
            stdout_limit_bytes=64 * 1024,
            stderr_limit_bytes=64 * 1024,
        )
    )
    if result.status is not ToolCommandStatus.EXITED or result.exit_code != 0:
        diagnostic = result.stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(f"pinned Harbor task digest failed: {diagnostic}")
    value = result.stdout.decode("ascii", errors="strict").strip()
    hexadecimal = value.removeprefix("sha256:")
    if len(hexadecimal) != 64:
        raise RuntimeError("pinned Harbor returned a malformed task digest")
    try:
        int(hexadecimal, 16)
    except ValueError as exc:
        raise RuntimeError("pinned Harbor returned a malformed task digest") from exc
    return "sha256:" + hexadecimal


def _task_contract(candidate: CalibrationCandidate) -> HarborTaskContract:
    if Path(candidate.dataset_id).name != candidate.dataset_id:
        raise RuntimeError("dataset id must be one path component")
    if Path(candidate.task_id).name != candidate.task_id:
        raise RuntimeError("task id must be one path component")
    task = _DATASETS / candidate.dataset_id / candidate.task_id
    if task.is_symlink() or not task.is_dir():
        raise RuntimeError(
            f"unknown Harbor task: {candidate.dataset_id}/{candidate.task_id}"
        )
    task_root = task.resolve(strict=True)
    task_toml = tomllib.loads((task_root / "task.toml").read_text(encoding="utf-8"))
    declared_name = task_toml.get("task", {}).get("name")
    if not isinstance(declared_name, str) or not declared_name.endswith(
        "/" + candidate.task_id
    ):
        raise RuntimeError(
            "task.toml identity does not match the calibration candidate"
        )
    public: dict[str, str] = {}
    for name, relative in _PUBLIC_FILES.items():
        path = task_root / relative
        if path.is_symlink() or not path.is_file():
            raise RuntimeError(f"missing regular public task file: {relative}")
        public[name] = file_digest(path)
    tests = task_root / "tests"
    verifier = {
        path.relative_to(tests).as_posix(): file_digest(path)
        for path in _regular_files(tests)
    }
    if not {"verifier.py", "verifier_support.py", "public_contract.json"}.issubset(
        verifier
    ):
        raise RuntimeError(
            "task does not expose the required clean-room verifier files"
        )
    return HarborTaskContract(
        dataset_id=candidate.dataset_id,
        task_id=candidate.task_id,
        path=task_root,
        harbor_digest=_harbor_task_digest(task_root),
        public_file_digests=public,
        verifier_file_digests=verifier,
    )


def _prepare_workspace(
    workspace: Path,
    spec: TrajectoryValueCalibrationSpec,
    candidate: CalibrationCandidate,
    task: HarborTaskContract,
) -> str:
    workspace.mkdir()
    for name, relative in _PUBLIC_FILES.items():
        shutil.copyfile(task.path / relative, workspace / name)
    (workspace / "evidence").mkdir()
    prompt = spec.agent_instructions.strip() + "\n"
    (workspace / "prompt.txt").write_text(prompt, encoding="utf-8")
    visible = {name: file_digest(workspace / name) for name in task.public_file_digests}
    if visible != dict(task.public_file_digests):
        raise RuntimeError("model-visible Harbor task files drifted during preparation")
    return prompt


def _codex_arguments(
    *,
    workspace: Path,
    spec: TrajectoryValueCalibrationSpec,
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
        spec.sandbox,
        "--json",
        "-m",
        spec.model.model_id,
        "-c",
        f"model_reasoning_effort={json.dumps(spec.model.reasoning_effort)}",
        "-c",
        f"web_search={json.dumps(spec.web_search)}",
        "-c",
        f"mcp_servers.jacobian.url={json.dumps(mcp_url)}",
        prompt,
    )


def _copy_workspace(source: Path, destination: Path) -> None:
    _regular_files(source)
    shutil.copytree(source, destination)


def _bounded_diagnostic(exc: BaseException) -> str:
    diagnostic = str(exc).replace("\x00", "\ufffd")
    if len(diagnostic) <= _DIAGNOSTIC_LIMIT:
        return diagnostic
    return diagnostic[: _DIAGNOSTIC_LIMIT - 14] + "...[truncated]"


def _unsafe_workspace_outcome(
    verifier: Mapping[str, Any], exc: BaseException
) -> dict[str, Any]:
    return {
        **dict(verifier),
        "acceptance": "INCONCLUSIVE",
        "reason": "UNSAFE_WORKSPACE_EVIDENCE",
        "workspace_evidence_status": "REJECTED",
        "workspace_evidence_diagnostic": _bounded_diagnostic(exc),
        "artifact_binding_valid": False,
        "reward": None,
    }


def _verification_outcome(
    *,
    task: HarborTaskContract,
    workspace: Path,
    run_dir: Path,
    command_status: ToolCommandStatus,
    exit_code: int | None,
) -> dict[str, Any]:
    base = {
        "clean_room": True,
        "task_harbor_digest": task.harbor_digest,
        "verifier_file_digests": dict(task.verifier_file_digests),
    }
    if command_status is ToolCommandStatus.TIMED_OUT:
        return {
            **base,
            "verifier_execution_status": "NOT_RUN",
            "acceptance": "INCONCLUSIVE",
            "reason": "MODEL_TIMEOUT",
            "reward": None,
        }
    if command_status is not ToolCommandStatus.EXITED or exit_code != 0:
        return {
            **base,
            "verifier_execution_status": "NOT_RUN",
            "acceptance": "INCONCLUSIVE",
            "reason": "MODEL_ERROR",
            "reward": None,
        }
    submission = workspace / "submission.json"
    if submission.is_symlink() or not submission.is_file():
        return {
            **base,
            "verifier_execution_status": "NOT_RUN",
            "acceptance": "INCONCLUSIVE",
            "reason": "MISSING_SUBMISSION",
            "reward": None,
        }
    logs = run_dir / "verification" / "logs"
    logs.mkdir(parents=True)
    try:
        reward = run_verifier_in_child(
            task=task.path,
            app=workspace,
            logs=logs,
            timeout_seconds=120.0,
        )
    except (VerifierExecutionError, ValueError) as exc:
        return {
            **base,
            "verifier_execution_status": "ERROR",
            "acceptance": "INCONCLUSIVE",
            "reason": type(exc).__name__,
            "reward": None,
        }
    # Missing binding diagnostics are unknown, not invalid: only an explicit
    # zero/false binding may force INCONCLUSIVE instead of REJECTED.
    if "input_binding" not in reward:
        input_binding_valid: bool | None = None
    else:
        input_binding_valid = reward.get("input_binding") == 1.0
    if "evidence_validity" not in reward:
        artifact_binding_valid: bool | None = None
    else:
        artifact_binding_valid = reward.get("evidence_validity") == 1.0
    accepted = reward.get("reward") == 1.0
    acceptance = (
        "INCONCLUSIVE"
        if input_binding_valid is False
        else "ACCEPTED"
        if accepted
        else "REJECTED"
    )
    return {
        **base,
        "verifier_execution_status": "COMPLETED",
        "acceptance": acceptance,
        "reason": (
            "TERMINAL_INPUT_BINDING_INVALID"
            if input_binding_valid is False
            else "TERMINAL_CLEAN_ROOM_REWARD"
        ),
        "input_binding_valid": input_binding_valid,
        "artifact_binding_valid": artifact_binding_valid,
        "false_certification": reward.get("false_certification"),
        "reward": reward,
    }


def _run_one(
    *,
    spec: TrajectoryValueCalibrationSpec,
    candidate: CalibrationCandidate,
    task: HarborTaskContract,
    repetition: int,
    output: Path,
) -> dict[str, Any]:
    trajectory_id = f"{candidate.task_id}-cal-r{repetition:02d}"
    run_dir = output / "runs" / trajectory_id
    run_dir.mkdir(parents=True)
    with tempfile.TemporaryDirectory(prefix=f"jacobian-{trajectory_id}-") as raw:
        isolated = Path(raw)
        workspace = isolated / "workspace"
        state_dir = isolated / "state"
        prompt = _prepare_workspace(workspace, spec, candidate, task)
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
                    spec=spec,
                    mcp_url=mcp_url,
                    prompt=prompt,
                ),
                cwd=workspace,
                timeout_seconds=spec.timeout_seconds,
                stdout_limit_bytes=32 * 1024 * 1024,
                stderr_limit_bytes=4 * 1024 * 1024,
                environment=operator_environment(include=_CODEX_ENVIRONMENT),
            )
            transcript = run_dir / "codex.jsonl"
            transcript.write_bytes(command.stdout)
            (run_dir / "codex.stderr").write_bytes(command.stderr)
            run_ids = _reasoning_run_ids(transcript)
            reasoning_text = _required_reasoning_log(mcp_url, run_ids)
            (run_dir / "reasoning-log.jsonl").write_text(
                reasoning_text, encoding="utf-8"
            )
        verifier = _verification_outcome(
            task=task,
            workspace=workspace,
            run_dir=run_dir,
            command_status=command.status,
            exit_code=command.exit_code,
        )
        workspace_copy = run_dir / "workspace"
        try:
            _copy_workspace(workspace, workspace_copy)
        except (RuntimeError, OSError, shutil.Error) as exc:
            if workspace_copy.exists():
                shutil.rmtree(workspace_copy, ignore_errors=True)
            verifier = _unsafe_workspace_outcome(verifier, exc)
        _write_json(run_dir / "verifier.json", verifier)
        telemetry = parse_agent_transcript(transcript)
        record = {
            "schema_version": "1",
            "trajectory_id": trajectory_id,
            "dataset_id": candidate.dataset_id,
            "task_id": candidate.task_id,
            "task_family": candidate.task_family,
            "calibration_tags": list(candidate.calibration_tags),
            "repetition": repetition,
            "task_contract": task.as_record(),
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
            "terminal": verifier,
            "surface_digest": surface["surface_digest"],
        }
        _write_json(run_dir / "run.json", record)
        return record


def _wilson(successes: int, denominator: int) -> tuple[float | None, float | None]:
    if denominator == 0:
        return None, None
    z = 1.959963984540054
    p = successes / denominator
    scale = 1 + z * z / denominator
    center = (p + z * z / (2 * denominator)) / scale
    radius = (
        z * math.sqrt(p * (1 - p) / denominator + z * z / (4 * denominator**2)) / scale
    )
    return max(0.0, center - radius), min(1.0, center + radius)


def _terminal_acceptance(record: Mapping[str, Any]) -> str:
    terminal = record["terminal"]
    if isinstance(terminal, Mapping) and terminal.get("input_binding_valid") is False:
        return "INCONCLUSIVE"
    if isinstance(terminal, Mapping):
        return str(terminal["acceptance"])
    return "INCONCLUSIVE"


def summarize(
    spec: TrajectoryValueCalibrationSpec,
    records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Apply the preregistered difficulty-selection rule without label tuning."""

    rows: list[dict[str, Any]] = []
    selected: list[dict[str, Any]] = []
    by_task: dict[tuple[str, str], list[Mapping[str, Any]]] = {}
    for record in records:
        key = (str(record["dataset_id"]), str(record["task_id"]))
        by_task.setdefault(key, []).append(record)
    minimum = spec.selection_rule.minimum_success_rate_millionths / 1_000_000
    maximum = spec.selection_rule.maximum_success_rate_millionths / 1_000_000
    for candidate in spec.candidates:
        task_records = by_task.get((candidate.dataset_id, candidate.task_id), [])
        acceptances = [_terminal_acceptance(record) for record in task_records]
        labelled = [value for value in acceptances if value != "INCONCLUSIVE"]
        accepted = sum(value == "ACCEPTED" for value in labelled)
        rejected = sum(value == "REJECTED" for value in labelled)
        rate = accepted / len(labelled) if labelled else None
        low, high = _wilson(accepted, len(labelled))
        eligible = bool(
            len(labelled) >= spec.selection_rule.minimum_labelled_rollouts
            and rate is not None
            and minimum <= rate <= maximum
        )
        row = {
            "dataset_id": candidate.dataset_id,
            "task_id": candidate.task_id,
            "task_family": candidate.task_family,
            "calibration_tags": list(candidate.calibration_tags),
            "planned_rollouts": spec.repetitions_per_candidate,
            "observed_rollouts": len(task_records),
            "accepted": accepted,
            "rejected": rejected,
            "inconclusive": len(acceptances) - len(labelled),
            "labelled": len(labelled),
            "success_rate": rate,
            "wilson_95_low": low,
            "wilson_95_high": high,
            "selection_eligible": eligible,
        }
        rows.append(row)
        if eligible and len(selected) < spec.selection_rule.maximum_selected_tasks:
            selected.append(row)
    return {
        "schema_version": "1",
        "calibration_id": spec.calibration_id,
        "evidence_class": "public-host-local-codex-difficulty-calibration",
        "terminal_reward": spec.terminal_reward,
        "selection_rule": spec.selection_rule.model_dump(mode="json"),
        "candidate_results": rows,
        "selected_tasks": selected,
        "selected_task_count": len(selected),
        "total_rollouts": len(records),
        "accepted": sum(
            _terminal_acceptance(record) == "ACCEPTED" for record in records
        ),
        "rejected": sum(
            _terminal_acceptance(record) == "REJECTED" for record in records
        ),
        "inconclusive": sum(
            _terminal_acceptance(record) == "INCONCLUSIVE" for record in records
        ),
        "training_performed": False,
        "scorer_intervention": False,
        "causal_claim_authorized": False,
    }


def _artifact_manifest(output: Path) -> dict[str, str]:
    root_manifest = output / "manifest.json"
    return {
        path.relative_to(output).as_posix(): file_digest(path)
        for path in _regular_files(output)
        if path != root_manifest
    }


def run_calibration(
    spec_path: Path,
    output: Path,
    *,
    execute: bool,
) -> dict[str, Any]:
    """Execute one immutable calibration matrix from a clean preregistration."""

    if not execute:
        raise RuntimeError("refusing external Codex execution without --execute")
    spec_path = spec_path.resolve(strict=True)
    output = output.resolve()
    if output.exists():
        raise RuntimeError(f"output directory already exists: {output}")
    spec = load_spec(spec_path)
    if _codex_version(_ROOT) != spec.model.codex_cli_version:
        raise RuntimeError("Codex CLI version differs from the preregistered version")
    model_record, model_cache_digest = _model_cache(spec)  # type: ignore[arg-type]
    if not _repository_is_clean():
        raise RuntimeError("calibration execution requires a clean source tree")
    revision = git_head_sha(_ROOT)
    if revision is None:
        raise RuntimeError("unable to bind the source revision")
    contracts = {
        (candidate.dataset_id, candidate.task_id): _task_contract(candidate)
        for candidate in spec.candidates
    }
    output.mkdir(parents=True)
    records = [
        _run_one(
            spec=spec,
            candidate=candidate,
            task=contracts[(candidate.dataset_id, candidate.task_id)],
            repetition=repetition,
            output=output,
        )
        for candidate in spec.candidates
        for repetition in range(1, spec.repetitions_per_candidate + 1)
    ]
    summary = summarize(spec, records)
    _write_json(output / "summary.json", summary)
    manifest = {
        "schema_version": "1",
        "calibration_id": spec.calibration_id,
        "evidence_class": "public-host-local-codex-difficulty-calibration",
        "source_revision": revision,
        "source_tree_clean_at_start": True,
        "spec": {
            "path": spec_path.relative_to(_ROOT).as_posix(),
            "digest": object_digest(spec.model_dump(mode="json")),
        },
        "codex": {
            "version": spec.model.codex_cli_version,
            "model": spec.model.model_id,
            "reasoning_effort": spec.model.reasoning_effort,
            "model_catalog_record": model_record,
            "model_cache_digest": model_cache_digest,
            "sandbox": spec.sandbox,
            "ephemeral": True,
            "user_config_loaded": False,
            "sampling_seed_available": False,
            "api_key_used": False,
        },
        "jacobian": {
            "reasoning_log_mode": spec.reasoning_log_mode,
            "one_ephemeral_state_directory_per_rollout": True,
        },
        "task_contracts": [
            contracts[(candidate.dataset_id, candidate.task_id)].as_record()
            for candidate in spec.candidates
        ],
        "versions": {
            "runner_digest": file_digest(Path(__file__).resolve(strict=True)),
            "calibration_schema_version": "1",
        },
        "budgets": {
            "candidates": len(spec.candidates),
            "repetitions_per_candidate": spec.repetitions_per_candidate,
            "rollouts": len(records),
            "timeout_seconds_per_rollout": spec.timeout_seconds,
            "retries_for_wrong_answers": 0,
        },
        "artifacts": _artifact_manifest(output),
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
    schema = subparsers.add_parser("schema")
    schema.add_argument("--output", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    if args.command == "run":
        summary = run_calibration(args.spec, args.output, execute=args.execute)
        print(json.dumps(summary, indent=2, sort_keys=True))
        return
    value = TrajectoryValueCalibrationSpec.model_json_schema()
    if args.output is None:
        print(json.dumps(value, indent=2, sort_keys=True))
    else:
        _write_json(args.output, value)


if __name__ == "__main__":
    main()
