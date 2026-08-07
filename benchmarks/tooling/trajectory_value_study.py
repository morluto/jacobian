"""Run and analyze the frozen real-Codex trajectory-value study.

Model execution is operator-only and opt-in.  Every rollout receives its own
workspace, Jacobian state directory, REQUIRED reasoning-log server, and Codex
process.  The model never receives this runner, the clean-room verifier, other
rollouts, or terminal labels.
"""

from __future__ import annotations

import argparse
import asyncio
import itertools
import json
import os
import shutil
import socket
import tempfile
import threading
import time
from collections import Counter, defaultdict
from collections.abc import Iterator, Mapping
from contextlib import contextmanager, suppress
from pathlib import Path
from typing import Any, Literal, Self

import httpx2
from mcp import Client
from mcp.client.streamable_http import streamable_http_client
from mcp_types import TextResourceContents
from pydantic import ConfigDict, Field, model_validator

from benchmarks.tooling.codex_visibility import inspect_surface
from benchmarks.tooling.command_runner import (
    ToolCommandRequest,
    ToolCommandResult,
    ToolCommandStatus,
    ToolResolver,
    git_head_sha,
    operator_environment,
    run_operator_command,
    run_tool_command,
)
from benchmarks.tooling.trajectory_value_study_verifier import (
    file_digest,
    object_digest,
    verifier_digest,
    verify_workspace,
)
from jacobian.contracts.results import ContractModel
from jacobian.eval.telemetry import parse_agent_transcript
from jacobian.eval.trajectory_score import replay_offline_values
from jacobian.eval.trajectory_state import (
    CleanRoomTerminalEvidence,
    StateBoundary,
    TerminalAcceptance,
    TrajectoryExtraction,
    extract_codex_trajectory,
)
from jacobian.eval.trajectory_value import (
    EstimatorEvaluation,
    EstimatorKind,
    LabelledTrajectory,
    OfflineValueComparison,
    StateValueEstimate,
    TrajectoryValueCorpus,
    ValueEstimatorConfig,
    evaluate_offline_trajectories,
)

_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_SPEC = _ROOT / "benchmarks/config/trajectory-value-study-v1.json"
_IDENTIFIER = r"^[a-z0-9][a-z0-9._-]{0,127}$"
_DIGEST = r"^sha256:[0-9a-f]{64}$"
_TASK_KINDS = (
    "INTEGER_BEZOUT",
    "MATRIX_DETERMINANT",
    "POLYNOMIAL_GCD_BEZOUT",
    "GRAPH_MAXIMUM_INDEPENDENT_SET",
)
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
_CRITICAL_TYPED_FIELDS = (
    "scope_digest",
    "assurance_level",
    "checker_state",
    "open_obligation_uris",
    "discharged_obligation_uris",
    "completeness_status",
    "binding_validity",
)


class StudyModel(ContractModel):
    model_id: str = Field(pattern=_IDENTIFIER)
    reasoning_effort: Literal["low", "medium", "high", "xhigh"]
    codex_cli_version: str = Field(min_length=1, max_length=128)


class StudyTask(ContractModel):
    task_id: str = Field(pattern=_IDENTIFIER)
    task_group: str = Field(pattern=_IDENTIFIER)
    task_family: str = Field(min_length=1, max_length=128)
    kind: Literal[
        "INTEGER_BEZOUT",
        "MATRIX_DETERMINANT",
        "POLYNOMIAL_GCD_BEZOUT",
        "GRAPH_MAXIMUM_INDEPENDENT_SET",
    ]
    statement: str = Field(min_length=1, max_length=4096)
    payload: dict[str, Any]
    answer_contract: dict[str, Any]


class TrajectoryValueStudySpec(ContractModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        json_schema_extra={
            "$id": "https://jacobian.invalid/docs/reference/evaluations/schemas/trajectory-value-study-v1.schema.json"
        },
    )

    schema_version: Literal["1"] = "1"
    study_id: str = Field(pattern=_IDENTIFIER)
    model: StudyModel
    repetitions_per_task: Literal[4] = 4
    timeout_seconds: int = Field(ge=60, le=600, strict=True)
    sandbox: Literal["workspace-write"] = "workspace-write"
    reasoning_log_mode: Literal["REQUIRED"] = "REQUIRED"
    tool_mode: Literal["direct"] = "direct"
    web_search: Literal["disabled"] = "disabled"
    evaluator_config: ValueEstimatorConfig = ValueEstimatorConfig()
    agent_instructions: str = Field(min_length=1, max_length=8192)
    tasks: tuple[StudyTask, ...] = Field(min_length=4, max_length=4)
    training_performed: Literal[False] = False
    scorer_intervention: Literal[False] = False
    exact_resume_supported: Literal[False] = False
    intermediate_value_surrogate: Literal[
        "leave-one-trajectory-out-success-frequency-among-compatible-cross-rollout-states"
    ] = "leave-one-trajectory-out-success-frequency-among-compatible-cross-rollout-states"

    @model_validator(mode="after")
    def require_frozen_balanced_suite(self) -> Self:
        if tuple(task.kind for task in self.tasks) != _TASK_KINDS:
            raise ValueError("v1 tasks must contain the four frozen kinds in order")
        identities = [task.task_id for task in self.tasks]
        groups = [task.task_group for task in self.tasks]
        if len(set(identities)) != len(identities):
            raise ValueError("task ids must be unique")
        if len(set(groups)) != len(groups):
            raise ValueError("task groups must be unique")
        return self


def load_spec(path: Path) -> TrajectoryValueStudySpec:
    """Load the completely validated preregistered study."""

    return TrajectoryValueStudySpec.model_validate_json(
        path.read_text(encoding="utf-8")
    )


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _module_digest(relative: str) -> str:
    return file_digest(_ROOT / relative)


def _codex_version(workspace: Path) -> str:
    result = run_operator_command(
        "codex",
        ("--version",),
        cwd=workspace,
        timeout_seconds=30,
        environment=operator_environment(include=_CODEX_ENVIRONMENT),
    )
    if result.status is not ToolCommandStatus.EXITED or result.exit_code != 0:
        raise RuntimeError("codex --version failed")
    return result.stdout.decode("utf-8", errors="replace").strip()


def _model_cache(spec: TrajectoryValueStudySpec) -> tuple[dict[str, Any], str]:
    codex_home = Path(os.environ.get("CODEX_HOME", str(Path.home() / ".codex")))
    path = codex_home / "models_cache.json"
    value = _read_json(path)
    if not isinstance(value, dict) or not isinstance(value.get("models"), list):
        raise RuntimeError("local Codex model cache is malformed")
    records = [
        record
        for record in value["models"]
        if isinstance(record, dict) and record.get("slug") == spec.model.model_id
    ]
    if len(records) != 1:
        raise RuntimeError(
            "the exact preregistered model is absent from the local catalog"
        )
    levels = records[0].get("supported_reasoning_levels")
    if not isinstance(levels, list):
        raise RuntimeError("the exact preregistered model has no reasoning levels")
    supported = {item.get("effort") for item in levels if isinstance(item, dict)}
    if spec.model.reasoning_effort not in supported:
        raise RuntimeError("the exact preregistered reasoning effort is unavailable")
    return records[0], file_digest(path)


def _repository_is_clean() -> bool:
    result = run_operator_command(
        "git", ("status", "--porcelain"), cwd=_ROOT, timeout_seconds=30
    )
    return bool(
        result.status is ToolCommandStatus.EXITED
        and result.exit_code == 0
        and not result.stdout.strip()
    )


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind(("127.0.0.1", 0))
        return int(listener.getsockname()[1])


def _wait_for_port(port: int, server_thread: threading.Thread) -> None:
    deadline = time.monotonic() + 45
    while time.monotonic() < deadline:
        if not server_thread.is_alive():
            raise RuntimeError("Jacobian MCP server exited during startup")
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
            probe.settimeout(0.25)
            if probe.connect_ex(("127.0.0.1", port)) == 0:
                return
        time.sleep(0.1)
    raise RuntimeError("Jacobian MCP server did not become ready")


@contextmanager
def _mcp_server(*, state_dir: Path, run_dir: Path, tenant_id: str) -> Iterator[str]:
    uv = ToolResolver(search_path=os.environ.get("PATH")).resolve("uv")
    if uv is None:
        raise RuntimeError("uv is unavailable")
    port = _free_port()
    url = f"http://127.0.0.1:{port}/mcp"
    stdout_path = run_dir / "jacobian-mcp.stdout"
    stderr_path = run_dir / "jacobian-mcp.stderr"
    environment = dict(operator_environment(include=("HOME", "PATH", "LANG", "LC_ALL")))
    cancellation = threading.Event()
    results: list[ToolCommandResult] = []
    with stdout_path.open("wb") as stdout, stderr_path.open("wb") as stderr:
        request = ToolCommandRequest(
            executable=uv,
            arguments=(
                "run",
                "--locked",
                "jacobian-mcp",
                "--transport",
                "streamable-http",
                "--host",
                "127.0.0.1",
                "--port",
                str(port),
                "--state-dir",
                str(state_dir),
                "--allow-anonymous",
                "--anonymous-tenant-id",
                tenant_id,
                "--reasoning-log-mode",
                "required",
            ),
            cwd=str(_ROOT.resolve(strict=True)),
            environment=environment,
            timeout_seconds=900,
            stdout_limit_bytes=16 * 1024 * 1024,
            stderr_limit_bytes=16 * 1024 * 1024,
            cancellation_event=cancellation,
            stdout_sink=stdout.write,
            stderr_sink=stderr.write,
        )
        server_thread = threading.Thread(
            target=lambda: results.append(run_tool_command(request)),
            name=f"mcp-server-{tenant_id}",
            daemon=True,
        )
        server_thread.start()
        try:
            _wait_for_port(port, server_thread)
            yield url
        finally:
            cancellation.set()
            server_thread.join(timeout=15)
            if server_thread.is_alive():
                raise RuntimeError(
                    "Jacobian MCP server did not stop after cancellation"
                )
            if not results:
                raise RuntimeError("Jacobian MCP server command produced no result")


def _submission_schema(task: StudyTask) -> dict[str, Any]:
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "type": "object",
        "additionalProperties": False,
        "required": ["task_id", "answer"],
        "properties": {
            "task_id": {"const": task.task_id},
            "answer": task.answer_contract,
        },
    }


def _task_payload(task: StudyTask) -> dict[str, Any]:
    return {
        "schema_version": "1",
        "task_id": task.task_id,
        "task_group": task.task_group,
        "task_family": task.task_family,
        "kind": task.kind,
        "statement": task.statement,
        "payload": task.payload,
    }


def _prepare_workspace(
    workspace: Path, spec: TrajectoryValueStudySpec, task: StudyTask
) -> str:
    payload = _task_payload(task)
    _write_json(workspace / "task.json", payload)
    _write_json(workspace / "submission-schema.json", _submission_schema(task))
    prompt = (
        spec.agent_instructions.strip()
        + "\n\nTask statement:\n"
        + task.statement.strip()
        + "\n"
    )
    (workspace / "prompt.txt").write_text(prompt, encoding="utf-8")
    return prompt


def _codex_arguments(
    *, workspace: Path, spec: TrajectoryValueStudySpec, mcp_url: str, prompt: str
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
        f"mcp_servers.jacobian.url={json.dumps(mcp_url)}",
        prompt,
    )


def _reasoning_run_ids(transcript: Path) -> tuple[str, ...]:
    run_ids: set[str] = set()
    for line in transcript.read_text(encoding="utf-8").splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        item = event.get("item") if isinstance(event, dict) else None
        if (
            not isinstance(item, dict)
            or item.get("type") != "mcp_tool_call"
            or not str(item.get("tool", "")).endswith("reasoning.write")
        ):
            continue
        result = item.get("result")
        if not isinstance(result, dict):
            continue
        structured = result.get("structured_content") or result.get("structuredContent")
        run_id = structured.get("run_id") if isinstance(structured, dict) else None
        if isinstance(run_id, str):
            run_ids.add(run_id)
    return tuple(sorted(run_ids))


async def _read_reasoning_resource(url: str, run_id: str) -> str:
    async with (
        httpx2.AsyncClient(trust_env=False, timeout=30) as http,
        Client(
            streamable_http_client(url, http_client=http),
            raise_exceptions=True,
        ) as client,
    ):
        result = await client.read_resource(f"reasoning://run/{run_id}")
        if len(result.contents) != 1 or not isinstance(
            result.contents[0], TextResourceContents
        ):
            raise RuntimeError("reasoning resource is not one text document")
        return result.contents[0].text


def _terminal_evidence(
    command_status: ToolCommandStatus,
    exit_code: int | None,
    verifier: Mapping[str, Any],
) -> CleanRoomTerminalEvidence:
    if command_status is ToolCommandStatus.TIMED_OUT:
        status: Literal["COMPLETED", "TIMEOUT", "CANCELLED", "ERROR"] = "TIMEOUT"
    elif command_status is not ToolCommandStatus.EXITED or exit_code != 0:
        status = "ERROR"
    else:
        status = str(verifier.get("verifier_execution_status", "ERROR"))
    acceptance_value = (
        str(verifier.get("acceptance", "INCONCLUSIVE"))
        if status == "COMPLETED"
        else "INCONCLUSIVE"
    )
    return CleanRoomTerminalEvidence(
        verifier_digest=str(verifier["verifier_digest"]),
        clean_room=True,
        verifier_execution_status=status,
        acceptance=TerminalAcceptance(acceptance_value),
        input_binding_valid=(
            bool(verifier.get("input_binding_valid")) if status == "COMPLETED" else None
        ),
        artifact_binding_valid=(
            bool(verifier.get("artifact_binding_valid"))
            if status == "COMPLETED"
            else None
        ),
    )


def _run_one(
    *,
    spec: TrajectoryValueStudySpec,
    task: StudyTask,
    repetition: int,
    output: Path,
) -> dict[str, Any]:
    trajectory_id = f"{task.task_id}-r{repetition:02d}"
    run_dir = output / "runs" / trajectory_id
    run_dir.mkdir(parents=True)
    with tempfile.TemporaryDirectory(prefix=f"jacobian-{trajectory_id}-") as raw:
        isolated = Path(raw)
        workspace = isolated / "workspace"
        state_dir = isolated / "state"
        workspace.mkdir()
        prompt = _prepare_workspace(workspace, spec, task)
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
            reasoning_text = ""
            if len(run_ids) == 1:
                with suppress(Exception):
                    reasoning_text = asyncio.run(
                        _read_reasoning_resource(mcp_url, run_ids[0])
                    )
            (run_dir / "reasoning-log.jsonl").write_text(
                reasoning_text, encoding="utf-8"
            )
        for name in ("task.json", "submission-schema.json", "prompt.txt"):
            shutil.copyfile(workspace / name, run_dir / name)
        submission = workspace / "submission.json"
        if submission.is_file() and not submission.is_symlink():
            shutil.copyfile(submission, run_dir / "submission.json")
        task_payload = _task_payload(task)
        verifier = verify_workspace(task_payload, workspace)
        _write_json(run_dir / "verifier.json", verifier)
        terminal = _terminal_evidence(command.status, command.exit_code, verifier)
        extraction = extract_codex_trajectory(
            transcript,
            task_family=task.task_family,
            terminal_evidence=terminal,
        )
        _write_json(run_dir / "extraction.json", extraction.model_dump(mode="json"))
        telemetry = parse_agent_transcript(transcript)
        record = {
            "schema_version": "1",
            "trajectory_id": trajectory_id,
            "task_id": task.task_id,
            "task_group": task.task_group,
            "task_family": task.task_family,
            "repetition": repetition,
            "command": {
                "status": command.status,
                "exit_code": command.exit_code,
                "diagnostic": command.diagnostic,
                "stdout_exceeded": command.stdout_exceeded,
                "stderr_exceeded": command.stderr_exceeded,
            },
            "reasoning_run_ids": run_ids,
            "reasoning_protocol": telemetry.get("reasoning_protocol"),
            "usage": telemetry.get("usage"),
            "terminal": terminal.model_dump(mode="json"),
            "surface_digest": surface["surface_digest"],
            "artifacts": {
                path.name: file_digest(path)
                for path in sorted(run_dir.iterdir())
                if path.is_file() and path.name != "run.json"
            },
        }
        _write_json(run_dir / "run.json", record)
        return record


def _corpus(
    spec: TrajectoryValueStudySpec, output: Path, records: list[dict[str, Any]]
) -> tuple[TrajectoryValueCorpus, list[dict[str, str]]]:
    task_by_id = {task.task_id: task for task in spec.tasks}
    labelled: list[LabelledTrajectory] = []
    exclusions: list[dict[str, str]] = []
    for record in records:
        terminal = record["terminal"]
        trajectory_id = record["trajectory_id"]
        if terminal["acceptance"] not in {"ACCEPTED", "REJECTED"}:
            exclusions.append(
                {
                    "trajectory_id": trajectory_id,
                    "reason": "terminal verifier outcome is inconclusive",
                }
            )
            continue
        extraction = TrajectoryExtraction.model_validate(
            _read_json(output / "runs" / trajectory_id / "extraction.json")
        )
        if not any(state.boundary is StateBoundary.PLAN for state in extraction.states):
            exclusions.append(
                {
                    "trajectory_id": trajectory_id,
                    "reason": "no successful PLAN boundary",
                }
            )
            continue
        task = task_by_id[record["task_id"]]
        labelled.append(
            LabelledTrajectory(
                trajectory_id=trajectory_id,
                task_group=task.task_group,
                extraction=extraction,
            )
        )
    return (
        TrajectoryValueCorpus(
            corpus_id=spec.study_id,
            evaluator_config=spec.evaluator_config,
            trajectories=tuple(labelled),
        ),
        exclusions,
    )


def _estimate_lookup(
    comparison: OfflineValueComparison, kind: EstimatorKind
) -> tuple[EstimatorEvaluation, dict[str, StateValueEstimate]]:
    evaluation = next(item for item in comparison.evaluations if item.estimator is kind)
    return evaluation, {item.observation_id: item for item in evaluation.estimates}


def _state_rows(
    corpus: TrajectoryValueCorpus, comparison: OfflineValueComparison
) -> list[dict[str, Any]]:
    _, estimates = _estimate_lookup(comparison, EstimatorKind.GROUP_ROLLOUT)
    rows: list[dict[str, Any]] = []
    for trajectory in corpus.trajectories:
        evidence = trajectory.extraction.terminal_evidence
        reward = int(
            evidence is not None and evidence.acceptance is TerminalAcceptance.ACCEPTED
        )
        for state in trajectory.extraction.states:
            observation_id = f"{trajectory.trajectory_id}:{state.index}"
            if observation_id not in estimates:
                continue
            rows.append(
                {
                    "observation_id": observation_id,
                    "trajectory_id": trajectory.trajectory_id,
                    "task_group": trajectory.task_group,
                    "reward": reward,
                    "state": state,
                }
            )
    return rows


def _trajectory_weighted_brier(values: list[tuple[str, float, int]]) -> float:
    grouped: dict[str, list[float]] = defaultdict(list)
    for trajectory_id, estimate, reward in values:
        grouped[trajectory_id].append((estimate - reward) ** 2)
    return round(
        sum(sum(errors) / len(errors) for errors in grouped.values()) / len(grouped),
        12,
    )


def _feature_value(state: Any, name: str) -> object:
    hard = state.hard_state
    if name == "boundary":
        return state.boundary.value
    if name == "typed_object_types":
        return tuple(sorted(item.object_type for item in hard.typed_objects))
    if name == "typed_object_identities":
        return tuple(
            sorted(
                (item.object_type, item.content_digest) for item in hard.typed_objects
            )
        )
    if name == "artifact_roles":
        return tuple(sorted(item.role for item in hard.artifacts))
    value = getattr(hard, name)
    if hasattr(value, "value"):
        return value.value
    if isinstance(value, tuple):
        return tuple(item.value if hasattr(item, "value") else item for item in value)
    return value


def _dimension_signal(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    names = (
        "boundary",
        "typed_object_types",
        "typed_object_identities",
        "artifact_roles",
        "candidate_state",
        "checker_state",
        "open_obligation_uris",
        "discharged_obligation_uris",
        "execution_status",
        "completeness_status",
        "completeness_assurance",
        "assurance_level",
        "scope_digest",
        "scope_escalation_errors",
        "binding_validity",
        "latest_meaningful_transitions",
        "reasoning_protocol_state",
    )
    results: list[dict[str, Any]] = []
    for name in names:
        estimates: list[tuple[str, float, int]] = []
        fallback_count = 0
        for target in rows:
            feature = _feature_value(target["state"], name)
            support = {
                row["trajectory_id"]: row["reward"]
                for row in rows
                if row["task_group"] == target["task_group"]
                and row["trajectory_id"] != target["trajectory_id"]
                and _feature_value(row["state"], name) == feature
            }
            if not support:
                fallback_count += 1
                support = {
                    row["trajectory_id"]: row["reward"]
                    for row in rows
                    if row["task_group"] == target["task_group"]
                    and row["trajectory_id"] != target["trajectory_id"]
                }
            estimate = sum(support.values()) / len(support)
            estimates.append((target["trajectory_id"], estimate, target["reward"]))
        results.append(
            {
                "dimension": name,
                "brier_score": _trajectory_weighted_brier(estimates),
                "fallback_count": fallback_count,
            }
        )
    return sorted(results, key=lambda item: (item["brier_score"], item["dimension"]))


def _value_drop_prediction(
    comparison: OfflineValueComparison,
) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    trajectory_ids = sorted(
        {
            estimate.trajectory_id
            for evaluation in comparison.evaluations
            for estimate in evaluation.estimates
        }
    )
    for kind in EstimatorKind:
        true_positive = false_positive = true_negative = false_negative = 0
        warnings: list[dict[str, Any]] = []
        for trajectory_id in trajectory_ids:
            replay = replay_offline_values(
                comparison, trajectory_id=trajectory_id, estimator=kind
            )
            negative = [
                state
                for state in replay.states
                if state.value_delta is not None and state.value_delta < 0
            ]
            predicted_failure = bool(negative)
            failed = replay.eventual_terminal_reward == 0
            if predicted_failure and failed:
                true_positive += 1
            elif predicted_failure:
                false_positive += 1
            elif failed:
                false_negative += 1
            else:
                true_negative += 1
            if negative:
                warnings.append(
                    {
                        "trajectory_id": trajectory_id,
                        "first_observation_id": negative[0].observation_id,
                        "first_value_delta": negative[0].value_delta,
                        "eventual_terminal_reward": replay.eventual_terminal_reward,
                    }
                )
        result[kind.value] = {
            "rule": "any strictly negative preterminal selected-state value delta",
            "true_positive": true_positive,
            "false_positive": false_positive,
            "true_negative": true_negative,
            "false_negative": false_negative,
            "precision": (
                round(true_positive / (true_positive + false_positive), 12)
                if true_positive + false_positive
                else None
            ),
            "recall": (
                round(true_positive / (true_positive + false_negative), 12)
                if true_positive + false_negative
                else None
            ),
            "warnings": warnings,
        }
    return result


def _pair_analysis(
    rows: list[dict[str, Any]], comparison: OfflineValueComparison
) -> tuple[dict[str, Any], dict[str, Any]]:
    _, typed = _estimate_lookup(comparison, EstimatorKind.JACOBIAN_TYPED)
    _, hybrid = _estimate_lookup(comparison, EstimatorKind.HYBRID_TYPED_TEXT)
    text_evaluation, _ = _estimate_lookup(comparison, EstimatorKind.REASONING_TEXT)
    text_cluster = {
        member: cluster.cluster_id
        for cluster in text_evaluation.clusters
        for member in cluster.member_observation_ids
    }
    same_typed_pairs = []
    text_alias_pairs = []
    critical_counts: Counter[str] = Counter()
    for left, right in itertools.combinations(rows, 2):
        if (
            left["trajectory_id"] == right["trajectory_id"]
            or left["task_group"] != right["task_group"]
        ):
            continue
        left_id = left["observation_id"]
        right_id = right["observation_id"]
        left_typed = typed[left_id]
        right_typed = typed[right_id]
        if (
            left_typed.typed_compatibility_digest
            == right_typed.typed_compatibility_digest
            and left_typed.reasoning_text_digest != right_typed.reasoning_text_digest
        ):
            same_typed_pairs.append(
                {
                    "left": left_id,
                    "right": right_id,
                    "mixed_terminal_outcome": left["reward"] != right["reward"],
                    "hybrid_separated": (
                        hybrid[left_id].cluster_id != hybrid[right_id].cluster_id
                    ),
                }
            )
        if (
            text_cluster[left_id] == text_cluster[right_id]
            and left_typed.typed_compatibility_digest
            != right_typed.typed_compatibility_digest
        ):
            differing = [
                name
                for name in _CRITICAL_TYPED_FIELDS
                if _feature_value(left["state"], name)
                != _feature_value(right["state"], name)
            ]
            if differing:
                critical_counts.update(differing)
                text_alias_pairs.append(
                    {"left": left_id, "right": right_id, "differing_fields": differing}
                )
    return (
        {
            "same_typed_different_text_pair_count": len(same_typed_pairs),
            "mixed_terminal_outcome_pair_count": sum(
                pair["mixed_terminal_outcome"] for pair in same_typed_pairs
            ),
            "hybrid_separated_pair_count": sum(
                pair["hybrid_separated"] for pair in same_typed_pairs
            ),
            "pairs": same_typed_pairs,
        },
        {
            "same_text_cluster_typed_incompatible_pair_count": len(text_alias_pairs),
            "critical_difference_counts": dict(sorted(critical_counts.items())),
            "pairs": text_alias_pairs,
        },
    )


def analyze(
    spec: TrajectoryValueStudySpec,
    output: Path,
    records: list[dict[str, Any]],
) -> dict[str, Any]:
    """Build the frozen comparison, observation replays, and A-F diagnostics."""

    corpus, exclusions = _corpus(spec, output, records)
    comparison = evaluate_offline_trajectories(corpus)
    _write_json(output / "corpus.json", corpus.model_dump(mode="json"))
    _write_json(output / "comparison.json", comparison.model_dump(mode="json"))
    for trajectory in corpus.trajectories:
        for kind in EstimatorKind:
            replay = replay_offline_values(
                comparison, trajectory_id=trajectory.trajectory_id, estimator=kind
            )
            _write_json(
                output
                / "replays"
                / kind.value.lower()
                / f"{trajectory.trajectory_id}.json",
                replay.model_dump(mode="json"),
            )
    metrics = {
        evaluation.estimator.value: evaluation.metrics.model_dump(mode="json")
        for evaluation in comparison.evaluations
    }
    rows = _state_rows(corpus, comparison)
    same_typed, text_alias = _pair_analysis(rows, comparison)
    group_brier = metrics[EstimatorKind.GROUP_ROLLOUT.value]["brier_score"]
    numca_brier = metrics[EstimatorKind.NUMCA_NUMERICAL.value]["brier_score"]
    typed_brier = metrics[EstimatorKind.JACOBIAN_TYPED.value]["brier_score"]
    hybrid_brier = metrics[EstimatorKind.HYBRID_TYPED_TEXT.value]["brier_score"]
    return {
        "schema_version": "1",
        "study_id": spec.study_id,
        "run_count": len(records),
        "labelled_trajectory_count": len(corpus.trajectories),
        "accepted_count": sum(
            trajectory.extraction.terminal_evidence is not None
            and trajectory.extraction.terminal_evidence.acceptance
            is TerminalAcceptance.ACCEPTED
            for trajectory in corpus.trajectories
        ),
        "rejected_count": sum(
            trajectory.extraction.terminal_evidence is not None
            and trajectory.extraction.terminal_evidence.acceptance
            is TerminalAcceptance.REJECTED
            for trajectory in corpus.trajectories
        ),
        "excluded": exclusions,
        "metrics": metrics,
        "question_a": {
            "typed_brier_delta_vs_group": round(typed_brier - group_brier, 12),
            "typed_brier_delta_vs_numca": round(typed_brier - numca_brier, 12),
            "hybrid_brier_delta_vs_group": round(hybrid_brier - group_brier, 12),
            "hybrid_brier_delta_vs_numca": round(hybrid_brier - numca_brier, 12),
        },
        "question_b": {
            "hybrid_brier_delta_vs_typed": round(hybrid_brier - typed_brier, 12)
        },
        "question_c": _value_drop_prediction(comparison),
        "question_d": same_typed,
        "question_e": text_alias,
        "question_f": {
            "univariate_exact_match_dimensions": _dimension_signal(rows),
            "fallback": "other rollouts in the same task group",
        },
        "intermediate_value_reference": {
            "kind": spec.intermediate_value_surrogate,
            "exact_resume_supported": False,
            "limitation": (
                "Codex CLI cannot resume an arbitrary intermediate tool boundary; "
                "the proxy is the leave-one-trajectory-out terminal-success mean "
                "of independently sampled compatible cross-rollout states."
            ),
        },
        "training_performed": False,
        "causal_claim_authorized": False,
    }


def _artifact_manifest(output: Path) -> dict[str, str]:
    return {
        path.relative_to(output).as_posix(): file_digest(path)
        for path in sorted(output.rglob("*"))
        if path.is_file() and path.name != "manifest.json"
    }


def run_study(spec_path: Path, output: Path, *, execute: bool) -> dict[str, Any]:
    """Execute the preregistered study and persist all observable evidence."""

    if not execute:
        raise RuntimeError("refusing paid/external Codex execution without --execute")
    spec_path = spec_path.resolve(strict=True)
    output = output.resolve()
    if output.exists():
        raise RuntimeError(f"output directory already exists: {output}")
    spec = load_spec(spec_path)
    if _codex_version(_ROOT) != spec.model.codex_cli_version:
        raise RuntimeError("Codex CLI version differs from the preregistered version")
    model_record, model_cache_digest = _model_cache(spec)
    if not _repository_is_clean():
        raise RuntimeError("study execution requires a clean preregistered source tree")
    revision = git_head_sha(_ROOT)
    if revision is None:
        raise RuntimeError("unable to bind the source revision")
    output.mkdir(parents=True)
    records = [
        _run_one(spec=spec, task=task, repetition=repetition, output=output)
        for task in spec.tasks
        for repetition in range(1, spec.repetitions_per_task + 1)
    ]
    summary = analyze(spec, output, records)
    _write_json(output / "summary.json", summary)
    manifest = {
        "schema_version": "1",
        "study_id": spec.study_id,
        "evidence_class": "public-host-local-codex-workflow-observation",
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
        },
        "jacobian": {
            "reasoning_log_mode": spec.reasoning_log_mode,
            "one_ephemeral_state_directory_per_rollout": True,
        },
        "versions": {
            "runner_digest": file_digest(Path(__file__).resolve(strict=True)),
            "verifier_digest": verifier_digest(),
            "extractor_digest": _module_digest("src/jacobian/eval/trajectory_state.py"),
            "evaluator_digest": _module_digest("src/jacobian/eval/trajectory_value.py"),
            "scorer_digest": _module_digest("src/jacobian/eval/trajectory_score.py"),
            "state_schema_version": "1",
            "evaluator_schema_version": "1",
            "scorer_schema_version": "1",
        },
        "budgets": {
            "tasks": len(spec.tasks),
            "repetitions_per_task": spec.repetitions_per_task,
            "rollouts": len(records),
            "timeout_seconds_per_rollout": spec.timeout_seconds,
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


def main() -> None:
    args = _parser().parse_args()
    if args.command == "schema":
        value = TrajectoryValueStudySpec.model_json_schema(mode="validation")
        rendered = json.dumps(value, indent=2, sort_keys=True) + "\n"
        if args.output is None:
            print(rendered, end="")
        else:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(rendered, encoding="utf-8")
        return
    summary = run_study(args.spec, args.output, execute=args.execute)
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()


__all__ = [
    "StudyModel",
    "StudyTask",
    "TrajectoryValueStudySpec",
    "analyze",
    "load_spec",
    "run_study",
]
