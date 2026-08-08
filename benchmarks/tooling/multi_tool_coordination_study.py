"""Run the preregistered public multi-tool coordination observation.

This is a host-local fallback for operators without a Docker runtime.  It is
not a Harbor result: the evaluated agent receives only the three public task
files in an isolated workspace, while the unchanged task-owned verifier runs
after Codex exits through the repository's fresh-interpreter verifier harness.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import shutil
import socket
import tempfile
import threading
import time
from collections import Counter
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, Protocol, Self

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
from benchmarks.tooling.harbor_suite import (
    get_suite,
    select_task_refs,
    task_digest,
    verifier_bundle_checksum,
)
from benchmarks.validation._verifier_child import (
    VerifierExecutionError,
    VerifierOutput,
    run_verifier_in_child,
)
from jacobian.contracts.results import ContractModel
from jacobian.eval.telemetry import parse_agent_transcript
from jacobian.eval.trajectory_state import (
    CleanRoomTerminalEvidence,
    TerminalAcceptance,
    extract_codex_trajectory,
)

_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_SPEC = _ROOT / "benchmarks/config/multi-tool-coordination-pr1.json"
_DEFAULT_OUTPUT = _ROOT / "benchmarks/results/multi-tool-coordination-pr1"
_TASK_ORDER = (
    "graph-artifact-composition",
    "rp2-homology-lattice",
    "polynomial-map-collision",
    "exact-farkas-ldl-slice",
    "apollonius-gap-repair",
    "hermite-normal-form",
)
_PUBLIC_FILES = (
    "instruction.md",
    "environment/input.json",
    "environment/submission_schema.json",
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


class StudyModel(ContractModel):
    model_id: Literal["gpt-5.4-mini"]
    reasoning_effort: Literal["medium"]
    codex_cli_version: Literal["codex-cli 0.147.0"]


class StudyTask(ContractModel):
    task_id: str = Field(pattern=r"^[a-z0-9][a-z0-9-]{0,127}$")
    harbor_task_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    domain: str = Field(min_length=1, max_length=128)
    coordination_probe: tuple[str, ...] = Field(min_length=1, max_length=8)


class CoordinationStudySpec(ContractModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["1"] = "1"
    study_id: Literal["multi-tool-coordination-pr1"]
    evidence_class: Literal["public-host-local-workflow-observation"]
    causal_claim_authorized: Literal[False] = False
    harbor_execution_claimed: Literal[False] = False
    source_base_revision: Literal["7edc7ba9035e5f9de5a04f406cfe50e7da28d8e1"]
    dataset: Literal["mathematical-benchmarks-v1"]
    snapshot_id: Literal[
        "sha256:26e558abcfda80f944ff1659f73b3c89b22ed4ddd2700d8340c067dc4ed7b323"
    ]
    model: StudyModel
    repetitions_per_task: Literal[2] = 2
    timeout_seconds_per_rollout: Literal[600] = 600
    sandbox: Literal["workspace-write"] = "workspace-write"
    reasoning_log_mode: Literal["REQUIRED"] = "REQUIRED"
    web_search: Literal["disabled"] = "disabled"
    wrong_answer_retries: Literal[0] = 0
    terminal_reward: Literal["task-owned-clean-room-verifier-only"]
    tool_call_reward: Literal[0] = 0
    tasks: tuple[StudyTask, ...] = Field(min_length=6, max_length=6)
    stop_rules: tuple[str, ...] = Field(min_length=3, max_length=8)
    agent_instructions: str = Field(min_length=1, max_length=4096)

    @model_validator(mode="after")
    def require_frozen_matrix(self) -> Self:
        if tuple(task.task_id for task in self.tasks) != _TASK_ORDER:
            raise ValueError("tasks must match the frozen PR1 order")
        if len({task.harbor_task_digest for task in self.tasks}) != len(self.tasks):
            raise ValueError("task digests must be unique")
        return self


class RunnableCoordinationSpec(Protocol):
    """Structural contract shared by frozen coordination runners."""

    agent_instructions: str
    model: StudyModel
    reasoning_log_mode: str
    sandbox: str
    timeout_seconds_per_rollout: int
    web_search: str


def load_spec(path: Path) -> CoordinationStudySpec:
    """Load the completely validated preregistration."""

    return CoordinationStudySpec.model_validate_json(path.read_text(encoding="utf-8"))


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _digest(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _object_digest(value: object) -> str:
    encoded = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode()
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


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
    return result.stdout.decode(errors="replace").strip()


def _model_record(spec: RunnableCoordinationSpec) -> tuple[dict[str, Any], str]:
    codex_home = Path(os.environ.get("CODEX_HOME", str(Path.home() / ".codex")))
    cache = codex_home / "models_cache.json"
    payload = json.loads(cache.read_text(encoding="utf-8"))
    records = [
        record
        for record in payload.get("models", [])
        if isinstance(record, dict) and record.get("slug") == spec.model.model_id
    ]
    if len(records) != 1:
        raise RuntimeError(
            "the exact preregistered model is absent from the local catalog"
        )
    levels = records[0].get("supported_reasoning_levels")
    supported = {item.get("effort") for item in levels or [] if isinstance(item, dict)}
    if spec.model.reasoning_effort not in supported:
        raise RuntimeError("the preregistered reasoning effort is unavailable")
    return records[0], _digest(cache)


def _repository_is_clean() -> bool:
    result = run_operator_command(
        "git", ("status", "--porcelain"), cwd=_ROOT, timeout_seconds=30
    )
    return bool(
        result.status is ToolCommandStatus.EXITED
        and result.exit_code == 0
        and not result.stdout.strip()
    )


def _parent_revision() -> str:
    result = run_operator_command(
        "git", ("rev-parse", "HEAD^"), cwd=_ROOT, timeout_seconds=30
    )
    if result.status is not ToolCommandStatus.EXITED or result.exit_code != 0:
        raise RuntimeError("unable to resolve the preregistration parent revision")
    return result.stdout.decode(errors="replace").strip()


def _tmux_session() -> str:
    if not os.environ.get("TMUX"):
        raise RuntimeError("model execution must run inside a named tmux session")
    result = run_operator_command(
        "tmux", ("display-message", "-p", "#S"), cwd=_ROOT, timeout_seconds=10
    )
    if result.status is not ToolCommandStatus.EXITED or result.exit_code != 0:
        raise RuntimeError("unable to identify the tmux session")
    name = result.stdout.decode(errors="replace").strip()
    if not name:
        raise RuntimeError("tmux session name is empty")
    return name


def _task_records(spec: CoordinationStudySpec) -> dict[str, dict[str, Any]]:
    suite = get_suite(spec.dataset)
    refs = select_task_refs(suite, tuple(task.task_id for task in spec.tasks))
    declared = {task.task_id: task for task in spec.tasks}
    records: dict[str, dict[str, Any]] = {}
    for ref in refs:
        expected = declared[ref.path.name]
        observed = task_digest(ref.path)
        if observed != expected.harbor_task_digest:
            raise RuntimeError(f"task digest drift: {ref.path.name}")
        public = {relative: _digest(ref.path / relative) for relative in _PUBLIC_FILES}
        records[ref.path.name] = {
            "path": ref.path,
            "harbor_task_digest": observed,
            "public_files": public,
            "public_bundle_digest": _object_digest(public),
            "verifier_bundle_digest": "sha256:"
            + verifier_bundle_checksum(ref.path / "tests"),
        }
    return records


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind(("127.0.0.1", 0))
        return int(listener.getsockname()[1])


def _wait_for_port(port: int, thread: threading.Thread) -> None:
    deadline = time.monotonic() + 45
    while time.monotonic() < deadline:
        if not thread.is_alive():
            raise RuntimeError("Jacobian MCP server exited during startup")
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
            probe.settimeout(0.25)
            if probe.connect_ex(("127.0.0.1", port)) == 0:
                return
        time.sleep(0.1)
    raise RuntimeError("Jacobian MCP server did not become ready")


@contextmanager
def _mcp_server(*, state: Path, run_dir: Path, tenant: str) -> Iterator[str]:
    uv = ToolResolver(search_path=os.environ.get("PATH")).resolve("uv")
    if uv is None:
        raise RuntimeError("uv is unavailable")
    port = _free_port()
    url = f"http://127.0.0.1:{port}/mcp"
    cancellation = threading.Event()
    results: list[ToolCommandResult] = []
    with (
        (run_dir / "jacobian-mcp.stdout").open("wb") as stdout,
        (run_dir / "jacobian-mcp.stderr").open("wb") as stderr,
    ):
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
                str(state),
                "--allow-anonymous",
                "--anonymous-tenant-id",
                tenant,
                "--reasoning-log-mode",
                "required",
            ),
            cwd=str(_ROOT),
            environment=operator_environment(
                include=("HOME", "PATH", "LANG", "LC_ALL")
            ),
            timeout_seconds=900,
            stdout_limit_bytes=16 * 1024 * 1024,
            stderr_limit_bytes=16 * 1024 * 1024,
            cancellation_event=cancellation,
            stdout_sink=stdout.write,
            stderr_sink=stderr.write,
        )
        thread = threading.Thread(
            target=lambda: results.append(run_tool_command(request)),
            name=f"mcp-{tenant}",
            daemon=True,
        )
        thread.start()
        try:
            _wait_for_port(port, thread)
            yield url
        finally:
            cancellation.set()
            thread.join(timeout=15)
            if thread.is_alive() or not results:
                raise RuntimeError("Jacobian MCP server did not stop cleanly")


def _prepare_workspace(
    workspace: Path, spec: RunnableCoordinationSpec, task_path: Path
) -> str:
    shutil.copyfile(task_path / "instruction.md", workspace / "instruction.md")
    shutil.copyfile(task_path / "environment/input.json", workspace / "input.json")
    shutil.copyfile(
        task_path / "environment/submission_schema.json",
        workspace / "submission_schema.json",
    )
    prompt = spec.agent_instructions.strip() + "\n"
    (workspace / "prompt.txt").write_text(prompt, encoding="utf-8")
    return prompt


def _codex_arguments(
    *, workspace: Path, spec: RunnableCoordinationSpec, mcp_url: str, prompt: str
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


def _reasoning_run_ids(transcript: Path) -> tuple[str, ...]:
    run_ids: set[str] = set()
    for line in transcript.read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        item = event.get("item") if isinstance(event, dict) else None
        if not isinstance(item, dict) or item.get("type") != "mcp_tool_call":
            continue
        if not str(item.get("tool", "")).endswith("reasoning.write"):
            continue
        result = item.get("result")
        structured = (
            result.get("structured_content") if isinstance(result, dict) else None
        )
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


def _reasoning_log(url: str, run_ids: tuple[str, ...]) -> tuple[str, str]:
    if len(run_ids) != 1:
        return "", "INCOMPLETE_AMBIGUOUS_RUN_ID"
    try:
        return asyncio.run(_read_reasoning_resource(url, run_ids[0])), "COMPLETE"
    except Exception as exc:  # evidence is preserved; the run is not retried
        return "", f"INCOMPLETE_RESOURCE_ERROR:{type(exc).__name__}"


def _status_value(status: ToolCommandStatus) -> str:
    return status.value if hasattr(status, "value") else str(status)


def _terminal(
    *,
    command_status: ToolCommandStatus,
    exit_code: int | None,
    verifier_execution_status: Literal["COMPLETED", "ERROR"],
    reasoning_status: str,
    reward: float,
) -> tuple[str, str]:
    if command_status is ToolCommandStatus.TIMED_OUT:
        return "INCONCLUSIVE", "MODEL_TIMEOUT"
    if command_status is not ToolCommandStatus.EXITED or exit_code != 0:
        return "INCONCLUSIVE", "MODEL_ERROR"
    if verifier_execution_status != "COMPLETED":
        return "INCONCLUSIVE", "VERIFIER_INFRASTRUCTURE_ERROR"
    if reasoning_status != "COMPLETE":
        return "INCONCLUSIVE", "REASONING_PROTOCOL_INCOMPLETE"
    if reward == 1.0:
        return "ACCEPTED", "EXACT_VERIFIER_FULL_REWARD"
    return "REJECTED", "EXACT_VERIFIER_REJECTED_TERMINAL_OBJECT"


def _clean_room_terminal(
    *,
    acceptance: str,
    verifier_digest: str,
    verifier_execution_status: Literal["COMPLETED", "ERROR"],
    details: Mapping[str, object],
) -> CleanRoomTerminalEvidence:
    completed = verifier_execution_status == "COMPLETED"
    raw_input_binding = details.get("input_binding", details.get("input_integrity"))
    raw_artifact_binding = details.get("evidence_validity")
    return CleanRoomTerminalEvidence(
        verifier_digest=verifier_digest,
        clean_room=True,
        verifier_execution_status=verifier_execution_status,
        acceptance=TerminalAcceptance(acceptance),
        input_binding_valid=(
            raw_input_binding == 1.0
            if completed and raw_input_binding is not None
            else None
        ),
        artifact_binding_valid=(
            raw_artifact_binding == 1.0
            if completed and raw_artifact_binding is not None
            else None
        ),
    )


_REJECTED_WORKSPACE_DETAILS: Mapping[str, object] = {
    "assurance_calibration": 0.0,
    "correctness": 0.0,
    "evidence_validity": 0.0,
    "false_certification": False,
    "input_binding": 0.0,
    "input_integrity": 0.0,
    "limitation_accuracy": 0.0,
    "protocol_compliance": 0.0,
    "scope_accuracy": 0.0,
}


def _execute_verifier(
    *, task: Path, workspace: Path, logs: Path
) -> tuple[VerifierOutput, Literal["COMPLETED", "ERROR"], str | None]:
    """Run the verifier without confusing invalid workspaces with verifier faults."""

    try:
        return (
            run_verifier_in_child(task=task, app=workspace, logs=logs),
            "COMPLETED",
            None,
        )
    except ValueError as exc:
        diagnostic = " ".join(str(exc).split())[:512]
        if "workspace entry" in diagnostic:
            return (
                VerifierOutput(reward=0.0, details=_REJECTED_WORKSPACE_DETAILS),
                "COMPLETED",
                f"INVALID_WORKSPACE:{diagnostic}",
            )
        return (
            VerifierOutput(reward=0.0, details={}),
            "ERROR",
            f"ValueError:{diagnostic}",
        )
    except VerifierExecutionError as exc:
        diagnostic = " ".join(str(exc).split())[:512]
        return (
            VerifierOutput(reward=0.0, details={}),
            "ERROR",
            f"VerifierExecutionError:{diagnostic}",
        )


def _copy_workspace(workspace: Path, destination: Path) -> None:
    if destination.exists():
        raise RuntimeError("workspace destination already exists")
    shutil.copytree(workspace, destination, symlinks=True)


def _run_one(
    *,
    spec: RunnableCoordinationSpec,
    task: StudyTask,
    task_record: Mapping[str, Any],
    repetition: int,
    output: Path,
) -> dict[str, Any]:
    trajectory_id = f"{task.task_id}-r{repetition:02d}"
    run_dir = output / "runs" / trajectory_id
    run_dir.mkdir(parents=True)
    started_at = _now()
    with tempfile.TemporaryDirectory(prefix=f"jacobian-{trajectory_id}-") as raw:
        isolated = Path(raw)
        workspace = isolated / "workspace"
        state = isolated / "state"
        workspace.mkdir()
        task_path = Path(task_record["path"])
        prompt = _prepare_workspace(workspace, spec, task_path)
        command: ToolCommandResult | None = None
        reasoning_status = "INCOMPLETE_NOT_STARTED"
        run_ids: tuple[str, ...] = ()
        surface: dict[str, Any] = {}
        try:
            with _mcp_server(state=state, run_dir=run_dir, tenant=trajectory_id) as url:
                surface = asyncio.run(inspect_surface(url, 30))
                _write_json(run_dir / "surface.json", surface)
                command = run_operator_command(
                    "codex",
                    _codex_arguments(
                        workspace=workspace, spec=spec, mcp_url=url, prompt=prompt
                    ),
                    cwd=workspace,
                    timeout_seconds=spec.timeout_seconds_per_rollout,
                    stdout_limit_bytes=32 * 1024 * 1024,
                    stderr_limit_bytes=4 * 1024 * 1024,
                    environment=operator_environment(include=_CODEX_ENVIRONMENT),
                )
                transcript = run_dir / "codex.jsonl"
                transcript.write_bytes(command.stdout)
                (run_dir / "codex.stderr").write_bytes(command.stderr)
                run_ids = _reasoning_run_ids(transcript)
                text, reasoning_status = _reasoning_log(url, run_ids)
                (run_dir / "reasoning-log.jsonl").write_text(text, encoding="utf-8")
        finally:
            _copy_workspace(workspace, run_dir / "workspace")

    transcript = run_dir / "codex.jsonl"
    logs = run_dir / "verification/logs"
    logs.mkdir(parents=True)
    verifier, verifier_execution_status, verifier_error = _execute_verifier(
        task=Path(task_record["path"]),
        workspace=run_dir / "workspace",
        logs=logs,
    )
    if command is None or not transcript.is_file():
        acceptance, reason = "INCONCLUSIVE", "RUNNER_ERROR_BEFORE_MODEL_RESULT"
        command_status = ToolCommandStatus.START_FAILED
        exit_code = None
        transcript.write_text("", encoding="utf-8")
    else:
        command_status = command.status
        exit_code = command.exit_code
        acceptance, reason = _terminal(
            command_status=command_status,
            exit_code=exit_code,
            verifier_execution_status=verifier_execution_status,
            reasoning_status=reasoning_status,
            reward=verifier.reward,
        )
    terminal = _clean_room_terminal(
        acceptance=acceptance,
        verifier_digest=str(task_record["verifier_bundle_digest"]),
        verifier_execution_status=verifier_execution_status,
        details=verifier.details,
    )
    extraction_error: str | None = None
    try:
        extraction = extract_codex_trajectory(
            transcript, task_family=task.domain, terminal_evidence=terminal
        )
        _write_json(run_dir / "extraction.json", extraction.model_dump(mode="json"))
    except Exception as exc:  # preserve raw evidence and continue the frozen batch
        extraction_error = f"{type(exc).__name__}: {' '.join(str(exc).split())[:512]}"
    try:
        telemetry = parse_agent_transcript(transcript)
    except Exception as exc:
        telemetry = {"parse_error": f"{type(exc).__name__}: {str(exc)[:512]}"}
    record = {
        "schema_version": "1",
        "trajectory_id": trajectory_id,
        "task_id": task.task_id,
        "domain": task.domain,
        "coordination_probe": list(task.coordination_probe),
        "repetition": repetition,
        "started_at": started_at,
        "ended_at": _now(),
        "command": {
            "status": _status_value(command_status),
            "exit_code": exit_code,
        },
        "reasoning_run_ids": list(run_ids),
        "reasoning_protocol": reasoning_status,
        "telemetry": telemetry,
        "terminal": {
            "acceptance": acceptance,
            "reason": reason,
            "reward": verifier.reward,
            "details": dict(verifier.details),
            "verifier_error": verifier_error,
            "evidence": terminal.model_dump(mode="json"),
        },
        "extraction_error": extraction_error,
        "surface_digest": surface.get("surface_digest"),
        "task_contract": {
            key: value for key, value in task_record.items() if key != "path"
        },
    }
    _write_json(run_dir / "run.json", record)
    return record


def _artifact_manifest(output: Path) -> dict[str, str]:
    return {
        path.relative_to(output).as_posix(): _digest(path)
        for path in sorted(output.rglob("*"))
        if path.is_file() and path.name != "manifest.json"
    }


def _infrastructure_failure_record(
    *,
    task: StudyTask,
    repetition: int,
    output: Path,
    error: Exception,
) -> dict[str, Any]:
    trajectory_id = f"{task.task_id}-r{repetition:02d}"
    run_dir = output / "runs" / trajectory_id
    run_dir.mkdir(parents=True, exist_ok=True)
    diagnostic = f"{type(error).__name__}: {' '.join(str(error).split())[:1024]}"
    failure = {
        "schema_version": "1",
        "trajectory_id": trajectory_id,
        "classification": "INFRASTRUCTURE_FAILURE",
        "diagnostic": diagnostic,
        "model_rerun_authorized": False,
    }
    _write_json(run_dir / "infrastructure-failure.json", failure)
    record = {
        "schema_version": "1",
        "trajectory_id": trajectory_id,
        "task_id": task.task_id,
        "domain": task.domain,
        "coordination_probe": list(task.coordination_probe),
        "repetition": repetition,
        "started_at": None,
        "ended_at": _now(),
        "command": {"status": "UNKNOWN", "exit_code": None},
        "reasoning_run_ids": [],
        "reasoning_protocol": "INCOMPLETE_INFRASTRUCTURE_FAILURE",
        "telemetry": {},
        "terminal": {
            "acceptance": "INCONCLUSIVE",
            "reason": "INFRASTRUCTURE_FAILURE",
            "reward": None,
            "details": {},
            "evidence": None,
        },
        "extraction_error": diagnostic,
        "surface_digest": None,
        "task_contract": None,
    }
    _write_json(run_dir / "run.json", record)
    return record


def run_study(spec_path: Path, output: Path) -> None:
    spec = load_spec(spec_path)
    if output.exists():
        raise RuntimeError(
            "output directory already exists; frozen runs are never overwritten"
        )
    if not _repository_is_clean():
        raise RuntimeError("source tree must be clean at model-execution start")
    session = _tmux_session()
    source_revision = git_head_sha(_ROOT)
    if _parent_revision() != spec.source_base_revision:
        raise RuntimeError(
            "the preregistration commit is not based directly on the frozen revision"
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
            except Exception as exc:  # never rerun a started frozen rollout
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
        "causal_claim_authorized": False,
        "harbor_execution_claimed": False,
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
        "outcomes": dict(sorted(outcomes.items())),
        "artifacts": _artifact_manifest(output),
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
    run_study(args.spec.resolve(), args.output.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
