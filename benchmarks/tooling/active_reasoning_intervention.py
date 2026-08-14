"""Collect and analyze the frozen internalcot reasoning intervention.

This operator-run study deliberately excludes hidden reasoning, ATIF
``reasoning_content``, tool arguments, and tool results from its derived
dataset. Raw Codex JSONL and task workspaces remain host-local inputs to the
post-run projection.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import shutil
import socket
import sys
import threading
import time
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path
from random import Random
from typing import Any, BinaryIO

from benchmarks.tooling.command_runner import (
    ToolCommandRequest,
    ToolCommandResult,
    ToolCommandStatus,
    git_head_sha,
    git_tracked_worktree_is_clean,
    operator_environment,
    run_operator_command,
    run_tool_command,
)
from benchmarks.tooling.errors import HarborSuiteError
from benchmarks.tooling.harbor_suite import ROOT, get_suite, task_digest
from benchmarks.validation.mathematical_benchmarks_v1._verifier import _run_verifier

_DEFAULT_CONFIG = ROOT / "benchmarks/config/internalcot-trajectory-intervention-v1.json"
_LEXICON = (
    "error",
    "fail",
    "retry",
    "fallback",
    "verify",
    "checked",
    "found",
    "compute",
    "wrote",
    "complete",
    "cannot",
    "timeout",
)
_SUMMARY_MAX_BYTES = 512
_SUMMARY_MAX_MESSAGES = 32
_SUMMARY_TOTAL_MAX_BYTES = 8192
_SERVER_EVENT_MARKER = re.compile(r"\bMCP (?:tool call|operation attempt)\b")
_TOOL_CALL = re.compile(
    r"\bMCP tool call tool=(math\.(?:find|run))\b"
    r".{0,512}?\bstatus=(success|error)\b"
    r".{0,512}?\brequest_digest=([0-9a-f]{16}|none)\b"
    r".{0,512}?\btrace_digest=([0-9a-f]{8}|none)\b"
    r".{0,512}?\btrace_source=([^\s]+)\b"
    r".{0,512}?\bduration_ms=([0-9]+(?:\.[0-9]+)?)\b"
    r".{0,512}?\bresponse_bytes=(-?[0-9]+)\b"
    r".{0,512}?\bargument_digest=(sha256:(?:[0-9a-f]\s*){64})",
    re.DOTALL,
)
_OPERATION_ATTEMPT = re.compile(
    r"\bMCP operation attempt argument_digest=(sha256:(?:[0-9a-f]\s*){64})"
    r".{0,512}?\boperation_id=([^\s]+)\b"
    r".{0,512}?\boperation_version=([^\s]+)\b"
    r".{0,512}?\bprovider=([^\s]+)\b"
    r".{0,512}?\bchecker_ids=([^\s]+)\b"
    r".{0,512}?\bexecution_status=([A-Z_]+)\b"
    r".{0,512}?\bverification_record_uri_present=(True|False)\b"
    r".{0,512}?\bdiagnostic_codes=([^\s]+)\b"
    r".{0,512}?\bartifact_count=([0-9]+)\b",
    re.DOTALL,
)
_REDACTIONS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"(?i)\bBearer\s+[^\s,;]+"), "[REDACTED_BEARER]"),
    (re.compile(r"\bsk-[A-Za-z0-9_-]{8,}\b"), "[REDACTED_API_KEY]"),
    (re.compile(r"/(?:Users|home)/[^/\s]+"), "[REDACTED_HOME]"),
    (
        re.compile(r"/(?:private/)?tmp/[^\s)\]]+|/var/folders/[^\s)\]]+"),
        "[REDACTED_TEMP_PATH]",
    ),
)
_BASE_CONDITION = "x+y+tau_tools"
_VISIBLE_CONDITION_BY_ARM = {
    "control": "x+y+b+tau_tools",
    "internalcot": "x+y+b_star+tau_tools",
}
_TARGETS = (
    "next_tool_action_class",
    "checker_state",
    "recovery_state",
    "mathematical_milestone",
    "tool_failure_state",
)
_HELDOUT_TARGETS = _TARGETS
_TAU_FIELD_GROUPS = (
    "call_structure",
    "operation_identity",
    "outcome",
    "cost",
    "binding",
)


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise HarborSuiteError(f"unable to read JSON {path}: {exc}") from exc


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _sha256_bytes(value: bytes) -> str:
    return "sha256:" + hashlib.sha256(value).hexdigest()


def _sha256(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _json_digest(value: object) -> str:
    return _sha256_bytes(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    )


def _collector_digest() -> str:
    return _sha256(Path(__file__).resolve(strict=True))


def _config(path: Path) -> dict[str, Any]:
    value = _read_json(path)
    if not isinstance(value, dict) or value.get("schema_version") != "1":
        raise HarborSuiteError("intervention config must be a schema-v1 object")
    if value.get("status") != "FROZEN_BEFORE_MODEL_RUNS":
        raise HarborSuiteError("intervention config is not frozen before runs")
    if value.get("production_change_authorized") is not False:
        raise HarborSuiteError("intervention config must remain research-only")
    return value


def _selected_tasks(
    config: Mapping[str, Any], selected: set[str]
) -> list[dict[str, str]]:
    dataset = config.get("dataset")
    tasks = dataset.get("tasks") if isinstance(dataset, Mapping) else None
    if not isinstance(tasks, list):
        raise HarborSuiteError("study config has no task list")
    values: list[dict[str, str]] = []
    seen: set[str] = set()
    for item in tasks:
        if not (
            isinstance(item, dict)
            and isinstance(item.get("id"), str)
            and isinstance(item.get("digest"), str)
        ):
            raise HarborSuiteError("study task records must bind IDs and digests")
        if item["id"] in seen:
            raise HarborSuiteError(f"duplicate study task ID: {item['id']}")
        seen.add(item["id"])
        if not isinstance(item.get("family"), str):
            raise HarborSuiteError("task records must bind a family")
        if not selected or item["id"] in selected:
            values.append(item)
    unknown = selected - {item["id"] for item in values}
    if unknown:
        raise HarborSuiteError(f"unknown selected tasks: {sorted(unknown)}")
    return values


def _validate_task_digest(task: Path, expected: str) -> None:
    actual = "sha256:" + task_digest(task)
    if actual != expected:
        raise HarborSuiteError(
            f"task digest drift for {task.name}: expected {expected}, got {actual}"
        )


def _copy_visible_task(task: Path, workspace: Path) -> None:
    workspace.mkdir(parents=True)
    shutil.copy2(task / "instruction.md", workspace / "instruction.md")
    shutil.copy2(task / "environment/input.json", workspace / "input.json")
    shutil.copy2(
        task / "environment/submission_schema.json",
        workspace / "submission_schema.json",
    )


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _run_server_command(
    request: ToolCommandRequest, results: list[ToolCommandResult]
) -> None:
    results.append(run_tool_command(request))


def _write_stream(stream: BinaryIO, payload: bytes) -> None:
    stream.write(payload)


def _wait_for_port(
    port: int,
    worker: threading.Thread,
    results: Sequence[ToolCommandResult],
) -> None:
    deadline = time.monotonic() + 120
    while time.monotonic() < deadline:
        if not worker.is_alive():
            status = results[0].status if results else "UNKNOWN"
            raise HarborSuiteError(
                f"Jacobian MCP server exited during startup: {status}"
            )
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.2):
                return
        except OSError:
            time.sleep(0.2)
    raise HarborSuiteError("Jacobian MCP server did not become ready")


def _server_command(*, port: int, trial_id: str) -> tuple[str, ...]:
    code = (
        "import logging,sys;"
        "logging.basicConfig(level=logging.INFO,stream=sys.stderr,"
        "format='%(levelname)s %(message)s');"
        "from jacobian.adapters.mcp.remote_cli import main;main()"
    )
    return (
        sys.executable,
        "-c",
        code,
        "--transport",
        "streamable-http",
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--allow-anonymous",
        "--anonymous-tenant-id",
        trial_id,
        "--operation-policy-profile",
        "COMPUTE_VERIFY_NO_RETRIEVAL",
    )


def _codex_arguments(
    *, workspace: Path, model: str, reasoning_effort: str, mcp_url: str, prompt: str
) -> tuple[str, ...]:
    return (
        "-a",
        "never",
        "exec",
        "--ephemeral",
        "--skip-git-repo-check",
        "--ignore-user-config",
        "-C",
        str(workspace),
        "-s",
        "workspace-write",
        "--json",
        "-m",
        model,
        "-c",
        f"model_reasoning_effort={json.dumps(reasoning_effort)}",
        "-c",
        f"mcp_servers.jacobian.url={json.dumps(mcp_url)}",
        "-c",
        'mcp_servers.jacobian.default_tools_approval_mode="approve"',
        "--enable",
        "unified_exec",
        prompt,
    )


def _run_trial(
    *,
    task: Path,
    task_record: Mapping[str, str],
    config: Mapping[str, Any],
    root: Path,
    trial_id: str,
    pair_id: str,
    arm: str,
    repetition: int,
    internalcot_prefix: Path,
) -> dict[str, Any]:
    trial = root / trial_id
    if trial.exists():
        raise HarborSuiteError(f"refusing to overwrite trial: {trial}")
    trial.mkdir(parents=True)
    workspace = trial / "workspace"
    verifier_logs = trial / "verifier"
    verifier_logs.mkdir()
    _copy_visible_task(task, workspace)
    if arm == "internalcot":
        skill_source = (
            internalcot_prefix / "node_modules/internalcot/skills/internalcot/SKILL.md"
        )
        skill_target = workspace / ".agents/skills/internalcot/SKILL.md"
        skill_target.parent.mkdir(parents=True)
        shutil.copy2(skill_source, skill_target)
    port = _free_port()
    server_stdout = trial / "server.stdout"
    server_log = trial / "server.log"
    server_environment = dict(operator_environment(include=("PATH",)))
    server_environment.update(
        {
            "NO_PROXY": "127.0.0.1,localhost",
            "no_proxy": "127.0.0.1,localhost",
        }
    )
    started = time.monotonic()
    with server_stdout.open("wb") as stdout_handle, server_log.open("wb") as log_handle:
        server_command = _server_command(port=port, trial_id=trial_id)
        server_cancel = threading.Event()
        server_results: list[ToolCommandResult] = []
        server_request = ToolCommandRequest(
            executable=server_command[0],
            arguments=server_command[1:],
            environment=server_environment,
            cwd=str(ROOT),
            timeout_seconds=float(config["runtime"]["task_timeout_seconds"]) + 180,
            stdout_limit_bytes=8 * 1024 * 1024,
            stderr_limit_bytes=32 * 1024 * 1024,
            cancellation_event=server_cancel,
            stdout_sink=lambda payload: _write_stream(stdout_handle, payload),
            stderr_sink=lambda payload: _write_stream(log_handle, payload),
        )
        server_worker = threading.Thread(
            target=_run_server_command,
            args=(server_request, server_results),
            daemon=True,
        )
        server_worker.start()
        try:
            _wait_for_port(port, server_worker, server_results)
            runtime = config["runtime"]
            environment = dict(
                operator_environment(
                    include=(
                        "HOME",
                        "PATH",
                        "CODEX_HOME",
                        "HTTP_PROXY",
                        "HTTPS_PROXY",
                        "ALL_PROXY",
                        "http_proxy",
                        "https_proxy",
                        "all_proxy",
                    )
                )
            )
            environment.update(
                {
                    "NO_PROXY": "127.0.0.1,localhost",
                    "no_proxy": "127.0.0.1,localhost",
                }
            )
            environment["PATH"] = (
                f"{internalcot_prefix / 'node_modules/.bin'}:"
                f"{environment.get('PATH', '')}"
            )
            prompt = str(config["common_prompt"])
            if arm == "internalcot":
                prompt = "$internalcot\n\n" + prompt
            result = run_operator_command(
                "codex",
                _codex_arguments(
                    workspace=workspace,
                    model=runtime["model"],
                    reasoning_effort=runtime["reasoning_effort"],
                    mcp_url=f"http://127.0.0.1:{port}/mcp",
                    prompt=prompt,
                ),
                cwd=workspace,
                timeout_seconds=float(runtime["task_timeout_seconds"]),
                stdout_limit_bytes=32 * 1024 * 1024,
                stderr_limit_bytes=4 * 1024 * 1024,
                environment=environment,
            )
        finally:
            server_cancel.set()
            server_worker.join(timeout=30)
            if server_worker.is_alive():
                raise HarborSuiteError(
                    "Jacobian MCP server did not stop within the tooling deadline"
                )
            if not server_results:
                raise HarborSuiteError("Jacobian MCP server produced no command result")
            server_result = server_results[0]
            if server_result.status not in {
                ToolCommandStatus.CANCELLED,
                ToolCommandStatus.EXITED,
            }:
                raise HarborSuiteError(
                    f"Jacobian MCP server ended with {server_result.status}"
                )
            if server_result.stdout_exceeded or server_result.stderr_exceeded:
                raise HarborSuiteError("Jacobian MCP server output exceeded its bound")
    transcript = trial / "codex.jsonl"
    stderr = trial / "codex.stderr"
    transcript.write_bytes(result.stdout)
    stderr.write_bytes(result.stderr)
    verifier = _run_verifier(task, workspace, verifier_logs)
    final_message = _final_agent_message(result.stdout)
    artifacts = _workspace_artifacts(workspace)
    record = {
        "schema_version": config["schema_version"],
        "trial_id": trial_id,
        "pair_id": pair_id,
        "arm": arm,
        "task_id": task.name,
        "family": task_record.get("family", task.name),
        "repetition": repetition,
        "task_digest": task_record["digest"],
        "source_sha": git_head_sha(ROOT),
        "command": {
            "status": result.status,
            "exit_code": result.exit_code,
            "diagnostic": result.diagnostic,
            "stdout_exceeded": result.stdout_exceeded,
            "stderr_exceeded": result.stderr_exceeded,
            "elapsed_seconds": round(time.monotonic() - started, 6),
        },
        "verifier": {"reward": verifier.reward, "details": dict(verifier.details)},
        "final_message_sha256": _sha256_bytes(final_message.encode("utf-8")),
        "final_message_utf8_bytes": len(final_message.encode("utf-8")),
        "workspace_artifacts": artifacts,
        "raw_artifacts": {
            "transcript": {"path": transcript.name, "sha256": _sha256(transcript)},
            "stderr": {"path": stderr.name, "sha256": _sha256(stderr)},
            "server_log": {"path": server_log.name, "sha256": _sha256(server_log)},
        },
    }
    _write_json(trial / "trial.json", record)
    return record


def _workspace_artifacts(workspace: Path) -> list[dict[str, object]]:
    values = []
    for relative in ("submission.json",):
        path = workspace / relative
        if path.is_file() and not path.is_symlink():
            values.append(
                {
                    "path": relative,
                    "sha256": _sha256(path),
                    "bytes": path.stat().st_size,
                }
            )
    evidence = workspace / "evidence"
    if evidence.is_dir() and not evidence.is_symlink():
        for path in sorted(evidence.rglob("*")):
            if path.is_file() and not path.is_symlink():
                values.append(
                    {
                        "path": path.relative_to(workspace).as_posix(),
                        "sha256": _sha256(path),
                        "bytes": path.stat().st_size,
                    }
                )
    return values


def _jsonl_events(payload: bytes) -> list[dict[str, Any]]:
    values = []
    for line in payload.decode("utf-8", errors="strict").splitlines():
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            values.append(value)
    return values


def _final_agent_message(payload: bytes) -> str:
    messages = []
    for event in _jsonl_events(payload):
        item = event.get("item")
        if (
            event.get("type") == "item.completed"
            and isinstance(item, dict)
            and item.get("type") == "agent_message"
            and isinstance(item.get("text"), str)
        ):
            messages.append(item["text"])
    return messages[-1] if messages else ""


def _bounded_message(text: str) -> tuple[str, int, bool, int]:
    value = text.strip()
    redactions = 0
    for pattern, replacement in _REDACTIONS:
        value, count = pattern.subn(replacement, value)
        redactions += count
    encoded = value.encode("utf-8")
    if len(encoded) <= _SUMMARY_MAX_BYTES:
        return value, len(encoded), False, redactions
    return (
        encoded[:_SUMMARY_MAX_BYTES].decode("utf-8", errors="ignore"),
        len(encoded),
        True,
        redactions,
    )


def _safe_int(value: object) -> int:
    return value if type(value) is int else 0


def _safe_float(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return 0.0
    result = float(value)
    return result if math.isfinite(result) else 0.0


def _safe_length(value: object) -> int:
    return len(value) if isinstance(value, list | tuple) else 0


def _visible_messages(payload: bytes) -> tuple[list[dict[str, object]], list[int]]:
    raw: list[tuple[int, str]] = []
    tool_message_counts: list[int] = []
    for position, event in enumerate(_jsonl_events(payload), start=1):
        item = event.get("item")
        if not (event.get("type") == "item.completed" and isinstance(item, dict)):
            continue
        if item.get("type") == "agent_message" and isinstance(item.get("text"), str):
            raw.append((position, item["text"]))
        elif item.get("type") == "mcp_tool_call" and item.get("tool") in {
            "math.find",
            "math.run",
        }:
            tool_message_counts.append(len(raw))
    if raw:
        raw = raw[:-1]
    summaries: list[dict[str, object]] = []
    retained_bytes = 0
    for position, text in raw[:_SUMMARY_MAX_MESSAGES]:
        bounded, original_bytes, truncated, redactions = _bounded_message(text)
        remaining = _SUMMARY_TOTAL_MAX_BYTES - retained_bytes
        if remaining <= 0:
            break
        encoded = bounded.encode("utf-8")
        if len(encoded) > remaining:
            bounded = encoded[:remaining].decode("utf-8", errors="ignore")
            truncated = True
        retained_bytes += len(bounded.encode("utf-8"))
        summaries.append(
            {
                "source_position": position,
                "text": bounded,
                "original_utf8_bytes": original_bytes,
                "truncated": truncated,
                "redaction_count": redactions,
            }
        )
    return summaries, tool_message_counts


def _command_trace(
    item: Mapping[str, object],
    *,
    position: int,
    expected_workflow_sha256: str,
) -> tuple[str, dict[str, object] | None, int]:
    command = item.get("command")
    output = item.get("aggregated_output")
    if not isinstance(command, str) or not isinstance(output, str):
        return "substantive", None, 0
    if "internalcot skill" in command:
        if _sha256_bytes(output.encode("utf-8")) == expected_workflow_sha256:
            return "skill", None, len(output.encode("utf-8"))
        return "ignored", None, 0
    if "internalcot note " not in command or not output.startswith("internalcot>"):
        return ("ignored" if "internalcot note " in command else "substantive"), None, 0
    bounded, original_bytes, truncated, redactions = _bounded_message(
        output.removeprefix("internalcot>").lstrip()
    )
    return (
        "note",
        {
            "source_position": position,
            "text": bounded,
            "original_utf8_bytes": original_bytes,
            "truncated": truncated,
            "redaction_count": redactions,
        },
        0,
    )


def _intervention_trace(
    payload: bytes, *, expected_workflow_sha256: str
) -> tuple[list[dict[str, object]], dict[str, object], dict[str, float]]:
    """Project note text only long enough to derive bounded features.

    The returned note summaries are never included in a committed projection.
    Adherence depends on successful completed command output and event order, not
    on model claims about following the intervention.
    """

    notes: list[dict[str, object]] = []
    skill_positions: list[int] = []
    skill_visible_bytes = 0
    note_positions: list[int] = []
    substantive_positions: list[int] = []
    final_agent_position = 0
    host_command_count = 0
    usage: dict[str, float] = {}
    for position, event in enumerate(_jsonl_events(payload), start=1):
        if event.get("type") == "turn.completed" and isinstance(
            event.get("usage"), Mapping
        ):
            usage = {
                name: float(_safe_int(event["usage"].get(name)))
                for name in (
                    "input_tokens",
                    "cached_input_tokens",
                    "output_tokens",
                    "reasoning_output_tokens",
                )
            }
        item = event.get("item")
        if not (event.get("type") == "item.completed" and isinstance(item, dict)):
            continue
        item_type = item.get("type")
        if item_type == "agent_message":
            final_agent_position = position
            continue
        if item_type == "mcp_tool_call":
            substantive_positions.append(position)
            continue
        if item_type != "command_execution":
            continue
        host_command_count += 1
        kind, note, visible_bytes = _command_trace(
            item,
            position=position,
            expected_workflow_sha256=expected_workflow_sha256,
        )
        if kind == "skill":
            skill_positions.append(position)
            skill_visible_bytes += visible_bytes
        elif kind == "note" and note is not None:
            notes.append(note)
            note_positions.append(position)
        elif kind == "substantive":
            substantive_positions.append(position)
    first_substantive = min(substantive_positions, default=final_agent_position)
    last_substantive = max(substantive_positions, default=0)
    adherence = {
        "official_skill_loaded": bool(skill_positions),
        "successful_note_count": len(note_positions),
        "first_note_before_substantive": bool(note_positions)
        and note_positions[0] < first_substantive,
        "final_note_after_substantive_before_final": bool(note_positions)
        and note_positions[-1] > last_substantive
        and note_positions[-1] < final_agent_position,
    }
    adherence["adherent"] = all(
        (
            adherence["official_skill_loaded"],
            _safe_int(adherence["successful_note_count"]) >= 2,
            adherence["first_note_before_substantive"],
            adherence["final_note_after_substantive_before_final"],
        )
    )
    behavior = {
        **usage,
        "host_command_count": float(host_command_count),
        "internalcot_command_count": float(len(skill_positions) + len(note_positions)),
        "internalcot_skill_bytes": float(skill_visible_bytes),
    }
    return notes, adherence, behavior


def _note_prefix_counts(payload: bytes) -> list[int]:
    count = 0
    values: list[int] = []
    for event in _jsonl_events(payload):
        item = event.get("item")
        if not (event.get("type") == "item.completed" and isinstance(item, dict)):
            continue
        if item.get("type") == "command_execution" and isinstance(
            item.get("command"), str
        ):
            if "internalcot note " in item["command"] and str(
                item.get("aggregated_output", "")
            ).startswith("internalcot>"):
                count += 1
        elif item.get("type") == "mcp_tool_call" and item.get("tool") in {
            "math.find",
            "math.run",
        }:
            values.append(count)
    return values


def _validate_internalcot(prefix: Path, config: Mapping[str, Any]) -> None:
    package = _read_json(prefix / "node_modules/internalcot/package.json")
    lock = _read_json(prefix / "package-lock.json")
    pinned = config["internalcot"]
    lock_package = lock.get("packages", {}).get("node_modules/internalcot", {})
    environment = dict(operator_environment(include=("PATH",)))
    environment["PATH"] = (
        f"{prefix / 'node_modules/.bin'}:{environment.get('PATH', '')}"
    )
    workflow = run_tool_command(
        ToolCommandRequest(
            executable=str(prefix / "node_modules/.bin/internalcot"),
            arguments=("skill",),
            environment=environment,
            cwd=str(prefix),
            timeout_seconds=30,
            stdout_limit_bytes=64 * 1024,
            stderr_limit_bytes=16 * 1024,
        )
    )
    if workflow.status is not ToolCommandStatus.EXITED or workflow.exit_code != 0:
        raise HarborSuiteError("pinned internalcot skill command failed")
    bindings = {
        "version": package.get("version"),
        "lock_version": lock_package.get("version"),
        "npm_integrity": lock_package.get("integrity"),
        "skill_sha256": _sha256(
            prefix / "node_modules/internalcot/skills/internalcot/SKILL.md"
        ),
        "workflow_sha256": _sha256_bytes(workflow.stdout),
    }
    expected = {
        "version": pinned["version"],
        "lock_version": pinned["version"],
        "npm_integrity": pinned["npm_integrity"],
        "skill_sha256": pinned["skill_sha256"],
        "workflow_sha256": pinned["workflow_sha256"],
    }
    if bindings != expected:
        raise HarborSuiteError(
            f"internalcot runtime binding mismatch: {bindings!r} != {expected!r}"
        )


def _server_events(payload: str) -> tuple[list[dict[str, object]], dict[str, int]]:
    events: list[tuple[int, dict[str, object]]] = []
    for match in _TOOL_CALL.finditer(payload):
        (
            tool,
            status,
            request_digest,
            trace_digest,
            trace_source,
            duration_ms,
            response_bytes,
            argument_digest,
        ) = match.groups()
        events.append(
            (
                match.start(),
                {
                    "kind": "TOOL_CALL",
                    "tool": tool,
                    "status": status,
                    "request_digest": request_digest,
                    "trace_digest": trace_digest,
                    "trace_source": trace_source,
                    "duration_ms": float(duration_ms),
                    "response_bytes": int(response_bytes),
                    "argument_digest": re.sub(r"\s", "", argument_digest),
                },
            )
        )
    for match in _OPERATION_ATTEMPT.finditer(payload):
        (
            argument_digest,
            operation_id,
            operation_version,
            provider,
            checker_ids,
            execution_status,
            verification_record_uri_present,
            diagnostic_codes,
            artifact_count,
        ) = match.groups()
        events.append(
            (
                match.start(),
                {
                    "kind": "OPERATION_ATTEMPT",
                    "argument_digest": re.sub(r"\s", "", argument_digest),
                    "operation_id": operation_id,
                    "operation_version": operation_version,
                    "provider": provider,
                    "checker_ids": (
                        [] if checker_ids == "none" else checker_ids.split(",")
                    ),
                    "execution_status": execution_status,
                    "assurance": None,
                    "verification_record_uri_present": (
                        verification_record_uri_present == "True"
                    ),
                    "diagnostic_codes": (
                        []
                        if diagnostic_codes in {"none", "-"}
                        else diagnostic_codes.split(",")[:8]
                    ),
                    "artifact_count": int(artifact_count),
                },
            )
        )
    ordered = [event for _, event in sorted(events, key=lambda item: item[0])]
    candidates = len(_SERVER_EVENT_MARKER.findall(payload))
    return ordered, {"candidates": candidates, "recorded": len(ordered)}


def _task_features(task: Path) -> dict[str, float]:
    task_toml = (task / "task.toml").read_text(encoding="utf-8")
    instruction = (task / "instruction.md").read_text(encoding="utf-8")
    input_bytes = (task / "environment/input.json").read_bytes()
    schema = _read_json(task / "environment/submission_schema.json")
    properties = schema.get("properties", {}) if isinstance(schema, dict) else {}
    return {
        "x:instruction_bytes": float(len(instruction.encode("utf-8"))),
        "x:instruction_lines": float(len(instruction.splitlines())),
        "x:input_bytes": float(len(input_bytes)),
        "x:schema_properties": float(
            len(properties) if isinstance(properties, dict) else 0
        ),
        "x:requires_verification_record": float(
            "verification_record_uri" in properties
        ),
        "x:difficulty_hard": float('difficulty = "hard"' in task_toml),
        "x:difficulty_medium": float('difficulty = "medium"' in task_toml),
    }


def _y_features(
    *, record: Mapping[str, Any], final_message: str, target: str
) -> dict[str, float]:
    verifier = record.get("verifier", {})
    details = verifier.get("details", {}) if isinstance(verifier, Mapping) else {}
    features = {
        "y:final_message_bytes": float(len(final_message.encode("utf-8"))),
        "y:final_message_lines": float(len(final_message.splitlines())),
        "y:workspace_artifact_count": float(len(record.get("workspace_artifacts", []))),
    }
    for key, value in sorted(details.items() if isinstance(details, Mapping) else ()):
        if target == "mathematical_milestone" and key == "correctness":
            continue
        if target == "terminal_verifier_success":
            continue
        if isinstance(value, bool) or (
            isinstance(value, int | float) and math.isfinite(float(value))
        ):
            features[f"y:verifier:{key}"] = float(value)
    if target != "terminal_verifier_success" and isinstance(verifier, Mapping):
        reward = verifier.get("reward")
        if isinstance(reward, int | float) and not isinstance(reward, bool):
            features["y:verifier_reward"] = float(reward)
    return features


def _b_features(
    messages: Sequence[Mapping[str, object]], *, prefix: str = "b"
) -> dict[str, float]:
    text = "\n".join(str(item["text"]) for item in messages).lower()
    features = {
        f"{prefix}:message_count": float(len(messages)),
        f"{prefix}:utf8_bytes": float(len(text.encode("utf-8"))),
        f"{prefix}:truncated_count": float(
            sum(bool(item["truncated"]) for item in messages)
        ),
        f"{prefix}:redaction_count": float(
            sum(_safe_int(item["redaction_count"]) for item in messages)
        ),
    }
    for token in _LEXICON:
        features[f"{prefix}:lex:{token}"] = float(
            len(re.findall(rf"\b{re.escape(token)}\w*\b", text))
        )
    return features


def _tool_action(
    event: Mapping[str, object], attempts: Mapping[str, Mapping[str, object]]
) -> str:
    if event.get("tool") == "math.find":
        return "FIND"
    attempt = attempts.get(str(event.get("argument_digest")))
    operation_id = attempt.get("operation_id") if isinstance(attempt, Mapping) else None
    return (
        "RUN_CHECKER"
        if isinstance(operation_id, str) and operation_id.endswith(".verify")
        else "RUN_PRODUCER"
    )


def _tau_features(events: Sequence[Mapping[str, object]]) -> dict[str, float]:
    attempts = {
        str(event["argument_digest"]): event
        for event in events
        if event.get("kind") == "OPERATION_ATTEMPT"
    }
    tools = [event for event in events if event.get("kind") == "TOOL_CALL"]
    actions = [_tool_action(event, attempts) for event in tools]
    features: dict[str, float] = {
        "tau:event_count": float(len(events)),
        "tau:tool_call_count": float(len(tools)),
        "tau:attempt_count": float(len(attempts)),
        "tau:error_count": float(
            sum(event.get("status") == "error" for event in tools)
        ),
        "tau:diagnostic_count": float(
            sum(
                _safe_length(event.get("diagnostic_codes"))
                for event in attempts.values()
            )
        ),
        "tau:duration_log1p_sum": sum(
            math.log1p(
                _safe_float(
                    event.get("duration_ms", event.get("attempt_duration_ms", 0.0))
                )
            )
            for event in events
        ),
        "tau:response_bytes_log1p_sum": sum(
            math.log1p(max(0, _safe_int(event.get("response_bytes"))))
            for event in events
        ),
        "tau:request_digest_available": float(
            sum(event.get("request_digest") != "none" for event in tools)
        ),
        "tau:argument_digest_available": float(
            sum(
                str(event.get("argument_digest", "")).startswith("sha256:")
                for event in events
            )
        ),
    }
    for action, count in Counter(actions).items():
        features[f"tau:action:{action}"] = float(count)
    for left, right in pairwise(actions):
        features[f"tau:bigram:{left}>{right}"] = (
            features.get(f"tau:bigram:{left}>{right}", 0.0) + 1.0
        )
    operation_ids = [str(event["operation_id"]) for event in attempts.values()]
    features["tau:unique_operation_count"] = float(len(set(operation_ids)))
    features["tau:unique_domain_count"] = float(
        len({value.split(".", 1)[0] for value in operation_ids})
    )
    features["tau:checker_attempt_count"] = float(
        sum(value.endswith(".verify") for value in operation_ids)
    )
    for operation_id, count in Counter(operation_ids).items():
        features[f"tau:operation:{operation_id}"] = float(count)
    for domain, count in Counter(
        value.split(".", 1)[0] for value in operation_ids
    ).items():
        features[f"tau:domain:{domain}"] = float(count)
    for event in attempts.values():
        features[f"tau:execution:{event.get('execution_status')}"] = (
            features.get(f"tau:execution:{event.get('execution_status')}", 0.0) + 1.0
        )
        assurance = event.get("assurance")
        if isinstance(assurance, str):
            features[f"tau:assurance:{assurance}"] = (
                features.get(f"tau:assurance:{assurance}", 0.0) + 1.0
            )
        evidence_present = event.get("verification_record_uri_present")
        if isinstance(evidence_present, bool):
            evidence_label = "present" if evidence_present else "absent"
            features[f"tau:evidence:{evidence_label}"] = (
                features.get(f"tau:evidence:{evidence_label}", 0.0) + 1.0
            )
    return features


def _checker_attempt_accepted(event: Mapping[str, object]) -> bool:
    return (
        event.get("execution_status") == "COMPLETED"
        and not event.get("diagnostic_codes")
        and (
            event.get("assurance") == "VERIFIED"
            or event.get("verification_record_uri_present") is True
        )
    )


def _checker_label(events: Sequence[Mapping[str, object]]) -> str:
    checkers = [
        event
        for event in events
        if event.get("kind") == "OPERATION_ATTEMPT"
        and str(event.get("operation_id", "")).endswith(".verify")
    ]
    if not checkers:
        return "NO_CHECKER"
    rejected = [event for event in checkers if not _checker_attempt_accepted(event)]
    if not rejected:
        return "SUCCESS_WITHOUT_REJECTION"
    last_rejection = max(events.index(event) for event in rejected)
    recovered = any(
        index > last_rejection
        and event.get("kind") == "OPERATION_ATTEMPT"
        and str(event.get("operation_id", "")).endswith(".verify")
        and _checker_attempt_accepted(event)
        for index, event in enumerate(events)
    )
    return "REJECTED_RECOVERED" if recovered else "REJECTED_UNRECOVERED"


def _checker_state(events: Sequence[Mapping[str, object]]) -> str:
    label = _checker_label(events)
    if label == "NO_CHECKER":
        return "NO_CHECKER"
    if label == "SUCCESS_WITHOUT_REJECTION":
        return "ACCEPTED_ONLY"
    return "REJECTED"


def _failure_label(events: Sequence[Mapping[str, object]]) -> str:
    tools = [event for event in events if event.get("kind") == "TOOL_CALL"]
    runs = [event for event in tools if event.get("tool") == "math.run"]
    if not tools:
        return "NO_SERVER_TOOL_USE"
    if not runs:
        return "DISCOVERY_ONLY"
    failures = [event for event in runs if event.get("status") != "success"]
    attempts = [event for event in events if event.get("kind") == "OPERATION_ATTEMPT"]
    failures.extend(
        event for event in attempts if event.get("execution_status") != "COMPLETED"
    )
    if not failures:
        return "CLEAN_EXECUTION"
    last_failure = max(events.index(event) for event in failures)
    recovered = any(
        index > last_failure
        and event.get("kind") == "TOOL_CALL"
        and event.get("tool") == "math.run"
        and event.get("status") == "success"
        for index, event in enumerate(events)
    )
    return "RECOVERED_AFTER_FAILURE" if recovered else "UNRECOVERED_FAILURE"


def _recovery_state(events: Sequence[Mapping[str, object]]) -> str:
    checker = _checker_label(events)
    failure = _failure_label(events)
    if checker == "REJECTED_RECOVERED" or failure == "RECOVERED_AFTER_FAILURE":
        return "RECOVERED"
    if checker == "REJECTED_UNRECOVERED" or failure == "UNRECOVERED_FAILURE":
        return "UNRECOVERED"
    return "NO_RECOVERY_OPPORTUNITY"


@dataclass(frozen=True)
class Row:
    task_id: str
    label: str
    x_y: Mapping[str, float]
    b: Mapping[str, float]
    tau: Mapping[str, float]
    family_id: str = ""
    arm: str = "control"
    pair_id: str = ""


def _project_trial(
    task: Path, trial: Path, config: Mapping[str, Any]
) -> tuple[dict[str, Any], dict[str, list[Row]]]:
    record = _read_json(trial / "trial.json")
    task_id = record.get("task_id", task.name)
    family_id = record.get("family", task_id)
    if not isinstance(task_id, str) or not isinstance(family_id, str):
        raise HarborSuiteError(f"invalid task/family identity: {trial.name}")
    transcript_path = trial / record["raw_artifacts"]["transcript"]["path"]
    server_path = trial / record["raw_artifacts"]["server_log"]["path"]
    if _sha256(transcript_path) != record["raw_artifacts"]["transcript"]["sha256"]:
        raise HarborSuiteError(f"transcript digest mismatch: {trial.name}")
    if _sha256(server_path) != record["raw_artifacts"]["server_log"]["sha256"]:
        raise HarborSuiteError(f"server-log digest mismatch: {trial.name}")
    transcript = transcript_path.read_bytes()
    final_message = _final_agent_message(transcript)
    messages, message_prefix_counts = _visible_messages(transcript)
    arm = record.get("arm")
    pair_id = record.get("pair_id")
    if arm not in {"control", "internalcot"} or not isinstance(pair_id, str):
        raise HarborSuiteError(f"invalid arm/pair binding: {trial.name}")
    notes, adherence, behavior = _intervention_trace(
        transcript,
        expected_workflow_sha256=config["internalcot"]["workflow_sha256"],
    )
    note_prefix_counts = _note_prefix_counts(transcript)
    if arm == "control" and (
        adherence["official_skill_loaded"]
        or _safe_int(adherence["successful_note_count"])
    ):
        raise HarborSuiteError(f"control contamination detected: {trial.name}")
    events, coverage = _server_events(server_path.read_text(encoding="utf-8"))
    if coverage["candidates"] != coverage["recorded"]:
        raise HarborSuiteError(f"incomplete server event projection: {trial.name}")
    base_x = _task_features(task)
    rows: dict[str, list[Row]] = defaultdict(list)
    trajectory_labels = {
        "checker_rejection_recovery": _checker_label(events),
        "checker_state": _checker_state(events),
        "recovery_state": _recovery_state(events),
        "mathematical_milestone": (
            "REACHED"
            if record["verifier"]["details"].get("correctness") == 1.0
            else "NOT_REACHED"
        ),
        "terminal_verifier_success": (
            "PASS" if record["verifier"]["reward"] == 1.0 else "FAIL"
        ),
        "tool_failure_state": {
            "NO_SERVER_TOOL_USE": "NO_TOOL",
            "DISCOVERY_ONLY": "DISCOVERY_ONLY",
            "CLEAN_EXECUTION": "CLEAN",
            "RECOVERED_AFTER_FAILURE": "RECOVERED",
            "UNRECOVERED_FAILURE": "UNRECOVERED",
        }[_failure_label(events)],
    }
    for target, label in trajectory_labels.items():
        rows[target].append(
            Row(
                task_id=task_id,
                label=label,
                x_y={
                    **base_x,
                    **_y_features(
                        record=record, final_message=final_message, target=target
                    ),
                },
                b=_b_features(
                    notes if arm == "internalcot" else messages,
                    prefix="b_star" if arm == "internalcot" else "b",
                ),
                tau=_tau_features(events),
                family_id=family_id,
                arm=arm,
                pair_id=pair_id,
            )
        )
    attempts = {
        str(event["argument_digest"]): event
        for event in events
        if event.get("kind") == "OPERATION_ATTEMPT"
    }
    tool_events = [event for event in events if event.get("kind") == "TOOL_CALL"]
    for index in range(len(tool_events) + 1):
        label = (
            "TERMINAL"
            if index == len(tool_events)
            else _tool_action(tool_events[index], attempts)
        )
        message_count = (
            message_prefix_counts[index]
            if index < len(message_prefix_counts)
            else len(messages)
        )
        prior_argument_digests = {
            str(event.get("argument_digest")) for event in tool_events[:index]
        }
        prefix_events = [
            event
            for event in events
            if (event.get("kind") == "TOOL_CALL" and event in tool_events[:index])
            or (
                event.get("kind") == "OPERATION_ATTEMPT"
                and str(event.get("argument_digest")) in prior_argument_digests
            )
        ]
        rows["next_tool_action_class"].append(
            Row(
                task_id=task_id,
                label=label,
                x_y={
                    **base_x,
                    **_y_features(
                        record=record,
                        final_message=final_message,
                        target="next_tool_action_class",
                    ),
                    "xy:prefix_index": float(index),
                },
                b=_b_features(
                    notes[
                        : (
                            note_prefix_counts[index]
                            if index < len(note_prefix_counts)
                            else len(notes)
                        )
                    ]
                    if arm == "internalcot"
                    else messages[:message_count],
                    prefix="b_star" if arm == "internalcot" else "b",
                ),
                tau=_tau_features(prefix_events),
                family_id=family_id,
                arm=arm,
                pair_id=pair_id,
            )
        )
    projection = {
        "trial_id": trial.name,
        "pair_id": pair_id,
        "arm": arm,
        "task_id": task_id,
        "family": family_id,
        "repetition": record.get("repetition", 1),
        "status": "COMPLETE",
        "command_status": record.get("command", {}).get("status", "UNKNOWN"),
        "server_event_coverage": coverage,
        "summary_metrics": {
            "ordinary_message_count": len(messages),
            "ordinary_visible_bytes": sum(
                len(str(item["text"]).encode("utf-8")) for item in messages
            ),
            "internalcot_note_count": len(notes),
            "internalcot_visible_bytes": sum(
                _safe_int(item["original_utf8_bytes"]) for item in notes
            )
            + _safe_int(behavior["internalcot_skill_bytes"]),
            "truncated_count": sum(
                bool(item["truncated"]) for item in [*messages, *notes]
            ),
            "redaction_count": sum(
                _safe_int(item["redaction_count"]) for item in messages
            )
            + sum(_safe_int(item["redaction_count"]) for item in notes),
        },
        "tool_metrics": {
            "event_count": len(events),
            "tool_call_count": sum(
                event.get("kind") == "TOOL_CALL" for event in events
            ),
            "operation_attempt_count": sum(
                event.get("kind") == "OPERATION_ATTEMPT" for event in events
            ),
            "successful_producer_checker_chain": _successful_producer_checker_chain(
                events
            ),
        },
        "intervention_adherence": adherence,
        "behavior_metrics": {
            "mathematical_correctness": float(
                record["verifier"]["details"].get("correctness") == 1.0
            ),
            "terminal_verifier_success": float(record["verifier"]["reward"] == 1.0),
            "tool_adoption": float(
                any(
                    event.get("kind") == "TOOL_CALL" and event.get("tool") == "math.run"
                    for event in events
                )
            ),
            "tool_error_count": float(
                sum(
                    event.get("kind") == "TOOL_CALL" and event.get("status") == "error"
                    for event in events
                )
            ),
            "retry_count": float(
                len(
                    [
                        event
                        for event in events
                        if event.get("kind") == "TOOL_CALL"
                        and event.get("tool") == "math.run"
                    ]
                )
                - len(
                    {
                        event.get("argument_digest")
                        for event in events
                        if event.get("kind") == "TOOL_CALL"
                        and event.get("tool") == "math.run"
                    }
                )
            ),
            "checker_use": float(_checker_state(events) != "NO_CHECKER"),
            "jacobian_call_count": float(
                sum(event.get("kind") == "TOOL_CALL" for event in events)
            ),
            **behavior,
            "ordinary_visible_bytes": float(
                sum(_safe_int(item["original_utf8_bytes"]) for item in messages)
            ),
            "internalcot_visible_bytes": float(
                sum(_safe_int(item["original_utf8_bytes"]) for item in notes)
            )
            + behavior["internalcot_skill_bytes"],
            "total_visible_bytes": float(
                sum(
                    _safe_int(item["original_utf8_bytes"])
                    for item in [*messages, *notes]
                )
            )
            + behavior["internalcot_skill_bytes"],
            "elapsed_seconds": float(record["command"]["elapsed_seconds"]),
        },
        "labels": trajectory_labels,
    }
    return projection, rows


def _tau_field_group(name: str) -> str:
    if name.startswith(("tau:action:", "tau:bigram:")) or name in {
        "tau:event_count",
        "tau:tool_call_count",
        "tau:attempt_count",
    }:
        return "call_structure"
    if name.startswith(("tau:operation:", "tau:domain:")) or name in {
        "tau:unique_operation_count",
        "tau:unique_domain_count",
        "tau:checker_attempt_count",
    }:
        return "operation_identity"
    if name.startswith(
        ("tau:execution:", "tau:assurance:", "tau:evidence:")
    ) or name in {
        "tau:error_count",
        "tau:diagnostic_count",
    }:
        return "outcome"
    if name in {"tau:duration_log1p_sum", "tau:response_bytes_log1p_sum"}:
        return "cost"
    if name in {"tau:request_digest_available", "tau:argument_digest_available"}:
        return "binding"
    raise HarborSuiteError(f"unclassified tau_tools feature: {name}")


def _condition_features(
    row: Row,
    condition: str,
    *,
    tau_groups: frozenset[str] | None = None,
) -> dict[str, float]:
    values = dict(row.x_y)
    if "+b" in condition:
        values.update(row.b)
    if "tau_tools" in condition:
        values.update(
            {
                name: value
                for name, value in row.tau.items()
                if tau_groups is None or _tau_field_group(name) in tau_groups
            }
        )
    return values


def _distance(
    left: Mapping[str, float],
    right: Mapping[str, float],
    ranges: Mapping[str, tuple[float, float]],
) -> float:
    total = 0.0
    for name, (minimum, maximum) in ranges.items():
        scale = maximum - minimum
        if scale <= 0:
            continue
        delta = (left.get(name, 0.0) - right.get(name, 0.0)) / scale
        total += delta * delta
    return math.sqrt(total)


def _predict(
    train: Sequence[Row],
    test: Row,
    condition: str,
    *,
    tau_groups: frozenset[str] | None = None,
) -> str:
    if not train:
        raise HarborSuiteError("empty training fold")
    vectors = [
        _condition_features(row, condition, tau_groups=tau_groups) for row in train
    ]
    names = sorted({name for vector in vectors for name in vector})
    ranges = {
        name: (
            min(vector.get(name, 0.0) for vector in vectors),
            max(vector.get(name, 0.0) for vector in vectors),
        )
        for name in names
    }
    test_vector = _condition_features(test, condition, tau_groups=tau_groups)
    neighbors = sorted(
        (
            (_distance(vector, test_vector, ranges), row.task_id, row.label)
            for row, vector in zip(train, vectors, strict=True)
        ),
        key=lambda item: (item[0], item[1], item[2]),
    )[: min(3, len(train))]
    counts = Counter(label for _, _, label in neighbors)
    return sorted(counts, key=lambda label: (-counts[label], label))[0]


def _family_predictions(
    rows: Sequence[Row],
    condition: str,
    *,
    tau_groups: frozenset[str] | None = None,
) -> list[dict[str, str]]:
    predictions = []
    families = sorted({row.family_id for row in rows})
    if "" in families or len(families) < 2:
        raise HarborSuiteError("held-out analysis requires at least two named families")
    for held_out in families:
        train = [row for row in rows if row.family_id != held_out]
        for row in rows:
            if row.family_id == held_out:
                predictions.append(
                    {
                        "task_id": row.task_id,
                        "family": row.family_id,
                        "truth": row.label,
                        "prediction": _predict(
                            train,
                            row,
                            condition,
                            tau_groups=tau_groups,
                        ),
                    }
                )
    return predictions


def _family_majority_predictions(rows: Sequence[Row]) -> list[dict[str, str]]:
    values: list[dict[str, str]] = []
    for held_out in sorted({row.family_id for row in rows}):
        counts = Counter(row.label for row in rows if row.family_id != held_out)
        if not counts:
            raise HarborSuiteError("empty family-held-out majority fold")
        prediction = sorted(counts, key=lambda label: (-counts[label], label))[0]
        values.extend(
            {
                "task_id": row.task_id,
                "family": row.family_id,
                "truth": row.label,
                "prediction": prediction,
            }
            for row in rows
            if row.family_id == held_out
        )
    return values


def _task_identity_upper(rows: Sequence[Row]) -> list[dict[str, str]]:
    values: list[dict[str, str]] = []
    for task_id in sorted({row.task_id for row in rows}):
        counts = Counter(row.label for row in rows if row.task_id == task_id)
        prediction = sorted(counts, key=lambda label: (-counts[label], label))[0]
        values.extend(
            {"task_id": row.task_id, "truth": row.label, "prediction": prediction}
            for row in rows
            if row.task_id == task_id
        )
    return values


def _metrics(predictions: Sequence[Mapping[str, str]]) -> dict[str, float]:
    classes = sorted({item["truth"] for item in predictions})
    if not predictions or not classes:
        return {"macro_f1": 0.0, "balanced_accuracy": 0.0, "accuracy": 0.0}
    f1s = []
    recalls = []
    for label in classes:
        true_positive = sum(
            item["truth"] == label == item["prediction"] for item in predictions
        )
        false_positive = sum(
            item["truth"] != label and item["prediction"] == label
            for item in predictions
        )
        false_negative = sum(
            item["truth"] == label and item["prediction"] != label
            for item in predictions
        )
        precision = (
            true_positive / (true_positive + false_positive)
            if true_positive + false_positive
            else 0.0
        )
        recall = (
            true_positive / (true_positive + false_negative)
            if true_positive + false_negative
            else 0.0
        )
        f1s.append(
            2 * precision * recall / (precision + recall) if precision + recall else 0.0
        )
        recalls.append(recall)
    return {
        "macro_f1": sum(f1s) / len(f1s),
        "balanced_accuracy": sum(recalls) / len(recalls),
        "accuracy": sum(item["truth"] == item["prediction"] for item in predictions)
        / len(predictions),
    }


def _percentile(values: Sequence[float], probability: float) -> float:
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, round(probability * (len(ordered) - 1))))
    return ordered[index]


def _planned_trials(
    config: Mapping[str, Any],
    selected: Sequence[Mapping[str, str]],
    tasks_by_name: Mapping[str, Path],
    *,
    subset_requested: bool,
) -> list[tuple[Mapping[str, str], Path, str, int, str]]:
    selected_by_id = {item["id"]: item for item in selected}
    pair_plan = config.get("pair_plan")
    if not isinstance(pair_plan, list) or len(pair_plan) != 16:
        raise HarborSuiteError("frozen study requires exactly 16 paired blocks")
    values: list[tuple[Mapping[str, str], Path, str, int, str]] = []
    for pair in pair_plan:
        if not isinstance(pair, Mapping):
            raise HarborSuiteError("pair plan entries must be objects")
        item = selected_by_id.get(str(pair.get("task_id")))
        if item is None and subset_requested:
            continue
        if item is None:
            raise HarborSuiteError(f"pair references unknown task: {pair!r}")
        task = tasks_by_name.get(item["id"])
        if task is None:
            raise HarborSuiteError(
                f"selected task is not a dataset member: {item['id']}"
            )
        _validate_task_digest(task, item["digest"])
        arm_order = pair.get("arm_order")
        if arm_order not in (["control", "internalcot"], ["internalcot", "control"]):
            raise HarborSuiteError(f"invalid frozen arm order: {arm_order!r}")
        pair_id = str(pair.get("pair_id"))
        repetition = _safe_int(pair.get("repetition"))
        values.extend((item, task, pair_id, repetition, arm) for arm in arm_order)
    return values


def run_study(args: argparse.Namespace) -> int:
    if not args.execute:
        raise SystemExit(
            "refusing paid/authenticated model execution without --execute"
        )
    config_path = args.config.resolve(strict=True)
    config = _config(config_path)
    internalcot_prefix = args.internalcot_prefix.resolve(strict=True)
    _validate_internalcot(internalcot_prefix, config)
    if not git_tracked_worktree_is_clean(ROOT):
        raise HarborSuiteError(
            "study execution requires a clean tracked worktree so collector bytes "
            "are bound by source SHA"
        )
    output = args.output.resolve()
    if output.exists():
        raise SystemExit(f"output directory already exists: {output}")
    output.mkdir(parents=True)
    tasks_by_name = {
        ref.path.name: ref.path for ref in get_suite("mathematical-benchmarks-v1").tasks
    }
    selected = _selected_tasks(config, set(args.task))
    manifest = {
        "schema_version": config["schema_version"],
        "study_id": config["study_id"],
        "config_sha256": _sha256(config_path),
        "collector_sha256": _collector_digest(),
        "source_sha": git_head_sha(ROOT),
        "codex_version": None,
        "internalcot": {
            "version": config["internalcot"]["version"],
            "npm_integrity": config["internalcot"]["npm_integrity"],
            "skill_sha256": config["internalcot"]["skill_sha256"],
            "workflow_sha256": config["internalcot"]["workflow_sha256"],
        },
        "trials": [],
    }
    version = run_operator_command(
        "codex", ("--version",), cwd=ROOT, timeout_seconds=30
    )
    if version.status is not ToolCommandStatus.EXITED or version.exit_code != 0:
        raise HarborSuiteError("codex --version failed")
    manifest["codex_version"] = version.stdout.decode("utf-8", errors="replace").strip()
    planned = _planned_trials(
        config, selected, tasks_by_name, subset_requested=bool(args.task)
    )
    for item, task, pair_id, repetition, arm in planned:
        trial_id = f"{pair_id}--{arm}"
        record = _run_trial(
            task=task,
            task_record=item,
            config=config,
            root=output,
            trial_id=trial_id,
            pair_id=pair_id,
            arm=arm,
            repetition=repetition,
            internalcot_prefix=internalcot_prefix,
        )
        manifest["trials"].append(
            {
                "trial_id": trial_id,
                "pair_id": pair_id,
                "arm": arm,
                "task_id": item["id"],
                "family": item.get("family", item["id"]),
                "repetition": repetition,
                "record": f"{trial_id}/trial.json",
                "record_sha256": _sha256(output / trial_id / "trial.json"),
                "command_status": record["command"]["status"],
                "verifier_reward": record["verifier"]["reward"],
            }
        )
        _write_json(output / "manifest.json", manifest)
        print(json.dumps(manifest["trials"][-1], sort_keys=True), flush=True)
    return 0


def _bootstrap_metric(
    predictions: Sequence[Mapping[str, str]],
    *,
    seed: int,
    repetitions: int,
) -> dict[str, float]:
    task_ids = sorted({item["task_id"] for item in predictions})
    if not task_ids:
        return {"lower_95": 0.0, "upper_95": 0.0}
    random = Random(seed)
    samples = []
    for _ in range(repetitions):
        selected = [random.choice(task_ids) for _ in task_ids]
        replicated = [
            item
            for task_id in selected
            for item in predictions
            if item["task_id"] == task_id
        ]
        samples.append(_metrics(replicated)["macro_f1"])
    return {
        "lower_95": _percentile(samples, 0.025),
        "upper_95": _percentile(samples, 0.975),
    }


def _summarize_heldout_diagnostics(
    all_rows: Mapping[str, Sequence[Row]],
    eligibility: Mapping[str, object],
    *,
    arm: str,
    seed: int,
    bootstrap_repetitions: int,
) -> tuple[
    dict[str, dict[str, Any]],
    list[str],
    dict[str, dict[str, list[dict[str, str]]]],
]:
    results: dict[str, dict[str, Any]] = {}
    eligible_targets: list[str] = []
    predictions: dict[str, dict[str, list[dict[str, str]]]] = {}
    minimum_classes = _safe_int(eligibility.get("diagnostic_requires_at_least_classes"))
    for target_index, target in enumerate(_HELDOUT_TARGETS):
        target_rows = all_rows[target]
        labels = sorted({row.label for row in target_rows})
        eligible = len(labels) >= minimum_classes
        if eligible:
            eligible_targets.append(target)
        predictions[target] = {}
        condition_metrics = {}
        conditions = (_BASE_CONDITION, _VISIBLE_CONDITION_BY_ARM[arm])
        for condition_index, condition in enumerate(conditions):
            heldout = _family_predictions(target_rows, condition)
            predictions[target][condition] = heldout
            condition_metrics[condition] = {
                **_metrics(heldout),
                "task_bootstrap_95": _bootstrap_metric(
                    heldout,
                    seed=seed + target_index * 10 + condition_index,
                    repetitions=bootstrap_repetitions,
                ),
            }
        results[target] = {
            "eligible": eligible,
            "row_count": len(target_rows),
            "task_count": len({row.task_id for row in target_rows}),
            "family_count": len({row.family_id for row in target_rows}),
            "class_counts": dict(
                sorted(Counter(row.label for row in target_rows).items())
            ),
            "conditions": condition_metrics,
            "baselines": {
                "family_heldout_global_majority": _metrics(
                    _family_majority_predictions(target_rows)
                ),
                "task_identity_resubstitution_upper_bound": _metrics(
                    _task_identity_upper(target_rows)
                ),
            },
        }
    return results, eligible_targets, predictions


def _successful_producer_checker_chain(events: Sequence[Mapping[str, object]]) -> bool:
    producer_positions = [
        index
        for index, event in enumerate(events)
        if event.get("kind") == "OPERATION_ATTEMPT"
        and not str(event.get("operation_id", "")).endswith(".verify")
        and event.get("execution_status") == "COMPLETED"
    ]
    checker_positions = [
        index
        for index, event in enumerate(events)
        if event.get("kind") == "OPERATION_ATTEMPT"
        and str(event.get("operation_id", "")).endswith(".verify")
        and _checker_attempt_accepted(event)
    ]
    return any(
        left < right for left in producer_positions for right in checker_positions
    )


def _load_projected_trials(
    *,
    config: Mapping[str, Any],
    manifest: Mapping[str, Any],
    results_root: Path,
    tasks_by_name: Mapping[str, Path],
) -> tuple[dict[str, list[Row]], list[dict[str, Any]]]:
    all_rows: dict[str, list[Row]] = defaultdict(list)
    projections = []
    for trial_record in manifest.get("trials", []):
        trial_id = trial_record["trial_id"]
        task_id = trial_record.get("task_id", trial_id)
        if not isinstance(trial_id, str) or not isinstance(task_id, str):
            raise HarborSuiteError("run manifest has invalid trial/task identity")
        trial_path = results_root / trial_id
        if _sha256(trial_path / "trial.json") != trial_record["record_sha256"]:
            raise HarborSuiteError(f"trial record digest mismatch: {trial_id}")
        if task_id not in tasks_by_name:
            raise HarborSuiteError(f"unknown task in run manifest: {task_id}")
        projection, projected_rows = _project_trial(
            tasks_by_name[task_id], trial_path, config
        )
        projections.append(projection)
        for target, values in projected_rows.items():
            all_rows[target].extend(values)
    return all_rows, projections


def _bootstrap_information_effects(
    predictions: Mapping[str, Mapping[str, Mapping[str, Sequence[Mapping[str, str]]]]],
    eligible: Sequence[str],
    *,
    seed: int,
    repetitions: int,
) -> dict[str, dict[str, float]]:
    task_ids = sorted(
        {
            item["task_id"]
            for arm in ("control", "internalcot")
            for target in eligible
            for item in predictions[arm][target][_BASE_CONDITION]
        }
    )
    random = Random(seed)
    samples = {"control": [], "internalcot": [], "b_star_minus_b": []}
    for _ in range(repetitions):
        selected = [random.choice(task_ids) for _ in task_ids]
        increments: dict[str, float] = {}
        for arm in ("control", "internalcot"):
            target_values = []
            for target in eligible:
                metrics = {}
                conditions = (_BASE_CONDITION, _VISIBLE_CONDITION_BY_ARM[arm])
                for condition in conditions:
                    source = predictions[arm][target][condition]
                    replicated = [
                        item
                        for task_id in selected
                        for item in source
                        if item["task_id"] == task_id
                    ]
                    metrics[condition] = _metrics(replicated)["macro_f1"]
                target_values.append(
                    metrics[_VISIBLE_CONDITION_BY_ARM[arm]] - metrics[_BASE_CONDITION]
                )
            increments[arm] = sum(target_values) / len(target_values)
            samples[arm].append(increments[arm])
        samples["b_star_minus_b"].append(
            increments["internalcot"] - increments["control"]
        )
    return {
        name: {
            "lower_95": _percentile(values, 0.025),
            "upper_95": _percentile(values, 0.975),
        }
        for name, values in samples.items()
    }


def _paired_behavior(
    projections: Sequence[Mapping[str, Any]], *, seed: int, repetitions: int
) -> dict[str, Any]:
    by_pair: dict[str, dict[str, Mapping[str, Any]]] = defaultdict(dict)
    for item in projections:
        by_pair[str(item["pair_id"])][str(item["arm"])] = item
    metrics = sorted(
        {
            name
            for item in projections
            for name in item["behavior_metrics"]
            if name != "cached_input_tokens"
        }
    )
    task_differences: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for arms in by_pair.values():
        if set(arms) != {"control", "internalcot"}:
            raise HarborSuiteError("behavior analysis requires complete paired arms")
        task_id = str(arms["control"]["task_id"])
        for metric in metrics:
            task_differences[metric][task_id].append(
                _safe_float(arms["internalcot"]["behavior_metrics"].get(metric))
                - _safe_float(arms["control"]["behavior_metrics"].get(metric))
            )
    task_ids = sorted({item["task_id"] for item in projections})
    random = Random(seed)
    report: dict[str, Any] = {}
    for metric in metrics:
        arm_values = {
            arm: [
                _safe_float(item["behavior_metrics"].get(metric))
                for item in projections
                if item["arm"] == arm
            ]
            for arm in ("control", "internalcot")
        }
        differences = [
            value for task_id in task_ids for value in task_differences[metric][task_id]
        ]
        samples = []
        for _ in range(repetitions):
            selected = [random.choice(task_ids) for _ in task_ids]
            values = [
                value
                for task_id in selected
                for value in task_differences[metric][task_id]
            ]
            samples.append(sum(values) / len(values))
        report[metric] = {
            "control_mean": sum(arm_values["control"]) / len(arm_values["control"]),
            "internalcot_mean": sum(arm_values["internalcot"])
            / len(arm_values["internalcot"]),
            "treatment_minus_control_mean": sum(differences) / len(differences),
            "task_bootstrap_95": {
                "lower_95": _percentile(samples, 0.025),
                "upper_95": _percentile(samples, 0.975),
            },
        }
    return report


def _analyze_intervention(
    *,
    config: Mapping[str, Any],
    config_path: Path,
    results_root: Path,
    manifest: Mapping[str, Any],
    collector_digest: str,
    all_rows: Mapping[str, Sequence[Row]],
    projections: Sequence[Mapping[str, Any]],
    output: Path,
) -> int:
    expected = {
        (str(pair["pair_id"]), arm)
        for pair in config["pair_plan"]
        for arm in ("control", "internalcot")
    }
    observed = {(str(item["pair_id"]), str(item["arm"])) for item in projections}
    if observed != expected or len(projections) != 32:
        raise HarborSuiteError("trials do not exactly cover the frozen paired matrix")
    repetitions = 2000
    seed = _safe_int(config["modeling"]["bootstrap_seed"])
    arm_results: dict[str, Any] = {}
    arm_predictions: dict[str, dict[str, dict[str, list[dict[str, str]]]]] = {}
    eligible_by_arm: dict[str, list[str]] = {}
    for arm in ("control", "internalcot"):
        arm_rows = {
            target: [row for row in rows if row.arm == arm]
            for target, rows in all_rows.items()
        }
        results, eligible, predictions = _summarize_heldout_diagnostics(
            arm_rows,
            config["eligibility"],
            arm=arm,
            seed=seed + (100 if arm == "internalcot" else 0),
            bootstrap_repetitions=repetitions,
        )
        arm_results[arm] = results
        arm_predictions[arm] = predictions
        eligible_by_arm[arm] = eligible
    eligible = sorted(
        set(eligible_by_arm["control"]) & set(eligible_by_arm["internalcot"])
    )
    information_intervals = (
        _bootstrap_information_effects(
            arm_predictions,
            eligible,
            seed=seed + 1000,
            repetitions=repetitions,
        )
        if eligible
        else {
            name: {"lower_95": 0.0, "upper_95": 0.0}
            for name in ("control", "internalcot", "b_star_minus_b")
        }
    )
    increments: dict[str, dict[str, Any]] = {}
    for arm in ("control", "internalcot"):
        per_target = {
            target: arm_results[arm][target]["conditions"][
                _VISIBLE_CONDITION_BY_ARM[arm]
            ]["macro_f1"]
            - arm_results[arm][target]["conditions"][_BASE_CONDITION]["macro_f1"]
            for target in eligible
        }
        increments[arm] = {
            "visible_source": "b" if arm == "control" else "b_star",
            "per_diagnostic_macro_f1": per_target,
            "mean_macro_f1": sum(per_target.values()) / len(per_target)
            if per_target
            else 0.0,
            "task_bootstrap_95": information_intervals[arm],
        }
    behavior = _paired_behavior(projections, seed=seed + 2000, repetitions=repetitions)
    correctness_metrics = ("mathematical_correctness", "tool_adoption", "checker_use")
    count_metrics = ("tool_error_count", "retry_count", "jacobian_call_count")
    policy_active = any(
        abs(behavior[name]["treatment_minus_control_mean"]) >= threshold
        and not (
            behavior[name]["task_bootstrap_95"]["lower_95"]
            <= 0
            <= behavior[name]["task_bootstrap_95"]["upper_95"]
        )
        for names, threshold in ((correctness_metrics, 0.10), (count_metrics, 0.50))
        for name in names
    )
    policy_equivalent = all(
        behavior[name]["task_bootstrap_95"]["lower_95"] >= -margin
        and behavior[name]["task_bootstrap_95"]["upper_95"] <= margin
        for names, margin in ((correctness_metrics, 0.10), (count_metrics, 0.50))
        for name in names
    )
    treatment_increments = increments["internalcot"]["per_diagnostic_macro_f1"]
    strong_b_star = bool(eligible) and (
        increments["internalcot"]["mean_macro_f1"] >= 0.05
        and sum(value >= 0.05 for value in treatment_increments.values()) >= 2
        and information_intervals["internalcot"]["lower_95"] > 0
        and information_intervals["b_star_minus_b"]["lower_95"] > 0
    )
    failure_counts = {
        arm: Counter(
            item["labels"]["tool_failure_state"]
            for item in projections
            if item["arm"] == arm
        )
        for arm in ("control", "internalcot")
    }
    checker_counts = Counter(item["labels"]["checker_state"] for item in projections)
    recovery_counts = Counter(item["labels"]["recovery_state"] for item in projections)
    per_arm_tool = {
        arm: sum(
            item["tool_metrics"]["tool_call_count"] > 0
            for item in projections
            if item["arm"] == arm
        )
        for arm in ("control", "internalcot")
    }
    treatment_adherent = sum(
        bool(item["intervention_adherence"]["adherent"])
        for item in projections
        if item["arm"] == "internalcot"
    )
    treatment_projections = [
        item for item in projections if item["arm"] == "internalcot"
    ]
    adherence_condition_counts = {
        "official_skill_loaded": sum(
            bool(item["intervention_adherence"]["official_skill_loaded"])
            for item in treatment_projections
        ),
        "minimum_two_successful_notes": sum(
            _safe_int(item["intervention_adherence"]["successful_note_count"]) >= 2
            for item in treatment_projections
        ),
        "first_note_before_substantive": sum(
            bool(item["intervention_adherence"]["first_note_before_substantive"])
            for item in treatment_projections
        ),
        "final_note_after_substantive_before_final": sum(
            bool(
                item["intervention_adherence"][
                    "final_note_after_substantive_before_final"
                ]
            )
            for item in treatment_projections
        ),
        "all_conditions": treatment_adherent,
    }
    eligibility = config["eligibility"]
    coverage_checks = {
        "completed_trials": sum(
            item["command_status"] == "EXITED" for item in projections
        )
        == _safe_int(eligibility["completed_trials"]),
        "complete_pairs": len({item["pair_id"] for item in projections})
        == _safe_int(eligibility["complete_pairs"]),
        "families": len({item["family"] for item in projections})
        == _safe_int(eligibility["families"]),
        "treatment_adherence": treatment_adherent
        == _safe_int(eligibility["treatment_adherent_trials"]),
        "eligible_diagnostics_per_arm": all(
            len(values)
            >= _safe_int(
                eligibility["minimum_eligible_observability_diagnostics_per_arm"]
            )
            for values in eligible_by_arm.values()
        ),
        "server_tool_trajectories_per_arm": all(
            value >= _safe_int(eligibility["minimum_server_tool_trajectories_per_arm"])
            for value in per_arm_tool.values()
        ),
        "no_tool_trajectories": sum(
            counts["NO_TOOL"] for counts in failure_counts.values()
        )
        >= _safe_int(eligibility["minimum_no_tool_trajectories"]),
        "checker_use_trajectories": sum(
            value != "NO_CHECKER"
            for value in (item["labels"]["checker_state"] for item in projections)
        )
        >= _safe_int(eligibility["minimum_checker_use_trajectories"]),
        "checker_rejections": checker_counts["REJECTED"]
        >= _safe_int(eligibility["minimum_checker_rejections"]),
        "recovered_trajectories": recovery_counts["RECOVERED"]
        >= _safe_int(eligibility["minimum_recovered_trajectories"]),
        "tool_failure_trajectories": sum(
            counts["RECOVERED"] + counts["UNRECOVERED"]
            for counts in failure_counts.values()
        )
        >= _safe_int(eligibility["minimum_tool_failure_trajectories"]),
    }
    report = {
        "schema_version": "1",
        "study_id": config["study_id"],
        "evidence_class": config["evidence_class"],
        "causal_claim_authorized": False,
        "production_change_authorized": False,
        "config_sha256": _sha256(config_path),
        "run_manifest_sha256": _sha256(results_root / "manifest.json"),
        "source_sha": manifest.get("source_sha"),
        "collector_sha256": collector_digest,
        "analyzer_sha256": _collector_digest(),
        "analysis": {
            "mode": "FROZEN_PAIRED_ACTIVE_INTERVENTION_FAMILY_HELD_OUT_KNN",
            "held_out": True,
            "transductive": False,
            "bootstrap_repetitions": repetitions,
            "predictive_association_is_not_behavioral_effect": True,
        },
        "dataset": {
            "trial_count": len(projections),
            "pair_count": len({item["pair_id"] for item in projections}),
            "task_count": len({item["task_id"] for item in projections}),
            "family_count": len({item["family"] for item in projections}),
            "treatment_adherent_count": treatment_adherent,
            "treatment_adherence_condition_counts": adherence_condition_counts,
            "eligible": all(coverage_checks.values()),
            "coverage_checks": coverage_checks,
        },
        "event_coverage": {
            "server_event_candidates": sum(
                _safe_int(item["server_event_coverage"]["candidates"])
                for item in projections
            ),
            "server_events_recorded": sum(
                _safe_int(item["server_event_coverage"]["recorded"])
                for item in projections
            ),
            "server_tool_trajectories_by_arm": per_arm_tool,
            "successful_producer_checker_chains": sum(
                bool(item["tool_metrics"]["successful_producer_checker_chain"])
                for item in projections
            ),
            "tool_failure_state_counts_by_arm": {
                arm: dict(sorted(values.items()))
                for arm, values in failure_counts.items()
            },
            "checker_state_counts": dict(sorted(checker_counts.items())),
            "recovery_state_counts": dict(sorted(recovery_counts.items())),
        },
        "observability": {
            "diagnostics_by_arm": arm_results,
            "eligible_diagnostics_by_arm": eligible_by_arm,
            "common_eligible_diagnostics": eligible,
            "conditional_visible_information": increments,
            "b_star_minus_b_mean_macro_f1": increments["internalcot"]["mean_macro_f1"]
            - increments["control"]["mean_macro_f1"],
            "b_star_minus_b_task_bootstrap_95": information_intervals["b_star_minus_b"],
            "strong_b_star_information": strong_b_star,
        },
        "paired_behavior": behavior,
        "policy_intervention_detected": policy_active,
        "policy_equivalence_supported": policy_equivalent,
        "decision": "INCONCLUSIVE_RESEARCH_ONLY",
        "decision_reasons": [
            "This active visible-reasoning intervention cannot authorize passive retention or a production observer change.",
            *(
                []
                if all(coverage_checks.values())
                else ["At least one frozen coverage gate failed."]
            ),
            *(
                []
                if strong_b_star
                else ["The frozen strong-b-star information rule was not satisfied."]
            ),
            *(
                []
                if policy_equivalent
                else ["The frozen policy-equivalence rule was not satisfied."]
            ),
        ],
        "projection_count": len(projections),
        "projection_sha256": _json_digest(list(projections)),
        "pre_run_failures": [
            {
                "stage": "pinned_internalcot_validation",
                "model_calls": 0,
                "accepted_trials": 0,
                "outcome": "The bounded command wrapper rejected the first absolute-path validation attempt; validation was corrected before collection.",
            },
            {
                "stage": "pinned_harbor_digest_validation",
                "model_calls": 0,
                "accepted_trials": 0,
                "outcome": "The lightweight environment lacked Harbor; all eight frozen digests were revalidated with Harbor 0.20.0 before collection.",
            },
        ],
        "retention": config["retention"],
        "limitations": [
            "The selected tasks were previously observed in passive #1259 work; blocking controls task identity but this is not an unseen-task operation evaluation.",
            "The paired contrast identifies the bundled prompt, skill, and CLI intervention, not internalcot notes in isolation.",
            "Conditional b and b_star gains are held-out predictive associations; paired behavior effects are a separate estimand.",
            "Thirty-two non-deterministic weak-model trials provide limited uncertainty resolution.",
            "No hidden reasoning, raw internalcot notes, raw agent messages, prompts, submissions, tool arguments, tool results, or verifier internals are committed.",
        ],
    }
    if output.exists():
        raise SystemExit(f"output already exists: {output}")
    _write_json(output, report)
    print(
        json.dumps(
            {"decision": report["decision"], "dataset": report["dataset"]},
            sort_keys=True,
        )
    )
    print(f"report_sha256={_sha256(output)}")
    return 0


def analyze_study(args: argparse.Namespace) -> int:
    config_path = args.config.resolve(strict=True)
    config = _config(config_path)
    results_root = args.results.resolve(strict=True)
    manifest = _read_json(results_root / "manifest.json")
    if manifest.get("config_sha256") != _sha256(config_path):
        raise HarborSuiteError("run manifest is not bound to the selected study config")
    collector_digest = manifest.get("collector_sha256")
    if not isinstance(collector_digest, str) or not re.fullmatch(
        r"sha256:[0-9a-f]{64}", collector_digest
    ):
        raise HarborSuiteError("run manifest does not bind collector bytes")
    tasks_by_name = {
        ref.path.name: ref.path for ref in get_suite("mathematical-benchmarks-v1").tasks
    }
    all_rows, projections = _load_projected_trials(
        config=config,
        manifest=manifest,
        results_root=results_root,
        tasks_by_name=tasks_by_name,
    )
    return _analyze_intervention(
        config=config,
        config_path=config_path,
        results_root=results_root,
        manifest=manifest,
        collector_digest=collector_digest,
        all_rows=all_rows,
        projections=projections,
        output=args.output.resolve(),
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    run = subparsers.add_parser("run")
    run.add_argument("--config", type=Path, default=_DEFAULT_CONFIG)
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--task", action="append", default=[])
    run.add_argument("--internalcot-prefix", type=Path, required=True)
    run.add_argument("--execute", action="store_true")
    run.set_defaults(function=run_study)
    analyze = subparsers.add_parser("analyze")
    analyze.add_argument("--config", type=Path, default=_DEFAULT_CONFIG)
    analyze.add_argument("--results", type=Path, required=True)
    analyze.add_argument("--output", type=Path, required=True)
    analyze.set_defaults(function=analyze_study)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    return int(args.function(args))


if __name__ == "__main__":
    raise SystemExit(main())
