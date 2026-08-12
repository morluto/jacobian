"""Operator-run diagnostic for Codex visibility of Jacobian's MCP affordances."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import tempfile
import time
from collections.abc import Mapping
from enum import StrEnum
from pathlib import Path
from typing import Any, Literal
from urllib.parse import urlsplit

import httpx2
from mcp import Client
from mcp.client.streamable_http import streamable_http_client
from mcp_types import TextResourceContents
from pydantic import BaseModel, ConfigDict, Field, model_validator

from benchmarks.tooling.command_runner import (
    ToolCommandStatus,
    git_head_sha,
    operator_environment,
    run_operator_command,
)
from jacobian.canonical import canonicalize_json
from jacobian.eval.telemetry import parse_agent_transcript

_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_CASES = _ROOT / "benchmarks/config/codex-visibility-v2.json"
_REQUIRED_TOOLS = frozenset({"math.find", "math.run"})
_LOCAL_VERIFICATION_URI_PREFIX = "artifact://"
_CODEX_ENVIRONMENT = (
    "HOME",
    "PATH",
    "CODEX_HOME",
    "JACOBIAN_MCP_BEARER_TOKEN",
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "ALL_PROXY",
    "NO_PROXY",
    "http_proxy",
    "https_proxy",
    "all_proxy",
    "no_proxy",
)
_MCP_TOOL_APPROVAL_MODE = "approve"


class CueLevel(StrEnum):
    """How directly a case exposes the existence of a specialized tool."""

    EXPLICIT = "EXPLICIT"
    AFFORDANCE = "AFFORDANCE"
    LATENT = "LATENT"


class AdoptionExpectation(StrEnum):
    """Whether the prompt should use or abstain from Jacobian MCP tools."""

    USE = "USE"
    ABSTAIN = "ABSTAIN"


class ToolMode(StrEnum):
    """How Codex receives and dispatches tools during one visibility run."""

    DIRECT = "direct"
    UNIFIED_EXEC = "unified_exec"


class VisibilityOutputOutcome(BaseModel):
    """One acceptable completed-operation output shape for a USE case."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    capability_id: str = Field(min_length=1)
    required_output_fields: tuple[str, ...] = Field(min_length=1)
    expected_output_values: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _valid_output_fields(self) -> VisibilityOutputOutcome:
        if len(set(self.required_output_fields)) != len(self.required_output_fields):
            raise ValueError("required_output_fields must be unique")
        if not set(self.expected_output_values).issubset(self.required_output_fields):
            raise ValueError("expected output values must name required output fields")
        return self


class VisibilityCase(BaseModel):
    """One agent-visible prompt plus hidden trajectory expectations."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    case_id: str = Field(pattern=r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
    cue_level: CueLevel
    prompt: str = Field(min_length=1)
    expectation: AdoptionExpectation = AdoptionExpectation.USE
    expected_capability_ids: tuple[str, ...] = ()
    diagnostic_capability_ids: tuple[str, ...] = ()
    acceptable_output_outcomes: tuple[VisibilityOutputOutcome, ...] = ()
    require_verified: bool = False

    @model_validator(mode="after")
    def _valid_expectation(self) -> VisibilityCase:
        if len(set(self.expected_capability_ids)) != len(self.expected_capability_ids):
            raise ValueError("expected_capability_ids must be unique")
        if len(set(self.diagnostic_capability_ids)) != len(
            self.diagnostic_capability_ids
        ):
            raise ValueError("diagnostic_capability_ids must be unique")
        if set(self.expected_capability_ids) & set(self.diagnostic_capability_ids):
            raise ValueError("required and diagnostic capability IDs must be disjoint")
        outcome_ids = {
            outcome.capability_id for outcome in self.acceptable_output_outcomes
        }
        if not outcome_ids.issubset(
            set(self.expected_capability_ids) | set(self.diagnostic_capability_ids)
        ):
            raise ValueError("output-outcome capability IDs must be tracked")
        if (
            self.expectation is AdoptionExpectation.USE
            and not self.expected_capability_ids
            and not self.acceptable_output_outcomes
        ):
            raise ValueError("USE cases require an operation or output outcome")
        if self.expectation is AdoptionExpectation.ABSTAIN and (
            self.expected_capability_ids
            or self.diagnostic_capability_ids
            or self.acceptable_output_outcomes
        ):
            raise ValueError("ABSTAIN cases cannot declare operations or outcomes")
        if self.expectation is AdoptionExpectation.ABSTAIN and self.require_verified:
            raise ValueError("ABSTAIN cases cannot require verification")
        return self


class VisibilitySuite(BaseModel):
    """Versioned visibility prompt suite."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["1", "2"]
    suite_id: str = Field(pattern=r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
    cases: tuple[VisibilityCase, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def _unique_cases(self) -> VisibilitySuite:
        case_ids = [case.case_id for case in self.cases]
        if len(set(case_ids)) != len(case_ids):
            raise ValueError("case_id values must be unique")
        if self.schema_version == "1" and any(
            case.expectation is not AdoptionExpectation.USE for case in self.cases
        ):
            raise ValueError("schema version 1 supports only USE cases")
        return self


def load_suite(path: Path) -> VisibilitySuite:
    """Load and fully validate a visibility suite."""

    return VisibilitySuite.model_validate_json(path.read_text(encoding="utf-8"))


def _sha256_bytes(value: bytes) -> str:
    return f"sha256:{hashlib.sha256(value).hexdigest()}"


def _json_digest(value: object) -> str:
    return _sha256_bytes(canonicalize_json(value))


def surface_snapshot_digest(surface: Mapping[str, Any]) -> str:
    """Digest exactly the server, instructions, tools, and catalog snapshot."""

    fields = ("server", "instructions", "tools", "catalog")
    if any(field not in surface for field in fields):
        raise ValueError("MCP surface snapshot is incomplete")
    return _json_digest({field: surface[field] for field in fields})


def _is_verified_invocation(invocation: object) -> bool:
    if not isinstance(invocation, Mapping):
        return False
    uri = invocation.get("verification_record_uri")
    return isinstance(uri, str) and uri.startswith(_LOCAL_VERIFICATION_URI_PREFIX)


def _output_field(output: object, path: str) -> tuple[bool, object]:
    current = output
    for component in path.split("."):
        if isinstance(current, Mapping) and component in current:
            current = current[component]
            continue
        if isinstance(current, list) and component.isdigit():
            index = int(component)
            if index < len(current):
                current = current[index]
                continue
        return False, None
    return True, current


def _substantive_output_value(value: object) -> bool:
    if value is None:
        return False
    if isinstance(value, str | list | tuple | Mapping):
        return bool(value)
    return True


def _output_outcome_matches(
    outcome: VisibilityOutputOutcome,
    invocation: object,
) -> bool:
    if not isinstance(invocation, Mapping) or (
        invocation.get("capability_id") != outcome.capability_id
    ):
        return False
    observed: dict[str, object] = {}
    for path in outcome.required_output_fields:
        present, value = _output_field(invocation.get("output"), path)
        if not present or not _substantive_output_value(value):
            return False
        observed[path] = value
    return all(
        observed[path] == expected
        for path, expected in outcome.expected_output_values.items()
    )


def classify_visibility(
    case: VisibilityCase,
    telemetry: Mapping[str, Any],
) -> dict[str, Any]:
    """Classify only observable adoption stages; do not grade answer prose."""

    expected = set(case.expected_capability_ids)
    diagnostic = set(case.diagnostic_capability_ids)
    outcome_ids = {outcome.capability_id for outcome in case.acceptable_output_outcomes}
    tracked = expected | diagnostic | outcome_ids
    described = {
        capability_id
        for description in telemetry.get("capability_descriptions", [])
        if isinstance(description, Mapping)
        for capability_id in (
            [description.get("capability_id")]
            if description.get("capability_id") is not None
            else description.get("match_ids", [])
        )
        if isinstance(capability_id, str)
    }
    attempted_sequence = [
        value
        for value in telemetry.get("capability_attempt_ids", [])
        if isinstance(value, str)
    ]
    attempted = set(attempted_sequence)
    completed_sequence = [
        value for value in telemetry.get("capability_ids", []) if isinstance(value, str)
    ]
    completed = set(completed_sequence)
    invocations = tuple(
        invocation
        for invocation in telemetry.get("capability_invocations", [])
        if isinstance(invocation, Mapping)
    )
    verified = any(
        isinstance(invocation, Mapping)
        and isinstance(invocation.get("capability_id"), str)
        and invocation.get("capability_id") in expected
        and _is_verified_invocation(invocation)
        for invocation in invocations
    )
    matched_outcomes = tuple(
        outcome
        for outcome in case.acceptable_output_outcomes
        if any(
            _output_outcome_matches(outcome, invocation) for invocation in invocations
        )
    )
    mcp_calls = [
        value for value in telemetry.get("mcp_calls", []) if isinstance(value, str)
    ]
    discovery_call_count = int(telemetry.get("capability_describe_index_calls", 0))
    inspection_call_count = int(telemetry.get("capability_describe_exact_calls", 0))
    resource_read_count = int(telemetry.get("mcp_resource_read_attempts", 0))
    expected_attempted = expected & attempted
    observed = {
        "discovered": bool(telemetry.get("capability_describe_index_calls", 0)),
        "inspected": bool(telemetry.get("capability_describe_exact_calls", 0)),
        "invoked": bool(attempted),
        "completed": bool(completed),
        "verified": verified,
        "discovery_free_invocation": bool(expected_attempted)
        and not discovery_call_count
        and not inspection_call_count,
        "abstained": not mcp_calls and not resource_read_count,
    }
    expected_observed = {
        "described": sorted(expected & described),
        "attempted": sorted(expected & attempted),
        "completed": sorted(expected & completed),
        "missing_completed": sorted(expected - completed),
    }
    diagnostic_observed = {
        "described": sorted(diagnostic & described),
        "attempted": sorted(diagnostic & attempted),
        "completed": sorted(diagnostic & completed),
        "not_completed": sorted(diagnostic - completed),
    }
    if case.expectation is AdoptionExpectation.ABSTAIN:
        contract_satisfied = observed["abstained"]
    else:
        contract_satisfied = (
            not expected_observed["missing_completed"]
            and (verified or not case.require_verified)
            and (bool(matched_outcomes) or not case.acceptable_output_outcomes)
        )
    usage = telemetry.get("usage")
    uncached_input_tokens = None
    if isinstance(usage, Mapping):
        input_tokens = usage.get("input_tokens")
        cached_input_tokens = usage.get("cached_input_tokens")
        if isinstance(input_tokens, int) and isinstance(cached_input_tokens, int):
            uncached_input_tokens = max(0, input_tokens - cached_input_tokens)
    return {
        "expectation": case.expectation,
        "observed": observed,
        "expected_capabilities": expected_observed,
        "diagnostic_capabilities": diagnostic_observed,
        "output_outcomes": {
            "required": bool(case.acceptable_output_outcomes),
            "satisfied": bool(matched_outcomes),
            "matched_capability_ids": sorted(
                {outcome.capability_id for outcome in matched_outcomes}
            ),
        },
        "unexpected_capabilities": {
            "attempted": sorted(attempted - tracked),
            "completed": sorted(completed - tracked),
        },
        "contract_satisfied": contract_satisfied,
        "tool_error_count": telemetry.get("tool_error_count", 0),
        "parameter_error_count": telemetry.get("parameter_error_count", 0),
        "shell_call_count": len(telemetry.get("shell_calls", [])),
        "usage": usage,
        "uncached_input_tokens": uncached_input_tokens,
        "mcp_call_count": len(mcp_calls),
        "math_find_call_count": discovery_call_count + inspection_call_count,
        "math_run_call_count": len(attempted_sequence),
        "mcp_resource_read_count": resource_read_count,
        "mcp_wire_bytes": telemetry.get("mcp_wire_bytes", 0),
        "mcp_model_visible_bytes": telemetry.get("mcp_model_visible_bytes", 0),
        "mcp_logical_payload_bytes": telemetry.get("mcp_logical_payload_bytes", 0),
    }


async def inspect_surface(
    url: str,
    timeout_seconds: float,
    *,
    require_deployment_identity: bool = False,
) -> dict[str, Any]:
    """Snapshot the exact MCP surface used by a visibility run."""

    token = os.environ.get("JACOBIAN_MCP_BEARER_TOKEN")
    headers = {"Authorization": f"Bearer {token}"} if token else None
    async with (
        httpx2.AsyncClient(
            headers=headers,
            trust_env=False,
            timeout=timeout_seconds,
        ) as http,
        Client(
            streamable_http_client(url, http_client=http),
            raise_exceptions=True,
        ) as client,
    ):
        server_info = client.server_info
        if server_info is None:
            raise RuntimeError("MCP server omitted implementation metadata")
        listed = await client.list_tools()
        tool_records = [
            tool.model_dump(mode="json", by_alias=True, exclude_none=True)
            for tool in listed.tools
        ]
        tool_names = {record["name"] for record in tool_records}
        missing = sorted(_REQUIRED_TOOLS - tool_names)
        if missing:
            raise RuntimeError(f"MCP surface is missing required tools: {missing}")
        catalog_result = await client.read_resource("capability://catalog")
        catalog_content = catalog_result.contents[0]
        if not isinstance(catalog_content, TextResourceContents):
            raise RuntimeError("capability catalog is not text")
        catalog = json.loads(catalog_content.text)
        catalog_digest = _sha256_bytes(
            canonicalize_json(
                {
                    "catalog_version": catalog["catalog_version"],
                    "capabilities": catalog["capabilities"],
                }
            )
        )
        snapshot = {
            "server": server_info.model_dump(
                mode="json", by_alias=True, exclude_none=True
            ),
            "instructions": client.instructions,
            "tools": sorted(tool_records, key=lambda item: item["name"]),
            "catalog": {
                "catalog_version": catalog["catalog_version"],
                "catalog_digest": catalog_digest,
                "policy_profile": catalog["policy_profile"],
                "policy_digest": catalog["policy_digest"],
                "capability_count": len(catalog["capabilities"]),
                "content_sha256": _sha256_bytes(catalog_content.text.encode("utf-8")),
            },
        }
        if require_deployment_identity:
            deployment_result = await client.read_resource("deployment://identity")
            deployment_content = deployment_result.contents[0]
            if not isinstance(deployment_content, TextResourceContents):
                raise RuntimeError("deployment identity is not text")
            deployment = json.loads(deployment_content.text)
            if (
                not isinstance(deployment, dict)
                or deployment.get("schema_version") != "1"
                or deployment.get("evidence") != "release-marker"
                or not isinstance(deployment.get("revision"), str)
                or deployment.get("package_version") != server_info.version
            ):
                raise RuntimeError("MCP deployment identity is malformed")
            snapshot["deployment"] = deployment
    return {**snapshot, "surface_digest": surface_snapshot_digest(snapshot)}


def _codex_arguments(
    *,
    workspace: Path,
    model: str,
    reasoning_effort: str,
    mcp_url: str,
    prompt: str,
    tool_mode: ToolMode,
) -> tuple[str, ...]:
    arguments = [
        "-a",
        "never",
        "exec",
        "--ephemeral",
        "--skip-git-repo-check",
        "--ignore-user-config",
        "-C",
        str(workspace),
        "-s",
        "read-only",
        "--json",
        "-m",
        model,
        "-c",
        f"model_reasoning_effort={json.dumps(reasoning_effort)}",
        "-c",
        f"mcp_servers.jacobian.url={json.dumps(mcp_url)}",
        "-c",
        (
            "mcp_servers.jacobian.default_tools_approval_mode="
            f"{json.dumps(_MCP_TOOL_APPROVAL_MODE)}"
        ),
    ]
    if tool_mode is ToolMode.UNIFIED_EXEC:
        arguments.extend(("--enable", "unified_exec"))
    if os.environ.get("JACOBIAN_MCP_BEARER_TOKEN"):
        arguments.extend(
            (
                "-c",
                'mcp_servers.jacobian.bearer_token_env_var="JACOBIAN_MCP_BEARER_TOKEN"',
            )
        )
    return (*arguments, prompt)


def _command_version(workspace: Path) -> str:
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


def _run_case(
    *,
    case: VisibilityCase,
    repetition: int,
    workspace: Path,
    output: Path,
    model: str,
    reasoning_effort: str,
    mcp_url: str,
    timeout_seconds: float,
    tool_mode: ToolMode,
) -> dict[str, Any]:
    stem = f"{case.case_id}-r{repetition:02d}"
    transcript_path = output / f"{stem}.jsonl"
    stderr_path = output / f"{stem}.stderr"
    environment = operator_environment(include=_CODEX_ENVIRONMENT)
    command_start = time.monotonic()
    result = run_operator_command(
        "codex",
        _codex_arguments(
            workspace=workspace,
            model=model,
            reasoning_effort=reasoning_effort,
            mcp_url=mcp_url,
            prompt=case.prompt,
            tool_mode=tool_mode,
        ),
        cwd=workspace,
        timeout_seconds=timeout_seconds,
        stdout_limit_bytes=16 * 1024 * 1024,
        stderr_limit_bytes=2 * 1024 * 1024,
        environment=environment,
    )
    elapsed_seconds = round(time.monotonic() - command_start, 6)
    transcript_path.write_bytes(result.stdout)
    stderr_path.write_bytes(result.stderr)
    telemetry = parse_agent_transcript(transcript_path)
    classification = classify_visibility(case, telemetry)
    command_completed = (
        result.status is ToolCommandStatus.EXITED and result.exit_code == 0
    )
    return {
        "case_id": case.case_id,
        "cue_level": case.cue_level,
        "expectation": case.expectation,
        "repetition": repetition,
        "command": {
            "status": result.status,
            "exit_code": result.exit_code,
            "diagnostic": result.diagnostic,
            "stdout_exceeded": result.stdout_exceeded,
            "stderr_exceeded": result.stderr_exceeded,
            "elapsed_seconds": elapsed_seconds,
        },
        "classification": {
            **classification,
            "contract_satisfied": (
                command_completed and classification["contract_satisfied"]
            ),
        },
        "artifacts": {
            "transcript": transcript_path.name,
            "transcript_sha256": _sha256_bytes(result.stdout),
            "stderr": stderr_path.name,
            "stderr_sha256": _sha256_bytes(result.stderr),
        },
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Measure whether Codex discovers and invokes Jacobian without grading "
            "mathematical answer prose. Model execution is opt-in."
        )
    )
    parser.add_argument("--cases", type=Path, default=_DEFAULT_CASES)
    parser.add_argument("--mcp-url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--reasoning-effort", default="high")
    parser.add_argument(
        "--tool-mode",
        type=ToolMode,
        choices=tuple(ToolMode),
        default=ToolMode.DIRECT,
        help="Codex tool dispatch mode; unified_exec matches Harbor Code Mode.",
    )
    parser.add_argument("--repetitions", type=int, default=1)
    parser.add_argument(
        "--case",
        action="append",
        default=[],
        help="case ID to run; repeatable, defaults to the complete suite",
    )
    parser.add_argument("--timeout-seconds", type=float, default=300)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--execute",
        action="store_true",
        help="confirm that paid/external Codex model calls may run",
    )
    return parser


def _validate_mcp_url(value: str) -> None:
    parsed = urlsplit(value)
    if (
        parsed.scheme not in {"http", "https"}
        or not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
    ):
        raise SystemExit(
            "--mcp-url must be an HTTP(S) URL without credentials, query, or fragment"
        )


def _build_summary(runs: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate per-run observations into the report summary block."""

    return {
        "run_count": len(runs),
        "command_failure_count": sum(
            run["command"]["status"] != ToolCommandStatus.EXITED
            or run["command"]["exit_code"] != 0
            for run in runs
        ),
        "contract_satisfied_count": sum(
            run["classification"]["contract_satisfied"] for run in runs
        ),
        "discovered_count": sum(
            run["classification"]["observed"]["discovered"] for run in runs
        ),
        "invoked_count": sum(
            run["classification"]["observed"]["invoked"] for run in runs
        ),
        "verified_count": sum(
            run["classification"]["observed"]["verified"] for run in runs
        ),
        "discovery_free_invocation_count": sum(
            run["classification"]["observed"]["discovery_free_invocation"]
            for run in runs
        ),
        "abstained_count": sum(
            run["classification"]["observed"]["abstained"] for run in runs
        ),
        "cost_totals": {
            "input_tokens": sum(
                (run["classification"]["usage"] or {}).get("input_tokens", 0)
                for run in runs
            ),
            "cached_input_tokens": sum(
                (run["classification"]["usage"] or {}).get("cached_input_tokens", 0)
                for run in runs
            ),
            "uncached_input_tokens": sum(
                run["classification"]["uncached_input_tokens"] or 0 for run in runs
            ),
            "output_tokens": sum(
                (run["classification"]["usage"] or {}).get("output_tokens", 0)
                for run in runs
            ),
            "mcp_calls": sum(run["classification"]["mcp_call_count"] for run in runs),
            "mcp_model_visible_bytes": sum(
                run["classification"]["mcp_model_visible_bytes"] for run in runs
            ),
        },
        "duration_totals": {
            "elapsed_seconds": round(
                sum(run["command"]["elapsed_seconds"] for run in runs), 6
            ),
        },
    }


def main() -> None:
    args = _parser().parse_args()
    if not args.execute:
        raise SystemExit("refusing model execution without --execute")
    if not 1 <= args.repetitions <= 20:
        raise SystemExit("--repetitions must be between 1 and 20")
    if args.timeout_seconds <= 0:
        raise SystemExit("--timeout-seconds must be positive")
    _validate_mcp_url(args.mcp_url)
    suite = load_suite(args.cases.resolve(strict=True))
    available_case_ids = {case.case_id for case in suite.cases}
    unknown_case_ids = sorted(set(args.case) - available_case_ids)
    if unknown_case_ids:
        raise SystemExit(f"unknown case IDs: {unknown_case_ids}")
    selected_cases = tuple(
        case for case in suite.cases if not args.case or case.case_id in args.case
    )
    output = args.output.resolve()
    if output.exists():
        raise SystemExit(f"output directory already exists: {output}")
    surface = asyncio.run(inspect_surface(args.mcp_url, args.timeout_seconds))
    output.mkdir(parents=True)
    with tempfile.TemporaryDirectory(prefix="jacobian-codex-visibility-") as raw:
        workspace = Path(raw)
        codex_version = _command_version(workspace)
        runs = [
            _run_case(
                case=case,
                repetition=repetition,
                workspace=workspace,
                output=output,
                model=args.model,
                reasoning_effort=args.reasoning_effort,
                mcp_url=args.mcp_url,
                timeout_seconds=args.timeout_seconds,
                tool_mode=args.tool_mode,
            )
            for case in selected_cases
            for repetition in range(1, args.repetitions + 1)
        ]
    suite_payload = suite.model_dump(mode="json")
    summary = _build_summary(runs)
    report = {
        "schema_version": "2",
        "suite": {
            "suite_id": suite.suite_id,
            "digest": _json_digest(suite_payload),
            "case_count": len(suite.cases),
            "selected_case_ids": [case.case_id for case in selected_cases],
        },
        "condition": {
            "mcp_url": args.mcp_url,
            "surface": surface,
            "evaluator": {
                "runner_sha256": _sha256_bytes(Path(__file__).read_bytes()),
                "telemetry_parser_sha256": _sha256_bytes(
                    (_ROOT / "src/jacobian/eval/telemetry.py").read_bytes()
                ),
            },
            "codex_version": codex_version,
            "model": args.model,
            "reasoning_effort": args.reasoning_effort,
            "tool_mode": args.tool_mode,
            "repetitions": args.repetitions,
            "repository_revision": git_head_sha(_ROOT),
        },
        "runs": runs,
        "summary": summary,
    }
    (output / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report["summary"], sort_keys=True))
    if summary["command_failure_count"]:
        raise SystemExit("one or more Codex commands failed; inspect report.json")


if __name__ == "__main__":
    main()
