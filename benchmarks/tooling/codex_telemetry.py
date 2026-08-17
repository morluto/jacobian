"""Codex JSONL telemetry parsing shared by executable evaluations."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from jacobian.canonical import canonicalize_json

_PARAMETER_ERROR_CODES = frozenset(
    {
        -32602,
        "INVALID_ARGUMENT",
        "INVALID_CONSTRAINT_RANGE",
        "INVALID_PARAMS",
        "INVALID_REQUEST",
        "SCHEMA_VALIDATION",
        "invalid_params",
    }
)
_RESOURCE_READ_TOOL_NAMES = frozenset(
    {"resources/read", "resources.read", "resources_read"}
)


@dataclass
class _McpResourceTelemetry:
    links_returned: int = 0
    link_uris: list[str] = field(default_factory=list)
    read_attempts: int = 0
    read_uris: list[str] = field(default_factory=list)
    read_successes: int = 0


def _contains_value(value: object, *, field: str, accepted: set[object]) -> bool:
    if isinstance(value, Mapping):
        candidate = value.get(field)
        if isinstance(candidate, str | int | float | bool | type(None)) and (
            candidate in accepted
        ):
            return True
        return any(
            _contains_value(item, field=field, accepted=accepted)
            for item in value.values()
        )
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return any(
            _contains_value(item, field=field, accepted=accepted) for item in value
        )
    return False


def _mcp_text_payload(item: Mapping[str, Any]) -> dict[str, Any] | None:
    result = item.get("result")
    blocks: list[object] = []
    if isinstance(result, Mapping):
        for key in ("content", "contents"):
            value = result.get(key)
            if isinstance(value, list):
                blocks.extend(value)
    for block in blocks:
        if not isinstance(block, Mapping) or not isinstance(block.get("text"), str):
            continue
        try:
            payload = json.loads(block["text"])
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            return payload
    return None


def _mcp_structured_payload(item: Mapping[str, Any]) -> dict[str, Any] | None:
    result = item.get("result")
    if not isinstance(result, Mapping):
        return None
    for key in ("structured_content", "structuredContent"):
        payload = result.get(key)
        if isinstance(payload, dict):
            return payload
    return None


def _mcp_resource_link_uris(item: Mapping[str, Any]) -> tuple[str, ...]:
    result = item.get("result")
    content = result.get("content") if isinstance(result, Mapping) else None
    if not isinstance(content, list):
        return ()
    return tuple(
        block["uri"]
        for block in content
        if isinstance(block, Mapping)
        and block.get("type") == "resource_link"
        and isinstance(block.get("uri"), str)
    )


def _mcp_resource_read_uri(item: Mapping[str, Any]) -> str | None:
    item_type = item.get("type")
    tool = item.get("tool")
    if item_type != "mcp_resource_read" and (
        not isinstance(tool, str) or tool not in _RESOURCE_READ_TOOL_NAMES
    ):
        return None
    for key in ("arguments", "params", "input"):
        value = item.get(key)
        if isinstance(value, Mapping):
            uri = value.get("uri")
            if isinstance(uri, str):
                return uri
    value = item.get("uri")
    return value if isinstance(value, str) else None


def _mcp_resource_read_failed(item: Mapping[str, Any]) -> bool:
    result = item.get("result")
    status = item.get("status")
    return bool(
        (
            isinstance(status, str)
            and status in {"error", "failed", "CANCELLED", "ERROR", "TIMEOUT"}
        )
        or item.get("error")
        or (
            isinstance(result, Mapping)
            and (result.get("isError") is True or result.get("is_error") is True)
        )
    )


def _record_mcp_resource_telemetry(
    telemetry: _McpResourceTelemetry,
    item: object,
) -> None:
    if not isinstance(item, Mapping):
        return
    link_uris = _mcp_resource_link_uris(item)
    telemetry.links_returned += len(link_uris)
    telemetry.link_uris.extend(link_uris)
    resource_read_uri = _mcp_resource_read_uri(item)
    if resource_read_uri is None:
        return
    telemetry.read_attempts += 1
    telemetry.read_uris.append(resource_read_uri)
    read_failed = _mcp_resource_read_failed(item)
    if read_failed:
        return
    telemetry.read_successes += 1


def _serialized_bytes(value: object) -> int:
    try:
        encoded = json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    except (TypeError, ValueError):
        return 0
    return len(encoded)


def _mcp_wire_bytes(item: Mapping[str, Any]) -> int:
    result = item.get("result")
    if result is None:
        return 0
    return _serialized_bytes(result)


def _mcp_model_visible_bytes(item: Mapping[str, Any]) -> int:
    result = item.get("result")
    content = result.get("content") if isinstance(result, Mapping) else None
    if not isinstance(content, list):
        return 0
    byte_count = 0
    for block in content:
        if isinstance(block, Mapping) and isinstance(block.get("text"), str):
            byte_count += len(block["text"].encode("utf-8"))
        else:
            byte_count += _serialized_bytes(block)
    return byte_count


def _mcp_logical_payload_bytes(
    item: Mapping[str, Any],
    *,
    text_payload: Mapping[str, Any] | None,
    structured_payload: Mapping[str, Any] | None,
) -> int | None:
    del item
    if structured_payload is not None:
        return _serialized_bytes(structured_payload)
    if text_payload is not None:
        return _serialized_bytes(text_payload)
    return None


def _mcp_call_signature(tool: str, arguments: object) -> tuple[str, str]:
    try:
        encoded = json.dumps(
            arguments,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    except (TypeError, ValueError):
        encoded = b"unserializable"
    return tool, f"sha256:{hashlib.sha256(encoded).hexdigest()}"


def _operation_recovery_metrics(
    attempts: list[dict[str, Any]],
) -> dict[str, int]:
    """Count neutral recovery signals without interpreting mathematical outcomes."""

    failed_signatures: Counter[tuple[str, str]] = Counter()
    empty_payload_probe_count = 0
    failed_operation_attempt_count = 0
    for attempt in attempts:
        if attempt.get("successful") is not False:
            continue
        failed_operation_attempt_count += 1
        if attempt.get("input") == {} and attempt.get("request_validation_failure"):
            empty_payload_probe_count += 1
        failed_signatures[
            _mcp_call_signature(
                "math.run.error",
                {
                    "operation_id": attempt.get("operation_id"),
                    "input": attempt.get("input"),
                    "terminal_status": attempt.get("terminal_status"),
                    "error_digest": attempt.get("error_digest"),
                    "diagnostic_codes": attempt.get("diagnostic_codes", []),
                    "diagnostics": attempt.get("diagnostics", []),
                },
            )
        ] += 1
    return {
        "empty_payload_probe_count": empty_payload_probe_count,
        "failed_operation_attempt_count": failed_operation_attempt_count,
        "repeated_error_count": sum(
            count - 1 for count in failed_signatures.values() if count > 1
        ),
    }


@dataclass
class _AgentTranscriptTelemetry:
    mcp_calls: list[str] = field(default_factory=list)
    successful_calls: list[str] = field(default_factory=list)
    operation_attempt_ids: list[str] = field(default_factory=list)
    operation_attempts: list[dict[str, Any]] = field(default_factory=list)
    operation_ids: list[str] = field(default_factory=list)
    operation_invocations: list[dict[str, Any]] = field(default_factory=list)
    operation_descriptions: list[dict[str, Any]] = field(default_factory=list)
    shell_calls: list[str] = field(default_factory=list)
    usage: dict[str, Any] | None = None
    tool_error_count: int = 0
    parameter_error_count: int = 0
    operation_rejection_count: int = 0
    mcp_wire_bytes: int = 0
    mcp_wire_bytes_by_tool: Counter[str] = field(default_factory=Counter)
    mcp_model_visible_bytes: int = 0
    mcp_model_visible_bytes_by_tool: Counter[str] = field(default_factory=Counter)
    mcp_logical_payload_bytes: int = 0
    mcp_logical_payload_bytes_by_tool: Counter[str] = field(default_factory=Counter)
    mcp_logical_payload_observed_calls: int = 0
    mcp_call_signatures: Counter[tuple[str, str]] = field(default_factory=Counter)
    operation_describe_index_calls: int = 0
    operation_describe_exact_calls: int = 0
    resource_telemetry: _McpResourceTelemetry = field(
        default_factory=_McpResourceTelemetry
    )


def _is_command_execution_event(event: dict[str, Any], item: object) -> bool:
    return (
        event.get("type") == "item.completed"
        and isinstance(item, dict)
        and item.get("type") == "command_execution"
    )


def _is_mcp_tool_call_event(event: dict[str, Any], item: object) -> bool:
    return (
        event.get("type") == "item.completed"
        and isinstance(item, dict)
        and item.get("type") == "mcp_tool_call"
        and isinstance(item.get("tool"), str)
    )


def _record_mcp_byte_metrics(
    telemetry: _AgentTranscriptTelemetry,
    item: Mapping[str, Any],
    tool: str,
) -> tuple[Mapping[str, Any] | None, Mapping[str, Any] | None]:
    wire_bytes = _mcp_wire_bytes(item)
    model_visible_bytes = _mcp_model_visible_bytes(item)
    text_response = _mcp_text_payload(item)
    structured_response = _mcp_structured_payload(item)
    logical_bytes = _mcp_logical_payload_bytes(
        item,
        text_payload=text_response,
        structured_payload=structured_response,
    )
    telemetry.mcp_wire_bytes += wire_bytes
    telemetry.mcp_model_visible_bytes += model_visible_bytes
    if wire_bytes:
        telemetry.mcp_wire_bytes_by_tool[tool] += wire_bytes
    if model_visible_bytes:
        telemetry.mcp_model_visible_bytes_by_tool[tool] += model_visible_bytes
    if logical_bytes is not None:
        telemetry.mcp_logical_payload_observed_calls += 1
        telemetry.mcp_logical_payload_bytes += logical_bytes
        telemetry.mcp_logical_payload_bytes_by_tool[tool] += logical_bytes
    return text_response, structured_response


def _record_describe_and_attempt(
    telemetry: _AgentTranscriptTelemetry,
    tool: str,
    arguments: object,
    *,
    successful: bool,
    item: Mapping[str, Any],
    response: Mapping[str, Any] | None,
) -> None:
    if tool == "math.find":
        request = arguments.get("request") if isinstance(arguments, Mapping) else None
        if isinstance(request, Mapping) and request.get("op") == "inspect":
            telemetry.operation_describe_exact_calls += 1
        else:
            telemetry.operation_describe_index_calls += 1
    if tool != "math.run":
        return
    operation_id = (
        arguments.get("operation_id") if isinstance(arguments, Mapping) else None
    )
    payload = arguments.get("payload") if isinstance(arguments, Mapping) else None
    attempt = {
        "operation_id": operation_id if isinstance(operation_id, str) else None,
        "input": payload,
        "successful": successful,
    }
    if not successful:
        attempt.update(_mcp_failure_metadata(item, response))
    diagnostic_codes = _operation_diagnostic_codes(response)
    if diagnostic_codes:
        attempt["diagnostic_codes"] = diagnostic_codes
    diagnostics = _operation_diagnostics(response)
    if diagnostics:
        attempt["diagnostics"] = diagnostics
    telemetry.operation_attempts.append(attempt)
    if isinstance(operation_id, str):
        telemetry.operation_attempt_ids.append(operation_id)


def _mcp_failure_metadata(
    item: Mapping[str, Any],
    response: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Return bounded failure identity for exact recovery aggregation."""

    execution = response.get("execution") if isinstance(response, Mapping) else None
    execution_status = (
        execution.get("status") if isinstance(execution, Mapping) else None
    )
    item_status = item.get("status")
    terminal_status = (
        execution_status
        if isinstance(execution_status, str)
        else item_status
        if isinstance(item_status, str)
        else None
    )
    output = response.get("output") if isinstance(response, Mapping) else None
    error_identity = {
        "item_error": item.get("error"),
        "response_error": response.get("error")
        if isinstance(response, Mapping)
        else None,
        "output_error": output.get("error") if isinstance(output, Mapping) else None,
        "diagnostics": (
            response.get("diagnostics") if isinstance(response, Mapping) else None
        ),
    }
    _, error_digest = _mcp_call_signature("math.run.error", error_identity)
    request_validation_failure = _contains_value(
        item,
        field="code",
        accepted=set(_PARAMETER_ERROR_CODES),
    ) or _contains_value(
        response,
        field="stage",
        accepted={"request_validation"},
    )
    return {
        "terminal_status": terminal_status,
        "error_digest": error_digest,
        "request_validation_failure": request_validation_failure,
    }


def _operation_diagnostic_codes(
    response: Mapping[str, Any] | None,
) -> list[str]:
    if not isinstance(response, Mapping):
        return []
    codes: list[str] = []
    diagnostics = response.get("diagnostics")
    if isinstance(diagnostics, list):
        codes.extend(
            diagnostic["code"]
            for diagnostic in diagnostics
            if isinstance(diagnostic, Mapping)
            and isinstance(diagnostic.get("code"), str)
        )
    output = response.get("output")
    error = output.get("error") if isinstance(output, Mapping) else None
    if isinstance(error, Mapping) and isinstance(error.get("code"), str):
        codes.append(error["code"])
    return list(dict.fromkeys(codes))


def _bounded_operation_diagnostic(diagnostic: Mapping[str, Any]) -> dict[str, Any]:
    retained = {
        key: diagnostic[key]
        for key in ("code", "phase", "stage", "path")
        if isinstance(diagnostic.get(key), str)
    }
    details = diagnostic.get("details")
    validation_errors = (
        details.get("validation_errors") if isinstance(details, Mapping) else None
    )
    if isinstance(validation_errors, list):
        retained_errors = [
            {
                key: error[key]
                for key in ("path", "reason", "type")
                if isinstance(error.get(key), str)
            }
            for error in validation_errors
            if isinstance(error, Mapping)
        ]
        if retained_errors:
            retained["details"] = {"validation_errors": retained_errors}
    return retained


def _operation_diagnostics(
    response: Mapping[str, Any] | None,
) -> list[dict[str, Any]]:
    if not isinstance(response, Mapping):
        return []
    candidates: list[Mapping[str, Any]] = []
    diagnostics = response.get("diagnostics")
    if isinstance(diagnostics, list):
        candidates.extend(
            diagnostic for diagnostic in diagnostics if isinstance(diagnostic, Mapping)
        )
    output = response.get("output")
    error = output.get("error") if isinstance(output, Mapping) else None
    if isinstance(error, Mapping):
        candidates.append(error)
    retained: list[dict[str, Any]] = []
    seen: set[bytes] = set()
    for candidate in candidates:
        bounded = _bounded_operation_diagnostic(candidate)
        if not isinstance(bounded.get("code"), str):
            continue
        identity = canonicalize_json(bounded)
        if identity in seen:
            continue
        seen.add(identity)
        retained.append(bounded)
    return retained


def _mcp_call_failed(
    item: Mapping[str, Any],
    result: object,
    response: Mapping[str, Any] | None,
    status: object,
) -> bool:
    execution = response.get("execution") if isinstance(response, Mapping) else None
    return bool(
        (isinstance(status, str) and status in {"error", "failed"})
        or item.get("error")
        or (
            isinstance(result, Mapping)
            and (result.get("isError") is True or result.get("is_error") is True)
        )
        or (
            isinstance(response, Mapping) and isinstance(response.get("error"), Mapping)
        )
        or (
            isinstance(execution, Mapping)
            and execution.get("status") in {"CANCELLED", "ERROR", "TIMEOUT"}
        )
        or _contains_value(
            item,
            field="status",
            accepted={"CANCELLED", "ERROR", "TIMEOUT"},
        )
    )


def _operation_match_ids(matches: object) -> list[str]:
    if not isinstance(matches, list):
        return []
    return [
        match["operation_id"]
        for match in matches
        if isinstance(match, Mapping) and isinstance(match.get("operation_id"), str)
    ]


def _build_operation_description(
    arguments: Mapping[str, Any],
    response: Mapping[str, Any] | None,
) -> dict[str, Any]:
    cards = None
    if isinstance(response, Mapping):
        cards = response.get("matches", response.get("operations"))
    request = arguments.get("request")
    request = request if isinstance(request, Mapping) else {}
    return {
        "kind": (
            response.get("kind")
            if isinstance(response, Mapping) and isinstance(response.get("kind"), str)
            else None
        ),
        "query": (
            request.get("query") if isinstance(request.get("query"), str) else None
        ),
        "domain": (
            request.get("domain") if isinstance(request.get("domain"), str) else None
        ),
        "operation_id": (
            request.get("operation_id")
            if isinstance(request.get("operation_id"), str)
            else None
        ),
        "match_ids": _operation_match_ids(cards),
    }


def _record_operation_invocation(
    telemetry: _AgentTranscriptTelemetry,
    tool: str,
    arguments: object,
    response: Mapping[str, Any] | None,
) -> None:
    execution = response.get("execution") if isinstance(response, Mapping) else None
    if not (
        tool == "math.run"
        and isinstance(arguments, Mapping)
        and isinstance(arguments.get("operation_id"), str)
        and isinstance(response, Mapping)
        and response.get("operation_id") == arguments["operation_id"]
        and isinstance(execution, Mapping)
        and execution.get("status") == "COMPLETED"
    ):
        return
    telemetry.operation_ids.append(arguments["operation_id"])
    telemetry.operation_invocations.append(
        {
            "operation_id": arguments["operation_id"],
            "input": arguments.get("payload"),
            "output": response.get("output"),
        }
    )


def _record_successful_mcp_call(
    telemetry: _AgentTranscriptTelemetry,
    tool: str,
    arguments: object,
    response: Mapping[str, Any] | None,
) -> None:
    telemetry.successful_calls.append(tool)
    if tool == "math.find" and isinstance(arguments, Mapping):
        telemetry.operation_descriptions.append(
            _build_operation_description(arguments, response)
        )
    if (
        tool == "math.run"
        and isinstance(response, Mapping)
        and _contains_value(
            response.get("output"),
            field="status",
            accepted={"REJECTED"},
        )
    ):
        telemetry.operation_rejection_count += 1
    _record_operation_invocation(telemetry, tool, arguments, response)


def _process_mcp_tool_call(
    telemetry: _AgentTranscriptTelemetry,
    item: dict[str, Any],
) -> None:
    tool = item["tool"]
    telemetry.mcp_calls.append(tool)
    arguments = item.get("arguments")
    text_response, structured_response = _record_mcp_byte_metrics(telemetry, item, tool)
    telemetry.mcp_call_signatures[_mcp_call_signature(tool, arguments)] += 1
    result = item.get("result")
    response = structured_response or text_response
    status = item.get("status")
    failed = _mcp_call_failed(item, result, response, status)
    _record_describe_and_attempt(
        telemetry,
        tool,
        arguments,
        successful=not failed,
        item=item,
        response=response,
    )
    if failed:
        telemetry.tool_error_count += 1
    else:
        _record_successful_mcp_call(telemetry, tool, arguments, response)
    if _contains_value(
        item,
        field="code",
        accepted=set(_PARAMETER_ERROR_CODES),
    ):
        telemetry.parameter_error_count += 1


def _transcript_payload(telemetry: _AgentTranscriptTelemetry) -> dict[str, Any]:
    return {
        "mcp_calls": telemetry.mcp_calls,
        "shell_calls": telemetry.shell_calls,
        "usage": telemetry.usage,
        "tool_error_count": telemetry.tool_error_count,
        "parameter_error_count": telemetry.parameter_error_count,
        "operation_rejection_count": telemetry.operation_rejection_count,
        "successful_tool_calls": telemetry.successful_calls,
        "operation_attempt_ids": telemetry.operation_attempt_ids,
        "operation_attempts": telemetry.operation_attempts,
        "operation_ids": telemetry.operation_ids,
        "operation_invocations": telemetry.operation_invocations,
        "operation_descriptions": telemetry.operation_descriptions,
        "mcp_wire_bytes": telemetry.mcp_wire_bytes,
        "mcp_wire_bytes_by_tool": dict(
            sorted(telemetry.mcp_wire_bytes_by_tool.items())
        ),
        "mcp_model_visible_bytes": telemetry.mcp_model_visible_bytes,
        "mcp_model_visible_bytes_by_tool": dict(
            sorted(telemetry.mcp_model_visible_bytes_by_tool.items())
        ),
        "mcp_logical_payload_bytes": telemetry.mcp_logical_payload_bytes,
        "mcp_logical_payload_bytes_by_tool": dict(
            sorted(telemetry.mcp_logical_payload_bytes_by_tool.items())
        ),
        "mcp_logical_payload_observed_calls": (
            telemetry.mcp_logical_payload_observed_calls
        ),
        "mcp_resource_links_returned": telemetry.resource_telemetry.links_returned,
        "mcp_resource_link_uris": telemetry.resource_telemetry.link_uris,
        "mcp_resource_read_attempts": telemetry.resource_telemetry.read_attempts,
        "mcp_resource_read_uris": telemetry.resource_telemetry.read_uris,
        "mcp_resource_read_successes": telemetry.resource_telemetry.read_successes,
        "repeated_mcp_call_count": sum(
            count - 1 for count in telemetry.mcp_call_signatures.values() if count > 1
        ),
        "repeated_mcp_calls": [
            {
                "tool": tool,
                "argument_digest": argument_digest,
                "count": count,
            }
            for (tool, argument_digest), count in sorted(
                telemetry.mcp_call_signatures.items()
            )
            if count > 1
        ],
        "operation_describe_index_calls": telemetry.operation_describe_index_calls,
        "operation_describe_exact_calls": telemetry.operation_describe_exact_calls,
        **_operation_recovery_metrics(telemetry.operation_attempts),
    }


def parse_agent_transcript_bytes(payload: bytes) -> dict[str, Any]:
    """Parse already-read transcript bytes without reopening mutable evidence."""

    telemetry = _AgentTranscriptTelemetry()
    for line in payload.decode("utf-8", errors="strict").splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(event, dict):
            continue
        if event.get("type") == "turn.completed" and isinstance(
            event.get("usage"), dict
        ):
            telemetry.usage = event["usage"]
        item = event.get("item")
        if not isinstance(item, dict):
            _record_mcp_resource_telemetry(telemetry.resource_telemetry, item)
            continue
        if _is_command_execution_event(event, item):
            command = item.get("command")
            telemetry.shell_calls.append(command if isinstance(command, str) else "")
        if _is_mcp_tool_call_event(event, item):
            _process_mcp_tool_call(telemetry, item)
        _record_mcp_resource_telemetry(telemetry.resource_telemetry, item)
    return _transcript_payload(telemetry)


def parse_agent_transcript(path: Path) -> dict[str, Any]:
    """Return calls, usage, failures, and successful operation dataflow."""

    return parse_agent_transcript_bytes(path.read_bytes())
