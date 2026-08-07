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
    uri_preservation_attempts: int = 0
    uri_preservation_successes: int = 0
    digest_preservation_successes: int = 0


@dataclass
class _ReasoningProtocolTelemetry:
    phase_counts: Counter[str] = field(default_factory=Counter)
    run_ids: set[str] = field(default_factory=set)
    before_call_ids: set[str] = field(default_factory=set)
    after_call_ids: set[str] = field(default_factory=set)
    bound_call_ids: set[str] = field(default_factory=set)
    bound_invoke_count: int = 0
    summary_characters: int = 0
    finalized_run_ids: set[str] = field(default_factory=set)
    unavailable_after_tool_count: int = 0
    reported_actual_mismatch_count: int = 0

    def record_attempt(self, tool: str, arguments: object) -> None:
        if tool != "math.run" or not isinstance(arguments, Mapping):
            return
        if isinstance(arguments.get("reasoning_run_id"), str) and isinstance(
            arguments.get("reasoning_call_id"), str
        ):
            self.bound_invoke_count += 1
            self.bound_call_ids.add(arguments["reasoning_call_id"])

    def record_write(
        self,
        tool: str,
        arguments: object,
        response: object,
    ) -> None:
        if tool != "reasoning.write" or not isinstance(arguments, Mapping):
            return
        phase = arguments.get("phase")
        if isinstance(phase, str):
            self.phase_counts[phase] += 1
        summary = arguments.get("summary")
        if isinstance(summary, str):
            self.summary_characters += len(summary)
        if (
            phase == "AFTER_TOOL"
            and arguments.get("interpretation_status") == "RESULT_UNAVAILABLE"
        ):
            self.unavailable_after_tool_count += 1
        if isinstance(response, Mapping):
            self.reported_actual_mismatch_count += sum(
                response.get(field) is False
                for field in (
                    "execution_status_matches",
                    "assurance_level_matches",
                    "completeness_status_matches",
                )
            )
            if phase == "FINAL" and response.get("state") == "FINALIZED":
                response_run_id = response.get("run_id")
                if isinstance(response_run_id, str):
                    self.finalized_run_ids.add(response_run_id)
        self._record_identity(phase, arguments, response)

    def _record_identity(
        self,
        phase: object,
        arguments: Mapping[str, Any],
        response: object,
    ) -> None:
        run_id = arguments.get("run_id")
        if isinstance(run_id, str):
            self.run_ids.add(run_id)
        if isinstance(response, Mapping) and isinstance(response.get("run_id"), str):
            self.run_ids.add(response["run_id"])
        call_id = arguments.get("call_id")
        if phase == "AFTER_TOOL" and isinstance(call_id, str):
            self.after_call_ids.add(call_id)
        if (
            phase == "BEFORE_TOOL"
            and isinstance(response, Mapping)
            and isinstance(response.get("call_id"), str)
        ):
            self.before_call_ids.add(response["call_id"])

    def payload(self) -> dict[str, int | str]:
        missing_after = self.before_call_ids - self.after_call_ids
        call_sets_match = (
            self.before_call_ids == self.after_call_ids == self.bound_call_ids
        )
        complete = (
            self.phase_counts["PLAN"] == 1
            and self.phase_counts["FINAL"] == 1
            and len(self.run_ids) == 1
            and len(self.finalized_run_ids) == 1
            and call_sets_match
        )
        return {
            "status": "COMPLETE" if complete else "INCOMPLETE",
            "plan_count": self.phase_counts["PLAN"],
            "before_tool_count": self.phase_counts["BEFORE_TOOL"],
            "after_tool_count": self.phase_counts["AFTER_TOOL"],
            "final_count": self.phase_counts["FINAL"],
            "run_count": len(self.run_ids),
            "bound_invoke_count": self.bound_invoke_count,
            "missing_after_tool_count": len(missing_after),
            "pending_call_count": len(
                (self.before_call_ids | self.bound_call_ids) - self.after_call_ids
            ),
            "unavailable_after_tool_count": self.unavailable_after_tool_count,
            "reported_actual_mismatch_count": self.reported_actual_mismatch_count,
            "summary_characters": self.summary_characters,
        }


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


def _mcp_resource_identity_preserved(
    item: Mapping[str, Any],
    uri: str,
) -> tuple[bool, bool]:
    payload = _mcp_structured_payload(item) or _mcp_text_payload(item)
    if payload is None:
        return False, False
    uri_preserved = payload.get("artifact_uri") == uri
    manifest = payload.get("manifest")
    digest_preserved = False
    if isinstance(manifest, Mapping) and uri.startswith("artifact://sha256/"):
        try:
            manifest_digest = (
                "sha256:" + hashlib.sha256(canonicalize_json(manifest)).hexdigest()
            )
            payload_digest = manifest.get("payload_digest")
            actual_payload_digest = (
                "sha256:"
                + hashlib.sha256(canonicalize_json(payload["payload"])).hexdigest()
            )
        except (TypeError, ValueError):
            manifest_digest = None
            actual_payload_digest = None
        digest_preserved = (
            uri_preserved
            and manifest_digest == "sha256:" + uri.removeprefix("artifact://sha256/")
            and payload_digest == actual_payload_digest
        )
    return uri_preserved, digest_preserved


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
    if resource_read_uri in telemetry.link_uris:
        telemetry.uri_preservation_attempts += 1
    if read_failed:
        return
    telemetry.read_successes += 1
    uri_preserved, digest_preserved = _mcp_resource_identity_preserved(
        item,
        resource_read_uri,
    )
    if resource_read_uri in telemetry.link_uris and uri_preserved:
        telemetry.uri_preservation_successes += 1
    if digest_preserved:
        telemetry.digest_preservation_successes += 1


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


def _reasoning_tool_name(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    normalized = value.replace("__", ".").replace("_", ".").lower()
    if normalized.endswith("reasoning.write"):
        return "reasoning.write"
    if normalized.endswith("math.run"):
        return "math.run"
    return None


def _atif_mapping_response(value: Mapping[str, Any]) -> dict[str, Any] | None:
    for key in ("structured_content", "structuredContent"):
        response = value.get(key)
        if isinstance(response, dict):
            return response
    if value.get("type") == "text" and isinstance(value.get("text"), str):
        return _atif_response(value["text"])
    content = value.get("content")
    if isinstance(content, list):
        response = _atif_response(content)
        if response is not None:
            return response
    return dict(value)


def _atif_response(value: object) -> dict[str, Any] | None:
    if isinstance(value, str):
        try:
            return _atif_response(json.loads(value))
        except json.JSONDecodeError:
            return None
    if isinstance(value, list):
        for item in value:
            response = _atif_response(item)
            if response is not None:
                return response
        return None
    if not isinstance(value, Mapping):
        return None
    return _atif_mapping_response(value)


def _parse_atif_reasoning_protocol(value: object) -> dict[str, int | str]:
    telemetry = _ReasoningProtocolTelemetry()
    if not isinstance(value, Mapping) or not isinstance(value.get("steps"), list):
        return telemetry.payload()
    for step in value["steps"]:
        if not isinstance(step, Mapping):
            continue
        observation = step.get("observation")
        raw_results = (
            observation.get("results") if isinstance(observation, Mapping) else None
        )
        results = raw_results if isinstance(raw_results, list) else []
        responses = {
            item["source_call_id"]: _atif_response(item.get("content"))
            for item in results
            if isinstance(item, Mapping) and isinstance(item.get("source_call_id"), str)
        }
        calls = step.get("tool_calls")
        if not isinstance(calls, list):
            continue
        for call in calls:
            if not isinstance(call, Mapping):
                continue
            tool = _reasoning_tool_name(call.get("function_name"))
            arguments = call.get("arguments")
            if tool is None or not isinstance(arguments, Mapping):
                continue
            telemetry.record_attempt(tool, arguments)
            call_id = call.get("tool_call_id")
            response = responses.get(call_id) if isinstance(call_id, str) else None
            if response is not None:
                telemetry.record_write(tool, arguments, response)
    return telemetry.payload()


def parse_reasoning_protocol_trace(path: Path) -> dict[str, int | str]:
    """Parse reasoning protocol facts from Codex JSONL or Harbor ATIF JSON."""

    if path.suffix == ".jsonl":
        protocol = parse_agent_transcript(path).get("reasoning_protocol")
        if not isinstance(protocol, dict):
            raise ValueError("agent transcript omitted reasoning protocol telemetry")
        return {
            str(key): item
            for key, item in protocol.items()
            if isinstance(item, int | str)
        }
    value = json.loads(path.read_text(encoding="utf-8"))
    return _parse_atif_reasoning_protocol(value)


@dataclass
class _AgentTranscriptTelemetry:
    mcp_calls: list[str] = field(default_factory=list)
    successful_calls: list[str] = field(default_factory=list)
    capability_attempt_ids: list[str] = field(default_factory=list)
    capability_ids: list[str] = field(default_factory=list)
    capability_invocations: list[dict[str, Any]] = field(default_factory=list)
    capability_descriptions: list[dict[str, Any]] = field(default_factory=list)
    shell_calls: list[str] = field(default_factory=list)
    usage: dict[str, Any] | None = None
    tool_error_count: int = 0
    parameter_error_count: int = 0
    capability_rejection_count: int = 0
    mcp_wire_bytes: int = 0
    mcp_wire_bytes_by_tool: Counter[str] = field(default_factory=Counter)
    mcp_model_visible_bytes: int = 0
    mcp_model_visible_bytes_by_tool: Counter[str] = field(default_factory=Counter)
    mcp_logical_payload_bytes: int = 0
    mcp_logical_payload_bytes_by_tool: Counter[str] = field(default_factory=Counter)
    mcp_logical_payload_observed_calls: int = 0
    mcp_call_signatures: Counter[tuple[str, str]] = field(default_factory=Counter)
    capability_describe_index_calls: int = 0
    capability_describe_exact_calls: int = 0
    reasoning_telemetry: _ReasoningProtocolTelemetry = field(
        default_factory=_ReasoningProtocolTelemetry
    )
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
) -> None:
    if tool == "math.find":
        if isinstance(arguments, Mapping) and isinstance(
            arguments.get("capability_id"), str
        ):
            telemetry.capability_describe_exact_calls += 1
        else:
            telemetry.capability_describe_index_calls += 1
    if (
        tool == "math.run"
        and isinstance(arguments, Mapping)
        and isinstance(arguments.get("capability_id"), str)
    ):
        telemetry.capability_attempt_ids.append(arguments["capability_id"])


def _mcp_call_failed(
    item: Mapping[str, Any],
    result: object,
    text_response: Mapping[str, Any] | None,
    status: object,
) -> bool:
    return bool(
        (isinstance(status, str) and status in {"error", "failed"})
        or item.get("error")
        or (
            isinstance(result, Mapping)
            and (result.get("isError") is True or result.get("is_error") is True)
        )
        or (
            isinstance(text_response, Mapping)
            and isinstance(text_response.get("error"), Mapping)
        )
        or _contains_value(
            item,
            field="status",
            accepted={"CANCELLED", "ERROR", "TIMEOUT"},
        )
    )


def _capability_match_ids(matches: object) -> list[str]:
    if not isinstance(matches, list):
        return []
    return [
        match["capability_id"]
        for match in matches
        if isinstance(match, Mapping) and isinstance(match.get("capability_id"), str)
    ]


def _build_capability_description(
    arguments: Mapping[str, Any],
    response: Mapping[str, Any] | None,
) -> dict[str, Any]:
    matches = response.get("matches") if isinstance(response, Mapping) else None
    return {
        "kind": (
            response.get("kind")
            if isinstance(response, Mapping) and isinstance(response.get("kind"), str)
            else None
        ),
        "query": (
            arguments.get("query") if isinstance(arguments.get("query"), str) else None
        ),
        "domain": (
            arguments.get("domain")
            if isinstance(arguments.get("domain"), str)
            else None
        ),
        "mode": (
            arguments.get("mode") if isinstance(arguments.get("mode"), str) else None
        ),
        "capability_id": (
            arguments.get("capability_id")
            if isinstance(arguments.get("capability_id"), str)
            else None
        ),
        "match_ids": _capability_match_ids(matches),
    }


def _record_capability_invocation(
    telemetry: _AgentTranscriptTelemetry,
    tool: str,
    arguments: object,
    response: Mapping[str, Any] | None,
) -> None:
    execution = response.get("execution") if isinstance(response, Mapping) else None
    if not (
        tool == "math.run"
        and isinstance(arguments, Mapping)
        and isinstance(arguments.get("capability_id"), str)
        and isinstance(response, Mapping)
        and response.get("capability_id") == arguments["capability_id"]
        and isinstance(execution, Mapping)
        and execution.get("status") == "COMPLETED"
    ):
        return
    telemetry.capability_ids.append(arguments["capability_id"])
    telemetry.capability_invocations.append(
        {
            "capability_id": arguments["capability_id"],
            "input": arguments.get("payload"),
            "output": response.get("output"),
            "artifact_uris": response.get("artifact_uris"),
            "assurance": response.get("assurance"),
            "completeness": response.get("completeness"),
        }
    )


def _record_successful_mcp_call(
    telemetry: _AgentTranscriptTelemetry,
    tool: str,
    arguments: object,
    response: Mapping[str, Any] | None,
) -> None:
    telemetry.successful_calls.append(tool)
    telemetry.reasoning_telemetry.record_write(tool, arguments, response)
    if tool == "math.find" and isinstance(arguments, Mapping):
        telemetry.capability_descriptions.append(
            _build_capability_description(arguments, response)
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
        telemetry.capability_rejection_count += 1
    _record_capability_invocation(telemetry, tool, arguments, response)


def _process_mcp_tool_call(
    telemetry: _AgentTranscriptTelemetry,
    item: dict[str, Any],
) -> None:
    tool = item["tool"]
    telemetry.mcp_calls.append(tool)
    arguments = item.get("arguments")
    telemetry.reasoning_telemetry.record_attempt(tool, arguments)
    text_response, structured_response = _record_mcp_byte_metrics(telemetry, item, tool)
    telemetry.mcp_call_signatures[_mcp_call_signature(tool, arguments)] += 1
    _record_describe_and_attempt(telemetry, tool, arguments)
    result = item.get("result")
    response = structured_response or text_response
    status = item.get("status")
    if _mcp_call_failed(item, result, text_response, status):
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
        "capability_rejection_count": telemetry.capability_rejection_count,
        "successful_tool_calls": telemetry.successful_calls,
        "capability_attempt_ids": telemetry.capability_attempt_ids,
        "capability_ids": telemetry.capability_ids,
        "capability_invocations": telemetry.capability_invocations,
        "capability_descriptions": telemetry.capability_descriptions,
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
        "mcp_resource_uri_preservation_attempts": (
            telemetry.resource_telemetry.uri_preservation_attempts
        ),
        "mcp_resource_uri_preservation_successes": (
            telemetry.resource_telemetry.uri_preservation_successes
        ),
        "mcp_resource_digest_preservation_successes": (
            telemetry.resource_telemetry.digest_preservation_successes
        ),
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
        "capability_describe_index_calls": telemetry.capability_describe_index_calls,
        "capability_describe_exact_calls": telemetry.capability_describe_exact_calls,
        "reasoning_protocol": telemetry.reasoning_telemetry.payload(),
    }


def parse_agent_transcript(path: Path) -> dict[str, Any]:
    """Return calls, usage, failures, and successful capability dataflow."""

    telemetry = _AgentTranscriptTelemetry()
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(event, dict):
            continue
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
        if event.get("type") == "turn.completed" and isinstance(
            event.get("usage"), dict
        ):
            telemetry.usage = event["usage"]
    return _transcript_payload(telemetry)
