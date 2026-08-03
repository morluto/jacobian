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
    if item_type != "mcp_resource_read" and tool not in _RESOURCE_READ_TOOL_NAMES:
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
    return bool(
        item.get("status") in {"error", "failed", "CANCELLED", "ERROR", "TIMEOUT"}
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


def parse_agent_transcript(path: Path) -> dict[str, Any]:
    """Return calls, usage, failures, and successful capability dataflow."""

    mcp_calls: list[str] = []
    successful_calls: list[str] = []
    capability_attempt_ids: list[str] = []
    capability_ids: list[str] = []
    capability_invocations: list[dict[str, Any]] = []
    capability_descriptions: list[dict[str, Any]] = []
    shell_calls: list[str] = []
    usage: dict[str, Any] | None = None
    tool_error_count = 0
    parameter_error_count = 0
    capability_rejection_count = 0
    mcp_wire_bytes = 0
    mcp_wire_bytes_by_tool: Counter[str] = Counter()
    mcp_model_visible_bytes = 0
    mcp_model_visible_bytes_by_tool: Counter[str] = Counter()
    mcp_logical_payload_bytes = 0
    mcp_logical_payload_bytes_by_tool: Counter[str] = Counter()
    mcp_logical_payload_observed_calls = 0
    mcp_call_signatures: Counter[tuple[str, str]] = Counter()
    resource_telemetry = _McpResourceTelemetry()
    capability_describe_index_calls = 0
    capability_describe_exact_calls = 0
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(event, dict):
            continue
        item = event.get("item")
        if (
            event.get("type") == "item.completed"
            and isinstance(item, dict)
            and item.get("type") == "command_execution"
        ):
            command = item.get("command")
            shell_calls.append(command if isinstance(command, str) else "")
        if (
            event.get("type") == "item.completed"
            and isinstance(item, dict)
            and item.get("type") == "mcp_tool_call"
            and isinstance(item.get("tool"), str)
        ):
            tool = item["tool"]
            mcp_calls.append(tool)
            arguments = item.get("arguments")
            wire_bytes = _mcp_wire_bytes(item)
            model_visible_bytes = _mcp_model_visible_bytes(item)
            text_response = _mcp_text_payload(item)
            structured_response = _mcp_structured_payload(item)
            logical_bytes = _mcp_logical_payload_bytes(
                item,
                text_payload=text_response,
                structured_payload=structured_response,
            )
            mcp_wire_bytes += wire_bytes
            mcp_model_visible_bytes += model_visible_bytes
            if wire_bytes:
                mcp_wire_bytes_by_tool[tool] += wire_bytes
            if model_visible_bytes:
                mcp_model_visible_bytes_by_tool[tool] += model_visible_bytes
            if logical_bytes is not None:
                mcp_logical_payload_observed_calls += 1
                mcp_logical_payload_bytes += logical_bytes
                mcp_logical_payload_bytes_by_tool[tool] += logical_bytes
            mcp_call_signatures[_mcp_call_signature(tool, arguments)] += 1
            if tool == "capability.describe":
                if isinstance(arguments, Mapping) and isinstance(
                    arguments.get("capability_id"), str
                ):
                    capability_describe_exact_calls += 1
                else:
                    capability_describe_index_calls += 1
            if (
                tool == "capability.invoke"
                and isinstance(arguments, Mapping)
                and isinstance(arguments.get("capability_id"), str)
            ):
                capability_attempt_ids.append(arguments["capability_id"])
            result = item.get("result")
            response = structured_response or text_response
            failed = bool(
                item.get("status") in {"error", "failed"}
                or item.get("error")
                or (
                    isinstance(result, Mapping)
                    and (
                        result.get("isError") is True or result.get("is_error") is True
                    )
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
            if failed:
                tool_error_count += 1
            else:
                successful_calls.append(tool)
                if tool == "capability.describe" and isinstance(arguments, Mapping):
                    matches = (
                        response.get("matches")
                        if isinstance(response, Mapping)
                        else None
                    )
                    capability_descriptions.append(
                        {
                            "kind": (
                                response.get("kind")
                                if isinstance(response, Mapping)
                                and isinstance(response.get("kind"), str)
                                else None
                            ),
                            "query": (
                                arguments.get("query")
                                if isinstance(arguments.get("query"), str)
                                else None
                            ),
                            "domain": (
                                arguments.get("domain")
                                if isinstance(arguments.get("domain"), str)
                                else None
                            ),
                            "mode": (
                                arguments.get("mode")
                                if isinstance(arguments.get("mode"), str)
                                else None
                            ),
                            "capability_id": (
                                arguments.get("capability_id")
                                if isinstance(arguments.get("capability_id"), str)
                                else None
                            ),
                            "match_ids": [
                                match["capability_id"]
                                for match in matches
                                if isinstance(match, Mapping)
                                and isinstance(match.get("capability_id"), str)
                            ]
                            if isinstance(matches, list)
                            else [],
                        }
                    )
                if (
                    tool == "capability.invoke"
                    and isinstance(response, Mapping)
                    and _contains_value(
                        response.get("output"),
                        field="status",
                        accepted={"REJECTED"},
                    )
                ):
                    capability_rejection_count += 1
                execution = (
                    response.get("execution") if isinstance(response, Mapping) else None
                )
                if (
                    tool == "capability.invoke"
                    and isinstance(arguments, Mapping)
                    and isinstance(arguments.get("capability_id"), str)
                    and isinstance(response, Mapping)
                    and response.get("capability_id") == arguments["capability_id"]
                    and isinstance(execution, Mapping)
                    and execution.get("status") == "COMPLETED"
                ):
                    capability_ids.append(arguments["capability_id"])
                    capability_invocations.append(
                        {
                            "capability_id": arguments["capability_id"],
                            "input": arguments.get("payload"),
                            "output": response.get("output"),
                            "artifact_uris": response.get("artifact_uris"),
                            "assurance": response.get("assurance"),
                            "completeness": response.get("completeness"),
                        }
                    )
            if _contains_value(
                item,
                field="code",
                accepted=set(_PARAMETER_ERROR_CODES),
            ):
                parameter_error_count += 1
        _record_mcp_resource_telemetry(resource_telemetry, item)
        if event.get("type") == "turn.completed" and isinstance(
            event.get("usage"), dict
        ):
            usage = event["usage"]
    return {
        "mcp_calls": mcp_calls,
        "shell_calls": shell_calls,
        "usage": usage,
        "tool_error_count": tool_error_count,
        "parameter_error_count": parameter_error_count,
        "capability_rejection_count": capability_rejection_count,
        "successful_tool_calls": successful_calls,
        "capability_attempt_ids": capability_attempt_ids,
        "capability_ids": capability_ids,
        "capability_invocations": capability_invocations,
        "capability_descriptions": capability_descriptions,
        "mcp_wire_bytes": mcp_wire_bytes,
        "mcp_wire_bytes_by_tool": dict(sorted(mcp_wire_bytes_by_tool.items())),
        "mcp_model_visible_bytes": mcp_model_visible_bytes,
        "mcp_model_visible_bytes_by_tool": dict(
            sorted(mcp_model_visible_bytes_by_tool.items())
        ),
        "mcp_logical_payload_bytes": mcp_logical_payload_bytes,
        "mcp_logical_payload_bytes_by_tool": dict(
            sorted(mcp_logical_payload_bytes_by_tool.items())
        ),
        "mcp_logical_payload_observed_calls": mcp_logical_payload_observed_calls,
        "mcp_resource_links_returned": resource_telemetry.links_returned,
        "mcp_resource_link_uris": resource_telemetry.link_uris,
        "mcp_resource_read_attempts": resource_telemetry.read_attempts,
        "mcp_resource_read_uris": resource_telemetry.read_uris,
        "mcp_resource_read_successes": resource_telemetry.read_successes,
        "mcp_resource_uri_preservation_attempts": (
            resource_telemetry.uri_preservation_attempts
        ),
        "mcp_resource_uri_preservation_successes": (
            resource_telemetry.uri_preservation_successes
        ),
        "mcp_resource_digest_preservation_successes": (
            resource_telemetry.digest_preservation_successes
        ),
        "repeated_mcp_call_count": sum(
            count - 1 for count in mcp_call_signatures.values() if count > 1
        ),
        "repeated_mcp_calls": [
            {
                "tool": tool,
                "argument_digest": argument_digest,
                "count": count,
            }
            for (tool, argument_digest), count in sorted(mcp_call_signatures.items())
            if count > 1
        ],
        "capability_describe_index_calls": capability_describe_index_calls,
        "capability_describe_exact_calls": capability_describe_exact_calls,
    }
