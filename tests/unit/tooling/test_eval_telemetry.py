from __future__ import annotations

import hashlib
import json
from pathlib import Path

from jacobian.canonical import canonicalize_json
from jacobian.eval.telemetry import (
    parse_agent_transcript,
    parse_agent_transcript_bytes,
)


def _tool_event(
    tool: str,
    arguments: dict[str, object],
    response: dict[str, object],
) -> dict[str, object]:
    return {
        "type": "item.completed",
        "item": {
            "type": "mcp_tool_call",
            "tool": tool,
            "arguments": arguments,
            "status": "completed",
            "result": {
                "isError": False,
                "content": [{"type": "text", "text": json.dumps(response)}],
            },
        },
    }


def test_agent_telemetry_records_itemless_turn_usage(tmp_path: Path) -> None:
    usage = {
        "input_tokens": 13550,
        "cached_input_tokens": 13056,
        "cache_write_input_tokens": 0,
        "output_tokens": 2522,
        "reasoning_output_tokens": 1552,
    }
    transcript = tmp_path / "transcript.jsonl"
    transcript.write_text(
        json.dumps({"type": "turn.completed", "usage": usage}) + "\n",
        encoding="utf-8",
    )

    telemetry = parse_agent_transcript(transcript)

    assert telemetry["usage"] == usage


def test_agent_telemetry_parses_preverified_bytes_without_a_path() -> None:
    payload = (
        json.dumps(
            {
                "type": "turn.completed",
                "usage": {"input_tokens": 7, "output_tokens": 3},
            }
        ).encode()
        + b"\n"
    )

    telemetry = parse_agent_transcript_bytes(payload)

    assert telemetry["usage"] == {"input_tokens": 7, "output_tokens": 3}


def test_agent_telemetry_preserves_discovery_to_invocation_dataflow(
    tmp_path: Path,
) -> None:
    events = [
        _tool_event(
            "math.find",
            {
                "request": {
                    "op": "search",
                    "query": "find a graph counterexample",
                    "domain": "graph",
                }
            },
            {
                "kind": "discovery",
                "matches": [
                    {"capability_id": "graph.search.atlas"},
                    {"capability_id": "graph.compute.properties"},
                ],
            },
        ),
        _tool_event(
            "math.find",
            {
                "request": {
                    "op": "inspect",
                    "capability_id": "graph.search.atlas",
                }
            },
            {
                "kind": "capability",
                "capability": {"capability_id": "graph.search.atlas"},
            },
        ),
        _tool_event(
            "math.run",
            {
                "capability_id": "graph.search.atlas",
                "payload": {"order": 7},
            },
            {
                "capability_id": "graph.search.atlas",
                "execution": {"status": "COMPLETED"},
                "output": {"status": "FOUND"},
                "artifact_uris": [],
                "verification_record_uri": None,
            },
        ),
    ]
    transcript = tmp_path / "transcript.jsonl"
    transcript.write_text(
        "\n".join(json.dumps(event) for event in events) + "\n",
        encoding="utf-8",
    )

    telemetry = parse_agent_transcript(transcript)

    assert telemetry["capability_descriptions"] == [
        {
            "kind": "discovery",
            "query": "find a graph counterexample",
            "domain": "graph",
            "capability_id": None,
            "match_ids": [
                "graph.search.atlas",
                "graph.compute.properties",
            ],
        },
        {
            "kind": "capability",
            "query": None,
            "domain": None,
            "capability_id": "graph.search.atlas",
            "match_ids": [],
        },
    ]
    assert telemetry["capability_attempt_ids"] == ["graph.search.atlas"]
    assert telemetry["capability_attempts"] == [
        {
            "capability_id": "graph.search.atlas",
            "input": {"order": 7},
            "successful": True,
        }
    ]
    assert telemetry["capability_ids"] == ["graph.search.atlas"]


def test_agent_telemetry_retains_failed_math_run_attempts(tmp_path: Path) -> None:
    failed = {
        "type": "item.completed",
        "item": {
            "type": "mcp_tool_call",
            "tool": "math.run",
            "arguments": {"payload": {"malformed": True}},
            "status": "failed",
            "result": None,
            "error": "invalid request",
        },
    }
    succeeded = _tool_event(
        "math.run",
        {"capability_id": "lean.check", "payload": {"statement": "True"}},
        {
            "capability_id": "lean.check",
            "execution": {"status": "COMPLETED"},
            "output": {"conclusion": "UNKNOWN"},
        },
    )
    domain_failure = _tool_event(
        "math.run",
        {
            "capability_id": "lean.retrieve.premises",
            "payload": {"statement": "True", "proof_prefix": ["by"]},
        },
        {
            "capability_id": "lean.retrieve.premises",
            "execution": {"status": "ERROR"},
            "output": {
                "error": {
                    "code": "INVALID_LEAN_RETRIEVAL_REQUEST",
                    "stage": "request_validation",
                    "message": "proof_prefix must not include by",
                }
            },
            "diagnostics": [
                {
                    "code": "INVALID_LEAN_RETRIEVAL_REQUEST",
                    "stage": "request_validation",
                    "message": "proof_prefix must not include by",
                }
            ],
        },
    )
    transcript = tmp_path / "transcript.jsonl"
    transcript.write_text(
        "\n".join(json.dumps(event) for event in (failed, domain_failure, succeeded))
        + "\n",
        encoding="utf-8",
    )

    telemetry = parse_agent_transcript(transcript)

    assert telemetry["capability_attempts"] == [
        {
            "capability_id": None,
            "input": {"malformed": True},
            "successful": False,
            "terminal_status": "failed",
            "error_digest": "sha256:"
            + hashlib.sha256(
                canonicalize_json(
                    {
                        "item_error": "invalid request",
                        "response_error": None,
                        "output_error": None,
                        "diagnostics": None,
                    }
                )
            ).hexdigest(),
            "request_validation_failure": False,
        },
        {
            "capability_id": "lean.retrieve.premises",
            "input": {"statement": "True", "proof_prefix": ["by"]},
            "successful": False,
            "terminal_status": "ERROR",
            "error_digest": "sha256:"
            + hashlib.sha256(
                canonicalize_json(
                    {
                        "item_error": None,
                        "response_error": None,
                        "output_error": {
                            "code": "INVALID_LEAN_RETRIEVAL_REQUEST",
                            "stage": "request_validation",
                            "message": "proof_prefix must not include by",
                        },
                        "diagnostics": [
                            {
                                "code": "INVALID_LEAN_RETRIEVAL_REQUEST",
                                "stage": "request_validation",
                                "message": "proof_prefix must not include by",
                            }
                        ],
                    }
                )
            ).hexdigest(),
            "request_validation_failure": True,
            "diagnostic_codes": ["INVALID_LEAN_RETRIEVAL_REQUEST"],
            "diagnostics": [
                {
                    "code": "INVALID_LEAN_RETRIEVAL_REQUEST",
                    "stage": "request_validation",
                }
            ],
        },
        {
            "capability_id": "lean.check",
            "input": {"statement": "True"},
            "successful": True,
        },
    ]
    assert telemetry["capability_attempt_ids"] == [
        "lean.retrieve.premises",
        "lean.check",
    ]
    assert telemetry["capability_ids"] == ["lean.check"]


def test_agent_telemetry_records_empty_payload_and_exact_repeated_errors(
    tmp_path: Path,
) -> None:
    invalid = {
        "capability_id": "lean.check",
        "execution": {"status": "ERROR"},
        "output": {
            "error": {
                "code": "INVALID_REQUEST",
                "stage": "request_validation",
                "path": "$",
            }
        },
    }
    events = [
        _tool_event(
            "math.run", {"capability_id": "lean.check", "payload": {}}, invalid
        ),
        _tool_event(
            "math.run", {"capability_id": "lean.check", "payload": {}}, invalid
        ),
        _tool_event(
            "math.run",
            {
                "capability_id": "lean.retrieve.premises",
                "payload": {"statement": "True", "proof_prefix": ["by"]},
            },
            {
                "capability_id": "lean.retrieve.premises",
                "execution": {"status": "ERROR"},
                "output": {
                    "error": {
                        "code": "INVALID_LEAN_RETRIEVAL_REQUEST",
                        "stage": "request_validation",
                    }
                },
            },
        ),
    ]
    transcript = tmp_path / "transcript.jsonl"
    transcript.write_text(
        "\n".join(json.dumps(event) for event in events) + "\n",
        encoding="utf-8",
    )

    telemetry = parse_agent_transcript(transcript)

    assert telemetry["empty_payload_probe_count"] == 2
    assert telemetry["failed_operation_attempt_count"] == 3
    assert telemetry["repeated_error_count"] == 1


def test_agent_telemetry_does_not_count_successful_empty_payload_as_probe(
    tmp_path: Path,
) -> None:
    event = _tool_event(
        "math.run",
        {"capability_id": "example.all_defaults", "payload": {}},
        {
            "capability_id": "example.all_defaults",
            "execution": {"status": "COMPLETED"},
            "output": {"value": 1},
        },
    )
    transcript = tmp_path / "transcript.jsonl"
    transcript.write_text(json.dumps(event) + "\n", encoding="utf-8")

    telemetry = parse_agent_transcript(transcript)

    assert telemetry["empty_payload_probe_count"] == 0


def test_agent_telemetry_distinguishes_terminal_failure_identity(
    tmp_path: Path,
) -> None:
    events = []
    for status, message in (
        ("TIMEOUT", "deadline exceeded"),
        ("CANCELLED", "request cancelled"),
        ("TIMEOUT", "deadline exceeded"),
    ):
        events.append(
            _tool_event(
                "math.run",
                {"capability_id": "lean.check", "payload": {"statement": "True"}},
                {
                    "capability_id": "lean.check",
                    "execution": {"status": status},
                    "output": {"error": {"code": status, "message": message}},
                },
            )
        )
    transcript = tmp_path / "transcript.jsonl"
    transcript.write_text(
        "\n".join(json.dumps(event) for event in events) + "\n",
        encoding="utf-8",
    )

    telemetry = parse_agent_transcript(transcript)

    assert telemetry["failed_operation_attempt_count"] == 3
    assert telemetry["repeated_error_count"] == 1


def test_agent_telemetry_counts_response_bytes_and_repeated_calls(
    tmp_path: Path,
) -> None:
    events = [
        _tool_event(
            "math.find",
            {"request": {"op": "search", "query": "SAT materialization"}},
            {"matches": [{"capability_id": "sat.cnf.materialize"}]},
        ),
        _tool_event(
            "math.find",
            {"request": {"op": "search", "query": "SAT materialization"}},
            {"matches": [{"capability_id": "sat.cnf.materialize"}]},
        ),
        _tool_event(
            "math.find",
            {
                "request": {
                    "op": "inspect",
                    "capability_id": "sat.cnf.materialize",
                }
            },
            {"capability": {"capability_id": "sat.cnf.materialize"}},
        ),
    ]
    transcript = tmp_path / "transcript.jsonl"
    transcript.write_text(
        "\n".join(json.dumps(event) for event in events) + "\n",
        encoding="utf-8",
    )

    telemetry = parse_agent_transcript(transcript)

    assert telemetry["mcp_wire_bytes"] > 0
    assert (
        telemetry["mcp_wire_bytes_by_tool"]["math.find"] == telemetry["mcp_wire_bytes"]
    )
    assert telemetry["repeated_mcp_call_count"] == 1
    assert telemetry["repeated_mcp_calls"][0]["tool"] == "math.find"
    assert telemetry["repeated_mcp_calls"][0]["count"] == 2
    assert telemetry["capability_describe_index_calls"] == 2
    assert telemetry["capability_describe_exact_calls"] == 1
    assert telemetry["mcp_model_visible_bytes"] > 0
    assert telemetry["mcp_logical_payload_observed_calls"] == 3


def test_agent_telemetry_ignores_non_string_mcp_status(tmp_path: Path) -> None:
    uri = "artifact://sha256/" + ("a" * 64)
    event = {
        "type": "item.completed",
        "item": {
            "type": "mcp_tool_call",
            "tool": "resources/read",
            "arguments": {"uri": uri},
            "status": [],
            "result": {},
        },
    }
    transcript = tmp_path / "transcript.jsonl"
    transcript.write_text(json.dumps(event) + "\n", encoding="utf-8")

    telemetry = parse_agent_transcript(transcript)

    assert telemetry["tool_error_count"] == 0
    assert telemetry["successful_tool_calls"] == ["resources/read"]
    assert telemetry["mcp_resource_read_attempts"] == 1
    assert telemetry["mcp_resource_read_successes"] == 1


def test_agent_telemetry_ignores_non_string_mcp_tool(tmp_path: Path) -> None:
    event = {
        "type": "item.completed",
        "item": {
            "type": "not_resource_read",
            "tool": [],
            "arguments": {"uri": "artifact://sha256/" + ("a" * 64)},
        },
    }
    transcript = tmp_path / "transcript.jsonl"
    transcript.write_text(json.dumps(event) + "\n", encoding="utf-8")

    telemetry = parse_agent_transcript(transcript)

    assert telemetry["mcp_calls"] == []
    assert telemetry["mcp_resource_read_attempts"] == 0


def test_agent_telemetry_separates_wire_model_and_logical_invocation_bytes(
    tmp_path: Path,
) -> None:
    canonical = {
        "capability_id": "graph.search.atlas",
        "execution": {"status": "COMPLETED"},
        "output": {"status": "FOUND", "graphs": [{"edges": [[0, 1]]}]},
        "artifact_uris": ["artifact://sha256/" + ("a" * 64)],
        "verification_record_uri": None,
    }
    event = {
        "type": "item.completed",
        "item": {
            "type": "mcp_tool_call",
            "tool": "math.run",
            "arguments": {
                "capability_id": "graph.search.atlas",
                "payload": {"order": 2},
            },
            "status": "completed",
            "result": {
                "isError": False,
                "content": [{"type": "text", "text": json.dumps(canonical)}],
                "structured_content": canonical,
            },
        },
    }
    transcript = tmp_path / "transcript.jsonl"
    transcript.write_text(json.dumps(event) + "\n", encoding="utf-8")

    telemetry = parse_agent_transcript(transcript)

    assert telemetry["mcp_logical_payload_bytes"] == len(
        json.dumps(
            canonical, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode()
    )
    assert telemetry["mcp_logical_payload_observed_calls"] == 1
    assert telemetry["mcp_wire_bytes"] > telemetry["mcp_model_visible_bytes"]
    assert telemetry["capability_invocations"][0]["output"] == canonical["output"]


def test_agent_telemetry_tracks_resource_link_follow_through_and_identity(
    tmp_path: Path,
) -> None:
    def read_event(seed: str) -> tuple[str, dict[str, object]]:
        payload = {"seed": seed}
        manifest = {
            "manifest_version": "1",
            "object_digest": "sha256:" + seed * 64,
            "payload_digest": "sha256:"
            + hashlib.sha256(canonicalize_json(payload)).hexdigest(),
            "schema_uri": "artifact://sha256/" + ("c" * 64),
            "semantics_uri": "artifact://sha256/" + ("d" * 64),
            "canonicalizer_digest": "sha256:" + ("e" * 64),
            "parents": [],
            "summary": "",
        }
        uri = (
            "artifact://sha256/"
            + hashlib.sha256(canonicalize_json(manifest)).hexdigest()
        )
        return uri, _tool_event(
            "resources/read",
            {"uri": uri},
            {"artifact_uri": uri, "manifest": manifest, "payload": payload},
        )

    uri, first_read = read_event("a")
    first_result = first_read["item"]["result"]
    first_result["contents"] = first_result.pop("content")
    unnecessary_uri, second_read = read_event("b")
    malformed_uri = "artifact://sha256/" + ("f" * 64)

    def link_event(link_uri: str) -> dict[str, object]:
        content = [{"type": "resource_link", "uri": link_uri}]
        return {
            "type": "item.completed",
            "item": {
                "type": "mcp_tool_call",
                "tool": "math.run",
                "arguments": {"capability_id": "graph.search.atlas"},
                "status": "completed",
                "result": {
                    "isError": False,
                    "content": content,
                },
            },
        }

    transcript = tmp_path / "transcript.jsonl"
    transcript.write_text(
        "\n".join(
            json.dumps(event)
            for event in (
                link_event(uri),
                link_event(unnecessary_uri),
                link_event(malformed_uri),
                first_read,
                second_read,
                _tool_event(
                    "resources/read",
                    {"uri": malformed_uri},
                    {"not_an_artifact": True},
                ),
            )
        )
        + "\n",
        encoding="utf-8",
    )

    telemetry = parse_agent_transcript(transcript)

    assert telemetry["mcp_resource_links_returned"] == 3
    assert telemetry["mcp_resource_link_uris"] == [uri, unnecessary_uri, malformed_uri]
    assert telemetry["mcp_resource_read_attempts"] == 3
    assert telemetry["mcp_resource_read_uris"] == [uri, unnecessary_uri, malformed_uri]
    assert telemetry["mcp_resource_read_successes"] == 3
    assert telemetry["mcp_resource_uri_preservation_attempts"] == 3
    assert telemetry["mcp_resource_uri_preservation_successes"] == 2
    assert telemetry["mcp_resource_digest_preservation_successes"] == 2


def test_agent_telemetry_handles_non_hashable_resource_tool_field(
    tmp_path: Path,
) -> None:
    uri = "artifact://sha256/" + ("f" * 64)
    malformed_tool_event = {
        "type": "item.completed",
        "item": {
            "type": "mcp_tool_call",
            "tool": ["resources", "read"],
            "status": "completed",
            "arguments": {"uri": uri},
            "result": {"isError": False, "content": []},
        },
    }
    valid_read = _tool_event(
        "resources/read",
        {"uri": uri},
        {
            "artifact_uri": uri,
            "manifest": {"payload_digest": "sha256:" + ("e" * 64)},
            "payload": {},
        },
    )
    transcript = tmp_path / "transcript.jsonl"
    transcript.write_text(
        "\n".join(json.dumps(event) for event in (malformed_tool_event, valid_read))
        + "\n",
        encoding="utf-8",
    )

    telemetry = parse_agent_transcript(transcript)

    assert telemetry["mcp_resource_read_attempts"] == 1
    assert telemetry["mcp_resource_read_uris"] == [uri]
    assert telemetry["mcp_resource_read_successes"] == 1
