from __future__ import annotations

import hashlib
import json
from pathlib import Path

from benchmarks.tooling.codex_telemetry import (
    parse_agent_transcript,
    parse_agent_transcript_bytes,
)

from jacobian.canonical import canonicalize_json


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
                    {"operation_id": "graph.search.atlas"},
                    {"operation_id": "graph.compute.properties"},
                ],
            },
        ),
        _tool_event(
            "math.find",
            {
                "request": {
                    "op": "inspect",
                    "operation_id": "graph.search.atlas",
                }
            },
            {
                "kind": "operation",
                "operation": {"operation_id": "graph.search.atlas"},
            },
        ),
        _tool_event(
            "math.run",
            {
                "operation_id": "graph.search.atlas",
                "payload": {"order": 7},
            },
            {
                "operation_id": "graph.search.atlas",
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

    assert telemetry["operation_descriptions"] == [
        {
            "kind": "discovery",
            "query": "find a graph counterexample",
            "domain": "graph",
            "operation_id": None,
            "match_ids": [
                "graph.search.atlas",
                "graph.compute.properties",
            ],
        },
        {
            "kind": "operation",
            "query": None,
            "domain": None,
            "operation_id": "graph.search.atlas",
            "match_ids": [],
        },
    ]
    assert telemetry["operation_attempt_ids"] == ["graph.search.atlas"]
    assert telemetry["operation_attempts"] == [
        {
            "operation_id": "graph.search.atlas",
            "input": {"order": 7},
            "successful": True,
        }
    ]
    assert telemetry["operation_ids"] == ["graph.search.atlas"]


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
        {
            "operation_id": "smt.solve",
            "payload": {"logic": "QF_LIA", "smtlib": "(set-logic QF_LIA)\n(check-sat)"},
        },
        {
            "operation_id": "smt.solve",
            "execution": {"status": "COMPLETED"},
            "output": {"conclusion": "UNKNOWN"},
        },
    )
    domain_failure = _tool_event(
        "math.run",
        {
            "operation_id": "sat.solve",
            "payload": {"cnf": {"variables": 0, "clauses": []}},
        },
        {
            "operation_id": "sat.solve",
            "execution": {"status": "ERROR"},
            "output": {
                "error": {
                    "code": "INVALID_SAT_REQUEST",
                    "stage": "request_validation",
                    "message": "CNF must contain at least one clause",
                }
            },
            "diagnostics": [
                {
                    "code": "INVALID_SAT_REQUEST",
                    "stage": "request_validation",
                    "message": "CNF must contain at least one clause",
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

    assert telemetry["operation_attempts"] == [
        {
            "operation_id": None,
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
            "operation_id": "sat.solve",
            "input": {"cnf": {"variables": 0, "clauses": []}},
            "successful": False,
            "terminal_status": "ERROR",
            "error_digest": "sha256:"
            + hashlib.sha256(
                canonicalize_json(
                    {
                        "item_error": None,
                        "response_error": None,
                        "output_error": {
                            "code": "INVALID_SAT_REQUEST",
                            "stage": "request_validation",
                            "message": "CNF must contain at least one clause",
                        },
                        "diagnostics": [
                            {
                                "code": "INVALID_SAT_REQUEST",
                                "stage": "request_validation",
                                "message": "CNF must contain at least one clause",
                            }
                        ],
                    }
                )
            ).hexdigest(),
            "request_validation_failure": True,
            "diagnostic_codes": ["INVALID_SAT_REQUEST"],
            "diagnostics": [
                {
                    "code": "INVALID_SAT_REQUEST",
                    "stage": "request_validation",
                }
            ],
        },
        {
            "operation_id": "smt.solve",
            "input": {"logic": "QF_LIA", "smtlib": "(set-logic QF_LIA)\n(check-sat)"},
            "successful": True,
        },
    ]
    assert telemetry["operation_attempt_ids"] == [
        "sat.solve",
        "smt.solve",
    ]
    assert telemetry["operation_ids"] == ["smt.solve"]


def test_agent_telemetry_records_empty_payload_and_exact_repeated_errors(
    tmp_path: Path,
) -> None:
    invalid = {
        "operation_id": "smt.solve",
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
        _tool_event("math.run", {"operation_id": "smt.solve", "payload": {}}, invalid),
        _tool_event("math.run", {"operation_id": "smt.solve", "payload": {}}, invalid),
        _tool_event(
            "math.run",
            {
                "operation_id": "sat.solve",
                "payload": {"cnf": {"variables": 0, "clauses": []}},
            },
            {
                "operation_id": "sat.solve",
                "execution": {"status": "ERROR"},
                "output": {
                    "error": {
                        "code": "INVALID_SAT_REQUEST",
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
        {"operation_id": "example.all_defaults", "payload": {}},
        {
            "operation_id": "example.all_defaults",
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
                {
                    "operation_id": "smt.solve",
                    "payload": {
                        "logic": "QF_LIA",
                        "smtlib": "(set-logic QF_LIA)\n(check-sat)",
                    },
                },
                {
                    "operation_id": "smt.solve",
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
            {"matches": [{"operation_id": "sat.cnf.materialize"}]},
        ),
        _tool_event(
            "math.find",
            {"request": {"op": "search", "query": "SAT materialization"}},
            {"matches": [{"operation_id": "sat.cnf.materialize"}]},
        ),
        _tool_event(
            "math.find",
            {
                "request": {
                    "op": "inspect",
                    "operation_id": "sat.cnf.materialize",
                }
            },
            {"operation": {"operation_id": "sat.cnf.materialize"}},
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
    assert telemetry["operation_describe_index_calls"] == 2
    assert telemetry["operation_describe_exact_calls"] == 1
    assert telemetry["mcp_model_visible_bytes"] > 0
    assert telemetry["mcp_logical_payload_observed_calls"] == 3


def test_agent_telemetry_ignores_non_string_mcp_status(tmp_path: Path) -> None:
    uri = "operation://catalog"
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
            "arguments": {"uri": "operation://catalog"},
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
        "operation_id": "graph.search.atlas",
        "execution": {"status": "COMPLETED"},
        "output": {"status": "FOUND", "graphs": [{"edges": [[0, 1]]}]},
    }
    event = {
        "type": "item.completed",
        "item": {
            "type": "mcp_tool_call",
            "tool": "math.run",
            "arguments": {
                "operation_id": "graph.search.atlas",
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
    assert telemetry["operation_invocations"][0]["output"] == canonical["output"]


def test_agent_telemetry_tracks_resource_link_follow_through(
    tmp_path: Path,
) -> None:
    def read_event(name: str) -> tuple[str, dict[str, object]]:
        uri = f"operation://{name}"
        return uri, _tool_event(
            "resources/read",
            {"uri": uri},
            {"name": name},
        )

    uri, first_read = read_event("a")
    first_result = first_read["item"]["result"]
    first_result["contents"] = first_result.pop("content")
    unnecessary_uri, second_read = read_event("b")
    malformed_uri = "operation://missing"

    def link_event(link_uri: str) -> dict[str, object]:
        content = [{"type": "resource_link", "uri": link_uri}]
        return {
            "type": "item.completed",
            "item": {
                "type": "mcp_tool_call",
                "tool": "math.run",
                "arguments": {"operation_id": "graph.search.atlas"},
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
                    {"not_found": True},
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


def test_agent_telemetry_handles_non_hashable_resource_tool_field(
    tmp_path: Path,
) -> None:
    uri = "operation://catalog"
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
        {"name": "catalog"},
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
