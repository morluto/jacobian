from __future__ import annotations

import hashlib
import json
from pathlib import Path

from jacobian.canonical import canonicalize_json
from jacobian.eval.telemetry import (
    parse_agent_transcript,
    parse_reasoning_protocol_trace,
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


def test_agent_telemetry_preserves_discovery_to_invocation_dataflow(
    tmp_path: Path,
) -> None:
    events = [
        _tool_event(
            "math.find",
            {
                "query": "find a graph counterexample",
                "domain": "graph",
                "mode": "EXPLORE",
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
            {"capability_id": "graph.search.atlas"},
            {
                "kind": "capability",
                "capability": {"capability_id": "graph.search.atlas"},
            },
        ),
        _tool_event(
            "math.run",
            {
                "capability_id": "graph.search.atlas",
                "mode": "EXPLORE",
                "payload": {"order": 7},
            },
            {
                "capability_id": "graph.search.atlas",
                "execution": {"status": "COMPLETED"},
                "output": {"status": "FOUND"},
                "artifact_uris": [],
                "assurance": {"level": "COMPUTED"},
                "completeness": {"status": "COMPLETE"},
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
            "mode": "EXPLORE",
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
            "mode": None,
            "capability_id": "graph.search.atlas",
            "match_ids": [],
        },
    ]
    assert telemetry["capability_attempt_ids"] == ["graph.search.atlas"]
    assert telemetry["capability_ids"] == ["graph.search.atlas"]


def test_agent_telemetry_counts_response_bytes_and_repeated_calls(
    tmp_path: Path,
) -> None:
    events = [
        _tool_event(
            "math.find",
            {},
            {"matches": [{"capability_id": "sat.cnf.materialize"}]},
        ),
        _tool_event(
            "math.find",
            {},
            {"matches": [{"capability_id": "sat.cnf.materialize"}]},
        ),
        _tool_event(
            "math.find",
            {"capability_id": "sat.cnf.materialize"},
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


def test_agent_telemetry_reports_reasoning_protocol_without_summary_text(
    tmp_path: Path,
) -> None:
    run_id = "00000000-0000-4000-8000-000000000000"
    call_id = "11111111-1111-4111-8111-111111111111"
    events = [
        _tool_event(
            "reasoning.write",
            {"phase": "PLAN", "summary": "private plan marker"},
            {"run_id": run_id},
        ),
        _tool_event(
            "reasoning.write",
            {
                "phase": "BEFORE_TOOL",
                "summary": "private before marker",
                "run_id": run_id,
            },
            {"run_id": run_id, "call_id": call_id},
        ),
        _tool_event(
            "math.run",
            {
                "capability_id": "integer.compute.gcd",
                "payload": {},
                "reasoning_run_id": run_id,
                "reasoning_call_id": call_id,
            },
            {"execution": {"status": "ERROR"}},
        ),
    ]
    transcript = tmp_path / "reasoning.jsonl"
    transcript.write_text(
        "\n".join(json.dumps(event) for event in events) + "\n",
        encoding="utf-8",
    )

    telemetry = parse_agent_transcript(transcript)

    assert telemetry["reasoning_protocol"] == {
        "status": "INCOMPLETE",
        "plan_count": 1,
        "before_tool_count": 1,
        "after_tool_count": 0,
        "final_count": 0,
        "run_count": 1,
        "bound_invoke_count": 1,
        "missing_after_tool_count": 1,
        "pending_call_count": 1,
        "unavailable_after_tool_count": 0,
        "reported_actual_mismatch_count": 0,
        "summary_characters": len("private plan markerprivate before marker"),
    }
    assert "private plan marker" not in json.dumps(telemetry)


def test_reasoning_protocol_parser_supports_harbor_atif(tmp_path: Path) -> None:
    run_id = "00000000-0000-4000-8000-000000000000"
    trajectory = {
        "schema_version": "ATIF-v1.7",
        "steps": [
            {
                "tool_calls": [
                    {
                        "tool_call_id": "plan",
                        "function_name": "mcp__jacobian__reasoning_write",
                        "arguments": {"phase": "PLAN", "summary": "private"},
                    }
                ],
                "observation": {
                    "results": [
                        {
                            "source_call_id": "plan",
                            "content": json.dumps({"run_id": run_id}),
                        }
                    ]
                },
            },
            {
                "tool_calls": [
                    {
                        "tool_call_id": "final",
                        "function_name": "reasoning.write",
                        "arguments": {
                            "phase": "FINAL",
                            "summary": "private final",
                            "run_id": run_id,
                        },
                    }
                ],
                "observation": {
                    "results": [
                        {
                            "source_call_id": "final",
                            "content": json.dumps(
                                {"run_id": run_id, "state": "FINALIZED"}
                            ),
                        }
                    ]
                },
            },
        ],
    }
    path = tmp_path / "trajectory.json"
    path.write_text(json.dumps(trajectory), encoding="utf-8")

    protocol = parse_reasoning_protocol_trace(path)

    assert protocol["status"] == "COMPLETE"
    assert protocol["plan_count"] == 1
    assert protocol["final_count"] == 1
    assert "private" not in json.dumps(protocol)


def test_agent_telemetry_separates_wire_model_and_logical_invocation_bytes(
    tmp_path: Path,
) -> None:
    canonical = {
        "capability_id": "graph.search.atlas",
        "execution": {"status": "COMPLETED"},
        "output": {"status": "FOUND", "graphs": [{"edges": [[0, 1]]}]},
        "artifact_uris": ["artifact://sha256/" + ("a" * 64)],
        "assurance": {"level": "COMPUTED"},
        "completeness": {"status": "COMPLETE"},
    }
    event = {
        "type": "item.completed",
        "item": {
            "type": "mcp_tool_call",
            "tool": "math.run",
            "arguments": {
                "capability_id": "graph.search.atlas",
                "mode": "EXPLORE",
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
