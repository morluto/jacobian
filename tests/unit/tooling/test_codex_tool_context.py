from __future__ import annotations

import json
from pathlib import Path

import pytest
from benchmarks.tooling.codex_tool_context import analyze_trajectory, build_report
from benchmarks.tooling.errors import HarborSuiteError


def _write_trajectory(
    path: Path,
    *,
    source: str,
    content: object,
    prompt_tokens: int = 100,
    cached_tokens: int = 60,
) -> None:
    path.write_text(
        json.dumps(
            {
                "schema_version": "ATIF-v1.7",
                "agent": {"name": "codex", "version": "test"},
                "final_metrics": {
                    "total_prompt_tokens": prompt_tokens,
                    "total_cached_tokens": cached_tokens,
                    "total_completion_tokens": 7,
                    "total_cost_usd": 0.125,
                },
                "steps": [
                    {
                        "source": "agent",
                        "tool_calls": [
                            {
                                "tool_call_id": "call-1",
                                "function_name": "exec",
                                "arguments": {"input": source},
                            }
                        ],
                        "observation": {
                            "results": [
                                {"source_call_id": "call-1", "content": content},
                                {"source_call_id": "other-call", "content": "ignored"},
                            ]
                        },
                    }
                ],
            }
        ),
        encoding="utf-8",
    )


def test_analyze_trajectory_measures_all_tools_projection(tmp_path: Path) -> None:
    path = tmp_path / "trajectory.json"
    _write_trajectory(
        path,
        source="const xs = ALL_TOOLS.filter(x => x.name.includes('jacobian')); text(xs);",
        content="catalog output",
    )

    result = analyze_trajectory(path)

    assert result["all_tools_scan_count"] == 1
    assert result["all_tools_model_visible_bytes"] == len(b"catalog output")
    assert result["all_tools_unbound_observation_count"] == 0
    assert result["tool_model_visible_bytes"] == len(b"catalog output")
    assert result["direct_jacobian_find_run_model_visible_bytes"] == 0
    assert result["direct_jacobian_find_references"] == 0
    assert result["uncached_prompt_tokens"] == 40


def test_analyze_trajectory_counts_direct_nested_calls_without_scan(
    tmp_path: Path,
) -> None:
    path = tmp_path / "trajectory.json"
    _write_trajectory(
        path,
        source=(
            "const a = await tools.mcp__jacobian__math_find({query:'gcd'});"
            "const b = await tools.mcp__jacobian__math_run({capability_id:'x'});"
            "text(a.structuredContent ?? a); text(b.structuredContent ?? b);"
        ),
        content={"kind": "result"},
    )

    result = analyze_trajectory(path)

    assert result["all_tools_scan_count"] == 0
    assert result["all_tools_model_visible_bytes"] == 0
    assert result["direct_jacobian_find_references"] == 1
    assert result["direct_jacobian_run_references"] == 1
    assert result["direct_jacobian_find_run_model_visible_bytes"] == len(
        json.dumps({"kind": "result"}, separators=(",", ":"), sort_keys=True).encode()
    )


def test_build_report_aggregates_trials_with_medians(tmp_path: Path) -> None:
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    _write_trajectory(first, source="text(ALL_TOOLS);", content="one", prompt_tokens=80)
    _write_trajectory(
        second, source="text('bounded');", content="two", prompt_tokens=120
    )

    report = build_report([first, second], label="paired-control")

    assert report["label"] == "paired-control"
    assert report["trial_count"] == 2
    assert report["summary"] == {
        "jacobian_invocation_trials": 0,
        "jacobian_execution_trials": 0,
        "jacobian_unused_trials": 2,
        "direct_jacobian_find_references": 0,
        "direct_jacobian_run_references": 0,
        "direct_jacobian_find_run_model_visible_bytes": 0,
        "all_tools_scan_trials": 1,
        "all_tools_scan_count": 1,
        "all_tools_model_visible_bytes": 3,
        "all_tools_unbound_observation_count": 0,
        "median_prompt_tokens": 100,
        "median_cached_prompt_tokens": 60,
        "median_uncached_prompt_tokens": 40,
        "median_completion_tokens": 7,
        "median_cost_usd": 0.125,
    }


def test_build_report_rejects_empty_input() -> None:
    with pytest.raises(HarborSuiteError, match="at least one"):
        build_report([], label="empty")


def test_analyze_trajectory_reports_unbound_scan_observation(tmp_path: Path) -> None:
    path = tmp_path / "trajectory.json"
    _write_trajectory(path, source="text(ALL_TOOLS);", content="catalog")
    value = json.loads(path.read_text(encoding="utf-8"))
    value["steps"][0]["observation"]["results"] = []
    path.write_text(json.dumps(value), encoding="utf-8")

    result = analyze_trajectory(path)

    assert result["all_tools_scan_count"] == 1
    assert result["all_tools_model_visible_bytes"] == 0
    assert result["all_tools_unbound_observation_count"] == 1
