"""Measure model-visible tool-catalog cost in Codex ATIF trajectories."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from pathlib import Path
from typing import Any

from benchmarks.tooling.errors import HarborSuiteError

_JACOBIAN_FIND = "tools.mcp__jacobian__math_find("
_JACOBIAN_RUN = "tools.mcp__jacobian__math_run("


def _read_trajectory(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise HarborSuiteError(f"unable to read ATIF trajectory {path}: {exc}") from exc
    if not isinstance(value, dict) or not isinstance(value.get("steps"), list):
        raise HarborSuiteError(
            f"invalid ATIF trajectory {path}: steps must be an array"
        )
    return value


def _visible_bytes(value: object) -> int:
    if isinstance(value, str):
        encoded = value.encode("utf-8")
    else:
        encoded = json.dumps(
            value, ensure_ascii=False, separators=(",", ":"), sort_keys=True
        ).encode("utf-8")
    return len(encoded)


def _integer(value: object) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _number(value: object) -> int | float | None:
    return (
        value
        if isinstance(value, (int, float)) and not isinstance(value, bool)
        else None
    )


def _visible_results(step: dict[str, Any]) -> dict[str, int]:
    observation = step.get("observation")
    results = observation.get("results", []) if isinstance(observation, dict) else []
    visible_by_call: dict[str, int] = {}
    if not isinstance(results, list):
        return visible_by_call
    for result in results:
        if not isinstance(result, dict):
            continue
        source_call_id = result.get("source_call_id")
        if isinstance(source_call_id, str) and "content" in result:
            visible_by_call[source_call_id] = _visible_bytes(result["content"])
    return visible_by_call


def _analyze_step(step: object) -> tuple[int, int, int, int, int, int, int]:
    if not isinstance(step, dict):
        return (0, 0, 0, 0, 0, 0, 0)
    visible_by_call = _visible_results(step)
    tool_calls = step.get("tool_calls", [])
    if not isinstance(tool_calls, list):
        return (0, 0, 0, 0, 0, 0, 0)
    scan_count = scan_bytes = unbound_scan_count = tool_output_bytes = 0
    jacobian_output_bytes = 0
    direct_find_references = direct_run_references = 0
    for call in tool_calls:
        if not isinstance(call, dict):
            continue
        call_id = call.get("tool_call_id")
        visible = visible_by_call.get(call_id, 0) if isinstance(call_id, str) else 0
        tool_output_bytes += visible
        arguments = call.get("arguments")
        source = arguments.get("input") if isinstance(arguments, dict) else None
        if call.get("function_name") != "exec" or not isinstance(source, str):
            continue
        find_references = source.count(_JACOBIAN_FIND)
        run_references = source.count(_JACOBIAN_RUN)
        direct_find_references += find_references
        direct_run_references += run_references
        if find_references or run_references:
            jacobian_output_bytes += visible
        if "ALL_TOOLS" in source:
            scan_count += 1
            scan_bytes += visible
            if not isinstance(call_id, str) or call_id not in visible_by_call:
                unbound_scan_count += 1
    return (
        scan_count,
        scan_bytes,
        unbound_scan_count,
        tool_output_bytes,
        jacobian_output_bytes,
        direct_find_references,
        direct_run_references,
    )


def analyze_trajectory(path: Path) -> dict[str, Any]:
    """Extract directory projection and token-cost facts from one ATIF trace."""

    trajectory = _read_trajectory(path)
    scan_count = 0
    scan_bytes = 0
    unbound_scan_count = 0
    tool_output_bytes = 0
    jacobian_output_bytes = 0
    direct_find_references = 0
    direct_run_references = 0

    for step in trajectory["steps"]:
        step_counts = _analyze_step(step)
        scan_count += step_counts[0]
        scan_bytes += step_counts[1]
        unbound_scan_count += step_counts[2]
        tool_output_bytes += step_counts[3]
        jacobian_output_bytes += step_counts[4]
        direct_find_references += step_counts[5]
        direct_run_references += step_counts[6]

    metrics = trajectory.get("final_metrics")
    metrics = metrics if isinstance(metrics, dict) else {}
    prompt_tokens = _integer(metrics.get("total_prompt_tokens"))
    cached_tokens = _integer(metrics.get("total_cached_tokens"))
    uncached_tokens = (
        max(0, prompt_tokens - cached_tokens)
        if prompt_tokens is not None and cached_tokens is not None
        else None
    )
    return {
        "trajectory": str(path),
        "trajectory_sha256": "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest(),
        "agent": trajectory.get("agent"),
        "all_tools_scan_count": scan_count,
        "all_tools_model_visible_bytes": scan_bytes,
        "all_tools_unbound_observation_count": unbound_scan_count,
        "tool_model_visible_bytes": tool_output_bytes,
        "direct_jacobian_find_run_model_visible_bytes": jacobian_output_bytes,
        "direct_jacobian_find_references": direct_find_references,
        "direct_jacobian_run_references": direct_run_references,
        "prompt_tokens": prompt_tokens,
        "cached_prompt_tokens": cached_tokens,
        "uncached_prompt_tokens": uncached_tokens,
        "completion_tokens": _integer(metrics.get("total_completion_tokens")),
        "cost_usd": _number(metrics.get("total_cost_usd")),
    }


def _median(trials: list[dict[str, Any]], field: str) -> int | float | None:
    values = [
        trial[field] for trial in trials if isinstance(trial.get(field), (int, float))
    ]
    return statistics.median(values) if values else None


def build_report(paths: list[Path], *, label: str) -> dict[str, Any]:
    """Build one digest-bound observation report for a set of trajectories."""

    if not paths:
        raise HarborSuiteError("at least one ATIF trajectory is required")
    trials = [analyze_trajectory(path) for path in paths]
    return {
        "schema_version": "1",
        "label": label,
        "trial_count": len(trials),
        "summary": {
            "jacobian_invocation_trials": sum(
                int(
                    trial["direct_jacobian_find_references"] > 0
                    or trial["direct_jacobian_run_references"] > 0
                )
                for trial in trials
            ),
            "jacobian_execution_trials": sum(
                int(trial["direct_jacobian_run_references"] > 0) for trial in trials
            ),
            "jacobian_unused_trials": sum(
                int(
                    trial["direct_jacobian_find_references"] == 0
                    and trial["direct_jacobian_run_references"] == 0
                )
                for trial in trials
            ),
            "direct_jacobian_find_references": sum(
                int(trial["direct_jacobian_find_references"]) for trial in trials
            ),
            "direct_jacobian_run_references": sum(
                int(trial["direct_jacobian_run_references"]) for trial in trials
            ),
            "direct_jacobian_find_run_model_visible_bytes": sum(
                int(trial["direct_jacobian_find_run_model_visible_bytes"])
                for trial in trials
            ),
            "all_tools_scan_trials": sum(
                int(trial["all_tools_scan_count"] > 0) for trial in trials
            ),
            "all_tools_scan_count": sum(
                int(trial["all_tools_scan_count"]) for trial in trials
            ),
            "all_tools_model_visible_bytes": sum(
                int(trial["all_tools_model_visible_bytes"]) for trial in trials
            ),
            "all_tools_unbound_observation_count": sum(
                int(trial["all_tools_unbound_observation_count"]) for trial in trials
            ),
            "median_prompt_tokens": _median(trials, "prompt_tokens"),
            "median_cached_prompt_tokens": _median(trials, "cached_prompt_tokens"),
            "median_uncached_prompt_tokens": _median(trials, "uncached_prompt_tokens"),
            "median_completion_tokens": _median(trials, "completion_tokens"),
            "median_cost_usd": _median(trials, "cost_usd"),
        },
        "trials": trials,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trajectories", nargs="+", type=Path)
    parser.add_argument("--label", default="observation")
    parser.add_argument("--output", type=Path)
    return parser


def main() -> int:
    args = _parser().parse_args()
    try:
        report = build_report(args.trajectories, label=args.label)
    except HarborSuiteError as exc:
        raise SystemExit(str(exc)) from exc
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output is None:
        print(rendered, end="")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["analyze_trajectory", "build_report"]
